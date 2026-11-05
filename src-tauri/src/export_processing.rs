use std::borrow::Cow;
use std::collections::HashMap;
use std::fs;
use std::io::Cursor;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use image::codecs::jpeg::JpegEncoder;
use image::codecs::png::PngEncoder;
use image::{
    DynamicImage, GenericImageView, GrayImage, ImageBuffer, ImageEncoder, ImageFormat, Luma,
    imageops,
};
use jxl_encoder::{
    LosslessConfig, LossyConfig, PixelLayout,
    api::{calibrated_jxl_quality, quality_to_distance},
};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tauri::Emitter;
use tauri::Manager;

use crate::AppState;
use crate::exif_processing;
use crate::file_management::{
    generate_filename_from_template, parse_virtual_path, read_file_mapped,
};
use crate::formats::is_raw_file;
use crate::image_loader::{
    composite_patches_on_image, load_and_composite, load_base_image_from_bytes,
};
use crate::image_processing::{
    AllAdjustments, Crop, GpuContext, RenderOutputPrecision, RenderRequest, downscale_f32_image,
    get_all_adjustments_from_json, get_or_init_gpu_context, process_and_get_dynamic_image,
    process_and_get_dynamic_image_with_precision, resolve_tonemapper_override_from_handle,
};
use crate::lut_processing::{
    convert_image_to_cube_lut, generate_identity_lut_image, get_or_load_lut,
};
use crate::mask_generation::{MaskDefinition, generate_mask_bitmap};

use crate::cache_utils::calculate_full_job_hash;
use crate::white_balance::as_shot_white_balance;
use crate::{
    apply_all_transformations, hydrate_adjustments, load_settings, resolve_warped_image_for_masks,
};

#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq, Default)]
#[serde(try_from = "u8", into = "u8")]
pub enum TiffBitDepth {
    Eight = 8,
    #[default]
    Sixteen = 16,
}

impl TryFrom<u8> for TiffBitDepth {
    type Error = String;

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value {
            8 => Ok(Self::Eight),
            16 => Ok(Self::Sixteen),
            _ => Err(format!(
                "Invalid TIFF bit depth '{}'; expected 8 or 16.",
                value
            )),
        }
    }
}

impl From<TiffBitDepth> for u8 {
    fn from(value: TiffBitDepth) -> Self {
        value as u8
    }
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(rename_all = "camelCase")]
pub enum ResizeMode {
    LongEdge,
    ShortEdge,
    Width,
    Height,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(rename_all = "camelCase")]
pub struct ResizeOptions {
    pub mode: ResizeMode,
    pub value: u32,
    pub dont_enlarge: bool,
}

#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq, Default)]
#[serde(rename_all = "camelCase")]
#[allow(clippy::enum_variant_names)]
pub enum BorderBasis {
    #[default]
    LongEdge,
    ShortEdge,
    EachEdge,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(rename_all = "camelCase")]
pub struct BorderOptions {
    #[serde(default)]
    pub basis: BorderBasis,
    pub horizontal_percent: f32,
    pub vertical_percent: f32,
    pub color: String,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(rename_all = "camelCase")]
pub struct PadOptions {
    pub ratio_width: f32,
    pub ratio_height: f32,
    pub color: String,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(rename_all = "camelCase")]
pub struct ExportSettings {
    pub jpeg_quality: u8,
    #[serde(default)]
    pub tiff_bit_depth: TiffBitDepth,
    pub resize: Option<ResizeOptions>,
    #[serde(default)]
    pub border: Option<BorderOptions>,
    #[serde(default)]
    pub pad: Option<PadOptions>,
    pub keep_metadata: bool,
    #[serde(default)]
    pub preserve_timestamps: bool,
    pub strip_gps: bool,
    pub filename_template: Option<String>,
    pub watermark: Option<WatermarkSettings>,
    #[serde(default)]
    pub export_masks: bool,
    #[serde(default)]
    pub preserve_folders: bool,
    #[serde(default)]
    pub destination_type: Option<String>,
    #[serde(default)]
    pub subfolder: Option<String>,
    /// 前端按路径绘制的拍摄信息条 PNG（导出时拼到图像底部）
    #[serde(default)]
    pub frame_png_by_path: Option<HashMap<String, Vec<u8>>>,
}

#[derive(Clone)]
pub(crate) enum ExportAdjustmentsMode {
    UseSidecars {
        active_path: Option<String>,
        active_adjustments: Option<Value>,
    },
    GlobalOverride(Value),
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(rename_all = "camelCase")]
pub enum WatermarkAnchor {
    TopLeft,
    TopCenter,
    TopRight,
    CenterLeft,
    Center,
    CenterRight,
    BottomLeft,
    BottomCenter,
    BottomRight,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(rename_all = "camelCase")]
pub struct WatermarkSettings {
    pub path: String,
    pub anchor: WatermarkAnchor,
    pub scale: f32,
    pub spacing: f32,
    pub opacity: f32,
}

fn apply_watermark(
    base_image: &mut DynamicImage,
    watermark_settings: &WatermarkSettings,
) -> Result<(), String> {
    let watermark_img = image::open(&watermark_settings.path)
        .map_err(|e| format!("Failed to open watermark image: {}", e))?;

    let (base_w, base_h) = base_image.dimensions();
    let base_min_dim = base_w.min(base_h) as f32;

    let watermark_scale_factor =
        (base_min_dim * (watermark_settings.scale / 100.0)) / watermark_img.width().max(1) as f32;
    let new_wm_w = (watermark_img.width() as f32 * watermark_scale_factor).round() as u32;
    let new_wm_h = (watermark_img.height() as f32 * watermark_scale_factor).round() as u32;

    if new_wm_w == 0 || new_wm_h == 0 {
        return Ok(());
    }

    let scaled_watermark =
        watermark_img.resize_exact(new_wm_w, new_wm_h, image::imageops::FilterType::Lanczos3);
    let opacity_factor = (watermark_settings.opacity / 100.0).clamp(0.0, 1.0);

    let spacing_pixels = (base_min_dim * (watermark_settings.spacing / 100.0)) as i64;
    let (wm_w, wm_h) = scaled_watermark.dimensions();

    let x = match watermark_settings.anchor {
        WatermarkAnchor::TopLeft | WatermarkAnchor::CenterLeft | WatermarkAnchor::BottomLeft => {
            spacing_pixels
        }
        WatermarkAnchor::TopCenter | WatermarkAnchor::Center | WatermarkAnchor::BottomCenter => {
            (base_w as i64 - wm_w as i64) / 2
        }
        WatermarkAnchor::TopRight | WatermarkAnchor::CenterRight | WatermarkAnchor::BottomRight => {
            base_w as i64 - wm_w as i64 - spacing_pixels
        }
    };

    let y = match watermark_settings.anchor {
        WatermarkAnchor::TopLeft | WatermarkAnchor::TopCenter | WatermarkAnchor::TopRight => {
            spacing_pixels
        }
        WatermarkAnchor::CenterLeft | WatermarkAnchor::Center | WatermarkAnchor::CenterRight => {
            (base_h as i64 - wm_h as i64) / 2
        }
        WatermarkAnchor::BottomLeft
        | WatermarkAnchor::BottomCenter
        | WatermarkAnchor::BottomRight => base_h as i64 - wm_h as i64 - spacing_pixels,
    };

    if matches!(
        base_image,
        DynamicImage::ImageRgb16(_) | DynamicImage::ImageRgba16(_)
    ) {
        let mut base_rgba = base_image.to_rgba16();
        let mut watermark_rgba = scaled_watermark.to_rgba16();
        for pixel in watermark_rgba.pixels_mut() {
            pixel[3] = (pixel[3] as f32 * opacity_factor).round() as u16;
        }
        image::imageops::overlay(&mut base_rgba, &watermark_rgba, x, y);
        *base_image = DynamicImage::ImageRgba16(base_rgba);
    } else if matches!(
        base_image,
        DynamicImage::ImageRgb32F(_) | DynamicImage::ImageRgba32F(_)
    ) {
        let mut base_rgba = base_image.to_rgba32f();
        let mut watermark_rgba = scaled_watermark.to_rgba32f();
        for pixel in watermark_rgba.pixels_mut() {
            pixel[3] *= opacity_factor;
        }
        image::imageops::overlay(&mut base_rgba, &watermark_rgba, x, y);
        *base_image = DynamicImage::ImageRgba32F(base_rgba);
    } else {
        let mut base_rgba = base_image.to_rgba8();
        let mut watermark_rgba = scaled_watermark.to_rgba8();
        for pixel in watermark_rgba.pixels_mut() {
            pixel[3] = (pixel[3] as f32 * opacity_factor).round() as u8;
        }
        image::imageops::overlay(&mut base_rgba, &watermark_rgba, x, y);
        *base_image = DynamicImage::ImageRgba8(base_rgba);
    }

    Ok(())
}

fn calculate_resize_target(
    current_w: u32,
    current_h: u32,
    resize_opts: &ResizeOptions,
) -> (u32, u32) {
    if resize_opts.dont_enlarge {
        let exceeds = match resize_opts.mode {
            ResizeMode::LongEdge => current_w.max(current_h) > resize_opts.value,
            ResizeMode::ShortEdge => current_w.min(current_h) > resize_opts.value,
            ResizeMode::Width => current_w > resize_opts.value,
            ResizeMode::Height => current_h > resize_opts.value,
        };
        if !exceeds {
            return (current_w, current_h);
        }
    }

    let fix_width = match resize_opts.mode {
        ResizeMode::LongEdge => current_w >= current_h,
        ResizeMode::ShortEdge => current_w <= current_h,
        ResizeMode::Width => true,
        ResizeMode::Height => false,
    };

    let value = resize_opts.value;
    if fix_width {
        let h = (value as f32 * (current_h as f32 / current_w as f32)).round() as u32;
        (value, h)
    } else {
        let w = (value as f32 * (current_w as f32 / current_h as f32)).round() as u32;
        (w, value)
    }
}

fn relative_dir_is_safe(rel_dir: &Path) -> bool {
    rel_dir.components().all(|component| {
        matches!(
            component,
            std::path::Component::Normal(_) | std::path::Component::CurDir
        )
    })
}

#[cfg(windows)]
fn component_matches(left: std::path::Component<'_>, right: std::path::Component<'_>) -> bool {
    left.as_os_str()
        .to_string_lossy()
        .eq_ignore_ascii_case(&right.as_os_str().to_string_lossy())
}

#[cfg(not(windows))]
fn component_matches(left: std::path::Component<'_>, right: std::path::Component<'_>) -> bool {
    left == right
}

fn strip_prefix_preserving_source_case(source_path: &Path, base_path: &Path) -> Option<PathBuf> {
    let source_components: Vec<_> = source_path.components().collect();
    let base_components: Vec<_> = base_path.components().collect();

    if base_components.len() > source_components.len() {
        return None;
    }

    if !source_components
        .iter()
        .zip(base_components.iter())
        .all(|(source, base)| component_matches(*source, *base))
    {
        return None;
    }

    Some(source_components[base_components.len()..].iter().collect())
}

fn relative_export_dir_for_preserved_folders(
    source_path: &Path,
    base_origin_folders: &[String],
) -> Option<PathBuf> {
    base_origin_folders
        .iter()
        .filter_map(|base| {
            let base_path = Path::new(base);
            strip_prefix_preserving_source_case(source_path, base_path)
                .map(|rel_path| (base_path.components().count(), rel_path))
        })
        .max_by_key(|(component_count, _)| *component_count)
        .and_then(|(_, rel_path)| {
            let rel_dir = rel_path.parent().unwrap_or_else(|| Path::new(""));
            if relative_dir_is_safe(rel_dir) {
                Some(rel_dir.to_path_buf())
            } else {
                None
            }
        })
}

const PAD_RATIO_EPSILON: f64 = 0.001;
const MAX_PAD_DIMENSION: u32 = 65_535;
const MAX_BORDER_PERCENT: f64 = 100.0;

#[derive(Debug, Clone, Copy)]
pub(crate) struct ExportGeometry {
    pub canvas_w: u32,
    pub canvas_h: u32,
    pub photo_w: u32,
    pub photo_h: u32,
    pub offset_x: u32,
    pub offset_y: u32,
    pub border_x: u32,
    pub border_y: u32,
}

impl ExportGeometry {
    fn has_border(&self) -> bool {
        self.border_x > 0 || self.border_y > 0
    }

    fn is_identity(&self) -> bool {
        self.canvas_w == self.photo_w && self.canvas_h == self.photo_h
    }
}

fn parse_hex_color(hex: &str) -> Option<[u8; 3]> {
    let raw = hex.trim().trim_start_matches('#');
    let normalized: String = match raw.len() {
        3 => raw.chars().flat_map(|c| [c, c]).collect(),
        6 => raw.to_string(),
        8 => raw[..6].to_string(),
        _ => return None,
    };
    Some([
        u8::from_str_radix(&normalized[0..2], 16).ok()?,
        u8::from_str_radix(&normalized[2..4], 16).ok()?,
        u8::from_str_radix(&normalized[4..6], 16).ok()?,
    ])
}

fn fill_color(hex: Option<&str>) -> [u8; 3] {
    hex.and_then(parse_hex_color).unwrap_or([0, 0, 0])
}

fn resize_target(width: u32, height: u32, resize_opts: Option<&ResizeOptions>) -> (u32, u32) {
    match resize_opts {
        Some(opts) => calculate_resize_target(width, height, opts),
        None => (width, height),
    }
}

fn bordered_size(width: u32, height: u32, border_x: u32, border_y: u32) -> (u32, u32) {
    (
        width.saturating_add(border_x.saturating_mul(2)),
        height.saturating_add(border_y.saturating_mul(2)),
    )
}

fn exceeds_max_dimension(label: &str, width: u32, height: u32) -> bool {
    let exceeds = width > MAX_PAD_DIMENSION || height > MAX_PAD_DIMENSION;
    if exceeds {
        log::warn!(
            "{} {}x{} exceeds the maximum supported dimension; skipping.",
            label,
            width,
            height
        );
    }
    exceeds
}

fn calculate_pad_target(current_w: u32, current_h: u32, pad: &PadOptions) -> Option<(u32, u32)> {
    let ratio_w = pad.ratio_width as f64;
    let ratio_h = pad.ratio_height as f64;
    if !ratio_w.is_finite() || !ratio_h.is_finite() || ratio_w <= 0.0 || ratio_h <= 0.0 {
        return None;
    }

    let target_ratio = ratio_w / ratio_h;
    let original_ratio = current_w as f64 / current_h as f64;
    if (original_ratio - target_ratio).abs() < PAD_RATIO_EPSILON {
        return None;
    }

    let (new_w, new_h) = if original_ratio > target_ratio {
        (current_w, (current_w as f64 / target_ratio).round() as u32)
    } else {
        ((current_h as f64 * target_ratio).round() as u32, current_h)
    };

    let new_w = new_w.max(current_w);
    let new_h = new_h.max(current_h);

    if exceeds_max_dimension("Pad target", new_w, new_h)
        || (new_w == current_w && new_h == current_h)
    {
        return None;
    }

    Some((new_w, new_h))
}

fn calculate_border_thickness(
    src_w: u32,
    src_h: u32,
    border: &BorderOptions,
) -> Option<(u32, u32)> {
    let horizontal = border.horizontal_percent as f64;
    let vertical = border.vertical_percent as f64;
    if !horizontal.is_finite() || !vertical.is_finite() {
        return None;
    }

    if horizontal < 0.0
        || vertical < 0.0
        || horizontal > MAX_BORDER_PERCENT
        || vertical > MAX_BORDER_PERCENT
    {
        log::warn!(
            "Border percentages {}/{} are outside 0..={}; skipping border.",
            horizontal,
            vertical,
            MAX_BORDER_PERCENT
        );
        return None;
    }

    let (basis_x, basis_y) = match border.basis {
        BorderBasis::LongEdge => {
            let edge = src_w.max(src_h) as f64;
            (edge, edge)
        }
        BorderBasis::ShortEdge => {
            let edge = src_w.min(src_h) as f64;
            (edge, edge)
        }
        BorderBasis::EachEdge => (src_w as f64, src_h as f64),
    };

    let border_x = (basis_x * horizontal / 100.0).round() as u32;
    let border_y = (basis_y * vertical / 100.0).round() as u32;

    if border_x == 0 && border_y == 0 {
        return None;
    }

    let (bordered_w, bordered_h) = bordered_size(src_w, src_h, border_x, border_y);
    if exceeds_max_dimension("Bordered size", bordered_w, bordered_h) {
        return None;
    }

    Some((border_x, border_y))
}

fn compute_export_geometry(
    src_w: u32,
    src_h: u32,
    settings: &ExportSettings,
) -> Option<ExportGeometry> {
    if src_w == 0 || src_h == 0 {
        return None;
    }

    let border = settings
        .border
        .as_ref()
        .and_then(|border| calculate_border_thickness(src_w, src_h, border));
    let (border_src_x, border_src_y) = border.unwrap_or((0, 0));
    let (bordered_w, bordered_h) = bordered_size(src_w, src_h, border_src_x, border_src_y);

    let pad_target = settings
        .pad
        .as_ref()
        .and_then(|pad| calculate_pad_target(bordered_w, bordered_h, pad));

    if border.is_none() && pad_target.is_none() {
        return None;
    }

    let (canvas_w, canvas_h) = pad_target.unwrap_or((bordered_w, bordered_h));

    Some(compute_fused_geometry(
        src_w,
        src_h,
        border_src_x,
        border_src_y,
        canvas_w,
        canvas_h,
        settings.resize.as_ref(),
    ))
}

#[allow(clippy::too_many_arguments)]
fn compute_fused_geometry(
    src_w: u32,
    src_h: u32,
    border_src_x: u32,
    border_src_y: u32,
    canvas_w: u32,
    canvas_h: u32,
    resize_opts: Option<&ResizeOptions>,
) -> ExportGeometry {
    let (final_w, final_h) = resize_target(canvas_w, canvas_h, resize_opts);
    let (final_w, final_h) = (final_w.max(1), final_h.max(1));

    let photo_w =
        ((src_w as f64 * final_w as f64 / canvas_w as f64).round() as u32).clamp(1, final_w);
    let photo_h =
        ((src_h as f64 * final_h as f64 / canvas_h as f64).round() as u32).clamp(1, final_h);

    let border_x = ((border_src_x as f64 * final_w as f64 / canvas_w as f64).round() as u32)
        .min(final_w.saturating_sub(photo_w) / 2);
    let border_y = ((border_src_y as f64 * final_h as f64 / canvas_h as f64).round() as u32)
        .min(final_h.saturating_sub(photo_h) / 2);

    ExportGeometry {
        canvas_w: final_w,
        canvas_h: final_h,
        photo_w,
        photo_h,
        offset_x: (final_w - photo_w) / 2,
        offset_y: (final_h - photo_h) / 2,
        border_x,
        border_y,
    }
}

fn compute_export_output_size(src_w: u32, src_h: u32, settings: &ExportSettings) -> (u32, u32) {
    match compute_export_geometry(src_w, src_h, settings) {
        Some(geometry) => (geometry.canvas_w, geometry.canvas_h),
        None => resize_target(src_w, src_h, settings.resize.as_ref()),
    }
}

trait FillChannel: image::Primitive {
    fn from_u8(value: u8) -> Self;
}

impl FillChannel for u8 {
    fn from_u8(value: u8) -> Self {
        value
    }
}

impl FillChannel for u16 {
    fn from_u8(value: u8) -> Self {
        value as u16 * 257
    }
}

impl FillChannel for f32 {
    fn from_u8(value: u8) -> Self {
        value as f32 / 255.0
    }
}

trait SolidFill: image::Pixel + 'static {
    fn from_rgb8(color: [u8; 3]) -> Self;
}

impl<T: FillChannel> SolidFill for image::Rgb<T>
where
    image::Rgb<T>: image::Pixel<Subpixel = T> + 'static,
{
    fn from_rgb8([r, g, b]: [u8; 3]) -> Self {
        image::Rgb([T::from_u8(r), T::from_u8(g), T::from_u8(b)])
    }
}

impl<T: FillChannel> SolidFill for image::Rgba<T>
where
    image::Rgba<T>: image::Pixel<Subpixel = T> + 'static,
{
    fn from_rgb8([r, g, b]: [u8; 3]) -> Self {
        image::Rgba([
            T::from_u8(r),
            T::from_u8(g),
            T::from_u8(b),
            T::DEFAULT_MAX_VALUE,
        ])
    }
}

fn fill_border_frame<P>(
    canvas: &mut ImageBuffer<P, Vec<P::Subpixel>>,
    geometry: &ExportGeometry,
    fill: P,
) where
    P: image::Pixel + 'static,
    P::Subpixel: 'static,
{
    let box_x = geometry.offset_x.saturating_sub(geometry.border_x);
    let box_y = geometry.offset_y.saturating_sub(geometry.border_y);
    let box_w = geometry
        .photo_w
        .saturating_add(geometry.border_x.saturating_mul(2));
    let below_photo = geometry.offset_y.saturating_add(geometry.photo_h);
    let right_of_photo = geometry.offset_x.saturating_add(geometry.photo_w);

    let strips = [
        (box_x, box_y, box_w, geometry.border_y),
        (box_x, below_photo, box_w, geometry.border_y),
        (
            box_x,
            geometry.offset_y,
            geometry.border_x,
            geometry.photo_h,
        ),
        (
            right_of_photo,
            geometry.offset_y,
            geometry.border_x,
            geometry.photo_h,
        ),
    ];

    for (x, y, width, height) in strips {
        let x_end = x.saturating_add(width).min(canvas.width());
        let y_end = y.saturating_add(height).min(canvas.height());
        for row in y..y_end {
            for col in x..x_end {
                canvas.put_pixel(col, row, fill);
            }
        }
    }
}

fn place_photo_on_canvas<P>(
    src: &ImageBuffer<P, Vec<P::Subpixel>>,
    geometry: &ExportGeometry,
    outer_fill: P,
    border_fill: Option<P>,
) -> ImageBuffer<P, Vec<P::Subpixel>>
where
    P: image::Pixel + 'static,
    P::Subpixel: 'static,
{
    let mut canvas = ImageBuffer::from_pixel(geometry.canvas_w, geometry.canvas_h, outer_fill);

    if let Some(fill) = border_fill
        && geometry.has_border()
    {
        fill_border_frame(&mut canvas, geometry, fill);
    }

    imageops::replace(
        &mut canvas,
        src,
        geometry.offset_x as i64,
        geometry.offset_y as i64,
    );
    canvas
}

fn compose_image_on_canvas(
    image: &DynamicImage,
    geometry: &ExportGeometry,
    pad_color: [u8; 3],
    border_color: [u8; 3],
) -> DynamicImage {
    fn compose<P: SolidFill>(
        src: &ImageBuffer<P, Vec<P::Subpixel>>,
        geometry: &ExportGeometry,
        pad_color: [u8; 3],
        border_color: [u8; 3],
    ) -> ImageBuffer<P, Vec<P::Subpixel>>
    where
        P::Subpixel: 'static,
    {
        place_photo_on_canvas(
            src,
            geometry,
            P::from_rgb8(pad_color),
            Some(P::from_rgb8(border_color)),
        )
    }

    match image {
        DynamicImage::ImageRgb8(buffer) => {
            DynamicImage::ImageRgb8(compose(buffer, geometry, pad_color, border_color))
        }
        DynamicImage::ImageRgba8(buffer) => {
            DynamicImage::ImageRgba8(compose(buffer, geometry, pad_color, border_color))
        }
        DynamicImage::ImageRgb16(buffer) => {
            DynamicImage::ImageRgb16(compose(buffer, geometry, pad_color, border_color))
        }
        DynamicImage::ImageRgba16(buffer) => {
            DynamicImage::ImageRgba16(compose(buffer, geometry, pad_color, border_color))
        }
        DynamicImage::ImageRgb32F(buffer) => {
            DynamicImage::ImageRgb32F(compose(buffer, geometry, pad_color, border_color))
        }
        DynamicImage::ImageRgba32F(buffer) => {
            DynamicImage::ImageRgba32F(compose(buffer, geometry, pad_color, border_color))
        }
        other => DynamicImage::ImageRgba8(compose(
            &other.to_rgba8(),
            geometry,
            pad_color,
            border_color,
        )),
    }
}

fn apply_export_geometry(
    mut image: DynamicImage,
    export_settings: &ExportSettings,
) -> Result<(DynamicImage, ExportGeometry), String> {
    let (src_w, src_h) = image.dimensions();

    match compute_export_geometry(src_w, src_h, export_settings) {
        None => {
            let (target_w, target_h) = resize_target(src_w, src_h, export_settings.resize.as_ref());
            if target_w != src_w || target_h != src_h {
                image = image.resize(target_w, target_h, imageops::FilterType::Lanczos3);
            }

            let (width, height) = image.dimensions();
            Ok((
                image,
                ExportGeometry {
                    canvas_w: width,
                    canvas_h: height,
                    photo_w: width,
                    photo_h: height,
                    offset_x: 0,
                    offset_y: 0,
                    border_x: 0,
                    border_y: 0,
                },
            ))
        }
        Some(geometry) => {
            if geometry.photo_w != src_w || geometry.photo_h != src_h {
                image = image.resize_exact(
                    geometry.photo_w,
                    geometry.photo_h,
                    imageops::FilterType::Lanczos3,
                );
            }

            let pad_color = fill_color(export_settings.pad.as_ref().map(|pad| pad.color.as_str()));
            let border_color = fill_color(
                export_settings
                    .border
                    .as_ref()
                    .map(|border| border.color.as_str()),
            );

            Ok((
                compose_image_on_canvas(&image, &geometry, pad_color, border_color),
                geometry,
            ))
        }
    }
}

fn apply_export_geometry_and_watermark(
    image: DynamicImage,
    export_settings: &ExportSettings,
) -> Result<(DynamicImage, ExportGeometry), String> {
    let (mut image, geometry) = apply_export_geometry(image, export_settings)?;

    if let Some(watermark_settings) = &export_settings.watermark {
        apply_watermark(&mut image, watermark_settings)?;
    }
    Ok((image, geometry))
}

/// 将前端绘制的白色信息条 PNG 拼接到图像底部；失败时记录日志并原样返回。
fn append_info_frame_bar(image: DynamicImage, frame_png: &[u8]) -> Result<DynamicImage, String> {
    if frame_png.is_empty() {
        return Ok(image);
    }
    let rgba = image.to_rgba8();
    let (out_w, out_h) = rgba.dimensions();
    if out_w == 0 || out_h == 0 {
        return Ok(DynamicImage::ImageRgba8(rgba));
    }
    let bar = image::load_from_memory(frame_png)
        .map_err(|e| format!("边框条解码失败: {}", e))?
        .to_rgba8();
    if bar.width() == 0 || bar.height() == 0 {
        return Err("边框条尺寸无效".to_string());
    }
    let bar_h = ((bar.height() as f64) * (out_w as f64) / (bar.width() as f64))
        .round()
        .max(1.0) as u32;
    let bar = if bar.width() != out_w || bar.height() != bar_h {
        imageops::resize(&bar, out_w, bar_h, imageops::FilterType::Lanczos3)
    } else {
        bar
    };
    let mut combined = ImageBuffer::new(out_w, out_h + bar_h);
    imageops::replace(&mut combined, &rgba, 0, 0);
    imageops::replace(&mut combined, &bar, 0, out_h as i64);
    Ok(DynamicImage::ImageRgba8(combined))
}

fn apply_info_frame_for_path(
    image: DynamicImage,
    path: &str,
    export_settings: &ExportSettings,
) -> DynamicImage {
    let Some(map) = export_settings.frame_png_by_path.as_ref() else {
        return image;
    };
    let Some(png) = map.get(path) else {
        return image;
    };
    if png.is_empty() {
        return image;
    }
    match append_info_frame_bar(image.clone(), png) {
        Ok(img) => img,
        Err(e) => {
            log::warn!("info frame append failed for {}: {}", path, e);
            image
        }
    }
}


fn ensure_export_not_cancelled(cancellation_token: &AtomicBool) -> Result<(), String> {
    if cancellation_token.load(Ordering::SeqCst) {
        Err("Export cancelled".to_string())
    } else {
        Ok(())
    }
}

/// 最终提交是完成边界；之后的取消不能抹掉已发布文件的任务结果。
fn completed_export_result(
    result: Result<(), String>,
    published: Option<PathBuf>,
    cancelled: bool,
) -> Result<Option<PathBuf>, String> {
    if published.is_some() {
        return Ok(published);
    }
    result?;
    if cancelled {
        Err("Export cancelled".into())
    } else {
        Ok(None)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
enum BatchExportOutcome {
    Success,
    Skipped,
    Failed,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct BatchExportItem {
    elapsed_ms: u64,
    outcome: BatchExportOutcome,
    output_path: Option<String>,
    path: String,
    reason: Option<String>,
    reshape_preparation_ms: u64,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct BatchExportPerformance {
    p50_ms: u64,
    p95_ms: u64,
    total_ms: u64,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct BatchExportSummary {
    failed: usize,
    items: Vec<BatchExportItem>,
    performance: BatchExportPerformance,
    skipped: usize,
    succeeded: usize,
}

fn batch_failure_reason(error: &str) -> &'static str {
    let error = error.to_ascii_lowercase();
    if error.contains("cancelled") || error.contains("取消") {
        "cancelled"
    } else if error.contains("memory") || error.contains("内存") {
        "outOfMemory"
    } else if error.contains("model") || error.contains("模型") {
        "modelError"
    } else if error.contains("pose") || error.contains("姿态") {
        "unsupportedPose"
    } else if error.contains("no face") || error.contains("noface") || error.contains("人脸") {
        "noFace"
    } else {
        "renderError"
    }
}

fn batch_outcome(error: Option<&str>) -> (BatchExportOutcome, Option<String>) {
    let Some(error) = error else {
        return (BatchExportOutcome::Success, None);
    };
    // 只有显式标记 batch-skip: 的才算跳过；其余错误一律按失败上报，避免关键字误判导致静默“成功”。
    match error.strip_prefix("batch-skip:") {
        Some(reason) => (
            BatchExportOutcome::Skipped,
            Some(batch_failure_reason(reason).to_string()),
        ),
        None => (
            BatchExportOutcome::Failed,
            Some(batch_failure_reason(error).to_string()),
        ),
    }
}

fn percentile(values: &mut [u64], numerator: usize, denominator: usize) -> u64 {
    if values.is_empty() {
        return 0;
    }
    values.sort_unstable();
    let index = values
        .len()
        .saturating_mul(numerator)
        .div_ceil(denominator)
        .saturating_sub(1);
    values[index]
}

fn batch_summary(items: Vec<BatchExportItem>, total_ms: u64) -> BatchExportSummary {
    let succeeded = items
        .iter()
        .filter(|item| item.outcome == BatchExportOutcome::Success)
        .count();
    let skipped = items
        .iter()
        .filter(|item| item.outcome == BatchExportOutcome::Skipped)
        .count();
    let failed = items.len().saturating_sub(succeeded + skipped);
    let mut elapsed: Vec<_> = items.iter().map(|item| item.elapsed_ms).collect();
    let mut median = elapsed.clone();
    let p50_ms = percentile(&mut median, 1, 2);
    let p95_ms = percentile(&mut elapsed, 95, 100);
    BatchExportSummary {
        failed,
        items,
        performance: BatchExportPerformance {
            p50_ms,
            p95_ms,
            total_ms,
        },
        skipped,
        succeeded,
    }
}

fn extrapolate_encoded_size(
    bytes: usize,
    full: [u32; 2],
    sample: [u32; 2],
) -> Result<usize, String> {
    if full.contains(&0) || sample.contains(&0) {
        return Err("导出估算尺寸为空".into());
    }
    let pixels = u128::from(full[0]) * u128::from(full[1]);
    let samples = u128::from(sample[0]) * u128::from(sample[1]);
    let total = (bytes as u128)
        .checked_mul(pixels)
        .ok_or("导出体积估算溢出")?
        / samples;
    usize::try_from(total).map_err(|_| "导出体积估算溢出".into())
}

fn estimate_dimensions(full: [u32; 2], edge: u32) -> Result<[u32; 2], String> {
    crate::portrait_tiles::check_dimensions(full)?;
    if edge == 0 {
        return Err("估算尺寸为空".into());
    }
    let scale = (edge as f64 / full[0].max(full[1]) as f64).min(1.0);
    Ok(full.map(|dimension| (f64::from(dimension) * scale).round().max(1.0) as u32))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ExportCancellationRequest {
    Requested,
    AlreadyRequested,
    NoActiveTask,
}

struct ExportTaskGuard {
    task_token: Arc<Mutex<Option<Arc<AtomicBool>>>>,
    cancellation_token: Arc<AtomicBool>,
    app_handle: Option<tauri::AppHandle>,
}

impl ExportTaskGuard {
    fn new(
        task_token: Arc<Mutex<Option<Arc<AtomicBool>>>>,
        cancellation_token: Arc<AtomicBool>,
    ) -> Self {
        Self {
            task_token,
            cancellation_token,
            app_handle: None,
        }
    }

    fn with_app_handle(
        task_token: Arc<Mutex<Option<Arc<AtomicBool>>>>,
        cancellation_token: Arc<AtomicBool>,
        app_handle: tauri::AppHandle,
    ) -> Self {
        let mut guard = Self::new(task_token, cancellation_token);
        guard.app_handle = Some(app_handle);
        guard
    }
}

fn register_export_task(
    task_token: &Mutex<Option<Arc<AtomicBool>>>,
) -> Result<Arc<AtomicBool>, String> {
    let mut active_token = task_token.lock().unwrap();
    if active_token.is_some() {
        return Err("An export is already in progress.".to_string());
    }

    let cancellation_token = Arc::new(AtomicBool::new(false));
    *active_token = Some(Arc::clone(&cancellation_token));
    Ok(cancellation_token)
}

fn request_export_cancellation<F>(
    task_token: &Mutex<Option<Arc<AtomicBool>>>,
    on_requested: F,
) -> ExportCancellationRequest
where
    F: FnOnce(),
{
    let active_token = task_token.lock().unwrap();
    let Some(cancellation_token) = active_token.as_ref() else {
        return ExportCancellationRequest::NoActiveTask;
    };

    if cancellation_token.swap(true, Ordering::SeqCst) {
        ExportCancellationRequest::AlreadyRequested
    } else {
        on_requested();
        ExportCancellationRequest::Requested
    }
}

fn finish_export_task<F>(
    task_token: &Mutex<Option<Arc<AtomicBool>>>,
    cancellation_token: &Arc<AtomicBool>,
    on_finish: F,
) -> bool
where
    F: FnOnce(bool),
{
    let mut active_token = task_token.lock().unwrap();
    let Some(current_token) = active_token.as_ref() else {
        return false;
    };
    if !Arc::ptr_eq(current_token, cancellation_token) {
        return false;
    }

    let cancelled = cancellation_token.load(Ordering::SeqCst);
    *active_token = None;

    on_finish(cancelled);
    true
}

impl Drop for ExportTaskGuard {
    fn drop(&mut self) {
        let app_handle = self.app_handle.clone();
        let _ = finish_export_task(
            &self.task_token,
            &self.cancellation_token,
            |cancelled| match (cancelled, app_handle) {
                (true, Some(app_handle)) => {
                    let _ = app_handle.emit("export-cancelled", ());
                }
                (false, Some(app_handle)) => {
                    let _ = app_handle.emit("export-error", "Export task terminated unexpectedly");
                }
                _ => {}
            },
        );
    }
}

#[allow(clippy::too_many_arguments)]
fn process_image_for_export_pipeline(
    path: &str,
    base_image: &DynamicImage,
    js_adjustments: &Value,
    context: &GpuContext,
    state: &tauri::State<AppState>,
    is_raw: bool,
    debug_tag: &str,
    app_handle: &tauri::AppHandle,
    output_precision: RenderOutputPrecision,
) -> Result<DynamicImage, String> {
    let (transformed_image, unscaled_crop_offset) =
        apply_all_transformations(Cow::Borrowed(base_image), js_adjustments);
    let (img_w, img_h) = transformed_image.dimensions();

    let mask_definitions: Vec<MaskDefinition> = js_adjustments
        .get("masks")
        .and_then(|m| serde_json::from_value(m.clone()).ok())
        .unwrap_or_default();

    let warped_image = resolve_warped_image_for_masks(state, js_adjustments, &mask_definitions);
    let mask_bitmaps: Vec<ImageBuffer<Luma<u8>, Vec<u8>>> = mask_definitions
        .iter()
        .filter_map(|def| {
            generate_mask_bitmap(
                def,
                img_w,
                img_h,
                1.0,
                unscaled_crop_offset,
                warped_image.as_deref(),
            )
        })
        .collect();

    let tm_override = resolve_tonemapper_override_from_handle(app_handle, is_raw);
    let mut all_adjustments = get_all_adjustments_from_json(
        js_adjustments,
        is_raw,
        as_shot_white_balance(path),
        tm_override,
    );
    all_adjustments.global.show_clipping = 0;

    let lut_path = js_adjustments["lutPath"].as_str();
    let lut = lut_path.and_then(|p| get_or_load_lut(state, p).ok());

    let unique_hash = calculate_full_job_hash(path, js_adjustments);

    process_and_get_dynamic_image_with_precision(
        context,
        state,
        transformed_image.as_ref(),
        unique_hash,
        RenderRequest {
            adjustments: all_adjustments,
            mask_bitmaps: &mask_bitmaps,
            lut,
            roi: None,
            portrait_maps: crate::portrait_processing::prepare_render(
                app_handle,
                state,
                path,
                js_adjustments,
            )?,
        },
        debug_tag,
        output_precision,
    )
}

fn render_output_precision(
    output_format: &str,
    export_settings: &ExportSettings,
) -> RenderOutputPrecision {
    if matches!(output_format.to_lowercase().as_str(), "tif" | "tiff")
        && export_settings.tiff_bit_depth == TiffBitDepth::Sixteen
    {
        RenderOutputPrecision::SixteenBit
    } else {
        RenderOutputPrecision::EightBit
    }
}

fn set_timestamps_from_exif(src: &Path, dst: &Path) {
    let capture_dt = exif_processing::get_creation_date_from_path(src);

    if capture_dt.timestamp() <= 0 {
        let now = filetime::FileTime::now();
        if let Err(e) = filetime::set_file_times(dst, now, now) {
            log::warn!("Could not set timestamps on '{}': {}", dst.display(), e);
        }
        return;
    }

    let ft = filetime::FileTime::from_unix_time(
        capture_dt.timestamp(),
        capture_dt.timestamp_subsec_nanos(),
    );
    if let Err(e) = filetime::set_file_times(dst, ft, ft) {
        log::warn!("Could not set timestamps on '{}': {}", dst.display(), e);
    }
}

fn reshape_export_intent(
    adjustments: &Value,
    output_format: &str,
    export_masks: bool,
    android_gallery: bool,
) -> Result<crate::portrait_render::RenderIntent, String> {
    use crate::portrait_render::RenderIntent;
    let intent = crate::portrait_render::render_intent(adjustments)?;
    if intent == RenderIntent::Prepare {
        if output_format.eq_ignore_ascii_case("cube") {
            return Err(".cube 不能表达活动人像重塑几何，请导出照片格式".into());
        }
        if export_masks {
            return Err("重塑蒙版图层的颜色／Alpha 同场输出尚未接通，禁止导出错位图层".into());
        }
        if android_gallery {
            return Err("Android 活动重塑图库事务尚未支持".into());
        }
    }
    Ok(intent)
}

/// 共享入口：严格原生照片先完成合成与重塑，输出尺寸只影响最后的浮点重建。
#[allow(clippy::too_many_arguments)]
fn prepare_reshape_output(
    path: &str,
    adjustments: &Value,
    export_settings: &ExportSettings,
    output_format: &str,
    app: &tauri::AppHandle,
    settings: &crate::AppSettings,
    estimate_edge: Option<u32>,
    frozen: Option<&crate::portrait_render::FrozenPhotoInput>,
    current: &dyn Fn() -> bool,
) -> Result<Option<(crate::portrait_render::PhotoRenderResult, [u32; 2])>, String> {
    use crate::portrait_render::{PhotoRenderRequest, RenderIntent, RenderPurpose, RenderQuality};
    if reshape_export_intent(
        adjustments,
        output_format,
        export_settings.export_masks,
        cfg!(target_os = "android") && estimate_edge.is_none(),
    )? == RenderIntent::Bypass
    {
        return Ok(None);
    }
    let state = app.state::<AppState>();
    let prepared;
    let frozen = match frozen {
        Some(frozen) => frozen,
        None => {
            prepared =
                state
                    .portrait_render
                    .freeze_output(app, path, adjustments, settings, current)?;
            &prepared
        }
    };
    let source = Arc::clone(frozen.source());
    let coordinates = crate::portrait_coordinates::CoordinateMap::new(
        [source.image().width(), source.image().height()],
        adjustments,
    )?;
    let crop =
        crate::portrait_tiles::effective_crop(coordinates.base_dimensions, &adjustments["crop"])?;
    let (width, height) = export_settings
        .resize
        .as_ref()
        .map_or((crop.width, crop.height), |resize| {
            calculate_resize_target(crop.width, crop.height, resize)
        });
    let full_output = [width, height];
    crate::portrait_tiles::check_dimensions(full_output)?;
    let output = match estimate_edge {
        Some(edge) => estimate_dimensions(full_output, edge)?,
        None => full_output,
    };
    let window = crate::portrait_tiles::OutputWindow::new(
        coordinates.base_dimensions,
        &adjustments["crop"],
        output,
        None,
    )?;
    let request_id = source.revision();
    let lease = state.portrait_render.begin_consumer(&source, request_id)?;
    let result = state.portrait_render.render_frozen_photo(
        app,
        PhotoRenderRequest {
            source,
            adjustments: adjustments.clone(),
            window,
            request_id,
            quality: RenderQuality::HighQuality,
            purpose: if estimate_edge.is_some() {
                RenderPurpose::Estimate
            } else {
                RenderPurpose::Export
            },
            sixteen_bit: render_output_precision(output_format, export_settings)
                == RenderOutputPrecision::SixteenBit,
        },
        lease,
        settings,
        frozen,
        current,
    )?;
    Ok(Some((result, full_output)))
}

/// 有效 identity 保留旧 GPU 像素管线，但所有输入来自同一冻结源，不借用编辑器缓存。
fn process_frozen_identity(
    frozen: &crate::portrait_render::FrozenPhotoInput,
    adjustments: &Value,
    app: &tauri::AppHandle,
    settings: &crate::AppSettings,
    precision: RenderOutputPrecision,
    current: &dyn Fn() -> bool,
) -> Result<DynamicImage, String> {
    let source = frozen.source();
    source.verify(current)?;
    let state = app.state::<AppState>();
    let base = crate::portrait_analysis_cache::analysis_base(adjustments)?;
    let warped = crate::adjustment_utils::prepare_patched_warped(source.image(), &base)?;
    let (native, _) =
        crate::adjustment_utils::apply_spatial_transformations(warped.as_ref(), &base);
    let dimensions = [native.width(), native.height()];
    let crop = crate::portrait_tiles::effective_crop(dimensions, &adjustments["crop"])?;
    let maps = crate::portrait_processing::prepare_native_render(
        app,
        &state,
        source.source_key(),
        &base,
        &native,
        source.is_raw(),
        adjustments,
        current,
    )?;
    let maps = maps.map(|maps| maps.analysis.for_render(adjustments));
    let definitions: Vec<MaskDefinition> = match adjustments.get("masks") {
        None | Some(Value::Null) => Vec::new(),
        Some(value) => {
            serde_json::from_value(value.clone()).map_err(|e| format!("照片蒙版参数非法：{e}"))?
        }
    };
    let mut masks = Vec::with_capacity(definitions.len());
    for definition in &definitions {
        crate::portrait_analysis::check_current(current)?;
        let mask = crate::mask_generation::generate_mask_bitmap_checked(
            definition,
            dimensions[0],
            dimensions[1],
            Some(&warped),
            current,
        )?;
        masks.push(match mask {
            Some(mask) => {
                imageops::crop_imm(&mask, crop.x as u32, crop.y as u32, crop.width, crop.height)
                    .to_image()
            }
            None => GrayImage::new(crop.width, crop.height),
        });
    }
    let cropped = crate::image_processing::apply_crop(native, &adjustments["crop"]);
    let mut params = get_all_adjustments_from_json(
        adjustments,
        source.is_raw(),
        as_shot_white_balance(source.path()),
        crate::image_processing::resolve_tonemapper_override(settings, source.is_raw()),
    );
    params.global.show_clipping = 0;
    let context = get_or_init_gpu_context(&state, app)?;
    let image = process_and_get_dynamic_image_with_precision(
        &context,
        &state,
        &cropped,
        calculate_full_job_hash(source.source_key(), adjustments),
        RenderRequest {
            adjustments: params,
            mask_bitmaps: &masks,
            lut: frozen.lut().map(Arc::new),
            roi: None,
            portrait_maps: maps,
        },
        "export_frozen_identity",
        precision,
    )?;
    source.verify(current)?;
    Ok(image)
}

/// 已完成最终浮点缩放的结果仅加水印，不再调用旧 resize 路径。
#[cfg(not(target_os = "android"))]
fn publish_reshape_export(
    output: &crate::portrait_render::PhotoRenderResult,
    image: &DynamicImage,
    target: &Path,
    export_settings: &ExportSettings,
    auto_number: bool,
    state: &tauri::State<AppState>,
    current: &dyn Fn() -> bool,
) -> Result<PathBuf, String> {
    crate::portrait_analysis::check_current(current)?;
    let extension = target
        .extension()
        .and_then(|s| s.to_str())
        .ok_or("导出目标缺少格式")?;
    let (source_path, _) = parse_virtual_path(output.source.path());
    let source_path_str = source_path.to_str().ok_or("导出源路径编码错误")?;
    let mut bytes = encode_image_to_bytes(
        image,
        extension,
        export_settings.jpeg_quality,
        export_settings.tiff_bit_depth,
    )?;
    crate::portrait_analysis::check_current(current)?;
    exif_processing::write_image_with_metadata_strict(
        &mut bytes,
        source_path_str,
        extension,
        export_settings.keep_metadata,
        export_settings.strip_gps,
    )?;
    let timestamp = if export_settings.preserve_timestamps {
        let date = exif_processing::get_creation_date_from_path(&source_path);
        Some(if date.timestamp() > 0 {
            filetime::FileTime::from_unix_time(date.timestamp(), date.timestamp_subsec_nanos())
        } else {
            filetime::FileTime::now()
        })
    } else {
        None
    };
    output.publish_encoded_guarded(
        target,
        &bytes,
        auto_number,
        current,
        &state.export_task_token,
        |file| {
            if let Some(time) = timestamp {
                filetime::set_file_handle_times(file, Some(time), Some(time))
                    .map_err(|e| e.to_string())?;
            }
            Ok(())
        },
    )
}

fn save_image_with_metadata(
    image: &DynamicImage,
    output_path: &std::path::Path,
    source_path_str: &str,
    export_settings: &ExportSettings,
) -> Result<(), String> {
    let extension = output_path
        .extension()
        .and_then(|s| s.to_str())
        .unwrap_or("")
        .to_lowercase();

    let mut image_bytes = encode_image_to_bytes(
        image,
        &extension,
        export_settings.jpeg_quality,
        export_settings.tiff_bit_depth,
    )?;

    exif_processing::write_image_with_metadata(
        &mut image_bytes,
        source_path_str,
        &extension,
        export_settings.keep_metadata,
        export_settings.strip_gps,
    )?;

    #[cfg(target_os = "android")]
    {
        let file_name = output_path
            .file_name()
            .and_then(|name| name.to_str())
            .ok_or_else(|| "Missing Android export file name".to_string())?;
        crate::android_integration::save_image_bytes_to_android_gallery(
            file_name,
            mime_type_for_extension(&extension),
            &image_bytes,
        )?;
    }

    #[cfg(not(target_os = "android"))]
    fs::write(output_path, image_bytes)
        .map_err(|e| format!("Failed to write file to '{}': {}", output_path.display(), e))?;

    Ok(())
}

#[cfg(target_os = "android")]
pub fn mime_type_for_extension(extension: &str) -> &'static str {
    match extension {
        "jpg" | "jpeg" => "image/jpeg",
        "png" => "image/png",
        "webp" => "image/webp",
        "bmp" => "image/bmp",
        "gif" => "image/gif",
        "tif" | "tiff" => "image/tiff",
        "jxl" => "image/jxl",
        _ => "application/octet-stream",
    }
}

#[allow(clippy::too_many_arguments)]
fn process_image_for_export(
    path: &str,
    base_image: &DynamicImage,
    js_adjustments: &Value,
    export_settings: &ExportSettings,
    context: &GpuContext,
    state: &tauri::State<AppState>,
    is_raw: bool,
    app_handle: &tauri::AppHandle,
    output_format: &str,
) -> Result<DynamicImage, String> {
    let processed_image = process_image_for_export_pipeline(
        path,
        base_image,
        js_adjustments,
        context,
        state,
        is_raw,
        "process_image_for_export",
        app_handle,
        render_output_precision(output_format, export_settings),
    )?;

    let (image, _) = apply_export_geometry_and_watermark(processed_image, export_settings)?;
    Ok(apply_info_frame_for_path(image, path, export_settings))
}

fn build_single_mask_adjustments(all: &AllAdjustments, mask_index: usize) -> AllAdjustments {
    let mut single = AllAdjustments {
        global: all.global,
        mask_adjustments: all.mask_adjustments,
        mask_count: 1,
        tile_offset_x: all.tile_offset_x,
        tile_offset_y: all.tile_offset_y,
        mask_atlas_cols: all.mask_atlas_cols,
        portrait: all.portrait,
    };
    single.mask_adjustments[0] = all.mask_adjustments[mask_index];
    for i in 1..single.mask_adjustments.len() {
        single.mask_adjustments[i] = Default::default();
    }
    single
}

fn encode_grayscale_to_png(bitmap: &GrayImage) -> Result<Vec<u8>, String> {
    let mut buf = Vec::new();
    let mut cursor = Cursor::new(&mut buf);
    bitmap
        .write_to(&mut cursor, ImageFormat::Png)
        .map_err(|e| e.to_string())?;
    Ok(buf)
}

/// Exports are sRGB-encoded; tagging them lets color-managed apps read them correctly.
const SRGB_ICC_PROFILE: &[u8] = include_bytes!("../icc/sRGB-v2-magic.icc");

fn encode_image_to_bytes(
    image: &DynamicImage,
    output_format: &str,
    jpeg_quality: u8,
    tiff_bit_depth: TiffBitDepth,
) -> Result<Vec<u8>, String> {
    let mut image_bytes = Vec::new();
    let mut cursor = Cursor::new(&mut image_bytes);

    match output_format.to_lowercase().as_str() {
        "jxl" => {
            let (width, height) = image.dimensions();
            let has_alpha = image.color().has_alpha();

            let jxl_data = if jpeg_quality == 100 {
                if has_alpha {
                    let rgba = image.to_rgba8();
                    LosslessConfig::new()
                        .encode(rgba.as_raw(), width, height, PixelLayout::Rgba8)
                        .map_err(|e| format!("Failed to encode lossless JXL: {}", e))?
                } else {
                    let rgb = image.to_rgb8();
                    LosslessConfig::new()
                        .encode(rgb.as_raw(), width, height, PixelLayout::Rgb8)
                        .map_err(|e| format!("Failed to encode lossless JXL: {}", e))?
                }
            } else {
                let jxl_quality = calibrated_jxl_quality(jpeg_quality as f32);
                let distance = quality_to_distance(jxl_quality);

                if has_alpha {
                    let rgba = image.to_rgba8();
                    LossyConfig::new(distance)
                        .encode(rgba.as_raw(), width, height, PixelLayout::Rgba8)
                        .map_err(|e| format!("Failed to encode lossy JXL: {}", e))?
                } else {
                    let rgb = image.to_rgb8();
                    LossyConfig::new(distance)
                        .encode(rgb.as_raw(), width, height, PixelLayout::Rgb8)
                        .map_err(|e| format!("Failed to encode lossy JXL: {}", e))?
                }
            };

            return Ok(jxl_data);
        }
        "webp" => {
            let encoder = webp::Encoder::from_image(image)
                .map_err(|_| "Failed to create WebP encoder".to_string())?;
            let webp_mem = encoder.encode(jpeg_quality as f32);
            return Ok(webp_mem.to_vec());
        }
        "jpg" | "jpeg" => {
            let rgb_image = image.to_rgb8();
            let mut encoder = JpegEncoder::new_with_quality(&mut cursor, jpeg_quality);
            encoder
                .set_icc_profile(SRGB_ICC_PROFILE.to_vec())
                .map_err(|e| e.to_string())?;
            rgb_image
                .write_with_encoder(encoder)
                .map_err(|e| e.to_string())?;
        }
        "png" => {
            let image_to_encode = if image.as_rgb32f().is_some() {
                DynamicImage::ImageRgb16(image.to_rgb16())
            } else {
                image.clone()
            };

            let mut encoder = PngEncoder::new(&mut cursor);
            encoder
                .set_icc_profile(SRGB_ICC_PROFILE.to_vec())
                .map_err(|e| e.to_string())?;
            image_to_encode
                .write_with_encoder(encoder)
                .map_err(|e| e.to_string())?;
        }
        "tif" | "tiff" => {
            let image_to_encode = match tiff_bit_depth {
                TiffBitDepth::Eight => DynamicImage::ImageRgb8(image.to_rgb8()),
                TiffBitDepth::Sixteen => DynamicImage::ImageRgb16(image.to_rgb16()),
            };
            image_to_encode
                .write_to(&mut cursor, image::ImageFormat::Tiff)
                .map_err(|e| e.to_string())?;
        }
        "avif" => {
            image
                .write_to(&mut cursor, image::ImageFormat::Avif)
                .map_err(|e| e.to_string())?;
        }
        _ => return Err(format!("Unsupported file format: {}", output_format)),
    };
    Ok(image_bytes)
}

#[allow(clippy::too_many_arguments)]
fn export_masks_for_image(
    base_image: &DynamicImage,
    js_adjustments: &Value,
    export_settings: &ExportSettings,
    output_path_obj: &std::path::Path,
    source_path_str: &str,
    virtual_path: &str,
    context: &Arc<GpuContext>,
    state: &tauri::State<AppState>,
    is_raw: bool,
    app_handle: &tauri::AppHandle,
    cancellation_token: &AtomicBool,
) -> Result<(), String> {
    ensure_export_not_cancelled(cancellation_token)?;
    let (transformed_image, unscaled_crop_offset) =
        apply_all_transformations(Cow::Borrowed(base_image), js_adjustments);
    ensure_export_not_cancelled(cancellation_token)?;
    let (img_w, img_h) = transformed_image.dimensions();
    let mask_definitions: Vec<MaskDefinition> = js_adjustments
        .get("masks")
        .and_then(|m| serde_json::from_value(m.clone()).ok())
        .unwrap_or_default();

    let warped_image = resolve_warped_image_for_masks(state, js_adjustments, &mask_definitions);
    let mut mask_bitmaps = Vec::with_capacity(mask_definitions.len());
    for definition in &mask_definitions {
        ensure_export_not_cancelled(cancellation_token)?;
        if let Some(bitmap) = generate_mask_bitmap(
            definition,
            img_w,
            img_h,
            1.0,
            unscaled_crop_offset,
            warped_image.as_deref(),
        ) {
            mask_bitmaps.push(bitmap);
        }
        ensure_export_not_cancelled(cancellation_token)?;
    }

    if !mask_bitmaps.is_empty() {
        let tm_override = resolve_tonemapper_override_from_handle(app_handle, is_raw);
        let all_adjustments = get_all_adjustments_from_json(
            js_adjustments,
            is_raw,
            as_shot_white_balance(source_path_str),
            tm_override,
        );
        let lut_path = js_adjustments["lutPath"].as_str();
        let lut = lut_path.and_then(|p| get_or_load_lut(state, p).ok());
        let unique_hash = calculate_full_job_hash(source_path_str, js_adjustments);
        let output_dir = output_path_obj.parent().unwrap_or(output_path_obj);
        let stem = output_path_obj
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("export");
        let extension = output_path_obj
            .extension()
            .and_then(|s| s.to_str())
            .unwrap_or("jpg");

        for (i, _) in mask_bitmaps.iter().enumerate() {
            ensure_export_not_cancelled(cancellation_token)?;
            let single_adjustments = build_single_mask_adjustments(&all_adjustments, i);
            let full_white_mask = ImageBuffer::from_fn(img_w, img_h, |_, _| Luma([255u8]));
            let single_bitmaps: Vec<ImageBuffer<Luma<u8>, Vec<u8>>> = vec![full_white_mask];

            let processed = process_and_get_dynamic_image_with_precision(
                context,
                state,
                transformed_image.as_ref(),
                unique_hash,
                RenderRequest {
                    adjustments: single_adjustments,
                    mask_bitmaps: &single_bitmaps,
                    lut: lut.clone(),
                    roi: None,
                    portrait_maps: crate::portrait_processing::prepare_render(
                        app_handle,
                        state,
                        virtual_path,
                        js_adjustments,
                    )?,
                },
                "export_mask_image",
                render_output_precision(extension, export_settings),
            )?;
            ensure_export_not_cancelled(cancellation_token)?;

            let (with_options, geometry) =
                apply_export_geometry_and_watermark(processed, export_settings)?;
            // 蒙版导出图不加信息边框（与主图区分）

            let alpha_scaled = imageops::resize(
                &mask_bitmaps[i],
                geometry.photo_w,
                geometry.photo_h,
                imageops::FilterType::Lanczos3,
            );

            let alpha_resized: GrayImage = if geometry.is_identity() {
                alpha_scaled
            } else {
                place_photo_on_canvas(&alpha_scaled, &geometry, Luma([0u8]), None)
            };
            ensure_export_not_cancelled(cancellation_token)?;

            let mask_image_path =
                output_dir.join(format!("{}_mask_{}_image.{}", stem, i, extension));
            let mask_alpha_path = output_dir.join(format!("{}_mask_{}_alpha.png", stem, i));

            save_image_with_metadata(
                &with_options,
                &mask_image_path,
                source_path_str,
                export_settings,
            )?;
            ensure_export_not_cancelled(cancellation_token)?;

            if export_settings.preserve_timestamps {
                set_timestamps_from_exif(Path::new(source_path_str), &mask_image_path);
            }

            let alpha_bytes = encode_grayscale_to_png(&alpha_resized)?;
            ensure_export_not_cancelled(cancellation_token)?;
            #[cfg(target_os = "android")]
            {
                let file_name = mask_alpha_path
                    .file_name()
                    .and_then(|name| name.to_str())
                    .ok_or_else(|| "Missing Android mask export file name".to_string())?;
                crate::android_integration::save_image_bytes_to_android_gallery(
                    file_name,
                    "image/png",
                    &alpha_bytes,
                )?;
            }

            #[cfg(not(target_os = "android"))]
            fs::write(&mask_alpha_path, alpha_bytes).map_err(|e| e.to_string())?;
            ensure_export_not_cancelled(cancellation_token)?;
        }
    }
    Ok(())
}

fn export_adjustments_as_lut(
    js_adjustments: &Value,
    source_path_str: &str,
    context: &Arc<GpuContext>,
    state: &tauri::State<AppState>,
    app_handle: &tauri::AppHandle,
    cancellation_token: &AtomicBool,
) -> Result<Vec<u8>, String> {
    ensure_export_not_cancelled(cancellation_token)?;
    reshape_export_intent(js_adjustments, "cube", false, false)?;
    let lut_size = 33;
    let identity_image = generate_identity_lut_image(lut_size);

    let tm_override = resolve_tonemapper_override_from_handle(app_handle, false);
    let mut all_adjustments = get_all_adjustments_from_json(
        js_adjustments,
        false,
        as_shot_white_balance(source_path_str),
        tm_override,
    );

    all_adjustments.global.show_clipping = 0;
    all_adjustments.portrait = Default::default();
    all_adjustments.global.vignette_amount = 0.0;
    all_adjustments.global.grain_amount = 0.0;
    all_adjustments.global.sharpness = 0.0;
    all_adjustments.global.clarity = 0.0;
    all_adjustments.global.dehaze = 0.0;
    all_adjustments.global.structure = 0.0;
    all_adjustments.global.centré = 0.0;
    all_adjustments.global.glow_amount = 0.0;
    all_adjustments.global.halation_amount = 0.0;
    all_adjustments.global.flare_amount = 0.0;
    all_adjustments.global.luma_noise_reduction = 0.0;
    all_adjustments.global.color_noise_reduction = 0.0;
    all_adjustments.global.chromatic_aberration_red_cyan = 0.0;
    all_adjustments.global.chromatic_aberration_blue_yellow = 0.0;

    let lut_path = js_adjustments["lutPath"].as_str();
    let lut = lut_path.and_then(|p| get_or_load_lut(state, p).ok());
    let unique_hash = calculate_full_job_hash(source_path_str, js_adjustments);

    let processed_lut = process_and_get_dynamic_image(
        context,
        state,
        &identity_image,
        unique_hash,
        RenderRequest {
            adjustments: all_adjustments,
            mask_bitmaps: &[],
            lut,
            roi: None,
            portrait_maps: None,
        },
        "export_lut",
    )?;
    ensure_export_not_cancelled(cancellation_token)?;

    let cube_lut = convert_image_to_cube_lut(&processed_lut, lut_size)?;
    ensure_export_not_cancelled(cancellation_token)?;
    Ok(cube_lut)
}

#[allow(clippy::too_many_arguments)]
pub(crate) async fn export_images_impl(
    paths: Vec<String>,
    output_folder_or_file: String,
    is_explicit_file_path: bool,
    base_origin_folders: Vec<String>,
    export_settings: ExportSettings,
    output_format: String,
    adjustments_mode: ExportAdjustmentsMode,
    state: tauri::State<'_, AppState>,
    app_handle: tauri::AppHandle,
    completion_tx: Option<tokio::sync::oneshot::Sender<Result<(), usize>>>,
) -> Result<(), String> {
    let cancellation_token = register_export_task(&state.export_task_token)?;
    let batch_started = Instant::now();
    let task_guard = ExportTaskGuard::with_app_handle(
        Arc::clone(&state.export_task_token),
        Arc::clone(&cancellation_token),
        app_handle.clone(),
    );
    // 参数在接受批任务时冻结，排队中的照片不再读取之后被编辑的 sidecar。
    let frozen_adjustments: Vec<_> = paths
        .iter()
        .map(|path| {
            let (_, sidecar) = parse_virtual_path(path);
            let (mut adjustments, is_current) = match &adjustments_mode {
                ExportAdjustmentsMode::UseSidecars {
                    active_path,
                    active_adjustments,
                } if active_path.as_ref() == Some(path) => (
                    active_adjustments
                        .clone()
                        .unwrap_or_else(|| exif_processing::load_sidecar(&sidecar).adjustments),
                    true,
                ),
                ExportAdjustmentsMode::UseSidecars { .. } => {
                    (exif_processing::load_sidecar(&sidecar).adjustments, false)
                }
                ExportAdjustmentsMode::GlobalOverride(adjustments) => (adjustments.clone(), false),
            };
            hydrate_adjustments(&state, &mut adjustments);
            (adjustments, is_current)
        })
        .collect();
    let settings = load_settings(app_handle.clone()).unwrap_or_default();
    // 进入 worker 队列前持有源和场。准备失败逐图报告，绝不在 worker 中重新取较晚源。
    let frozen_outputs: Vec<_> = paths
        .iter()
        .zip(&frozen_adjustments)
        .map(|(path, (adjustments, _))| {
            let preparation_started = Instant::now();
            let frozen = (|| {
                let current = || !cancellation_token.load(Ordering::SeqCst);
                let intent = reshape_export_intent(
                    adjustments,
                    &output_format,
                    export_settings.export_masks,
                    cfg!(target_os = "android"),
                )?;
                if intent == crate::portrait_render::RenderIntent::Bypass {
                    return Ok(None);
                }
                if is_explicit_file_path && paths.len() == 1 {
                    let actual = Path::new(&output_folder_or_file)
                        .extension()
                        .and_then(|s| s.to_str())
                        .ok_or("导出目标缺少格式")?;
                    reshape_export_intent(
                        adjustments,
                        actual,
                        export_settings.export_masks,
                        cfg!(target_os = "android"),
                    )?;
                }
                state
                    .portrait_render
                    .freeze_output(&app_handle, path, adjustments, &settings, &current)
                    .map(Some)
            })();
            (
                frozen,
                u64::try_from(preparation_started.elapsed().as_millis()).unwrap_or(u64::MAX),
            )
        })
        .collect::<Vec<_>>();
    tokio::time::sleep(std::time::Duration::from_millis(10)).await;

    let initialization = if paths.is_empty() || cancellation_token.load(Ordering::SeqCst) {
        Ok(None)
    } else {
        get_or_init_gpu_context(&state, &app_handle).map(Some)
    };
    if initialization.as_ref().is_err()
        || initialization.as_ref().is_ok_and(Option::is_none)
        || cancellation_token.load(Ordering::SeqCst)
    {
        let error = initialization.err();
        let cancelled = cancellation_token.load(Ordering::SeqCst);
        finish_export_task(&state.export_task_token, &cancellation_token, |cancelled| {
            let payload = serde_json::json!({ "completedOutputs": [] });
            if cancelled {
                let _ = app_handle.emit("export-cancelled", payload);
            } else if let Some(error) = &error {
                let _ = app_handle.emit("export-error", error);
            } else {
                let _ = app_handle.emit("export-complete", payload);
            }
        });
        if let Some(tx) = completion_tx {
            let _ = tx.send(if error.is_some() || (cancelled && !paths.is_empty()) {
                Err(paths.len().max(1))
            } else {
                Ok(())
            });
        }
        return if cancelled {
            Ok(())
        } else {
            error.map_or(Ok(()), Err)
        };
    }
    let context = Arc::new(initialization?.ok_or("导出 GPU 上下文缺失")?);
    let progress_counter = Arc::new(AtomicUsize::new(0));

    let available_cores = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);

    let mut sys = sysinfo::System::new();
    sys.refresh_memory();

    let available_ram_gb = sys.available_memory() as f64 / 1024.0 / 1024.0 / 1024.0;
    let ram_based_limit = (available_ram_gb / 4.0).floor() as usize;

    let num_threads = if paths.len() == 1 {
        1
    } else {
        available_cores.min(ram_based_limit).clamp(1, 4)
    };

    log::info!(
        "Batch Export: {} cores, {:.1} GB free RAM -> {} threads",
        available_cores,
        available_ram_gb,
        num_threads
    );

    let _export_task = tokio::spawn(async move {
        let _task_guard = task_guard;
        let output_folder_path = std::path::Path::new(&output_folder_or_file);
        let total_paths = paths.len();

        let mut base_path_counts: HashMap<String, usize> = HashMap::new();
        let mut export_items = Vec::with_capacity(total_paths);

        for (i, path_str) in paths.into_iter().enumerate() {
            let (source_path, _) = parse_virtual_path(&path_str);
            let source_str = source_path.to_string_lossy().to_string();
            let count = base_path_counts.entry(source_str.clone()).or_insert(0);
            *count += 1;

            let mut explicit_vc = None;
            if let Some(idx) = path_str.rfind("vc=") {
                let id_str = path_str[idx + 3..].split('&').next().unwrap_or("");
                if let Ok(id) = id_str.parse::<u32>() {
                    explicit_vc = Some(id);
                }
            }
            if explicit_vc.is_none() {
                let lower = path_str.to_lowercase();
                if let Some(idx) = lower.rfind("_vc") {
                    let id_str: String = lower[idx + 3..]
                        .chars()
                        .take_while(|c| c.is_ascii_digit())
                        .collect();
                    if let Ok(id) = id_str.parse::<u32>() {
                        explicit_vc = Some(id);
                    }
                }
            }
            export_items.push((i, path_str, *count, explicit_vc));
        }

        let used_paths = Arc::new(Mutex::new(std::collections::HashSet::new()));
        let semaphore = Arc::new(tokio::sync::Semaphore::new(num_threads));
        let mut join_handles = Vec::new();

        for (
            (
                (global_index, image_path_str, appearance_count, explicit_vc),
                (js_adjustments, is_current_edit),
            ),
            (frozen_output, reshape_preparation_ms),
        ) in export_items
            .into_iter()
            .zip(frozen_adjustments)
            .zip(frozen_outputs)
        {
            if cancellation_token.load(Ordering::SeqCst) {
                break;
            }
            let permit = semaphore.clone().acquire_owned().await.unwrap();
            if cancellation_token.load(Ordering::SeqCst) {
                drop(permit);
                break;
            }

            let app_handle_clone = app_handle.clone();
            let context_clone = Arc::clone(&context);
            let progress_counter_clone = Arc::clone(&progress_counter);
            let output_folder_path = output_folder_path.to_path_buf();
            let base_origin_folders = base_origin_folders.clone();
            let export_settings = export_settings.clone();
            let output_format = output_format.clone();
            let settings = settings.clone();
            let cancellation_token_clone = Arc::clone(&cancellation_token);
            let used_paths_clone = Arc::clone(&used_paths);

            let handle = tokio::task::spawn_blocking(move || {
                let image_started = Instant::now();
                let mut published_path: Option<PathBuf> = None;
                let result: Result<(), String> = (|| {
                    ensure_export_not_cancelled(&cancellation_token_clone)?;
                    // 无可处理人脸时不再跳过整张图：与预览一致，重塑对该脸退化为 identity，其余调整照常导出。
                    let frozen_output = frozen_output?;
                    let state = app_handle_clone.state::<AppState>();
                    let (source_path, _) = parse_virtual_path(&image_path_str);
                    let source_path_str = source_path.to_string_lossy().to_string();
                    let is_raw = is_raw_file(&source_path_str);
                    let original_path = std::path::Path::new(&source_path_str);
                    let file_date = exif_processing::get_creation_date_from_path(original_path);

                    let filename_template = export_settings
                        .filename_template
                        .as_deref()
                        .unwrap_or("{original_filename}_edited");

                    let mut new_stem = generate_filename_from_template(
                        filename_template,
                        original_path,
                        global_index + 1,
                        total_paths,
                        &file_date,
                    );

                    if let Some(vc_id) = explicit_vc {
                        new_stem = format!("{}_VC{:02}", new_stem, vc_id);
                    } else if appearance_count > 1 {
                        new_stem = format!("{}_VC{:02}", new_stem, appearance_count - 1);
                    }

                    let new_filename = format!("{}.{}", new_stem, output_format);

                    let mut output_path =
                        if export_settings.destination_type.as_deref() == Some("originalFolder") {
                            let mut dir = source_path
                                .parent()
                                .unwrap_or(std::path::Path::new(""))
                                .to_path_buf();
                            if let Some(sub) = &export_settings.subfolder {
                                let mut trimmed = sub.trim();

                                while trimmed.starts_with('/') || trimmed.starts_with('\\') {
                                    trimmed = &trimmed[1..];
                                }

                                if !trimmed.is_empty() {
                                    dir = dir.join(trimmed);
                                }
                            }

                            if let Err(e) = std::fs::create_dir_all(&dir) {
                                return Err(format!(
                                    "Failed to create export subdirectory '{}': {}",
                                    dir.display(),
                                    e
                                ));
                            }

                            dir.join(&new_filename)
                        } else if is_explicit_file_path && total_paths == 1 {
                            output_folder_path.clone()
                        } else if export_settings.preserve_folders {
                            if let Some(rel_dir) = relative_export_dir_for_preserved_folders(
                                source_path.as_path(),
                                &base_origin_folders,
                            ) {
                                let full_dir = output_folder_path.join(rel_dir);
                                if let Err(e) = std::fs::create_dir_all(&full_dir) {
                                    return Err(format!(
                                        "Failed to create export subdirectory '{}': {}",
                                        full_dir.display(),
                                        e
                                    ));
                                }
                                full_dir.join(&new_filename)
                            } else {
                                output_folder_path.join(&new_filename)
                            }
                        } else {
                            output_folder_path.join(&new_filename)
                        };

                    let extension = output_format.to_lowercase();

                    // 编号避让不能把“目标就是源”悄悄改成成功导出。
                    if let Some(frozen) = &frozen_output {
                        frozen.source().reject_source_target(&output_path)?;
                    }
                    if !(is_explicit_file_path && total_paths == 1) {
                        let mut used = used_paths_clone.lock().unwrap();
                        let parent_dir = output_path
                            .parent()
                            .unwrap_or(std::path::Path::new(""))
                            .to_path_buf();
                        let mut counter = 1;

                        while output_path.exists() || used.contains(&output_path) {
                            let incremented_filename =
                                format!("{}_{}.{}", new_stem, counter, extension);
                            output_path = parent_dir.join(&incremented_filename);
                            counter += 1;
                        }
                        used.insert(output_path.clone());
                    }

                    let current = || !cancellation_token_clone.load(Ordering::SeqCst);
                    let actual_output_format = output_path
                        .extension()
                        .and_then(|s| s.to_str())
                        .unwrap_or(output_format.as_str());
                    if let Some((output, _)) = prepare_reshape_output(
                        &image_path_str,
                        &js_adjustments,
                        &export_settings,
                        actual_output_format,
                        &app_handle_clone,
                        &settings,
                        None,
                        frozen_output.as_ref(),
                        &current,
                    )? {
                        #[cfg(target_os = "android")]
                        return Err("Android 活动重塑图库事务尚未支持".into());
                        #[cfg(not(target_os = "android"))]
                        {
                            // 有效 identity 保留旧像素算法，严格源与冻结 lease 持有至最终发布。
                            let image = match &output.image {
                                Some(image) => {
                                    let (img, _) = apply_export_geometry_and_watermark(
                                        image.clone(),
                                        &export_settings,
                                    )?;
                                    apply_info_frame_for_path(img, &image_path_str, &export_settings)
                                }
                                None => {
                                    let frozen = frozen_output
                                        .as_ref()
                                        .ok_or("有效 identity 缺少启动快照")?;
                                    let image = process_frozen_identity(
                                        frozen,
                                        &js_adjustments,
                                        &app_handle_clone,
                                        &settings,
                                        render_output_precision(
                                            actual_output_format,
                                            &export_settings,
                                        ),
                                        &current,
                                    )?;
                                    let (img, _) =
                                        apply_export_geometry_and_watermark(image, &export_settings)?;
                                    apply_info_frame_for_path(img, &image_path_str, &export_settings)
                                }
                            };
                            published_path = Some(publish_reshape_export(
                                &output,
                                &image,
                                &output_path,
                                &export_settings,
                                !(is_explicit_file_path && total_paths == 1),
                                &state,
                                &current,
                            )?);
                            return Ok(());
                        }
                    }

                    if extension == "cube" {
                        let cube_bytes = export_adjustments_as_lut(
                            &js_adjustments,
                            &source_path_str,
                            &context_clone,
                            &state,
                            &app_handle_clone,
                            &cancellation_token_clone,
                        )?;
                        ensure_export_not_cancelled(&cancellation_token_clone)?;
                        #[cfg(target_os = "android")]
                        {
                            let file_name = output_path
                                .file_name()
                                .and_then(|name| name.to_str())
                                .ok_or_else(|| "Missing Android LUT file name".to_string())?;
                            crate::android_integration::save_file_bytes_to_android_downloads(
                                file_name,
                                "application/octet-stream",
                                &cube_bytes,
                            )?;
                        }
                        #[cfg(not(target_os = "android"))]
                        fs::write(&output_path, cube_bytes).map_err(|e| e.to_string())?;
                        ensure_export_not_cancelled(&cancellation_token_clone)?;
                        return Ok(());
                    }

                    let base_image = if is_current_edit
                        && !crate::portrait_processing::enabled(&js_adjustments)
                    {
                        let snapshot = state
                            .original_image
                            .lock()
                            .unwrap()
                            .clone()
                            .filter(|image| image.path == image_path_str);
                        match snapshot {
                            Some(image) => {
                                composite_patches_on_image(&image.image, &js_adjustments)
                                    .map_err(|e| format!("Failed to composite AI patches: {}", e))?
                            }
                            None => {
                                let bytes =
                                    fs::read(&source_path_str).map_err(|e| e.to_string())?;
                                load_and_composite(
                                    &bytes,
                                    &source_path_str,
                                    &js_adjustments,
                                    false,
                                    &settings,
                                    None,
                                )
                                .map_err(|e| format!("Failed to load fallback image: {}", e))?
                            }
                        }
                    } else {
                        match read_file_mapped(Path::new(&source_path_str)) {
                            Ok(mmap) => load_and_composite(
                                &mmap,
                                &source_path_str,
                                &js_adjustments,
                                false,
                                &settings,
                                None,
                            )
                            .map_err(|e| format!("Failed to load from mmap: {}", e))?,
                            Err(_) => {
                                let bytes =
                                    fs::read(&source_path_str).map_err(|e| e.to_string())?;
                                load_and_composite(
                                    &bytes,
                                    &source_path_str,
                                    &js_adjustments,
                                    false,
                                    &settings,
                                    None,
                                )
                                .map_err(|e| format!("Failed to load from bytes: {}", e))?
                            }
                        }
                    };
                    ensure_export_not_cancelled(&cancellation_token_clone)?;

                    let mut main_export_adjustments = js_adjustments.clone();
                    if export_settings.export_masks
                        && let Some(obj) = main_export_adjustments.as_object_mut()
                    {
                        obj.insert("masks".to_string(), serde_json::json!([]));
                    }

                    let actual_output_format = output_path
                        .extension()
                        .and_then(|extension| extension.to_str())
                        .unwrap_or(output_format.as_str());

                    let final_image = process_image_for_export(
                        &image_path_str,
                        &base_image,
                        &main_export_adjustments,
                        &export_settings,
                        &context_clone,
                        &state,
                        is_raw,
                        &app_handle_clone,
                        actual_output_format,
                    )?;
                    ensure_export_not_cancelled(&cancellation_token_clone)?;
                    save_image_with_metadata(
                        &final_image,
                        &output_path,
                        &source_path_str,
                        &export_settings,
                    )?;
                    ensure_export_not_cancelled(&cancellation_token_clone)?;

                    if export_settings.preserve_timestamps {
                        set_timestamps_from_exif(Path::new(&source_path_str), &output_path);
                    }
                    ensure_export_not_cancelled(&cancellation_token_clone)?;

                    if export_settings.export_masks {
                        export_masks_for_image(
                            &base_image,
                            &js_adjustments,
                            &export_settings,
                            &output_path,
                            &source_path_str,
                            &image_path_str,
                            &context_clone,
                            &state,
                            is_raw,
                            &app_handle_clone,
                            &cancellation_token_clone,
                        )?;
                    }

                    Ok(())
                })();

                let completed = completed_export_result(
                    result,
                    published_path,
                    cancellation_token_clone.load(Ordering::SeqCst),
                );
                let (outcome, reason) = match &completed {
                    Ok(_) => batch_outcome(None),
                    Err(error) => {
                        log::error!("Export failed for {}: {}", image_path_str, error);
                        batch_outcome(Some(error))
                    }
                };
                let item = BatchExportItem {
                    elapsed_ms: u64::try_from(image_started.elapsed().as_millis())
                        .unwrap_or(u64::MAX),
                    outcome,
                    output_path: completed
                        .as_ref()
                        .ok()
                        .and_then(Option::as_ref)
                        .map(|path| path.to_string_lossy().into_owned()),
                    path: image_path_str.clone(),
                    reason,
                    reshape_preparation_ms,
                };
                if item.output_path.is_some() || !cancellation_token_clone.load(Ordering::SeqCst) {
                    let current_progress =
                        progress_counter_clone.fetch_add(1, Ordering::SeqCst) + 1;
                    let _ = app_handle_clone.emit(
                        "batch-export-progress",
                        serde_json::json!({
                            "current": current_progress,
                            "total": total_paths,
                            "path": &item.path,
                            "outputPath": &item.output_path,
                            "batchItem": &item,
                        }),
                    );
                }

                drop(permit);
                item
            });

            join_handles.push(handle);
        }

        let mut results = Vec::new();
        for handle in join_handles {
            match handle.await {
                Ok(item) => results.push(item),
                Err(error) => {
                    let message = format!("Thread crashed: {error}");
                    let (outcome, reason) = batch_outcome(Some(&message));
                    results.push(BatchExportItem {
                        elapsed_ms: 0,
                        outcome,
                        output_path: None,
                        path: String::new(),
                        reason,
                        reshape_preparation_ms: 0,
                    });
                }
            }
        }

        let batch_summary = batch_summary(
            results,
            u64::try_from(batch_started.elapsed().as_millis()).unwrap_or(u64::MAX),
        );
        let completed_outputs: Vec<_> = batch_summary
            .items
            .iter()
            .filter(|item| item.outcome == BatchExportOutcome::Success)
            .filter_map(|item| item.output_path.clone())
            .collect();
        let completed_count = batch_summary.succeeded;
        // 跳过的图同样没有产出文件，终态必须按未成功上报，不能落到 export-complete。
        let errors: Vec<_> = batch_summary
            .items
            .iter()
            .filter(|item| item.outcome != BatchExportOutcome::Success)
            .filter_map(|item| item.reason.as_deref())
            .collect();
        let error_count = batch_summary.failed + batch_summary.skipped;
        let export_state = app_handle.state::<AppState>();
        let finalized = finish_export_task(
            &export_state.export_task_token,
            &cancellation_token,
            |cancelled| {
                if cancelled {
                    log::info!("Batch export cancelled and worker cleanup completed");
                    let _ = app_handle.emit(
                        "export-cancelled",
                        serde_json::json!({ "completedOutputs": completed_outputs, "batchSummary": &batch_summary })
                    );
                    return;
                }

                for error in &errors {
                    log::error!("Export error: {}", error);
                    if total_paths == 1 {
                        let _ = app_handle.emit(
                            "export-error",
                            serde_json::json!({
                                "message": error, "completedOutputs": &completed_outputs, "batchSummary": &batch_summary,
                            }),
                        );
                    }
                }

                if error_count > 0 && total_paths > 1 {
                    let _ = app_handle.emit(
                        "export-error",
                        serde_json::json!({
                            "message": format!("{error_count} of {total_paths} exports failed"),
                            "completedOutputs": &completed_outputs,
                            "batchSummary": &batch_summary,
                        }),
                    );
                } else if error_count == 0 {
                    let _ = app_handle.emit(
                        "batch-export-progress",
                        serde_json::json!({ "current": total_paths, "total": total_paths, "path": "" }),
                    );
                    let _ = app_handle.emit(
                        "export-complete",
                        serde_json::json!({ "completedOutputs": completed_outputs, "batchSummary": &batch_summary })
                    );
                }
            },
        );

        if !finalized {
            log::warn!("Ignoring terminal events from a stale export task");
        }

        if let Some(tx) = completion_tx {
            let unfinished = total_paths.saturating_sub(completed_count);
            if error_count > 0 || (cancellation_token.load(Ordering::SeqCst) && unfinished > 0) {
                let _ = tx.send(Err(error_count.max(unfinished)));
            } else {
                let _ = tx.send(Ok(()));
            }
        }
    });

    Ok(())
}

#[allow(clippy::too_many_arguments)]
#[tauri::command]
pub async fn export_images(
    paths: Vec<String>,
    output_folder_or_file: String,
    is_explicit_file_path: bool,
    base_origin_folders: Vec<String>,
    export_settings: ExportSettings,
    output_format: String,
    current_edit_path: Option<String>,
    current_edit_adjustments: Option<Value>,
    state: tauri::State<'_, AppState>,
    app_handle: tauri::AppHandle,
) -> Result<(), String> {
    export_images_impl(
        paths,
        output_folder_or_file,
        is_explicit_file_path,
        base_origin_folders,
        export_settings,
        output_format,
        ExportAdjustmentsMode::UseSidecars {
            active_path: current_edit_path,
            active_adjustments: current_edit_adjustments,
        },
        state,
        app_handle,
        None,
    )
    .await
}

pub async fn run_headless_export(
    session: crate::launch_request::HeadlessExportSession,
    app_handle: tauri::AppHandle,
) -> Result<(), String> {
    println!("Starting headless export...");
    // 命令行导出同样需要已在线校验的登录态
    if crate::qr_login::verify_saved(&app_handle).await?.is_none() {
        return Err("请先在修了么桌面端扫码登录后再使用命令行导出".to_string());
    }
    let state = app_handle.state::<crate::AppState>();

    let source_path = std::path::Path::new(&session.source);
    if !source_path.exists() {
        return Err(format!("Source path does not exist: {}", session.source));
    }

    let mut paths = Vec::new();
    if source_path.is_dir() {
        let images = crate::file_management::list_images_recursive(
            session.source.clone(),
            app_handle.clone(),
        )?;
        paths = images.into_iter().map(|img| img.path).collect();
    } else {
        paths.push(session.source.clone());
    }

    if paths.is_empty() {
        return Err("No supported images found at the source path.".to_string());
    }

    let output_path = std::path::Path::new(&session.output);
    let is_explicit_file_path =
        paths.len() == 1 && output_path.extension().is_some() && !output_path.is_dir();

    if is_explicit_file_path {
        if let Some(parent) = output_path.parent() {
            std::fs::create_dir_all(parent)
                .map_err(|e| format!("Failed to create output parent directory: {}", e))?;
        }
    } else {
        std::fs::create_dir_all(&session.output)
            .map_err(|e| format!("Failed to create output directory: {}", e))?;
    }

    println!("Found {} images to export. Processing...", paths.len());

    let export_settings = ExportSettings {
        jpeg_quality: session.quality,
        tiff_bit_depth: session.tiff_bit_depth,
        resize: None,
        border: None,
        pad: None,
        keep_metadata: session.keep_metadata,
        preserve_timestamps: true,
        strip_gps: false,
        filename_template: None,
        watermark: None,
        export_masks: false,
        preserve_folders: true,
        destination_type: None,
        subfolder: None,
        frame_png_by_path: None,
    };

    let mut custom_adjustments = None;
    if let Some(adj_path) = &session.adjustments_override {
        let content = std::fs::read_to_string(adj_path)
            .map_err(|e| format!("Failed to read adjustments file: {}", e))?;
        let json: serde_json::Value = serde_json::from_str(&content)
            .map_err(|e| format!("Failed to parse adjustments JSON: {}", e))?;
        custom_adjustments = Some(json);
        println!(
            "Loaded custom adjustments to override sidecars from: {}",
            adj_path
        );
    }

    let (tx, rx) = tokio::sync::oneshot::channel();

    let mode = if let Some(adj) = custom_adjustments {
        ExportAdjustmentsMode::GlobalOverride(adj)
    } else {
        ExportAdjustmentsMode::UseSidecars {
            active_path: None,
            active_adjustments: None,
        }
    };

    export_images_impl(
        paths,
        session.output,
        is_explicit_file_path,
        vec![session.source],
        export_settings,
        session.format,
        mode,
        state.clone(),
        app_handle.clone(),
        Some(tx),
    )
    .await?;

    match rx.await {
        Ok(Ok(())) => Ok(()),
        Ok(Err(errors)) => Err(format!("Export completed with {} errors.", errors)),
        Err(_) => Err("Export task panicked or was cancelled.".to_string()),
    }
}

#[tauri::command]
pub fn cancel_export(
    state: tauri::State<AppState>,
    app_handle: tauri::AppHandle,
) -> Result<(), String> {
    match request_export_cancellation(&state.export_task_token, || {
        let _ = app_handle.emit("export-cancelling", ());
    }) {
        ExportCancellationRequest::Requested => {
            log::info!("Export cancellation requested; workers will stop at the next checkpoint");
        }
        ExportCancellationRequest::AlreadyRequested => {
            log::info!("Export cancellation was already requested");
        }
        ExportCancellationRequest::NoActiveTask => {
            return Err("No export task is currently running.".to_string());
        }
    }
    Ok(())
}

#[tauri::command]
pub async fn estimate_export_sizes(
    paths: Vec<String>,
    export_settings: ExportSettings,
    output_format: String,
    current_edit_path: Option<String>,
    current_edit_adjustments: Option<Value>,
    state: tauri::State<'_, AppState>,
    app_handle: tauri::AppHandle,
) -> Result<usize, String> {
    use crate::portrait_render::RenderIntent;
    if paths.is_empty() {
        return Ok(0);
    }
    let has_info_frame = export_settings
        .frame_png_by_path
        .as_ref()
        .is_some_and(|m| !m.is_empty());
    let mut frozen = Vec::with_capacity(paths.len());
    let mut has_reshape = false;
    for path in &paths {
        let (_, sidecar) = parse_virtual_path(path);
        let mut adjustments = if Some(path) == current_edit_path.as_ref() {
            current_edit_adjustments
                .clone()
                .unwrap_or_else(|| exif_processing::load_sidecar(&sidecar).adjustments)
        } else {
            exif_processing::load_sidecar(&sidecar).adjustments
        };
        hydrate_adjustments(&state, &mut adjustments);
        let intent = reshape_export_intent(
            &adjustments,
            &output_format,
            export_settings.export_masks,
            false,
        )?;
        has_reshape |= intent == RenderIntent::Prepare;
        frozen.push((adjustments, intent));
    }
    let settings = load_settings(app_handle.clone()).unwrap_or_default();
    let estimated = tauri::async_runtime::spawn_blocking(move || {
        let state = app_handle.state::<AppState>();
        if !has_reshape {
            let first_adjustments = frozen.into_iter().next().ok_or("估算参数缺失")?.0;
            return estimate_export_sizes_legacy(
                paths,
                export_settings,
                output_format,
                current_edit_path,
                current_edit_adjustments,
                first_adjustments,
                &settings,
                state,
                app_handle.clone(),
            );
        }
        let mut total = 0usize;
        for (path, (adjustments, intent)) in paths.into_iter().zip(frozen) {
            let size = if intent == RenderIntent::Prepare {
                estimate_reshape_image(
                    &path,
                    &adjustments,
                    &export_settings,
                    &output_format,
                    &app_handle,
                    &settings,
                )?
            } else {
                estimate_export_sizes_legacy(
                    vec![path],
                    export_settings.clone(),
                    output_format.clone(),
                    current_edit_path.clone(),
                    Some(adjustments.clone()),
                    adjustments,
                    &settings,
                    state.clone(),
                    app_handle.clone(),
                )?
            };
            total = total.checked_add(size).ok_or("导出体积估算溢出")?;
        }
        Ok(total)
    })
    .await
    .map_err(|e| e.to_string())??;
    // 信息边框约增加 11% 高度，体积粗略上浮
    Ok(if has_info_frame {
        estimated.saturating_add(estimated / 9)
    } else {
        estimated
    })
}

fn estimate_reshape_image(
    path: &str,
    adjustments: &Value,
    export_settings: &ExportSettings,
    output_format: &str,
    app: &tauri::AppHandle,
    settings: &crate::AppSettings,
) -> Result<usize, String> {
    let state = app.state::<AppState>();
    let frozen = state
        .portrait_render
        .freeze_output(app, path, adjustments, settings, &|| true)?;
    let (mut output, dimensions) = prepare_reshape_output(
        path,
        adjustments,
        export_settings,
        output_format,
        app,
        settings,
        Some(1280),
        Some(&frozen),
        &|| true,
    )?
    .ok_or("重塑估算缺少冻结请求")?;
    let sample = estimate_dimensions(dimensions, 1280)?;
    let mut image = if let Some(image) = output.image.take() {
        image
    } else {
        let rendered = process_frozen_identity(
            &frozen,
            adjustments,
            app,
            settings,
            render_output_precision(output_format, export_settings),
            &|| true,
        )?;
        rendered.resize_exact(sample[0], sample[1], imageops::FilterType::Lanczos3)
    };
    if [image.width(), image.height()] != sample {
        return Err("估算输出尺寸不匹配冻结窗口".into());
    }
    if let Some(watermark) = &export_settings.watermark {
        apply_watermark(&mut image, watermark)?;
    }
    let bytes = encode_image_to_bytes(
        &image,
        output_format,
        export_settings.jpeg_quality,
        export_settings.tiff_bit_depth,
    )?;
    let estimate = extrapolate_encoded_size(bytes.len(), dimensions, sample)?;
    output
        .lease
        .commit(&output.source, &output.stamp, &|| true, || Ok(estimate))
}

#[allow(clippy::too_many_arguments)]
fn estimate_export_sizes_legacy(
    paths: Vec<String>,
    export_settings: ExportSettings,
    output_format: String,
    current_edit_path: Option<String>,
    current_edit_adjustments: Option<Value>,
    frozen_first_adjustments: Value,
    settings: &crate::AppSettings,
    state: tauri::State<'_, AppState>,
    app_handle: tauri::AppHandle,
) -> Result<usize, String> {
    if paths.is_empty() {
        return Ok(0);
    }
    // 此层只接受外层已 hydrate 的冻结值，禁止重新读 sidecar 或 id 缓存。
    if reshape_export_intent(
        &frozen_first_adjustments,
        &output_format,
        export_settings.export_masks,
        false,
    )? != crate::portrait_render::RenderIntent::Bypass
    {
        return Err("活动重塑不能进入旧估算路径".into());
    }
    if output_format.eq_ignore_ascii_case("cube") {
        return 1_050_000usize
            .checked_mul(paths.len())
            .ok_or_else(|| "导出体积估算溢出".into());
    }

    let first_path = &paths[0];
    let (source_path, _) = parse_virtual_path(first_path);
    let source_path_str = source_path.to_string_lossy().to_string();

    let context = get_or_init_gpu_context(&state, &app_handle)?;
    let is_current_edit = Some(first_path) == current_edit_path.as_ref();
    let is_raw = is_raw_file(&source_path_str);

    let single_image_extrapolated_size: usize = if is_current_edit
        && current_edit_adjustments.is_some()
    {
        let _publication = state.lock_preview_epoch();
        let loaded_image = state
            .original_image
            .lock()
            .unwrap()
            .clone()
            .filter(|image| image.path == *first_path)
            .ok_or("当前照片已切换，请重新估算导出大小")?;
        let adjustments_clone = frozen_first_adjustments;
        drop(_publication);
        // 编辑器旧 transform hash 不覆盖等长补丁变化，估算不复用这个缓存。
        let warped = crate::adjustment_utils::prepare_patched_warped(
            &loaded_image.image,
            &adjustments_clone,
        )?;
        let (transformed, unscaled_crop_offset) =
            crate::adjustment_utils::apply_spatial_transformations(
                warped.as_ref(),
                &adjustments_clone,
            );
        let full_dimensions = transformed.dimensions();
        let preview_dim = settings.editor_preview_resolution.unwrap_or(1920);
        let preview_image = if full_dimensions.0 > preview_dim || full_dimensions.1 > preview_dim {
            downscale_f32_image(&transformed, preview_dim, preview_dim)
        } else {
            transformed.into_owned()
        };
        let scale = preview_image.width() as f32 / full_dimensions.0 as f32;

        let (img_w, img_h) = preview_image.dimensions();
        let mask_definitions: Vec<MaskDefinition> = adjustments_clone
            .get("masks")
            .and_then(|m| serde_json::from_value(m.clone()).ok())
            .unwrap_or_default();

        let scaled_crop_offset = (
            unscaled_crop_offset.0 * scale,
            unscaled_crop_offset.1 * scale,
        );

        let mask_bitmaps: Vec<ImageBuffer<Luma<u8>, Vec<u8>>> = mask_definitions
            .iter()
            .filter_map(|def| {
                generate_mask_bitmap(
                    def,
                    img_w,
                    img_h,
                    scale,
                    scaled_crop_offset,
                    Some(warped.as_ref()),
                )
            })
            .collect();

        let tm_override = crate::image_processing::resolve_tonemapper_override(settings, is_raw);
        let mut all_adjustments = get_all_adjustments_from_json(
            &adjustments_clone,
            is_raw,
            loaded_image.as_shot_white_balance,
            tm_override,
        );
        all_adjustments.global.show_clipping = 0;

        let lut = adjustments_clone["lutPath"]
            .as_str()
            .and_then(|p| get_or_load_lut(&state, p).ok());
        let unique_hash =
            calculate_full_job_hash(&loaded_image.path, &adjustments_clone).wrapping_add(1);

        let processed_preview = process_and_get_dynamic_image_with_precision(
            &context,
            &state,
            &preview_image,
            unique_hash,
            RenderRequest {
                adjustments: all_adjustments,
                mask_bitmaps: &mask_bitmaps,
                lut,
                roi: None,
                portrait_maps: crate::portrait_processing::prepare_render(
                    &app_handle,
                    &state,
                    &loaded_image.path,
                    &adjustments_clone,
                )?,
            },
            "estimate_export_size",
            render_output_precision(&output_format, &export_settings),
        )?;

        let preview_bytes = encode_image_to_bytes(
            &processed_preview,
            &output_format,
            export_settings.jpeg_quality,
            export_settings.tiff_bit_depth,
        )?;
        let preview_byte_size = preview_bytes.len();

        let (full_w, full_h) = full_dimensions;

        let (final_full_w, final_full_h) =
            compute_export_output_size(full_w, full_h, &export_settings);

        extrapolate_encoded_size(
            preview_byte_size,
            [final_full_w, final_full_h],
            [processed_preview.width(), processed_preview.height()],
        )?
    } else {
        let mut js_adjustments = frozen_first_adjustments;

        const ESTIMATE_DIM: u32 = 1280;

        let file_slice: Vec<u8>;
        let mmap_guard;
        let file_data: &[u8] = match read_file_mapped(Path::new(&source_path_str)) {
            Ok(mmap) => {
                mmap_guard = Some(mmap);
                mmap_guard.as_ref().unwrap()
            }
            Err(_) => {
                file_slice = fs::read(&source_path_str).map_err(|io_err| io_err.to_string())?;
                &file_slice
            }
        };

        let portrait_enabled = crate::portrait_processing::enabled(&js_adjustments);
        let original_image = if portrait_enabled {
            load_and_composite(
                file_data,
                &source_path_str,
                &js_adjustments,
                false,
                &settings,
                None,
            )
        } else {
            load_base_image_from_bytes(file_data, &source_path_str, true, &settings, None)
        }
        .map_err(|e| e.to_string())?;

        let raw_scale_factor = if is_raw && !portrait_enabled {
            crate::raw_processing::get_fast_demosaic_scale_factor(
                file_data,
                original_image.width(),
                original_image.height(),
            )
        } else {
            1.0
        };

        let portrait_adjustments = js_adjustments.clone();
        if let Some(crop_val) = js_adjustments.get_mut("crop")
            && let Ok(c) = serde_json::from_value::<Crop>(crop_val.clone())
        {
            *crop_val = serde_json::to_value(Crop {
                x: c.x * raw_scale_factor as f64,
                y: c.y * raw_scale_factor as f64,
                width: c.width * raw_scale_factor as f64,
                height: c.height * raw_scale_factor as f64,
            })
            .unwrap_or(serde_json::Value::Null);
        }

        let warped = crate::image_processing::apply_geometry_warp(
            Cow::Borrowed(&original_image),
            &js_adjustments,
        );
        let warped = crate::lens_blur::apply_lens_blur(warped, &js_adjustments);
        let (transformed_shrunk_res, unscaled_crop_offset) =
            crate::adjustment_utils::apply_spatial_transformations(
                warped.as_ref(),
                &js_adjustments,
            );
        let (shrunk_w, shrunk_h) = transformed_shrunk_res.dimensions();

        let preview_base = if shrunk_w > ESTIMATE_DIM || shrunk_h > ESTIMATE_DIM {
            downscale_f32_image(transformed_shrunk_res.as_ref(), ESTIMATE_DIM, ESTIMATE_DIM)
        } else {
            transformed_shrunk_res.into_owned()
        };

        let (preview_w, preview_h) = preview_base.dimensions();
        let gpu_scale = if shrunk_w > 0 {
            preview_w as f32 / shrunk_w as f32
        } else {
            1.0
        };
        let total_scale = gpu_scale * raw_scale_factor;

        let mask_definitions: Vec<MaskDefinition> = js_adjustments
            .get("masks")
            .and_then(|m| serde_json::from_value(m.clone()).ok())
            .unwrap_or_default();
        let scaled_crop_offset = (
            unscaled_crop_offset.0 * gpu_scale,
            unscaled_crop_offset.1 * gpu_scale,
        );

        let mask_bitmaps: Vec<ImageBuffer<Luma<u8>, Vec<u8>>> = mask_definitions
            .iter()
            .filter_map(|def| {
                generate_mask_bitmap(
                    def,
                    preview_w,
                    preview_h,
                    total_scale,
                    scaled_crop_offset,
                    Some(warped.as_ref()),
                )
            })
            .collect();

        let tm_override = crate::image_processing::resolve_tonemapper_override(settings, is_raw);
        let mut all_adjustments = get_all_adjustments_from_json(
            &js_adjustments,
            is_raw,
            as_shot_white_balance(&source_path_str),
            tm_override,
        );
        all_adjustments.global.show_clipping = 0;

        let lut = js_adjustments["lutPath"]
            .as_str()
            .and_then(|p| get_or_load_lut(&state, p).ok());
        let unique_hash =
            calculate_full_job_hash(&source_path_str, &js_adjustments).wrapping_add(1);

        let processed_preview = process_and_get_dynamic_image_with_precision(
            &context,
            &state,
            &preview_base,
            unique_hash,
            RenderRequest {
                adjustments: all_adjustments,
                mask_bitmaps: &mask_bitmaps,
                lut,
                roi: None,
                portrait_maps: crate::portrait_processing::prepare_render(
                    &app_handle,
                    &state,
                    first_path,
                    &portrait_adjustments,
                )?,
            },
            "estimate_batch_export_size",
            render_output_precision(&output_format, &export_settings),
        )?;

        let preview_bytes = encode_image_to_bytes(
            &processed_preview,
            &output_format,
            export_settings.jpeg_quality,
            export_settings.tiff_bit_depth,
        )?;
        let single_image_estimated_size = preview_bytes.len();

        let full_w = (shrunk_w as f32 / raw_scale_factor).round() as u32;
        let full_h = (shrunk_h as f32 / raw_scale_factor).round() as u32;

        let (final_full_w, final_full_h) =
            compute_export_output_size(full_w, full_h, &export_settings);

        extrapolate_encoded_size(
            single_image_estimated_size,
            [final_full_w, final_full_h],
            [processed_preview.width(), processed_preview.height()],
        )?
    };

    single_image_extrapolated_size
        .checked_mul(paths.len())
        .ok_or_else(|| "导出体积估算溢出".into())
}

#[cfg(test)]
mod portrait_export_tests {
    use super::*;
    use serde_json::json;

    fn settings() -> ExportSettings {
        serde_json::from_value(json!({
            "jpegQuality":90,"tiffBitDepth":16,"resize":null,"keepMetadata":false,
            "stripGps":true,"filenameTemplate":null,"watermark":null
        }))
        .unwrap()
    }

    fn active(amount: u32) -> Value {
        json!({"portrait":{"slimFace":amount},"portraitReshape":{
            "schemaVersion":1,"engineVersion":"mesh-arap-v1","enabled":true,"faces":[]
        }})
    }

    #[test]
    fn portrait_export_format_gates_preserve_legacy_and_reject_active_exceptions() {
        use crate::portrait_render::RenderIntent;
        for format in ["cube", "CuBe"] {
            assert!(reshape_export_intent(&active(30), format, false, false).is_err());
            assert_eq!(
                reshape_export_intent(&active(0), format, true, true).unwrap(),
                RenderIntent::Bypass
            );
            assert_eq!(
                reshape_export_intent(&json!({"portrait":{"slimFace":100}}), format, false, false)
                    .unwrap(),
                RenderIntent::Bypass
            );
        }
        assert!(reshape_export_intent(&active(30), "png", true, false).is_err());
        assert!(reshape_export_intent(&active(30), "png", false, true).is_err());
        assert_eq!(
            reshape_export_intent(&active(30), "TIFF", false, false).unwrap(),
            RenderIntent::Prepare
        );
        let mut invalid = active(0);
        invalid["portraitReshape"]["schemaVersion"] = json!(99);
        assert!(reshape_export_intent(&invalid, "jpeg", false, false).is_err());
    }

    #[test]
    fn portrait_export_size_arithmetic_is_checked_and_never_zero_sized() {
        assert_eq!(extrapolate_encoded_size(17, [12, 20], [3, 5]).unwrap(), 272);
        assert_eq!(extrapolate_encoded_size(0, [12, 20], [3, 5]).unwrap(), 0);
        assert!(extrapolate_encoded_size(17, [0, 20], [3, 5]).is_err());
        assert!(extrapolate_encoded_size(17, [12, 20], [0, 5]).is_err());
        assert!(extrapolate_encoded_size(usize::MAX, [2, 2], [1, 1]).is_err());
        assert_eq!(estimate_dimensions([32768, 1], 1280).unwrap(), [1280, 1]);
        assert_eq!(estimate_dimensions([513, 257], 1280).unwrap(), [513, 257]);
        assert_eq!(
            estimate_dimensions([3001, 2001], 1280).unwrap(),
            [1280, 853]
        );
        assert!(estimate_dimensions([100, 100], 0).is_err());
    }

    #[test]
    fn portrait_batch_summary_classifies_skips_and_percentiles_from_items() {
        assert_eq!(batch_failure_reason("模型初始化失败"), "modelError");
        assert_eq!(batch_failure_reason("unsupported pose"), "unsupportedPose");
        assert_eq!(batch_failure_reason("Export cancelled"), "cancelled");
        let (outcome, reason) = batch_outcome(Some("batch-skip:no face available"));
        assert_eq!(outcome, BatchExportOutcome::Skipped);
        assert_eq!(reason.as_deref(), Some("noFace"));
        let (outcome, reason) = batch_outcome(Some("Surface lost"));
        assert_eq!(outcome, BatchExportOutcome::Failed);
        assert_eq!(reason.as_deref(), Some("renderError"));
        let summary = batch_summary(
            vec![
                BatchExportItem {
                    elapsed_ms: 10,
                    outcome: BatchExportOutcome::Success,
                    output_path: Some("a.jpg".into()),
                    path: "a.raw".into(),
                    reason: None,
                    reshape_preparation_ms: 3,
                },
                BatchExportItem {
                    elapsed_ms: 30,
                    outcome: BatchExportOutcome::Skipped,
                    output_path: None,
                    path: "b.raw".into(),
                    reason: Some("noFace".into()),
                    reshape_preparation_ms: 5,
                },
                BatchExportItem {
                    elapsed_ms: 50,
                    outcome: BatchExportOutcome::Failed,
                    output_path: None,
                    path: "c.raw".into(),
                    reason: Some("modelError".into()),
                    reshape_preparation_ms: 9,
                },
            ],
            100,
        );
        assert_eq!(
            (summary.succeeded, summary.skipped, summary.failed),
            (1, 1, 1)
        );
        assert_eq!(summary.performance.p50_ms, 30);
        assert_eq!(summary.performance.p95_ms, 50);
    }

    #[test]
    fn portrait_export_completion_survives_later_cancellation() {
        let path = PathBuf::from("completed.tiff");
        assert_eq!(
            completed_export_result(Ok(()), Some(path.clone()), true).unwrap(),
            Some(path)
        );
        assert!(completed_export_result(Ok(()), None, true).is_err());
        assert!(completed_export_result(Err("encode failed".into()), None, false).is_err());
        assert_eq!(completed_export_result(Ok(()), None, false).unwrap(), None);
    }

    #[test]
    fn portrait_export_task_token_cleanup_is_once_and_does_not_clear_new_task() {
        let slot = Arc::new(Mutex::new(None));
        let first = register_export_task(&slot).unwrap();
        let guard = ExportTaskGuard::new(Arc::clone(&slot), Arc::clone(&first));
        assert!(register_export_task(&slot).is_err());
        assert_eq!(
            request_export_cancellation(&slot, || {}),
            ExportCancellationRequest::Requested
        );
        assert_eq!(
            request_export_cancellation(&slot, || panic!("取消事件只发一次")),
            ExportCancellationRequest::AlreadyRequested
        );
        assert!(finish_export_task(&slot, &first, |cancelled| assert!(
            cancelled
        )));
        let next = register_export_task(&slot).unwrap();
        drop(guard);
        assert!(Arc::ptr_eq(slot.lock().unwrap().as_ref().unwrap(), &next));
        assert!(finish_export_task(&slot, &next, |cancelled| assert!(
            !cancelled
        )));
        assert!(!finish_export_task(&slot, &first, |_| panic!(
            "旧任务不能再发终态"
        )));
    }

    #[test]
    fn portrait_export_tiff16_watermark_and_encoding_preserve_non_u8_steps() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("watermark.png");
        image::RgbaImage::from_pixel(1, 1, image::Rgba([255, 255, 255, 255]))
            .save(&path)
            .unwrap();
        let mut image = DynamicImage::ImageRgba16(ImageBuffer::from_fn(513, 9, |x, _| {
            image::Rgba([x as u16 * 100, 12345, 54321, 65535])
        }));
        let watermark = WatermarkSettings {
            path: path.to_string_lossy().into_owned(),
            anchor: WatermarkAnchor::TopLeft,
            scale: 25.0,
            spacing: 0.0,
            opacity: 50.0,
        };
        let original = image.to_rgba16();
        apply_watermark(&mut image, &watermark).unwrap();
        assert!(image.as_rgba16().is_some());
        assert_ne!(image.to_rgba16().get_pixel(0, 0), original.get_pixel(0, 0));
        let bytes = encode_image_to_bytes(&image, "TiFf", 90, TiffBitDepth::Sixteen).unwrap();
        let decoded = image::load_from_memory(&bytes).unwrap();
        assert!(decoded.as_rgb16().is_some());
        assert_eq!(
            decoded.to_rgb16().get_pixel(400, 8).0,
            [40000, 12345, 54321]
        );
        assert!(decoded.to_rgb16().pixels().any(|p| p[0] % 257 != 0));
        let settings = settings();
        assert_eq!(
            render_output_precision("TiF", &settings),
            RenderOutputPrecision::SixteenBit
        );
        assert_eq!(
            render_output_precision("jpeg", &settings),
            RenderOutputPrecision::EightBit
        );
    }

    #[test]
    fn portrait_export_metadata_write_failure_is_strict_without_claiming_tiff_support() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("source.png");
        image::RgbaImage::from_pixel(2, 2, image::Rgba([1, 2, 3, 255]))
            .save(&path)
            .unwrap();
        let source = path.to_str().unwrap();
        let invalid = b"not a jpeg".to_vec();
        assert!(
            exif_processing::write_image_with_metadata_strict(
                &mut invalid.clone(),
                source,
                "jpeg",
                true,
                true
            )
            .is_err()
        );
        assert!(
            exif_processing::write_image_with_metadata(
                &mut invalid.clone(),
                source,
                "jpeg",
                true,
                true
            )
            .is_ok()
        );
        for format in ["tiff", "tif", "jxl", "avif"] {
            let mut bytes = invalid.clone();
            exif_processing::write_image_with_metadata_strict(
                &mut bytes, source, format, true, true,
            )
            .unwrap();
            assert_eq!(bytes, invalid);
        }
    }
}
