#[cfg(not(all(target_os = "windows", target_arch = "aarch64")))]
use mimalloc::MiMalloc;

#[cfg(not(all(target_os = "windows", target_arch = "aarch64")))]
#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

// Like println!/eprintln!, but ignores write errors: when stdout/stderr is a
// closed pipe (e.g. headless export | head), println! panics.
macro_rules! cli_println {
    ($($arg:tt)*) => {{
        use std::io::Write as _;
        let _ = writeln!(std::io::stdout(), $($arg)*);
    }};
}

macro_rules! cli_eprintln {
    ($($arg:tt)*) => {{
        use std::io::Write as _;
        let _ = writeln!(std::io::stderr(), $($arg)*);
    }};
}

mod adjustment_utils;
mod ai_commands;
mod ai_connector;
mod ai_processing;
mod android_integration;
mod announcement;
mod requirement_pool;
mod app_config;
mod app_settings;
mod app_state;
mod app_update;
mod apple_raw;
mod cache_utils;
mod camera_tethering;
mod crash_report;
mod culling;
mod denoising;
mod exif_processing;
mod export_processing;
mod face_reshape;
mod file_management;
mod focus_stacking;
mod formats;
mod gpu_processing;
mod guided_perspective;
mod hdr_deghosting;
mod image_loader;
mod image_processing;
mod inpainting;
mod bug_feedback;
mod invite;
mod launch_request;
mod lens_blur;
mod lens_correction;
mod lut_processing;
mod mask_generation;
mod multi_exposure;
mod negative_conversion;
mod panorama_stitching;
mod panorama_utils;
mod pixel_billing;
mod portrait_adapters;
mod portrait_analysis;
mod portrait_analysis_cache;
mod portrait_coordinates;
mod portrait_engine;
mod portrait_gpu;
mod portrait_gpu_field;
mod portrait_gpupixel;
mod portrait_models;
mod portrait_processing;
mod portrait_protection;
mod portrait_render;
mod portrait_reshape;
mod portrait_reshape_fill;
mod portrait_safety;
mod portrait_snapshot;
mod portrait_tiles;
mod portrait_uv_warp;
mod preset_converter;
mod qr_login;
mod raw_processing;
mod sys_config;
mod tagging;
mod tagging_utils;
mod usage_tracking;
mod white_balance;
mod window_customizer;

use std::collections::{HashMap, hash_map::DefaultHasher};
use std::fs;
use std::hash::{Hash, Hasher};
use std::io::Cursor;
use std::io::Write;
use std::panic;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::mpsc::{self, Receiver, Sender};

use std::borrow::Cow;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use base64::{Engine as _, engine::general_purpose};
use image::codecs::jpeg::JpegEncoder;
use image::{DynamicImage, GenericImageView, ImageBuffer, ImageFormat, Luma, RgbImage};
use image_hdr::hdr_merge_images;
use image_hdr::input::HDRInput;
use imgref::ImgRef;
use mozjpeg_rs::{Encoder, Preset};
use rgb::{FromSlice, RGBA8};

use serde::{Deserialize, Serialize};
use serde_json::Value;
use tauri::{Emitter, Manager, ipc::Response};
use tempfile::NamedTempFile;
use tokio::sync::Mutex as TokioMutex;

use crate::cache_utils::{
    DecodedImageCache, calculate_full_job_hash, calculate_geometry_hash, calculate_transform_hash,
    calculate_visual_hash,
};
use crate::file_management::{parse_virtual_path, read_file_mapped};
use crate::formats::is_raw_file;
use crate::hdr_deghosting::{align_hdr_frames, assert_uniform_dimensions, load_hdr_frames};
use crate::image_loader::{composite_patches_on_image, load_and_composite};
use crate::image_processing::{
    Crop, RenderRequest, apply_coarse_rotation, apply_cpu_default_raw_processing, apply_flip,
    apply_geometry_warp, apply_linear_to_srgb, downscale_f32_image, get_all_adjustments_from_json,
    get_or_init_gpu_context, process_and_get_dynamic_image, resolve_tonemapper_override,
    resolve_tonemapper_override_from_handle, warp_image_geometry,
};
use crate::mask_generation::{
    MaskDefinition, generate_mask_bitmap, get_cached_or_generate_mask,
    resolve_warped_image_for_masks,
};
use crate::white_balance::WhiteBalance;
use crate::window_customizer::PinchZoomDisablePlugin;
pub use adjustment_utils::*;
pub use android_integration::*;
pub use app_settings::*;
pub use app_state::*;
pub use launch_request::*;
use tagging_utils::{candidates, hierarchy};

#[cfg(target_os = "macos")]
extern "C" fn force_exit(_signal: libc::c_int) {
    unsafe {
        libc::_exit(0);
    }
}

#[cfg(target_os = "macos")]
pub fn register_exit_handler() {
    unsafe {
        libc::signal(libc::SIGABRT, force_exit as *const () as libc::sighandler_t);
    }
}

#[cfg(not(target_os = "macos"))]
pub fn register_exit_handler() {}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct CommunityPreset {
    pub name: String,
    pub creator: String,
    pub adjustments: Value,
    #[serde(rename = "includeMasks")]
    pub include_masks: Option<bool>,
    #[serde(rename = "includeCropTransform")]
    pub include_crop_transform: Option<bool>,
}

#[derive(serde::Serialize)]
struct ImageDimensions {
    width: u32,
    height: u32,
}

#[derive(serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct WgpuTransformPayload {
    pub window_width: f32,
    pub window_height: f32,
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
    pub clip_x: f32,
    pub clip_y: f32,
    pub clip_width: f32,
    pub clip_height: f32,
    pub bg_primary: [f32; 4],
    pub bg_secondary: [f32; 4],
    pub pixelated: bool,
}

pub fn generate_transformed_preview(
    state: &tauri::State<AppState>,
    loaded_image: &LoadedImage,
    adjustments: &serde_json::Value,
    preview_dim: u32,
) -> Result<(DynamicImage, f32, (f32, f32)), String> {
    let transform_hash = calculate_transform_hash(adjustments);

    let (transformed_full_res, unscaled_crop_offset) = {
        let mut cache_lock = state
            .full_transformed_cache
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        if let Some((hash, img, offset)) = cache_lock.as_ref() {
            if *hash == transform_hash {
                (Arc::clone(img), *offset)
            } else {
                let (arc_img, offset) =
                    compute_full_transformed_res(state, loaded_image, adjustments)?;
                *cache_lock = Some((transform_hash, Arc::clone(&arc_img), offset));
                (arc_img, offset)
            }
        } else {
            let (arc_img, offset) = compute_full_transformed_res(state, loaded_image, adjustments)?;
            *cache_lock = Some((transform_hash, Arc::clone(&arc_img), offset));
            (arc_img, offset)
        }
    };

    let (full_res_w, full_res_h) = transformed_full_res.dimensions();

    let final_preview_base = if full_res_w > preview_dim || full_res_h > preview_dim {
        downscale_f32_image(&transformed_full_res, preview_dim, preview_dim)
    } else {
        (*transformed_full_res).clone()
    };

    let scale_for_gpu = if full_res_w > 0 {
        final_preview_base.width() as f32 / full_res_w as f32
    } else {
        1.0
    };

    Ok((final_preview_base, scale_for_gpu, unscaled_crop_offset))
}

fn compute_full_transformed_res(
    state: &tauri::State<AppState>,
    loaded_image: &LoadedImage,
    adjustments: &serde_json::Value,
) -> Result<(Arc<DynamicImage>, (f32, f32)), String> {
    let geo_hash = crate::cache_utils::calculate_patched_warped_hash(adjustments);

    let warped_arc = {
        let mut cache_lock = state
            .patched_warped_cache
            .lock()
            .unwrap_or_else(|e| e.into_inner());

        if let Some((hash, img)) = cache_lock.as_ref() {
            if *hash == geo_hash {
                Arc::clone(img)
            } else {
                let new_img = compute_patched_and_warped(loaded_image, adjustments)?;
                *cache_lock = Some((geo_hash, Arc::clone(&new_img)));
                new_img
            }
        } else {
            let new_img = compute_patched_and_warped(loaded_image, adjustments)?;
            *cache_lock = Some((geo_hash, Arc::clone(&new_img)));
            new_img
        }
    };

    let (transformed_img, offset) = crate::adjustment_utils::apply_spatial_transformations(
        Cow::Borrowed(warped_arc.as_ref()),
        adjustments,
    );

    Ok((Arc::new(transformed_img.into_owned()), offset))
}

fn compute_patched_and_warped(
    loaded_image: &LoadedImage,
    adjustments: &serde_json::Value,
) -> Result<Arc<DynamicImage>, String> {
    let base = crate::adjustment_utils::prepare_patched_warped(&loaded_image.image, adjustments)?;
    Ok(Arc::new(base.into_owned()))
}

#[tauri::command]
fn get_image_dimensions(path: String) -> Result<ImageDimensions, String> {
    let (source_path, _) = parse_virtual_path(&path);
    image::image_dimensions(&source_path)
        .map(|(width, height)| ImageDimensions { width, height })
        .map_err(|e| e.to_string())
}

#[tauri::command]
fn cancel_thumbnail_generation(
    state: tauri::State<AppState>,
    app_handle: tauri::AppHandle,
) -> Result<(), String> {
    state
        .thumbnail_cancellation_token
        .store(true, Ordering::SeqCst);

    let mut tracker = state.thumbnail_progress.lock().unwrap();
    tracker.total = 0;
    tracker.completed = 0;
    drop(tracker);

    let _ = app_handle.emit(
        "thumbnail-progress",
        serde_json::json!({ "current": 0, "total": 0 }),
    );
    Ok(())
}

pub fn get_cached_full_warped_image(
    state: &tauri::State<AppState>,
    js_adjustments: &serde_json::Value,
) -> Result<Arc<DynamicImage>, String> {
    let geo_hash = calculate_geometry_hash(js_adjustments);

    {
        let cache_lock = state
            .full_warped_cache
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        if let Some((hash, img)) = cache_lock.as_ref()
            && *hash == geo_hash
        {
            return Ok(Arc::clone(img));
        }
    }

    let (base_arc, is_raw) = get_original_image(state)?;
    let mut cow_image = Cow::Borrowed(base_arc.as_ref());

    if is_raw {
        apply_cpu_default_raw_processing(cow_image.to_mut());
    }

    let warped_image = apply_geometry_warp(cow_image, js_adjustments).into_owned();
    let warped_arc = Arc::new(warped_image);

    {
        let mut cache_lock = state
            .full_warped_cache
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        *cache_lock = Some((geo_hash, Arc::clone(&warped_arc)));
    }

    Ok(warped_arc)
}

#[tauri::command]
async fn update_wgpu_transform(
    payload: WgpuTransformPayload,
    state: tauri::State<'_, AppState>,
) -> Result<(), String> {
    let context = match state
        .gpu_context
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .as_ref()
    {
        Some(c) => c.clone(),
        None => return Ok(()),
    };

    tokio::task::spawn_blocking(move || {
        let mut display_lock = context.display.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(display) = display_lock.as_mut() {
            display.latest_transform.rect = [payload.x, payload.y, payload.width, payload.height];
            display.latest_transform.clip = [
                payload.clip_x,
                payload.clip_y,
                payload.clip_width,
                payload.clip_height,
            ];
            display.latest_transform.window = [payload.window_width, payload.window_height];
            display.latest_transform.bg_primary = payload.bg_primary;
            display.latest_transform.bg_secondary = payload.bg_secondary;
            display.latest_transform.pixelated = if payload.pixelated { 1.0 } else { 0.0 };

            context.queue.write_buffer(
                &display.transform_buffer,
                0,
                bytemuck::bytes_of(&display.latest_transform),
            );
            display.render(&context.device, &context.queue);
        }
    })
    .await
    .map_err(|e| format!("Task panicked: {}", e))?;

    Ok(())
}

fn default_levels_json() -> serde_json::Value {
    let channel = serde_json::json!({
        "inputBlack": 0,
        "gamma": 1.0,
        "inputWhite": 255,
        "outputBlack": 0,
        "outputWhite": 255
    });
    serde_json::json!({
        "rgb": channel,
        "red": channel,
        "green": channel,
        "blue": channel
    })
}

/// Force global + mask levels to identity so a re-grade yields the Levels panel input histogram.
fn neutralize_levels_for_input_histogram(adjustments: &mut serde_json::Value) {
    let defaults = default_levels_json();
    adjustments["levels"] = defaults.clone();
    if let Some(masks) = adjustments.get_mut("masks").and_then(|m| m.as_array_mut()) {
        for mask in masks {
            if let Some(mask_adj) = mask.get_mut("adjustments") {
                mask_adj["levels"] = defaults.clone();
            }
        }
    }
}

fn channel_levels_are_identity(obj: &serde_json::Value) -> bool {
    let ib = obj.get("inputBlack").and_then(|v| v.as_f64()).unwrap_or(0.0);
    let iw = obj.get("inputWhite").and_then(|v| v.as_f64()).unwrap_or(255.0);
    let g = obj.get("gamma").and_then(|v| v.as_f64()).unwrap_or(1.0);
    let ob = obj.get("outputBlack").and_then(|v| v.as_f64()).unwrap_or(0.0);
    let ow = obj.get("outputWhite").and_then(|v| v.as_f64()).unwrap_or(255.0);
    (ib - 0.0).abs() < 1e-6
        && (iw - 255.0).abs() < 1e-6
        && (g - 1.0).abs() < 1e-6
        && (ob - 0.0).abs() < 1e-6
        && (ow - 255.0).abs() < 1e-6
}

fn levels_settings_are_identity(levels: &serde_json::Value) -> bool {
    ["rgb", "red", "green", "blue"]
        .iter()
        .all(|ch| channel_levels_are_identity(&levels.get(ch).cloned().unwrap_or_default()))
}

fn adjustments_levels_are_identity(adjustments: &serde_json::Value) -> bool {
    if !levels_settings_are_identity(&adjustments.get("levels").cloned().unwrap_or_default()) {
        return false;
    }
    if let Some(masks) = adjustments.get("masks").and_then(|m| m.as_array()) {
        for mask in masks {
            let levels = mask
                .get("adjustments")
                .and_then(|a| a.get("levels"))
                .cloned()
                .unwrap_or_default();
            if !levels_settings_are_identity(&levels) {
                return false;
            }
        }
    }
    true
}

#[allow(clippy::too_many_arguments)]
fn process_preview_job(
    app_handle: &tauri::AppHandle,
    state: tauri::State<AppState>,
    adjustments_json: serde_json::Value,
    loaded_image: LoadedImage,
    epoch: usize,
    request_id: u64,
    source_revision: u64,
    is_interactive: bool,
    target_resolution: Option<u32>,
    roi: Option<(f32, f32, f32, f32)>,
    request_analytics: bool,
    request_levels_input_analytics: bool,
    compute_waveform: bool,
    active_waveform_channel: Option<&str>,
) -> Result<Vec<u8>, String> {
    use portrait_render::{
        PhotoRenderRequest, PreviewMetadata, PreviewMode, RenderIntent, RenderPurpose,
        RenderQuality,
    };
    use portrait_tiles::PixelRect;
    let fn_start = std::time::Instant::now();
    // 预览执行期间压住后台 AI 美肤预热，首帧不与其争抢模型锁与 CPU。
    let _foreground = state.portrait_state.foreground();
    let adjustments_clone = adjustments_json;
    let current = || {
        state
            .portrait_render
            .editor_is_current(source_revision, epoch as u64)
    };
    portrait_analysis::check_current(&current)?;
    let intent = portrait_render::render_intent(&adjustments_clone)?;
    let intent_label = match intent {
        RenderIntent::Bypass => "bypass",
        RenderIntent::Prepare => "prepare",
    };
    let settings = load_settings(app_handle.clone()).unwrap_or_default();
    let live_quality = settings.live_preview_quality.as_deref().unwrap_or("high");

    let default_preview_dim = settings.editor_preview_resolution.unwrap_or(1920);
    let preview_dim = target_resolution.unwrap_or(default_preview_dim);
    #[cfg(not(any(target_os = "linux", target_os = "android")))]
    let use_wgpu_renderer = settings.use_wgpu_renderer.unwrap_or(true);
    #[cfg(any(target_os = "linux", target_os = "android"))]
    let use_wgpu_renderer = false;

    let has_roi = roi.is_some();
    let (interactive_divisor, interactive_quality) = match live_quality {
        "full" => (1.0_f32, 85_u8),
        "performance" => (if has_roi { 1.8_f32 } else { 1.5_f32 }, 65_u8),
        _ => (if has_roi { 1.4_f32 } else { 1.0_f32 }, 75_u8),
    };
    // 人像交互帧的工作尺寸也受限；单独修肤仍共享原来的分析/AI 层缓存。
    let interactive_divisor = if portrait_processing::enabled(&adjustments_clone)
        || portrait_gpupixel::requested(&adjustments_clone)
    {
        portrait_render::portrait_preview_limit(live_quality).map_or(interactive_divisor, |limit| {
            interactive_divisor.max(preview_dim as f32 / limit as f32)
        })
    } else {
        interactive_divisor
    };

    let mut prepared = if intent == RenderIntent::Prepare {
        let source = state.portrait_render.editor_source(
            &loaded_image.path,
            &settings,
            source_revision,
            &current,
        )?;
        let lease =
            state
                .portrait_render
                .begin_editor_consumer(&source, request_id, epoch as u64)?;
        let coordinates = portrait_coordinates::CoordinateMap::new(
            [source.image().width(), source.image().height()],
            &adjustments_clone,
        )?;
        let target = if is_interactive {
            (preview_dim as f32 / interactive_divisor).max(1.0) as u32
        } else {
            preview_dim
        };
        // 活动几何的 JPEG 和原生显示均提交完整最终预览，避免新 ROI 与旧脸型混合。
        // 协调器仍支持独立 ROI 输出；显示复用 ROI 前须具备同版本完整基底。
        let window = portrait_render::preview_window(
            coordinates.base_dimensions,
            &adjustments_clone["crop"],
            target,
            None,
        )?;
        let mut output = state.portrait_render.render_photo(
            app_handle,
            PhotoRenderRequest {
                source,
                adjustments: adjustments_clone.clone(),
                window: window.clone(),
                quality: if is_interactive {
                    RenderQuality::Interactive
                } else {
                    RenderQuality::HighQuality
                },
                purpose: RenderPurpose::Preview,
                request_id,
                sixteen_bit: false,
            },
            lease,
            &settings,
            &current,
        )?;
        if let Some(image) = output.image.take() {
            let image = Arc::new(image);
            let metadata = PreviewMetadata {
                path: loaded_image.path.clone(),
                stamp: output.stamp.clone(),
                dimensions: window.output_dimensions,
                roi: window.roi,
                mode: if use_wgpu_renderer {
                    PreviewMode::Native
                } else {
                    PreviewMode::Jpeg
                },
                reshape_attenuation: output.reshape_attenuation,
                body_attenuation: output.body_attenuation,
            };
            let rgba = image.as_rgba8().ok_or("预览量化结果不是 RGBA8")?;
            let body = if use_wgpu_renderer {
                Vec::new()
            } else {
                Encoder::new(Preset::BaselineFastest)
                    .quality(if is_interactive {
                        interactive_quality
                    } else {
                        94
                    })
                    .fast_color(true)
                    .encode_imgref(ImgRef::new(
                        rgba.as_raw().as_rgba(),
                        rgba.width() as usize,
                        rgba.height() as usize,
                    ))
                    .map_err(|e| format!("预览编码失败：{e}"))?
            };
            let response = metadata.encode(&body)?;
            if use_wgpu_renderer {
                let context = get_or_init_gpu_context(&state, app_handle)?;
                // 交互预览保留 legacy GPU 热缓存；仅高质/松手帧释放整图工作池腾预算。
                if !is_interactive {
                    let mut processor = state
                        .gpu_processor
                        .lock()
                        .unwrap_or_else(|e| e.into_inner());
                    let mut cache = state
                        .gpu_image_cache
                        .lock()
                        .unwrap_or_else(|e| e.into_inner());
                    drop(processor.take());
                    drop(cache.take());
                }
                let staged =
                    gpu_processing::PreparedPreviewDisplay::prepare(&context, rgba, &current)?;
                state
                    .portrait_render
                    .publish_editor(&output, epoch as u64, &current, || staged.publish(&context))?;
                state
                    .portrait_render
                    .with_committed_preview(&output.stamp, epoch as u64, || {
                        let _ = app_handle.emit("wgpu-frame-ready", &metadata);
                    });
            } else {
                state
                    .portrait_render
                    .publish_editor(&output, epoch as u64, &current, || Ok(()))?;
            }
            if request_analytics
                && window.roi == PixelRect::full(window.output_dimensions)?
                && let Some(sender) = state
                    .analytics_worker_tx
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .as_ref()
            {
                // Prepare 路径：levels 已是 identity 时复用最终图；否则再渲一次 identity levels（尽力而为）。
                let reuse_final_as_levels_input = request_levels_input_analytics
                    && adjustments_levels_are_identity(&adjustments_clone);
                let levels_input_histogram = if !request_levels_input_analytics
                    || reuse_final_as_levels_input
                {
                    None
                } else {
                    (|| -> Result<image_processing::HistogramData, String> {
                        let mut identity_adj = adjustments_clone.clone();
                        neutralize_levels_for_input_histogram(&mut identity_adj);
                        let source = state.portrait_render.editor_source(
                            &loaded_image.path,
                            &settings,
                            source_revision,
                            &current,
                        )?;
                        let lease = state.portrait_render.begin_editor_consumer(
                            &source,
                            request_id,
                            epoch as u64,
                        )?;
                        let mut input_output = state.portrait_render.render_photo(
                            app_handle,
                            PhotoRenderRequest {
                                source,
                                adjustments: identity_adj,
                                window: window.clone(),
                                quality: if is_interactive {
                                    RenderQuality::Interactive
                                } else {
                                    RenderQuality::HighQuality
                                },
                                purpose: RenderPurpose::Preview,
                                request_id,
                                sixteen_bit: false,
                            },
                            lease,
                            &settings,
                            &current,
                        )?;
                        let img = input_output
                            .image
                            .take()
                            .ok_or_else(|| "levels input preview missing image".to_string())?;
                        image_processing::calculate_histogram_from_image(&img)
                    })()
                    .map_err(|e| {
                        log::warn!("levels input histogram (prepare) failed: {e}");
                        e
                    })
                    .ok()
                };
                let _ = sender.send(AnalyticsJob {
                    path: loaded_image.path.clone(),
                    stamp: output.stamp.clone(),
                    editor_epoch: epoch as u64,
                    image,
                    compute_waveform,
                    active_waveform_channel: if is_interactive {
                        active_waveform_channel.map(str::to_owned)
                    } else {
                        None
                    },
                    levels_input_histogram,
                    reuse_final_as_levels_input,
                });
            }
            log_preview_total(
                request_id,
                is_interactive,
                intent_label,
                window.output_dimensions,
                fn_start.elapsed(),
            );
            return Ok(response);
        }
        Some(output)
    } else {
        None
    };
    // 合法关闭／零值／有效 identity 继续使用旧像素算法。
    let context = get_or_init_gpu_context(&state, app_handle)?;
    let prepare_started = std::time::Instant::now();
    let portrait_maps = portrait_processing::prepare_render(
        app_handle,
        &state,
        &loaded_image.path,
        &adjustments_clone,
    )?;
    log::debug!(
        "[portrait-preview] prepare_render interactive={} maps={} elapsed={:?}",
        is_interactive,
        portrait_maps.is_some(),
        prepare_started.elapsed()
    );
    let publication = state.lock_preview_epoch();
    if *publication != epoch || !current() {
        return Err("Superseded or worker failed".into());
    }
    let new_transform_hash = calculate_transform_hash(&adjustments_clone);
    let mut cached_preview_lock = state
        .cached_preview
        .lock()
        .unwrap_or_else(|e| e.into_inner());

    let base_valid = cached_preview_lock
        .as_ref()
        .is_some_and(|c| c.transform_hash == new_transform_hash && c.preview_dim == preview_dim);
    let small_valid = base_valid
        && cached_preview_lock
            .as_ref()
            .is_some_and(|c| c.interactive_divisor == interactive_divisor);

    let (final_preview_base, scale_for_gpu, unscaled_crop_offset) = if base_valid {
        let cached = cached_preview_lock.as_ref().unwrap();
        (
            Arc::clone(&cached.image),
            cached.scale,
            cached.unscaled_crop_offset,
        )
    } else {
        *state
            .gpu_image_cache
            .lock()
            .unwrap_or_else(|e| e.into_inner()) = None;

        let (base, scale, offset) =
            generate_transformed_preview(&state, &loaded_image, &adjustments_clone, preview_dim)?;
        (Arc::new(base), scale, offset)
    };

    let small_preview_base = if small_valid {
        Arc::clone(&cached_preview_lock.as_ref().unwrap().small_image)
    } else {
        let small = if interactive_divisor > 1.0 {
            let (w, h) = final_preview_base.dimensions();
            let target_size = ((preview_dim as f32 / interactive_divisor) as u32)
                .min(w.max(h))
                .max(1);
            let (small_w, small_h) = if w > h {
                let ratio = h as f32 / w as f32;
                (target_size, ((target_size as f32 * ratio) as u32).max(1))
            } else {
                let ratio = w as f32 / h as f32;
                (((target_size as f32 * ratio) as u32).max(1), target_size)
            };
            Arc::new(image_processing::downscale_f32_image(
                &final_preview_base,
                small_w,
                small_h,
            ))
        } else {
            Arc::clone(&final_preview_base)
        };

        // 交互降采样尺寸变化时由 process_and_get_dynamic_image 按 width/height 失效缓存，
        // 不再在只改 tone 的拖动路径上主动清空输入纹理。
        small
    };

    *cached_preview_lock = Some(CachedPreview {
        image: Arc::clone(&final_preview_base),
        small_image: Arc::clone(&small_preview_base),
        transform_hash: new_transform_hash,
        scale: scale_for_gpu,
        unscaled_crop_offset,
        preview_dim,
        interactive_divisor,
    });

    drop(cached_preview_lock);

    let (processing_image, effective_scale, jpeg_quality) = if is_interactive {
        let orig_w = final_preview_base.width() as f32;
        let small_w = small_preview_base.width() as f32;
        let scale_factor = if orig_w > 0.0 { small_w / orig_w } else { 1.0 };
        let new_scale = scale_for_gpu * scale_factor;
        (small_preview_base, new_scale, interactive_quality)
    } else {
        (final_preview_base, scale_for_gpu, 94)
    };

    let (preview_width, preview_height) = processing_image.dimensions();

    let preview_roi = portrait_render::normalized_preview_roi(
        [preview_width, preview_height],
        if is_interactive && !(use_wgpu_renderer && prepared.is_some()) {
            roi
        } else {
            None
        },
    )?;
    let pixel_roi = preview_roi.map(|r| crate::gpu_processing::Roi {
        x: r.x as u32,
        y: r.y as u32,
        width: r.width,
        height: r.height,
    });
    let full = PixelRect::full([preview_width, preview_height])?;
    let window_key = serde_json::json!([
        "legacy-output-v2",
        preview_width,
        preview_height,
        preview_roi,
        is_interactive,
        jpeg_quality
    ]);
    let stamp = if let Some(output) = prepared.as_mut() {
        output.stamp.window_key = portrait_render::content_key(&window_key);
        output.stamp.clone()
    } else {
        state.portrait_render.legacy_stamp(
            request_id,
            source_revision,
            &loaded_image.path,
            &adjustments_clone,
            &window_key,
        )?
    };
    let metadata = PreviewMetadata {
        path: loaded_image.path.clone(),
        stamp: stamp.clone(),
        dimensions: [preview_width, preview_height],
        roi: if use_wgpu_renderer {
            full
        } else {
            preview_roi.unwrap_or(full)
        },
        mode: if use_wgpu_renderer {
            PreviewMode::Native
        } else {
            PreviewMode::Jpeg
        },
        reshape_attenuation: prepared
            .as_ref()
            .and_then(|output| output.reshape_attenuation),
        body_attenuation: prepared
            .as_ref()
            .and_then(|output| output.body_attenuation),
    };
    let commit = || -> Result<(), String> {
        if let Some(output) = prepared.as_ref() {
            state
                .portrait_render
                .publish_editor(output, epoch as u64, &current, || Ok(()))
        } else {
            state
                .portrait_render
                .publish_legacy_preview(&stamp, epoch as u64, || Ok(()))
        }
    };

    let mask_definitions: Vec<MaskDefinition> = adjustments_clone
        .get("masks")
        .and_then(|m| serde_json::from_value(m.clone()).ok())
        .unwrap_or_default();

    let scaled_crop_offset = (
        unscaled_crop_offset.0 * effective_scale,
        unscaled_crop_offset.1 * effective_scale,
    );

    let mask_bitmaps: Vec<ImageBuffer<Luma<u8>, Vec<u8>>> = mask_definitions
        .iter()
        .filter_map(|def| {
            get_cached_or_generate_mask(
                &state,
                def,
                preview_width,
                preview_height,
                effective_scale,
                scaled_crop_offset,
                &adjustments_clone,
            )
        })
        .collect();

    let is_raw = loaded_image.is_raw;
    // 交互帧与松手共用非破坏 GPU portrait_ev：不再 neural-bake EV 进低分底图。
    // 旧路径先 Burn 再 clarity，且预览 luma 对 atlas 边权失配，中性灰立体拖动会出现块状死黑。
    let tm_override = resolve_tonemapper_override_from_handle(app_handle, is_raw);
    let final_adjustments = get_all_adjustments_from_json(
        &adjustments_clone,
        is_raw,
        loaded_image.as_shot_white_balance,
        tm_override,
    );
    let lut_path = adjustments_clone["lutPath"].as_str();
    let lut = lut_path.and_then(|p| lut_processing::get_or_load_lut(&state, p).ok());

    let wants_analytics = !(is_interactive && pixel_roi.is_some()) && request_analytics;
    let channel_filter = if is_interactive {
        active_waveform_channel.map(|s| s.to_string())
    } else {
        None
    };

    let reuse_final_as_levels_input =
        wants_analytics && request_levels_input_analytics && adjustments_levels_are_identity(&adjustments_clone);
    let levels_input_histogram = if wants_analytics
        && request_levels_input_analytics
        && !reuse_final_as_levels_input
    {
        let mut identity_adj = adjustments_clone.clone();
        neutralize_levels_for_input_histogram(&mut identity_adj);
        let identity_adjustments = get_all_adjustments_from_json(
            &identity_adj,
            is_raw,
            loaded_image.as_shot_white_balance,
            tm_override,
        );
        // Distinct hash so GPU cache does not return the leveled grade.
        let mut hasher = DefaultHasher::new();
        new_transform_hash.hash(&mut hasher);
        "levels-input-hist".hash(&mut hasher);
        let input_hash = hasher.finish();
        match process_and_get_dynamic_image(
            &context,
            &state,
            &processing_image,
            input_hash,
            RenderRequest {
                adjustments: identity_adjustments,
                mask_bitmaps: &mask_bitmaps,
                lut: lut.clone(),
                roi: pixel_roi,
                portrait_maps: portrait_maps.clone(),
            },
            "levels_input_histogram",
        ) {
            Ok(img) => image_processing::calculate_histogram_from_image(&img).ok(),
            Err(e) => {
                log::warn!("levels input histogram (grade) failed: {e}");
                None
            }
        }
    } else {
        None
    };

    let analytics_config = if wants_analytics {
        state
            .analytics_worker_tx
            .lock()
            .unwrap()
            .clone()
            .map(|tx| crate::AnalyticsConfig {
                path: loaded_image.path.clone(),
                stamp: stamp.clone(),
                editor_epoch: epoch as u64,
                compute_waveform,
                active_waveform_channel: channel_filter,
                sender: tx,
                levels_input_histogram,
                reuse_final_as_levels_input,
            })
    } else {
        None
    };

    let grade_started = std::time::Instant::now();
    let grade_has_portrait_maps = portrait_maps.is_some();
    let final_processed_image_result =
        crate::image_processing::process_and_get_dynamic_image_with_analytics(
            &context,
            &state,
            &processing_image,
            new_transform_hash,
            RenderRequest {
                adjustments: final_adjustments,
                mask_bitmaps: &mask_bitmaps,
                lut,
                roi: pixel_roi,
                portrait_maps,
            },
            "apply_adjustments",
            use_wgpu_renderer && prepared.is_none(),
            analytics_config,
        );
    log::debug!(
        "[portrait-preview] grade interactive={} portrait_maps={} elapsed={:?} total={:?}",
        is_interactive,
        grade_has_portrait_maps,
        grade_started.elapsed(),
        fn_start.elapsed()
    );

    if let Ok(final_processed_image) = final_processed_image_result {
        if use_wgpu_renderer {
            let _ = context.device.poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: Some(std::time::Duration::from_millis(500)),
            });
            let response = metadata.encode(&[])?;
            if let Some(output) = prepared.as_ref() {
                let rgba = final_processed_image
                    .as_rgba8()
                    .ok_or("identity 预览回读不是 RGBA8")?;
                // 交互预览保留 legacy GPU 热缓存；仅高质/松手帧释放整图工作池腾预算。
                if !is_interactive {
                    let mut processor = state
                        .gpu_processor
                        .lock()
                        .unwrap_or_else(|e| e.into_inner());
                    let mut cache = state
                        .gpu_image_cache
                        .lock()
                        .unwrap_or_else(|e| e.into_inner());
                    drop(processor.take());
                    drop(cache.take());
                }
                let staged =
                    gpu_processing::PreparedPreviewDisplay::prepare(&context, rgba, &current)?;
                state
                    .portrait_render
                    .publish_editor(output, epoch as u64, &current, || staged.publish(&context))?;
            } else {
                commit()?;
            }
            state
                .portrait_render
                .with_committed_preview(&stamp, epoch as u64, || {
                    let _ = app_handle.emit("wgpu-frame-ready", &metadata);
                });
            log_preview_total(
                request_id,
                is_interactive,
                intent_label,
                metadata.dimensions,
                fn_start.elapsed(),
            );
            return Ok(response);
        }

        let final_processed_image = Arc::new(final_processed_image);
        let final_rgba_image = match &*final_processed_image {
            DynamicImage::ImageRgba8(img) => img,
            _ => return Err("Expected Rgba8 image from GPU for encoding".to_string()),
        };

        let raw_bytes: &[u8] = final_rgba_image.as_raw();
        let rgba8_pixels: &[RGBA8] = raw_bytes.as_rgba();

        let img_ref = ImgRef::new(
            rgba8_pixels,
            final_rgba_image.width() as usize,
            final_rgba_image.height() as usize,
        );

        let step_start = std::time::Instant::now();

        let encode_result = Encoder::new(Preset::BaselineFastest)
            .quality(jpeg_quality)
            .fast_color(true)
            .encode_imgref(img_ref);

        match encode_result {
            Ok(jpeg_bytes) => {
                if final_rgba_image.dimensions() != (metadata.roi.width, metadata.roi.height) {
                    return Err("预览像素与版本窗口不一致".into());
                }
                let response = metadata.encode(&jpeg_bytes)?;
                commit()?;
                log::info!(
                    "[process_preview_job] RRV2 {}x{} encode in {:.2?}, total {:.2?}",
                    metadata.roi.width,
                    metadata.roi.height,
                    step_start.elapsed(),
                    fn_start.elapsed()
                );
                log_preview_total(
                    request_id,
                    is_interactive,
                    intent_label,
                    metadata.dimensions,
                    fn_start.elapsed(),
                );
                Ok(response)
            }
            Err(e) => Err(format!("Failed to encode preview: {}", e)),
        }
    } else {
        log::error!(
            "[process_preview_job] processing failed after {:.2?}",
            fn_start.elapsed()
        );
        Err(final_processed_image_result.unwrap_err())
    }
}

/// 预览总耗时：松手/高质帧记 info，用于从日志读出调色首帧（bypass）与完整人像帧（prepare）的时刻。
fn log_preview_total(
    request_id: u64,
    is_interactive: bool,
    intent: &str,
    output: [u32; 2],
    total: std::time::Duration,
) {
    let level = if is_interactive {
        log::Level::Debug
    } else {
        log::Level::Info
    };
    log::log!(
        level,
        "[portrait-preview] request={request_id} interactive={is_interactive} intent={intent} output={output:?} total={total:?}"
    );
}

fn start_analytics_worker(app_handle: tauri::AppHandle) {
    let state = app_handle.state::<AppState>();
    let (tx, rx): (Sender<AnalyticsJob>, Receiver<AnalyticsJob>) = mpsc::channel();
    *state.analytics_worker_tx.lock().unwrap() = Some(tx);

    std::thread::spawn(move || {
        while let Ok(mut job) = rx.recv() {
            while let Ok(latest) = rx.try_recv() {
                job = latest;
            }

            let state = app_handle.state::<AppState>();
            if !state
                .portrait_render
                .editor_is_current(job.stamp.source_revision, job.editor_epoch)
            {
                continue;
            }
            let histogram_data = image_processing::calculate_histogram_from_image(&job.image).ok();

            let waveform_data = if job.compute_waveform {
                image_processing::calculate_waveform_from_image(
                    &job.image,
                    job.active_waveform_channel.as_deref(),
                )
                .ok()
            } else {
                None
            };

            let levels_input_histogram = job.levels_input_histogram.or_else(|| {
                if job.reuse_final_as_levels_input {
                    histogram_data.clone()
                } else {
                    None
                }
            });

            if histogram_data.is_some() || waveform_data.is_some() || levels_input_histogram.is_some()
            {
                // 旧 GPU 异步回读可能早于主响应；等旧发布临界区结束，再核对精确 stamp。
                let _publication = state.lock_preview_epoch();
                state
                    .portrait_render
                    .with_committed_preview(&job.stamp, job.editor_epoch, || {
                        let mut payload = serde_json::json!({
                            "path": job.path, "stamp": job.stamp,
                            "histogram": histogram_data, "waveform": waveform_data,
                        });
                        if let Some(levels_hist) = levels_input_histogram {
                            payload["levelsInputHistogram"] = serde_json::to_value(levels_hist)
                                .unwrap_or(serde_json::Value::Null);
                        }
                        let _ = app_handle.emit("analytics-update", payload);
                    });
            }
        }
    });
}

fn start_preview_worker(app_handle: tauri::AppHandle) {
    let state = app_handle.state::<AppState>();
    let (tx, rx): (Sender<PreviewJob>, Receiver<PreviewJob>) = mpsc::channel();

    *state.preview_worker_tx.lock().unwrap() = Some(tx);

    std::thread::spawn(move || {
        while let Ok(mut job) = rx.recv() {
            while let Ok(latest_job) = rx.try_recv() {
                job = latest_job;
            }

            let state = app_handle.state::<AppState>();
            let responder = job.responder;
            // GPU/GL 争用可能 panic 并毒化 preview_epoch；捕获后转成 Err，保持 worker 存活。
            let result = panic::catch_unwind(panic::AssertUnwindSafe(|| {
                process_preview_job(
                    &app_handle,
                    state,
                    job.adjustments,
                    job.loaded_image,
                    job.epoch,
                    job.request_id,
                    job.source_revision,
                    job.is_interactive,
                    job.target_resolution,
                    job.roi,
                    job.request_analytics,
                    job.request_levels_input_analytics,
                    job.compute_waveform,
                    job.active_waveform_channel.as_deref(),
                )
            }));
            match result {
                Ok(Ok(bytes)) => {
                    let _ = responder.send(Ok(bytes));
                }
                Ok(Err(e)) => {
                    if e == portrait_analysis::CANCELLED || e == "Superseded or worker failed" {
                        log::debug!("Preview worker superseded: {}", e);
                    } else {
                        log::error!("Preview worker error: {}", e);
                    }
                    let _ = responder.send(Err(e));
                }
                Err(_) => {
                    log::error!("Preview worker panicked（多为 GPU/GL 上下文争用）");
                    let _ = responder.send(Err(
                        "预览渲染异常（GPU 上下文争用），请重试或重启应用".into()
                    ));
                }
            }
        }
    });
}

#[allow(clippy::too_many_arguments)]
#[tauri::command]
async fn apply_adjustments(
    path: String,
    request_id: u64,
    mut js_adjustments: serde_json::Value,
    is_interactive: bool,
    target_resolution: Option<u32>,
    roi: Option<(f32, f32, f32, f32)>,
    request_analytics: bool,
    request_levels_input_analytics: Option<bool>,
    compute_waveform: bool,
    active_waveform_channel: Option<String>,
    state: tauri::State<'_, AppState>,
) -> Result<Response, String> {
    if request_id == 0 || request_id > 9_007_199_254_740_991 {
        return Err("预览请求编号非法".into());
    }
    let request_levels_input_analytics = request_levels_input_analytics.unwrap_or(false);
    let (tx, rx) = tokio::sync::oneshot::channel();

    {
        let mut epoch = state.lock_preview_epoch();
        let loaded_image = state
            .original_image
            .lock()
            .unwrap()
            .clone()
            .filter(|image| image.path == path)
            .ok_or("Superseded or worker failed")?;
        *epoch = epoch.checked_add(1).ok_or("预览代次已耗尽")?;
        state.portrait_render.advance_editor(
            state.load_image_generation.load(Ordering::SeqCst) as u64,
            *epoch as u64,
        )?;
        hydrate_adjustments(&state, &mut js_adjustments);
        let tx_guard = state.preview_worker_tx.lock().unwrap();
        if let Some(worker_tx) = tx_guard.as_ref() {
            let job = PreviewJob {
                epoch: *epoch,
                request_id,
                source_revision: state.load_image_generation.load(Ordering::SeqCst) as u64,
                loaded_image,
                adjustments: js_adjustments,
                is_interactive,
                target_resolution,
                roi,
                request_analytics,
                request_levels_input_analytics,
                compute_waveform,
                active_waveform_channel,
                responder: tx,
            };
            worker_tx
                .send(job)
                .map_err(|e| format!("Failed to send to preview worker: {}", e))?;
        } else {
            return Err("Preview worker not running".to_string());
        }
    }

    match rx.await {
        Ok(result) => result.map(Response::new),
        Err(_) => Err("Superseded or worker failed".to_string()),
    }
}

#[tauri::command]
async fn generate_uncropped_preview(
    path: String,
    js_adjustments: serde_json::Value,
    state: tauri::State<'_, AppState>,
    app_handle: tauri::AppHandle,
) -> Result<String, String> {
    let context = get_or_init_gpu_context(&state, &app_handle)?;
    let mut adjustments_clone = js_adjustments.clone();
    let (loaded_image, generation) = {
        let _publication = state.lock_preview_epoch();
        let loaded_image = state
            .original_image
            .lock()
            .unwrap()
            .clone()
            .filter(|image| image.path == path)
            .ok_or("Superseded or worker failed")?;
        hydrate_adjustments(&state, &mut adjustments_clone);
        (
            loaded_image,
            state.load_image_generation.load(Ordering::SeqCst),
        )
    };

    tokio::task::spawn_blocking(move || {
        let state = app_handle.state::<AppState>();
        let path = loaded_image.path.clone();
        let is_raw = loaded_image.is_raw;
        if portrait_render::render_intent(&adjustments_clone)?
            == portrait_render::RenderIntent::Prepare
        {
            // 未裁剪共轭路径：U 基底（定向／镜像后、自由旋转前）＋ F_U=R⁻¹∘F_B∘R，
            // 与主预览共享同一冻结分析场；不清空 rotation 重新分析，不反旋截断 B 图。
            let settings = load_settings(app_handle.clone()).unwrap_or_default();
            let target = settings.editor_preview_resolution.unwrap_or(1920);
            let data_url = state.portrait_render.render_uncropped_photo(
                &app_handle,
                &path,
                &adjustments_clone,
                target,
                &settings,
                &|| state.load_image_generation.load(Ordering::SeqCst) == generation,
                |image| {
                    let bytes = Encoder::new(Preset::BaselineFastest)
                        .quality(80)
                        .encode_rgb(image.to_rgb8().as_raw(), image.width(), image.height())
                        .map_err(|e| e.to_string())?;
                    if state.load_image_generation.load(Ordering::SeqCst) != generation {
                        return Err("Superseded or worker failed".into());
                    }
                    Ok(format!(
                        "data:image/jpeg;base64,{}",
                        general_purpose::STANDARD.encode(bytes)
                    ))
                },
            )?;
            if let Some(url) = data_url {
                return Ok(url);
            }
            // 有效 identity：落回旧美肤／像素路径，不返回未重塑预览。
        }
        if portrait_processing::enabled(&adjustments_clone) {
            let mut uncropped = adjustments_clone.clone();
            uncropped["crop"] = Value::Null;
            uncropped["rotation"] = serde_json::json!(0);
            let settings = load_settings(app_handle.clone()).unwrap_or_default();
            let image = portrait_processing::render_reduced(
                &app_handle,
                &state,
                &path,
                &uncropped,
                settings.editor_preview_resolution.unwrap_or(1920),
            )?;
            let bytes = Encoder::new(Preset::BaselineFastest)
                .quality(80)
                .encode_rgb(image.to_rgb8().as_raw(), image.width(), image.height())
                .map_err(|e| e.to_string())?;
            if state.load_image_generation.load(Ordering::SeqCst) != generation {
                return Err("Superseded or worker failed".into());
            }
            return Ok(format!(
                "data:image/jpeg;base64,{}",
                general_purpose::STANDARD.encode(bytes)
            ));
        }
        let _publication = state.lock_preview_epoch();
        if state.load_image_generation.load(Ordering::SeqCst) != generation {
            return Err("Superseded or worker failed".into());
        }
        let visual_hash = calculate_visual_hash(&path, &adjustments_clone);

        let pre_geometry_base = {
            let mut cache = state.geometry_cache.lock().unwrap();
            if let Some(cached) = cache.get(&visual_hash).cloned() {
                cached
            } else {
                let has_patches = adjustments_clone
                    .get("aiPatches")
                    .and_then(|v| v.as_array())
                    .is_some_and(|a| !a.is_empty());
                let patched_image = if has_patches {
                    Cow::Owned(
                        composite_patches_on_image(&loaded_image.image, &adjustments_clone)
                            .unwrap_or_else(|_| loaded_image.image.as_ref().clone()),
                    )
                } else {
                    Cow::Borrowed(loaded_image.image.as_ref())
                };

                let blurred_image =
                    crate::lens_blur::apply_lens_blur(patched_image, &adjustments_clone);

                let settings = load_settings(app_handle.clone()).unwrap_or_default();
                let target_dim = (settings.editor_preview_resolution.unwrap_or(1920) as f32) as u32;

                let downscaled = downscale_f32_image(&blurred_image, target_dim, target_dim);

                if cache.len() > 5 {
                    cache.clear();
                }
                cache.insert(visual_hash, downscaled.clone());
                downscaled
            }
        };

        let scale_for_gpu = if loaded_image.image.width() > 0 {
            pre_geometry_base.width() as f32 / loaded_image.image.width() as f32
        } else {
            1.0
        };

        let params = crate::image_processing::get_geometry_params_from_json(&adjustments_clone);
        let mut adjusted_params = params;
        adjusted_params.lens_vignette_amount *= if is_raw { 0.4 } else { 0.8 };

        let warped_image = warp_image_geometry(&pre_geometry_base, adjusted_params);
        let orientation_steps = adjustments_clone["orientationSteps"].as_u64().unwrap_or(0) as u8;
        let flipped_image = apply_flip(
            apply_coarse_rotation(Cow::Owned(warped_image), orientation_steps),
            adjustments_clone["flipHorizontal"]
                .as_bool()
                .unwrap_or(false),
            adjustments_clone["flipVertical"].as_bool().unwrap_or(false),
        )
        .into_owned();

        let (preview_width, preview_height) = flipped_image.dimensions();

        let mask_definitions: Vec<MaskDefinition> = adjustments_clone
            .get("masks")
            .and_then(|m| serde_json::from_value(m.clone()).ok())
            .unwrap_or_default();

        let mask_bitmaps: Vec<ImageBuffer<Luma<u8>, Vec<u8>>> = mask_definitions
            .iter()
            .filter_map(|def| {
                get_cached_or_generate_mask(
                    &state,
                    def,
                    preview_width,
                    preview_height,
                    scale_for_gpu,
                    (0.0, 0.0),
                    &adjustments_clone,
                )
            })
            .collect();

        let tm_override = resolve_tonemapper_override_from_handle(&app_handle, is_raw);
        let mut uncropped_adjustments = get_all_adjustments_from_json(
            &adjustments_clone,
            is_raw,
            loaded_image.as_shot_white_balance,
            tm_override,
        );
        uncropped_adjustments.global.show_clipping = 0;
        let lut_path = adjustments_clone["lutPath"].as_str();
        let lut = lut_path.and_then(|p| lut_processing::get_or_load_lut(&state, p).ok());

        let full_hash = calculate_full_job_hash(&path, &adjustments_clone);

        if let Ok(processed_image) = process_and_get_dynamic_image(
            &context,
            &state,
            &flipped_image,
            full_hash,
            RenderRequest {
                adjustments: uncropped_adjustments,
                mask_bitmaps: &mask_bitmaps,
                lut,
                roi: None,
                portrait_maps: None,
            },
            "generate_uncropped_preview",
        ) {
            let (width, height) = processed_image.dimensions();
            let rgb_pixels = processed_image.to_rgb8().into_vec();

            if let Ok(bytes) = Encoder::new(Preset::BaselineFastest)
                .quality(80)
                .encode_rgb(&rgb_pixels, width, height)
            {
                let base64_str = general_purpose::STANDARD.encode(&bytes);
                let data_url = format!("data:image/jpeg;base64,{}", base64_str);
                return Ok(data_url);
            }
        }

        Err("Failed to process uncropped preview".to_string())
    })
    .await
    .map_err(|e| format!("Task execution failed: {}", e))?
}

pub fn get_original_image(
    state: &tauri::State<AppState>,
) -> Result<(std::sync::Arc<image::DynamicImage>, bool), String> {
    let original_image_lock = state.original_image.lock().unwrap();
    let loaded_image = original_image_lock
        .as_ref()
        .ok_or("No original image loaded")?;
    Ok((
        std::sync::Arc::clone(&loaded_image.image),
        loaded_image.is_raw,
    ))
}

#[tauri::command]
fn generate_preset_preview(
    js_adjustments: serde_json::Value,
    state: tauri::State<AppState>,
    app_handle: tauri::AppHandle,
) -> Result<Response, String> {
    const PRESET_PREVIEW_DIM: u32 = 400;

    // 活动重塑：共享渲染（严格源→冻结场→原生合成→最终浮点缩放到预设预览尺寸）。
    // 旁路／identity 返回 None，继续旧像素路径。先取路径再渲染，不跨渲染持有编辑器锁。
    let preset_settings = load_settings(app_handle.clone()).unwrap_or_default();
    let preset_path = state
        .original_image
        .lock()
        .unwrap()
        .clone()
        .ok_or("No original image loaded for preset preview")?
        .path;
    let reshape_preview = state.portrait_render.render_reduced_photo(
        &app_handle,
        &preset_path,
        &js_adjustments,
        PRESET_PREVIEW_DIM,
        crate::portrait_render::RenderQuality::Interactive,
        false,
        &preset_settings,
        &|| true,
        |image| {
            let mut buf = Cursor::new(Vec::new());
            image
                .to_rgb8()
                .write_with_encoder(JpegEncoder::new_with_quality(&mut buf, 80))
                .map_err(|e| e.to_string())?;
            Ok(buf.into_inner())
        },
    )?;
    if let Some(bytes) = reshape_preview {
        return Ok(Response::new(bytes));
    }

    let context = get_or_init_gpu_context(&state, &app_handle)?;

    let loaded_image = state
        .original_image
        .lock()
        .unwrap()
        .clone()
        .ok_or("No original image loaded for preset preview")?;
    let is_raw = loaded_image.is_raw;
    let unique_hash = calculate_full_job_hash(&loaded_image.path, &js_adjustments);

    let (preview_image, scale_for_gpu, unscaled_crop_offset) =
        generate_transformed_preview(&state, &loaded_image, &js_adjustments, PRESET_PREVIEW_DIM)?;

    let (img_w, img_h) = preview_image.dimensions();

    let mask_definitions: Vec<MaskDefinition> = js_adjustments
        .get("masks")
        .and_then(|m| serde_json::from_value(m.clone()).ok())
        .unwrap_or_default();

    let scaled_crop_offset = (
        unscaled_crop_offset.0 * scale_for_gpu,
        unscaled_crop_offset.1 * scale_for_gpu,
    );

    let mask_bitmaps: Vec<ImageBuffer<Luma<u8>, Vec<u8>>> = mask_definitions
        .iter()
        .filter_map(|def| {
            get_cached_or_generate_mask(
                &state,
                def,
                img_w,
                img_h,
                scale_for_gpu,
                scaled_crop_offset,
                &js_adjustments,
            )
        })
        .collect();

    let tm_override = resolve_tonemapper_override_from_handle(&app_handle, is_raw);
    let mut all_adjustments = get_all_adjustments_from_json(
        &js_adjustments,
        is_raw,
        loaded_image.as_shot_white_balance,
        tm_override,
    );
    all_adjustments.global.show_clipping = 0;
    let lut_path = js_adjustments["lutPath"].as_str();
    let lut = lut_path.and_then(|p| lut_processing::get_or_load_lut(&state, p).ok());

    let processed_image = process_and_get_dynamic_image(
        &context,
        &state,
        &preview_image,
        unique_hash,
        RenderRequest {
            adjustments: all_adjustments,
            mask_bitmaps: &mask_bitmaps,
            lut,
            roi: None,
            portrait_maps: portrait_processing::prepare_render(
                &app_handle,
                &state,
                &loaded_image.path,
                &js_adjustments,
            )?,
        },
        "generate_preset_preview",
    )?;

    let mut buf = Cursor::new(Vec::new());
    processed_image
        .to_rgb8()
        .write_with_encoder(JpegEncoder::new_with_quality(&mut buf, 80))
        .map_err(|e| e.to_string())?;

    Ok(Response::new(buf.into_inner()))
}

#[tauri::command]
async fn fetch_community_presets() -> Result<Vec<CommunityPreset>, String> {
    // Community presets are no longer fetched from an external repository.
    Ok(Vec::new())
}

#[tauri::command]
async fn generate_all_community_previews(
    image_paths: Vec<String>,
    presets: Vec<CommunityPreset>,
    state: tauri::State<'_, AppState>,
    app_handle: tauri::AppHandle,
) -> Result<HashMap<String, Vec<u8>>, String> {
    let context = get_or_init_gpu_context(&state, &app_handle)?;
    let mut results: HashMap<String, Vec<u8>> = HashMap::new();

    const TILE_DIM: u32 = 360;
    const PROCESSING_DIM: u32 = TILE_DIM * 2;

    let settings = load_settings(app_handle.clone()).unwrap_or_default();

    let mut base_thumbnails: Vec<(DynamicImage, bool, WhiteBalance, f32)> = Vec::new();
    for image_path in image_paths.iter() {
        let (source_path, _) = parse_virtual_path(image_path);
        let source_path_str = source_path.to_string_lossy().to_string();
        let image_bytes = fs::read(&source_path).map_err(|e| e.to_string())?;
        let original_image = crate::image_loader::load_base_image_from_bytes(
            &image_bytes,
            &source_path_str,
            true,
            &settings,
            None,
        )
        .map_err(|e| e.to_string())?;

        let is_raw = is_raw_file(&source_path_str);
        let (orig_w, orig_h) = original_image.dimensions();
        let (base_image, base_scale) = if orig_w > PROCESSING_DIM || orig_h > PROCESSING_DIM {
            let downscaled = downscale_f32_image(&original_image, PROCESSING_DIM, PROCESSING_DIM);
            let scale = downscaled.width() as f32 / orig_w as f32;
            (downscaled, scale)
        } else {
            (original_image, 1.0)
        };

        base_thumbnails.push((
            base_image,
            is_raw,
            white_balance::as_shot_white_balance(&source_path_str),
            base_scale,
        ));
    }

    for preset in presets.iter() {
        let mut processed_tiles: Vec<RgbImage> = Vec::new();
        let js_adjustments = &preset.adjustments;

        let mut preset_hasher = DefaultHasher::new();
        preset.name.hash(&mut preset_hasher);
        let preset_hash = preset_hasher.finish();

        for (i, (base_image, is_raw, as_shot_white_balance, base_scale)) in
            base_thumbnails.iter().enumerate()
        {
            // 活动重塑：共享渲染，中心方形裁剪并入照片最终输出窗口（TILE_DIM 方形）；
            // 不先缩小到低清再分析或求解脸型。旁路／identity 继续旧低清路径。
            let reshape_tile = state.portrait_render.render_reduced_photo(
                &app_handle,
                &image_paths[i],
                js_adjustments,
                TILE_DIM,
                crate::portrait_render::RenderQuality::Interactive,
                true,
                &settings,
                &|| true,
                |image| Ok(image.to_rgb8()),
            )?;
            if let Some(tile) = reshape_tile {
                // 共享输出已是最终方形窗口；UI 拼贴仅在已渲染 tile 之后叠加。
                processed_tiles.push(tile);
                continue;
            }

            let mut scaled_adjustments = js_adjustments.clone();
            if let Some(crop_val) = scaled_adjustments.get_mut("crop")
                && let Ok(c) = serde_json::from_value::<Crop>(crop_val.clone())
            {
                *crop_val = serde_json::to_value(Crop {
                    x: c.x * (*base_scale as f64),
                    y: c.y * (*base_scale as f64),
                    width: c.width * (*base_scale as f64),
                    height: c.height * (*base_scale as f64),
                })
                .unwrap_or(serde_json::Value::Null);
            }

            let (transformed_image, _scaled_crop_offset) =
                crate::apply_all_transformations(Cow::Borrowed(base_image), &scaled_adjustments);
            let (img_w, img_h) = transformed_image.dimensions();

            let mask_definitions: Vec<MaskDefinition> = scaled_adjustments
                .get("masks")
                .and_then(|m| serde_json::from_value(m.clone()).ok())
                .unwrap_or_else(Vec::new);

            let unscaled_crop_offset = js_adjustments
                .get("crop")
                .and_then(|c| serde_json::from_value::<Crop>(c.clone()).ok())
                .map_or((0.0, 0.0), |c| (c.x as f32, c.y as f32));
            let actual_scaled_crop_offset = (
                unscaled_crop_offset.0 * base_scale,
                unscaled_crop_offset.1 * base_scale,
            );

            let mask_bitmaps: Vec<ImageBuffer<Luma<u8>, Vec<u8>>> = mask_definitions
                .iter()
                .filter_map(|def| {
                    generate_mask_bitmap(
                        def,
                        img_w,
                        img_h,
                        *base_scale,
                        actual_scaled_crop_offset,
                        None,
                    )
                })
                .collect();

            let tm_override = resolve_tonemapper_override_from_handle(&app_handle, *is_raw);
            let all_adjustments = get_all_adjustments_from_json(
                &scaled_adjustments,
                *is_raw,
                *as_shot_white_balance,
                tm_override,
            );
            let lut_path = js_adjustments["lutPath"].as_str();
            let lut = lut_path.and_then(|p| lut_processing::get_or_load_lut(&state, p).ok());

            let unique_hash = preset_hash.wrapping_add(i as u64);

            let processed_image_dynamic = if portrait_processing::enabled(js_adjustments) {
                portrait_processing::render_reduced(
                    &app_handle,
                    &state,
                    &image_paths[i],
                    js_adjustments,
                    PROCESSING_DIM,
                )?
            } else {
                crate::image_processing::process_and_get_dynamic_image(
                    &context,
                    &state,
                    transformed_image.as_ref(),
                    unique_hash,
                    RenderRequest {
                        adjustments: all_adjustments,
                        mask_bitmaps: &mask_bitmaps,
                        lut,
                        roi: None,
                        portrait_maps: portrait_processing::prepare_render(
                            &app_handle,
                            &state,
                            &image_paths[i],
                            js_adjustments,
                        )?,
                    },
                    "generate_all_community_previews",
                )?
            };

            let processed_image = processed_image_dynamic.to_rgb8();

            let (proc_w, proc_h) = processed_image.dimensions();
            let size = proc_w.min(proc_h);
            let cropped_processed_image = image::imageops::crop_imm(
                &processed_image,
                (proc_w - size) / 2,
                (proc_h - size) / 2,
                size,
                size,
            )
            .to_image();

            let final_tile = image::imageops::resize(
                &cropped_processed_image,
                TILE_DIM,
                TILE_DIM,
                image::imageops::FilterType::Lanczos3,
            );
            processed_tiles.push(final_tile);
        }

        let final_image_buffer = match processed_tiles.len() {
            1 => processed_tiles.remove(0),
            2 => {
                let mut canvas = RgbImage::new(TILE_DIM * 2, TILE_DIM);
                image::imageops::overlay(&mut canvas, &processed_tiles[0], 0, 0);
                image::imageops::overlay(&mut canvas, &processed_tiles[1], TILE_DIM as i64, 0);
                canvas
            }
            4 => {
                let mut canvas = RgbImage::new(TILE_DIM * 2, TILE_DIM * 2);
                image::imageops::overlay(&mut canvas, &processed_tiles[0], 0, 0);
                image::imageops::overlay(&mut canvas, &processed_tiles[1], TILE_DIM as i64, 0);
                image::imageops::overlay(&mut canvas, &processed_tiles[2], 0, TILE_DIM as i64);
                image::imageops::overlay(
                    &mut canvas,
                    &processed_tiles[3],
                    TILE_DIM as i64,
                    TILE_DIM as i64,
                );
                canvas
            }
            _ => continue,
        };

        let mut buf = Cursor::new(Vec::new());
        if final_image_buffer
            .write_with_encoder(JpegEncoder::new_with_quality(&mut buf, 75))
            .is_ok()
        {
            results.insert(preset.name.clone(), buf.into_inner());
        }
    }

    Ok(results)
}

#[tauri::command]
async fn save_temp_file(bytes: Vec<u8>) -> Result<String, String> {
    let mut temp_file = NamedTempFile::new().map_err(|e| e.to_string())?;
    temp_file.write_all(&bytes).map_err(|e| e.to_string())?;
    let (_file, path) = temp_file.keep().map_err(|e| e.to_string())?;
    Ok(path.to_string_lossy().to_string())
}

#[tauri::command]
async fn merge_hdr(
    paths: Vec<String>,
    app_handle: tauri::AppHandle,
    state: tauri::State<'_, AppState>,
) -> Result<(), String> {
    if paths.len() < 2 {
        return Err("Please select at least two images to merge.".to_string());
    }

    let hdr_result_handle = state.hdr_result.clone();
    let settings = load_settings(app_handle.clone()).unwrap_or_default();

    let mut frames = load_hdr_frames(&paths, &app_handle, &settings)?;
    assert_uniform_dimensions(&frames)?;
    align_hdr_frames(&mut frames, &app_handle);

    let images: Vec<HDRInput> = frames
        .iter()
        .map(|(path, img, exposure, gains)| {
            HDRInput::with_image(img, *exposure, *gains)
                .map_err(|e| format!("Failed to prepare HDR input for {}: {}", path, e))
        })
        .collect::<Result<Vec<HDRInput>, String>>()?;

    log::info!("Starting HDR merge of {} images", images.len());
    let mut hdr_merged = hdr_merge_images(&mut images.into()).map_err(|e| e.to_string())?;
    hdr_merged =
        image_hdr::stretch::apply_histogram_stretch(&hdr_merged).map_err(|e| e.to_string())?;
    hdr_merged = apply_linear_to_srgb(hdr_merged);
    log::info!("HDR merge completed");

    let mut buf = Cursor::new(Vec::new());
    if let Err(e) = hdr_merged.to_rgb8().write_to(&mut buf, ImageFormat::Png) {
        return Err(format!("Failed to encode hdr preview: {}", e));
    }

    let base64_str = general_purpose::STANDARD.encode(buf.get_ref());
    let final_base64 = format!("data:image/png;base64,{}", base64_str);

    let _ = app_handle.emit("hdr-progress", "Creating preview...");

    *hdr_result_handle.lock().unwrap() = Some(hdr_merged);

    let _ = app_handle.emit(
        "hdr-complete",
        serde_json::json!({
            "base64": final_base64,
        }),
    );
    Ok(())
}

#[tauri::command]
async fn save_hdr(
    first_path_str: String,
    state: tauri::State<'_, AppState>,
) -> Result<String, String> {
    let hdr_image = state.hdr_result.lock().unwrap().take().ok_or_else(|| {
        "No hdr image found in memory to save. It might have already been saved.".to_string()
    })?;

    let (first_path, _) = parse_virtual_path(&first_path_str);
    let parent_dir = first_path
        .parent()
        .ok_or_else(|| "Could not determine parent directory of the first image.".to_string())?;
    let stem = first_path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("hdr");

    let (output_filename, image_to_save): (String, DynamicImage) = if hdr_image.color().has_alpha()
    {
        (
            format!("{}_Hdr.png", stem),
            DynamicImage::ImageRgba8(hdr_image.to_rgba8()),
        )
    } else if hdr_image.as_rgb32f().is_some() {
        (format!("{}_Hdr.tiff", stem), hdr_image)
    } else {
        (
            format!("{}_Hdr.png", stem),
            DynamicImage::ImageRgb8(hdr_image.to_rgb8()),
        )
    };

    let output_path = parent_dir.join(output_filename);

    image_to_save
        .save(&output_path)
        .map_err(|e| format!("Failed to save hdr image: {}", e))?;

    let (real_path, _) = crate::file_management::parse_virtual_path(&first_path_str);
    let _ =
        crate::exif_processing::write_rrexif_sidecar(&real_path.to_string_lossy(), &output_path);

    Ok(output_path.to_string_lossy().to_string())
}

#[tauri::command]
async fn save_collage(base64_data: String, first_path_str: String) -> Result<String, String> {
    let data_url_prefix = "data:image/png;base64,";
    if !base64_data.starts_with(data_url_prefix) {
        return Err("Invalid base64 data format".to_string());
    }
    let encoded_data = &base64_data[data_url_prefix.len()..];

    let decoded_bytes = general_purpose::STANDARD
        .decode(encoded_data)
        .map_err(|e| format!("Failed to decode base64: {}", e))?;

    let (first_path, _) = parse_virtual_path(&first_path_str);
    let parent_dir = first_path
        .parent()
        .ok_or_else(|| "Could not determine parent directory of the first image.".to_string())?;
    let stem = first_path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("collage");

    let output_filename = format!("{}_Collage.png", stem);
    let output_path = parent_dir.join(output_filename);

    fs::write(&output_path, &decoded_bytes)
        .map_err(|e| format!("Failed to save collage image: {}", e))?;

    Ok(output_path.to_string_lossy().to_string())
}

#[tauri::command]
async fn generate_preview_for_path(
    path: String,
    js_adjustments: Value,
    app_handle: tauri::AppHandle,
) -> Result<Response, String> {
    tokio::task::spawn_blocking(move || {
        let state = app_handle.state::<AppState>();
        let context = get_or_init_gpu_context(&state, &app_handle)?;
        let (source_path, _) = parse_virtual_path(&path);
        let source_path_str = source_path.to_string_lossy().to_string();
        let is_raw = is_raw_file(&source_path_str);
        let settings = load_settings(app_handle.clone()).unwrap_or_default();

        // 活动重塑：共享渲染（严格源→冻结场→原生合成→最终浮点缩放到完整输出尺寸）；
        // 不先缩小再分析，也不借用编辑器蒙版／预览缓存。
        let reshape_preview = state.portrait_render.render_reduced_photo(
            &app_handle,
            &path,
            &js_adjustments,
            crate::portrait_tiles::MAX_SIDE,
            crate::portrait_render::RenderQuality::HighQuality,
            false,
            &settings,
            &|| true,
            |image| {
                let (width, height) = image.dimensions();
                let rgb_pixels = image.to_rgb8().into_vec();
                Encoder::new(Preset::BaselineFastest)
                    .quality(92)
                    .encode_rgb(&rgb_pixels, width, height)
                    .map_err(|e| format!("Failed to encode with mozjpeg-rs: {}", e))
            },
        )?;
        if let Some(bytes) = reshape_preview {
            return Ok(Response::new(bytes));
        }

        let base_image = match read_file_mapped(&source_path) {
            Ok(mmap) => load_and_composite(
                &mmap,
                &source_path_str,
                &js_adjustments,
                false,
                &settings,
                None,
            )
            .map_err(|e| e.to_string())?,
            Err(e) => {
                log::warn!(
                    "Failed to memory-map file '{}': {}. Falling back to standard read.",
                    source_path_str,
                    e
                );
                let bytes = fs::read(&source_path).map_err(|io_err| io_err.to_string())?;
                load_and_composite(
                    &bytes,
                    &source_path_str,
                    &js_adjustments,
                    false,
                    &settings,
                    None,
                )
                .map_err(|e| e.to_string())?
            }
        };

        let (transformed_image, unscaled_crop_offset) =
            apply_all_transformations(Cow::Borrowed(&base_image), &js_adjustments);
        let (img_w, img_h) = transformed_image.dimensions();
        let mask_definitions: Vec<MaskDefinition> = js_adjustments
            .get("masks")
            .and_then(|m| serde_json::from_value(m.clone()).ok())
            .unwrap_or_default();

        let warped_image =
            resolve_warped_image_for_masks(&state, &js_adjustments, &mask_definitions);
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

        let tm_override = resolve_tonemapper_override(&settings, is_raw);
        let mut all_adjustments = get_all_adjustments_from_json(
            &js_adjustments,
            is_raw,
            white_balance::as_shot_white_balance(&source_path_str),
            tm_override,
        );
        all_adjustments.global.show_clipping = 0;
        let lut_path = js_adjustments["lutPath"].as_str();
        let lut = lut_path.and_then(|p| lut_processing::get_or_load_lut(&state, p).ok());
        let unique_hash = calculate_full_job_hash(&source_path_str, &js_adjustments);

        let final_image = process_and_get_dynamic_image(
            &context,
            &state,
            transformed_image.as_ref(),
            unique_hash,
            RenderRequest {
                adjustments: all_adjustments,
                mask_bitmaps: &mask_bitmaps,
                lut,
                roi: None,
                portrait_maps: portrait_processing::prepare_render(
                    &app_handle,
                    &state,
                    &path,
                    &js_adjustments,
                )?,
            },
            "generate_preview_for_path",
        )?;

        let (width, height) = final_image.dimensions();
        let rgb_pixels = final_image.to_rgb8().into_vec();

        let bytes = Encoder::new(Preset::BaselineFastest)
            .quality(92)
            .encode_rgb(&rgb_pixels, width, height)
            .map_err(|e| format!("Failed to encode with mozjpeg-rs: {}", e))?;

        Ok(Response::new(bytes))
    })
    .await
    .map_err(|e| format!("Task execution failed: {}", e))?
}

fn setup_logging(app_handle: &tauri::AppHandle, enable_crash_report: bool) {
    let log_dir = match app_handle.path().app_log_dir() {
        Ok(dir) => dir,
        Err(e) => {
            eprintln!("Failed to get app log directory: {}", e);
            return;
        }
    };

    if let Err(e) = fs::create_dir_all(&log_dir) {
        eprintln!("Failed to create log directory at {:?}: {}", log_dir, e);
    }

    // 须在新 app.log 截断前执行：补报上次异常退出并保留上次日志
    if enable_crash_report {
        crash_report::on_startup(app_handle, &log_dir);
    }

    let log_file_path = log_dir.join(crash_report::LOG_FILE_NAME);

    let log_file = fs::OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .open(&log_file_path)
        .ok();

    let var = std::env::var("RUST_LOG").unwrap_or_else(|_| "info".to_string());
    let level: log::LevelFilter = var.parse().unwrap_or(log::LevelFilter::Info);

    let mut dispatch = fern::Dispatch::new()
        .format(|out, message, record| {
            out.finish(format_args!(
                "{} [{}] {}",
                chrono::Local::now().format("%Y-%m-%d %H:%M:%S"),
                record.level(),
                message
            ))
        })
        .level(level)
        .chain(std::io::stderr());

    if let Some(file) = log_file {
        dispatch = dispatch.chain(file);
    } else {
        eprintln!(
            "Failed to open log file at {:?}. Logging to console only.",
            log_file_path
        );
    }

    if let Err(e) = dispatch.apply() {
        eprintln!("Failed to apply logger configuration: {}", e);
    }

    panic::set_hook(Box::new(|info| {
        let message = if let Some(s) = info.payload().downcast_ref::<&'static str>() {
            s.to_string()
        } else if let Some(s) = info.payload().downcast_ref::<String>() {
            s.clone()
        } else {
            format!("{:?}", info.payload())
        };
        let location = info.location().map_or_else(
            || "at an unknown location".to_string(),
            |loc| format!("at {}:{}:{}", loc.file(), loc.line(), loc.column()),
        );
        log::error!("PANIC! {} - {}", location, message.trim());
        crash_report::record_panic(info, &message);
    }));

    log::info!(
        "Logger initialized successfully. Log file at: {:?}",
        log_file_path
    );
}

#[tauri::command]
fn get_log_file_path(app_handle: tauri::AppHandle) -> Result<String, String> {
    let log_dir = app_handle.path().app_log_dir().map_err(|e| e.to_string())?;
    let log_file_path = log_dir.join("app.log");
    Ok(log_file_path.to_string_lossy().to_string())
}

#[tauri::command]
fn frontend_log(level: String, message: String) -> Result<(), String> {
    let trimmed = message.trim();
    if trimmed.is_empty() {
        return Ok(());
    }

    let log_line = |line: &str| match level.to_lowercase().as_str() {
        "error" => log::error!("[frontend] {}", line),
        "warn" => log::warn!("[frontend] {}", line),
        "debug" => log::debug!("[frontend] {}", line),
        "trace" => log::trace!("[frontend] {}", line),
        _ => log::info!("[frontend] {}", line),
    };

    for line in trimmed
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
    {
        log_line(line);
    }

    Ok(())
}

#[derive(Clone, Copy, Debug)]
struct MonitorBounds {
    x: i32,
    y: i32,
    width: u32,
    height: u32,
}

fn saved_window_state_is_usable(state: &WindowState, monitors: &[MonitorBounds]) -> bool {
    if state.width < 800 || state.height < 600 {
        return false;
    }

    if monitors.is_empty() {
        return true;
    }

    let window_left = state.x as i64;
    let window_top = state.y as i64;
    let window_right = window_left + state.width as i64;
    let window_bottom = window_top + state.height as i64;

    monitors.iter().any(|monitor| {
        let monitor_left = monitor.x as i64;
        let monitor_top = monitor.y as i64;
        let monitor_right = monitor_left + monitor.width as i64;
        let monitor_bottom = monitor_top + monitor.height as i64;

        let overlap_width = window_right.min(monitor_right) - window_left.max(monitor_left);
        let overlap_height = window_bottom.min(monitor_bottom) - window_top.max(monitor_top);

        overlap_width >= 100 && overlap_height >= 100
    })
}

#[cfg(not(target_os = "android"))]
fn available_monitor_bounds(window: &tauri::WebviewWindow) -> Vec<MonitorBounds> {
    window
        .available_monitors()
        .map(|monitors| {
            monitors
                .into_iter()
                .map(|monitor| {
                    let position = monitor.position();
                    let size = monitor.size();
                    MonitorBounds {
                        x: position.x,
                        y: position.y,
                        width: size.width,
                        height: size.height,
                    }
                })
                .collect()
        })
        .unwrap_or_default()
}

#[cfg(target_os = "android")]
fn available_monitor_bounds(_window: &tauri::WebviewWindow) -> Vec<MonitorBounds> {
    Vec::new()
}

#[tauri::command]
fn frontend_ready(
    app_handle: tauri::AppHandle,
    window: tauri::Window,
    state: tauri::State<AppState>,
) -> Result<LaunchPayload, String> {
    let is_first_run = !state
        .window_setup_complete
        .swap(true, std::sync::atomic::Ordering::Relaxed);
    #[cfg(target_os = "android")]
    let _ = (is_first_run, &window, &app_handle);

    #[cfg(not(target_os = "android"))]
    {
        #[cfg(any(windows, target_os = "linux"))]
        let mut should_maximize = false;
        #[cfg(any(windows, target_os = "linux"))]
        let mut should_fullscreen = false;
        #[cfg(not(any(windows, target_os = "linux")))]
        let _ = (&app_handle, is_first_run);

        #[cfg(any(windows, target_os = "linux"))]
        if is_first_run && let Ok(config_dir) = app_handle.path().app_config_dir() {
            let path = config_dir.join("window_state.json");

            if let Ok(contents) = std::fs::read_to_string(&path)
                && let Ok(saved_state) = serde_json::from_str::<WindowState>(&contents)
            {
                #[cfg(any(windows, target_os = "linux"))]
                {
                    should_maximize = saved_state.maximized;
                    should_fullscreen = saved_state.fullscreen;
                }

                if (should_maximize || should_fullscreen)
                    && let Some(monitor) = window
                        .current_monitor()
                        .ok()
                        .flatten()
                        .or_else(|| window.primary_monitor().ok().flatten())
                        .or_else(|| {
                            window
                                .available_monitors()
                                .ok()
                                .and_then(|m| m.into_iter().next())
                        })
                {
                    let monitor_size = monitor.size();
                    let monitor_pos = monitor.position();
                    let default_width = 1280i32;
                    let default_height = 720i32;
                    let center_x = monitor_pos.x + (monitor_size.width as i32 - default_width) / 2;
                    let center_y =
                        monitor_pos.y + (monitor_size.height as i32 - default_height) / 2;

                    let _ = window.set_size(tauri::PhysicalSize::new(
                        default_width as u32,
                        default_height as u32,
                    ));
                    let _ = window.set_position(tauri::PhysicalPosition::new(center_x, center_y));
                }
            }
        }

        if let Err(e) = window.show() {
            log::error!("Failed to show window: {}", e);
        }
        if let Err(e) = window.set_focus() {
            log::error!("Failed to focus window: {}", e);
        }
        #[cfg(any(windows, target_os = "linux"))]
        if is_first_run {
            if should_maximize {
                let _ = window.maximize();
            }
            if should_fullscreen {
                let _ = window.set_fullscreen(true);
            }
        }
    }

    let open_with_file = state.initial_file_path.lock().unwrap().take();
    let edit_session = state.pending_edit_session.lock().unwrap().take();
    if let Some(path) = &open_with_file {
        log::info!("Frontend is ready, returning initial path: {}", path);
    }
    if let Some(session) = &edit_session {
        log::info!(
            "Frontend is ready, returning external edit session for: {}",
            session.source
        );
    }
    Ok(LaunchPayload {
        open_with_file,
        edit_session,
    })
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    let _ = rayon::ThreadPoolBuilder::new()
        .stack_size(8 * 1024 * 1024)
        .build_global();

    let args: Vec<String> = std::env::args().skip(1).collect();
    let launch_req = parse_launch_args(&args);
    if let LaunchRequest::InvalidHeadless(error) = &launch_req {
        eprintln!("Headless export failed: {}", error);
        std::process::exit(2);
    }
    let is_headless = matches!(launch_req, LaunchRequest::HeadlessExport(_));

    let mut builder = tauri::Builder::default();

    #[cfg(target_os = "linux")]
    {
        if !is_headless {
            builder = builder.plugin(tauri_plugin_wayland_nvidia_quirk::init());
        }
    }

    #[cfg(not(any(target_os = "android", target_os = "ios")))]
    {
        if !is_headless {
            builder = builder.plugin(tauri_plugin_single_instance::init(|app, argv, _cwd| {
                log::info!(
                    "New instance launched with args: {:?}. Focusing main window.",
                    argv
                );
                if let Some(window) = app.get_webview_window("main") {
                    if let Err(e) = window.unminimize() {
                        log::error!("Failed to unminimize window: {}", e);
                    }
                    if let Err(e) = window.set_focus() {
                        log::error!("Failed to set focus on window: {}", e);
                    }
                }

                let forwarded_args = argv.get(1..).unwrap_or(&[]);
                emit_launch_request(app, parse_launch_args(forwarded_args));
            }));
        }
    }

    builder
        .plugin(tauri_plugin_os::init())
        .plugin(tauri_plugin_fs::init())
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_process::init())
        .plugin(tauri_plugin_shell::init())
        .plugin(PinchZoomDisablePlugin)
        .on_window_event(|window, event| if let tauri::WindowEvent::Resized(size) = event {
            let state = window.state::<AppState>();
            if let Some(ctx) = state.gpu_context.lock().unwrap().as_ref()
                && let Ok(mut display_lock) = ctx.display.try_lock()
                    && let Some(display) = display_lock.as_mut() {
                        display.config.width = size.width.max(1);
                        display.config.height = size.height.max(1);
                        display.surface.configure(&ctx.device, &display.config);
                        display.render(&ctx.device, &ctx.queue);
                    }
        })
        .setup(move |app| {
            // 确保外部配置文件存在（首次运行生成默认 config.json，失败仅记录日志）
            app_config::ensure_default_config_file();

            #[cfg(feature = "tethering")]
            {
                std::thread::spawn(|| {
                    match gphoto2::Context::new() {
                        Ok(context) => {
                            log::info!("gphoto2 context initialized successfully.");
                            match gphoto2::Camera::autodetect(&context) {
                                Ok(cameras) => {
                                    log::info!("Found {} attached camera(s)", cameras.len());
                                }
                                Err(e) => log::warn!("Failed to autodetect cameras: {}", e),
                            }
                        }
                        Err(e) => log::error!("Failed to initialize gphoto2 context: {}", e),
                    }
                });
            }

            let state = app.state::<AppState>();

            #[cfg(any(windows, target_os = "linux", target_os = "macos"))]
            {
                match launch_req.clone() {
                    LaunchRequest::EditSession(session) => {
                        log::info!("Initial launch with external edit session for: {}", session.source);
                        *state.pending_edit_session.lock().unwrap() = Some(session);
                    }
                    LaunchRequest::OpenFile(path) => {
                        log::info!("Initial open: Storing path {} for later.", path);
                        *state.initial_file_path.lock().unwrap() = Some(path);
                    }
                    _ => {}
                }
            }

            let app_handle = app.handle().clone();

            if let Ok(cache_dir) = app_handle.path().app_cache_dir() {
                crate::exif_processing::initialize_cache_dir(cache_dir);
            }

            {
                let disks_app_handle = app_handle.clone();
                std::thread::spawn(move || {
                    let disks = sysinfo::Disks::new_with_refreshed_list();
                    let state = disks_app_handle.state::<AppState>();
                    *state.disks_cache.lock().unwrap() = Some(disks);
                });
            }

            let config_dir = app_handle.path().app_config_dir().expect("Failed to get config dir");
            let crash_flag_path = config_dir.join(".gpu_init_crash_flag");

            {
                let state = app.state::<AppState>();
                *state.gpu_crash_flag_path.lock().unwrap() = Some(crash_flag_path.clone());
            }

            let mut settings: AppSettings = load_settings(app_handle.clone()).unwrap_or_default();

            {
                let state = app.state::<AppState>();
                let cache_size = settings.image_cache_size.unwrap_or(5) as usize;
                state.decoded_image_cache.lock().unwrap().set_capacity(cache_size);
            }

            if crash_flag_path.exists() {
                // Windows 上 OpenGL/GLES 与 GPUPixel 争用会死锁；崩溃回退改走 DX12。
                #[cfg(target_os = "windows")]
                {
                    log::warn!(
                        "GPU Driver crash detected on last run! Falling back to DX12 backend (not OpenGL)."
                    );
                    settings.processing_backend = Some("dx12".to_string());
                }
                #[cfg(not(target_os = "windows"))]
                {
                    log::warn!(
                        "GPU Driver crash detected on last run! Falling back to OpenGL backend."
                    );
                    settings.processing_backend = Some("gl".to_string());
                }
                let _ = crate::save_settings(settings.clone(), app_handle.clone());
                let _ = std::fs::remove_file(&crash_flag_path);
            }

            // 迁移旧配置：Windows 上残留的 gl 会与 GPUPixel 冲突。
            #[cfg(target_os = "windows")]
            if settings.processing_backend.as_deref() == Some("gl") {
                log::warn!(
                    "Windows processingBackend=gl 与 GPUPixel 冲突，已自动改回 auto（DX12/Vulkan）"
                );
                settings.processing_backend = Some("auto".to_string());
                let _ = crate::save_settings(settings.clone(), app_handle.clone());
            }

            let lens_db = lens_correction::load_lensfun_db(&app_handle);
            {
                let state = app.state::<AppState>();
                *state.lens_db.lock().unwrap() = Some(Arc::new(lens_db));
            }

            unsafe {
                if let Some(backend) = &settings.processing_backend
                    && backend != "auto"
                {
                    #[cfg(target_os = "windows")]
                    let skip_gl = backend.eq_ignore_ascii_case("gl");
                    #[cfg(not(target_os = "windows"))]
                    let skip_gl = false;
                    if !skip_gl {
                        std::env::set_var("WGPU_BACKEND", backend);
                    }
                }

                #[cfg(target_os = "linux")]
                {
                    if settings.linux_gpu_optimization.unwrap_or(false) {
                        std::env::set_var("WEBKIT_DISABLE_DMABUF_RENDERER", "1");
                        std::env::set_var("WEBKIT_DISABLE_COMPOSITING_MODE", "1");
                        std::env::set_var("NODEVICE_SELECT", "1");
                    }
                }

                #[cfg(not(target_os = "android"))]
                {
                    let resource_path = app_handle
                        .path()
                        .resolve("resources", tauri::path::BaseDirectory::Resource)
                        .expect("failed to resolve resource directory");

                    let ort_library_name = {
                        #[cfg(target_os = "windows")]
                        { "onnxruntime.dll" }
                        #[cfg(target_os = "linux")]
                        { "libonnxruntime.so" }
                        #[cfg(target_os = "macos")]
                        { "libonnxruntime.dylib" }
                        #[cfg(not(any(windows, target_os = "linux", target_os = "macos")))]
                        { "libonnxruntime.so" }
                    };
                    let ort_library_path = resource_path.join(ort_library_name);
                    std::env::set_var("ORT_DYLIB_PATH", &ort_library_path);
                    println!("Set ORT_DYLIB_PATH to: {}", ort_library_path.display());

                    let gpupixel_dir = resource_path.join("gpupixel");
                    portrait_gpupixel::set_resource_directory(gpupixel_dir.clone());
                    println!(
                        "GPUPixel resource dir: {}",
                        gpupixel_dir.display()
                    );

                    portrait_engine::set_resource_directory(
                        resource_path.clone(),
                        Some(ort_library_path.clone()),
                    );
                }

                #[cfg(target_os = "android")]
                {
                    // jniLibs resolve by soname; assets/gpupixel holds VNN + lookup.
                    if let Ok(dir) = app_handle.path().resolve(
                        "assets/gpupixel",
                        tauri::path::BaseDirectory::Resource,
                    ) {
                        portrait_gpupixel::set_resource_directory(dir);
                    }
                }
            }

            // 无头导出可能与桌面实例并存，不参与会话标记与崩溃上报
            setup_logging(&app_handle, !is_headless);
            if !is_headless {
                crash_report::spawn_background_uploader();
            }

            if let Some(backend) = &settings.processing_backend
                && backend != "auto" {
                    log::info!("Applied processing backend setting: {}", backend);
                }

            #[cfg(target_os = "linux")]
            if settings.linux_gpu_optimization.unwrap_or(false) {
                log::info!("Applied Linux Compatibility Mode (forced software compositing).");
            } else {
                log::info!(
                    "Wayland Nvidia quirk status: {:?}",
                    tauri_plugin_wayland_nvidia_quirk::status()
                );
            }

            match launch_req {
                LaunchRequest::HeadlessExport(session) => {
                    let app_handle_clone = app_handle.clone();
                    tauri::async_runtime::spawn(async move {
                        match crate::export_processing::run_headless_export(session, app_handle_clone.clone()).await {
                            Ok(_) => {
                                println!("Headless export completed successfully.");
                                app_handle_clone.exit(0);
                            }
                            Err(e) => {
                                eprintln!("Headless export failed: {}", e);
                                app_handle_clone.exit(1);
                            }
                        }
                    });

                    return Ok(());
                }
                LaunchRequest::InvalidHeadless(_) => unreachable!("invalid headless arguments exit before app setup"),
                _ => {}
            }

            start_preview_worker(app_handle.clone());
            start_analytics_worker(app_handle.clone());
            file_management::start_thumbnail_workers(app_handle.clone());
            file_management::start_metadata_workers(app_handle.clone());
            jxl_oxide::integration::register_image_decoding_hook();

            let window_cfg = app.config().app.windows.first().unwrap().clone();
            let decorations = settings.decorations.unwrap_or(window_cfg.decorations);
            #[cfg(target_os = "android")]
            let _ = decorations;

            let main_window_cfg = app
                .config()
                .app
                .windows
                .iter()
                .find(|w| w.label == "main")
                .expect("Main window config not found")
                .clone();

            let mut window_builder =
                tauri::WebviewWindowBuilder::from_config(app.handle(), &main_window_cfg)
                    .unwrap();

            #[cfg(not(target_os = "android"))]
            {
                window_builder = window_builder.decorations(decorations).visible(false);
            }

            let window = window_builder.build().expect("Failed to build window");

            #[cfg(target_os = "android")]
            android_integration::initialize_android(&window);

            #[cfg(not(target_os = "android"))]
            {
                let app_state = app.state::<AppState>();
                if let Err(error) = get_or_init_gpu_context(&app_state, app.handle()) {
                    log::warn!(
                        "GPU pre-initialization failed (editing and thumbnails may be degraded): {}",
                        error
                    );
                }

                if let Ok(config_dir) = app.path().app_config_dir() {
                    let path = config_dir.join("window_state.json");
                    if let Ok(contents) = std::fs::read_to_string(&path) {
                        if let Ok(state) = serde_json::from_str::<WindowState>(&contents) {
                            let monitor_bounds = available_monitor_bounds(&window);
                            if saved_window_state_is_usable(&state, &monitor_bounds) {
                                let _ = window.set_size(tauri::Size::Physical(
                                    tauri::PhysicalSize::new(state.width, state.height),
                                ));
                                let _ = window.set_position(tauri::Position::Physical(
                                    tauri::PhysicalPosition::new(state.x, state.y),
                                ));
                            } else {
                                log::warn!(
                                    "Saved window state was unusable ({}x{} at {},{}), centering instead.",
                                    state.width,
                                    state.height,
                                    state.x,
                                    state.y
                                );
                                let _ = window.center();
                            }
                        } else {
                            let _ = window.center();
                        }
                    } else {
                        let _ = window.center();
                    }
                } else {
                    let _ = window.center();
                }

                let window_failsafe = window.clone();
                tauri::async_runtime::spawn(async move {
                    tokio::time::sleep(std::time::Duration::from_secs(4)).await;
                    if let Ok(false) = window_failsafe.is_visible() {
                        log::warn!(
                            "Frontend failed to report ready within timeout. Forcing window visibility."
                        );
                        let _ = window_failsafe.show();
                        let _ = window_failsafe.set_focus();
                    }
                });

                let pending_window_state = Arc::new(Mutex::new(None::<WindowState>));
                let pending_state_for_saver = pending_window_state.clone();
                let app_handle_for_saver = app.handle().clone();

                tauri::async_runtime::spawn(async move {
                    loop {
                        tokio::time::sleep(Duration::from_millis(500)).await;

                        let state_to_save = {
                            let mut lock = pending_state_for_saver.lock().unwrap();
                            lock.take()
                        };

                        if let Some(state) = state_to_save
                            && let Ok(config_dir) =
                                app_handle_for_saver.path().app_config_dir()
                        {
                            let path = config_dir.join("window_state.json");
                            let _ = std::fs::create_dir_all(&config_dir);
                            if let Ok(json) = serde_json::to_string(&state) {
                                let _ = std::fs::write(&path, json);
                            }
                        }
                    }
                });

                let window_for_handler = window.clone();
                let pending_state_for_handler = pending_window_state.clone();

                window.on_window_event(move |event| match event {
                    tauri::WindowEvent::Resized(_) | tauri::WindowEvent::Moved(_) => {
                        #[cfg(any(windows, target_os = "linux"))]
                        let maximized = window_for_handler.is_maximized().unwrap_or(false);
                        #[cfg(not(any(windows, target_os = "linux")))]
                        let maximized = false;

                        #[cfg(any(windows, target_os = "linux"))]
                        let fullscreen = window_for_handler.is_fullscreen().unwrap_or(false);
                        #[cfg(not(any(windows, target_os = "linux")))]
                        let fullscreen = false;

                        if window_for_handler.is_minimized().unwrap_or(false) {
                            return;
                        }

                        let mut state = WindowState {
                            width: 1280,
                            height: 720,
                            x: 0,
                            y: 0,
                            maximized,
                            fullscreen,
                        };

                        if let Ok(position) = window_for_handler.outer_position() {
                            state.x = position.x;
                            state.y = position.y;
                        }

                        if !maximized
                            && !fullscreen
                            && let Ok(size) = window_for_handler.outer_size()
                            && size.width >= 800
                            && size.height >= 600
                        {
                            state.width = size.width;
                            state.height = size.height;
                        }

                        *pending_state_for_handler.lock().unwrap() = Some(state);
                    }
                    _ => {}
                });
            }

            crate::register_exit_handler();
            Ok(())
        })
        .manage(AppState {
            window_setup_complete: AtomicBool::new(false),
            gpu_crash_flag_path: Mutex::new(None),
            original_image: Mutex::new(None),
            cached_preview: Mutex::new(None),
            gpu_context: Mutex::new(None),
            gpu_image_cache: Mutex::new(None),
            gpu_processor: Mutex::new(None),
            ai_state: Mutex::new(None),
            ai_init_lock: TokioMutex::new(()),
            portrait_state: portrait_processing::PortraitState::default(),
            reshape_analysis: portrait_analysis_cache::ReshapeAnalysisState::default(),
            portrait_render: portrait_render::PortraitRenderState::default(),
            export_task_token: Arc::new(Mutex::new(None)),
            hdr_result: Arc::new(Mutex::new(None)),
            panorama_result: Arc::new(Mutex::new(None)),
            focus_stack_result: Arc::new(Mutex::new(None)),
            denoise_result: Arc::new(Mutex::new(None)),
            indexing_task_handle: Mutex::new(None),
            lut_cache: Mutex::new(HashMap::new()),
            initial_file_path: Mutex::new(None),
            pending_edit_session: Mutex::new(None),
            thumbnail_cancellation_token: Arc::new(AtomicBool::new(false)),
            thumbnail_progress: Mutex::new(ThumbnailProgressTracker { total: 0, completed: 0 }),
            preview_worker_tx: Mutex::new(None),
            analytics_worker_tx: Mutex::new(None),
            mask_cache: Mutex::new(HashMap::new()),
            patch_cache: Mutex::new(HashMap::new()),
            geometry_cache: Mutex::new(HashMap::new()),
            thumbnail_geometry_cache: Mutex::new(HashMap::new()),
            lens_db: Mutex::new(None),
            load_image_generation: Arc::new(AtomicUsize::new(0)),
            preview_epoch: Mutex::new(0),
            full_warped_cache: Mutex::new(None),
            patched_warped_cache: Mutex::new(None),
            full_transformed_cache: Mutex::new(None),
            decoded_image_cache: Mutex::new(DecodedImageCache::new(5)),
            thumbnail_manager: ThumbnailManager::new(),
            metadata_manager: MetadataManager::new(),
            disks_cache: Mutex::new(None),
            disks_cache_refreshing: AtomicBool::new(false),
            camera_session: Mutex::new(camera_tethering::CameraSession::new()),
        })
        .invoke_handler(tauri::generate_handler![
            apply_adjustments,
            portrait_processing::prepare_portrait_analysis,
            portrait_processing::cancel_portrait_analysis,
            portrait_analysis_cache::prepare_portrait_reshape_analysis,
            portrait_analysis_cache::cancel_portrait_reshape_analysis,
            generate_preview_for_path,
            generate_preset_preview,
            generate_uncropped_preview,
            get_log_file_path,
            frontend_log,
            save_collage,
            merge_hdr,
            save_hdr,
            lut_processing::load_and_parse_lut,
            lut_processing::list_luts,
            lut_processing::import_luts,
            lut_processing::remove_lut,
            lut_processing::generate_lut_previews,
            fetch_community_presets,
            generate_all_community_previews,
            save_temp_file,
            get_image_dimensions,
            frontend_ready,
            cancel_thumbnail_generation,
            update_wgpu_transform,
            android_integration::resolve_android_content_uri_name,
            cache_utils::clear_session_caches,
            cache_utils::clear_image_caches,
            app_settings::load_settings,
            app_settings::save_settings,
            app_settings::is_tethering_supported,
            qr_login::qr_login_create_session,
            qr_login::qr_login_poll,
            qr_login::qr_login_verify_saved,
            qr_login::qr_login_logout,
            usage_tracking::report_desktop_usage,
            crash_report::report_frontend_error,
            qr_login::fetch_pixel_balance,
            qr_login::fetch_pixel_details,
            pixel_billing::fetch_pixel_earn_qr_code,
            invite::fetch_invite_info,
            invite::bind_invite_code,
            bug_feedback::fetch_bug_feedback_info,
            bug_feedback::submit_bug_feedback,
            app_update::fetch_app_version_policy,
            announcement::fetch_announcements,
            requirement_pool::fetch_requirement_info,
            requirement_pool::fetch_requirements,
            requirement_pool::submit_requirement,
            requirement_pool::toggle_requirement_like,
            ai_commands::generate_ai_subject_mask,
            ai_commands::precompute_ai_subject_mask,
            ai_commands::generate_ai_foreground_mask,
            ai_commands::generate_ai_sky_mask,
            ai_commands::generate_ai_depth_mask,
            ai_commands::check_ai_connector_status,
            ai_commands::test_ai_connector_connection,
            ai_commands::generate_full_image_depth_map,
            inpainting::invoke_generative_replace_with_mask_def,
            inpainting::generate_manual_cleanup_patch,
            inpainting::generate_liquify_patch,
            inpainting::generate_retouch_patch,
            denoising::apply_denoising,
            denoising::batch_denoise_images,
            denoising::save_denoised_image,
            focus_stacking::stitch_focus_stack,
            focus_stacking::save_focus_stack,
            image_loader::load_image,
            image_loader::is_image_cached,
            panorama_stitching::stitch_panorama,
            panorama_stitching::save_panorama,
            export_processing::export_images,
            export_processing::cancel_export,
            export_processing::estimate_export_sizes,
            image_processing::calculate_auto_adjustments,
            image_processing::sample_white_balance,
            apple_raw::is_raw9_available,
            mask_generation::generate_mask_overlay,
            file_management::update_exif_fields,
            file_management::get_supported_file_types,
            file_management::read_exif_for_paths,
            file_management::list_images_in_dir,
            file_management::list_images_recursive,
            file_management::get_folder_tree,
            file_management::get_folder_children,
            file_management::get_pinned_folder_trees,
            file_management::update_thumbnail_queue,
            file_management::create_folder,
            file_management::delete_folder,
            file_management::copy_files,
            file_management::move_files,
            file_management::rename_folder,
            file_management::rename_files,
            file_management::duplicate_file,
            file_management::show_in_finder,
            file_management::delete_files_from_disk,
            file_management::delete_files_with_associated,
            file_management::save_metadata_and_update_thumbnail,
            file_management::apply_adjustments_to_paths,
            file_management::load_metadata,
            file_management::load_presets,
            file_management::save_presets,
            file_management::get_or_create_internal_library_root,
            file_management::reset_adjustments_for_paths,
            file_management::apply_auto_lens_correction_to_paths,
            file_management::apply_auto_adjustments_to_paths,
            file_management::handle_import_presets_from_file,
            file_management::handle_import_legacy_presets_from_file,
            file_management::handle_import_presets_from_files,
            file_management::handle_export_presets_to_file,
            file_management::save_community_preset,
            file_management::clear_all_sidecars,
            file_management::clear_thumbnail_cache,
            file_management::set_color_label_for_paths,
            file_management::set_rating_for_paths,
            file_management::set_flag_for_paths,
            file_management::import_files,
            file_management::create_virtual_copy,
            file_management::get_albums,
            file_management::save_albums,
            file_management::add_to_album,
            file_management::get_album_images,
            tagging::start_background_indexing,
            tagging::clear_ai_tags,
            tagging::clear_all_tags,
            tagging::add_tag_for_paths,
            tagging::remove_tag_for_paths,
            culling::cull_images,
            lens_correction::get_lensfun_makers,
            lens_correction::get_lensfun_lenses_for_maker,
            lens_correction::autodetect_lens,
            lens_correction::get_lens_distortion_params,
            negative_conversion::preview_negative_conversion,
            negative_conversion::convert_negatives,
            camera_tethering::tether_list_cameras,
            camera_tethering::tether_connect,
            camera_tethering::tether_get_settings,
            camera_tethering::tether_set_setting,
            camera_tethering::tether_capture,
            camera_tethering::tether_get_preview,
            camera_tethering::tether_autofocus,
            guided_perspective::calculate_guided_perspective,
        ])
        .build(tauri::generate_context!())
        .expect("error while building tauri application")
        .run(#[allow(unused_variables)] |app_handle, event| {
            match event {
                #[cfg(target_os = "macos")]
				tauri::RunEvent::Opened { urls } => {
				    if let Some(url) = urls.first()
				        && let Ok(path) = url.to_file_path()
				        && let Some(path_str) = path.to_str()
				    {
				        let state = app_handle.state::<AppState>();

				        if state
				            .window_setup_complete
				            .load(std::sync::atomic::Ordering::Relaxed)
				        {
				            if let Some(window) = app_handle.get_webview_window("main") {
				                let _ = window.unminimize();
				                let _ = window.show();
				                let _ = window.set_focus();
				            }

				            log::info!("macOS runtime open: Opening {}.", path_str);

				            emit_launch_request(
				                app_handle,
				                LaunchRequest::OpenFile(path_str.to_string()),
				            );
				        } else {
				            *state.initial_file_path.lock().unwrap() =
				                Some(path_str.to_string());

				            log::info!(
				                "macOS initial open: Stored path {} for later.",
				                path_str
				            );
				        }
				    }
				}
                tauri::RunEvent::ExitRequested { api, .. } => {
                    api.prevent_exit();
                    crash_report::mark_clean_exit();

                    #[cfg(target_os = "macos")]
                    unsafe { libc::_exit(0); }

                    #[cfg(not(target_os = "macos"))]
                    std::process::exit(0);
                }
                tauri::RunEvent::Exit => {
                    crash_report::mark_clean_exit();

                    #[cfg(target_os = "macos")]
                    unsafe { libc::_exit(0); }

                    #[cfg(not(target_os = "macos"))]
                    std::process::exit(0);
                }
                _ => {}
            }
        });
}
