//! 08：严格源快照、共享冻结资产和分层身份；不接受外部网格或跨会话快照。
#![allow(dead_code)]
use crate::portrait_analysis::{ReshapeAnalysis, check_current, match_faces};
use crate::portrait_analysis_cache::{TaskCache, analysis_base};
use crate::portrait_protection::LimitReason;
use crate::portrait_reshape::{KEYS, read_parameters};
use crate::portrait_safety::SafeScene;
use crate::portrait_snapshot::FrozenPlan;
use crate::portrait_tiles::{CanonicalField, OutputWindow, check_dimensions};
use image::DynamicImage;
#[cfg(test)]
use image::RgbaImage;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, VecDeque};
use std::fs::File;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::UNIX_EPOCH;
use tauri::Manager;

pub const RENDER_VERSION: &str = "portrait-render-v2";
const ASSET_BUDGET: usize = 64 * 1024 * 1024;
const PORTRAIT_BASE_BUDGET: usize = 256 * 1024 * 1024;
// 至少容纳最近两张常见 1792px RGBA32F 输入代理及其 bake 结果，支持 A→B→A 热切回。
const PORTRAIT_PROXY_BUDGET: usize = 128 * 1024 * 1024;

/// 交互实时预览的长边上限；`"full"` 表示不额外收紧（仍可被输出窗裁切）。
pub(crate) fn portrait_preview_limit(quality: &str) -> Option<u32> {
    match quality {
        "full" => None,
        "performance" => Some(960),
        _ => Some(1280),
    }
}

/// 编辑器预览（含松手高质帧）在代理尺寸上做人像 bake／调色；导出与非 identity 场不走捷径。
/// `output_long_edge` 为当前预览窗长边，避免工作分辨率超过最终显示。
fn interactive_portrait_proxy_limit(
    purpose: RenderPurpose,
    quality: RenderQuality,
    identity: bool,
    live_quality: &str,
    output_long_edge: u32,
) -> Option<u32> {
    if purpose != RenderPurpose::Preview || !identity {
        return None;
    }
    let edge = output_long_edge.max(1);
    match quality {
        RenderQuality::Interactive => Some(
            portrait_preview_limit(live_quality)
                .map_or(edge, |limit| limit.min(edge)),
        ),
        RenderQuality::HighQuality => Some(edge),
    }
}

fn portrait_base_key(source_key: &str, adjustments: &Value) -> Result<String, String> {
    Ok(content_key(&json!([
        "portrait-pre-color-v2",
        source_key,
        analysis_base(adjustments)?,
        crate::portrait_gpupixel::parameter_hash(adjustments),
    ])))
}

/// GPUPixel bake：瘦脸／大眼／口红／腮红整图 RGBA。
fn bake_gpupixel_base(
    _app: &tauri::AppHandle,
    request: &PhotoRenderRequest,
    _spatial: &Value,
    input: &DynamicImage,
    _current: &dyn Fn() -> bool,
) -> Result<DynamicImage, String> {
    let params = crate::portrait_gpupixel::params_from_adjustments(&request.adjustments);
    crate::portrait_gpupixel::bake_dynamic(input, params)
}

#[derive(Clone)]
struct PortraitBasePixels {
    image: Arc<DynamicImage>,
    original_dimensions: [u32; 2],
}

/// 按内容键保留最近编辑源；缓存失败不改变照片处理结果。
#[derive(Default)]
struct PortraitBaseCache {
    full: Option<(String, PortraitBasePixels)>,
    proxies: VecDeque<(String, u32, PortraitBasePixels)>,
    hits: u64,
    bakes: u64,
}

impl PortraitBaseCache {
    fn full(&self, key: &str) -> Option<PortraitBasePixels> {
        self.full
            .as_ref()
            .filter(|entry| entry.0 == key)
            .map(|entry| entry.1.clone())
    }

    fn proxy(&mut self, key: &str, limit: u32) -> Option<PortraitBasePixels> {
        let index = self
            .proxies
            .iter()
            .position(|entry| entry.0 == key && entry.1 == limit)?;
        let entry = self.proxies.remove(index)?;
        let result = entry.2.clone();
        self.proxies.push_back(entry);
        Some(result)
    }

    fn insert_full(&mut self, key: &str, pixels: &PortraitBasePixels, budget: usize) {
        self.full =
            (pixels.image.as_bytes().len() <= budget).then(|| (key.to_owned(), pixels.clone()));
    }

    fn insert_proxy(&mut self, key: &str, limit: u32, pixels: &PortraitBasePixels) {
        let bytes = pixels.image.as_bytes().len();
        if bytes > PORTRAIT_PROXY_BUDGET {
            return;
        }
        self.proxies
            .retain(|entry| entry.0 == key && entry.1 != limit);
        while self
            .proxies
            .iter()
            .map(|entry| entry.2.image.as_bytes().len())
            .sum::<usize>()
            > PORTRAIT_PROXY_BUDGET - bytes
        {
            self.proxies.pop_front();
        }
        self.proxies
            .push_back((key.to_owned(), limit, pixels.clone()));
    }
}

/// 从任意位深基底做面积采样到代理浮点图；预览路径可能是 RAW 浮点或 RGB8，先量化到 RGBA8
///（与 GPUPixel bake 一致），只分配代理浮点图，不复制整图到浮点内存。
fn portrait_proxy(
    image: &DynamicImage,
    limit: u32,
    current: &dyn Fn() -> bool,
) -> Result<DynamicImage, String> {
    check_current(current)?;
    let rgba8;
    let rgba = if let Some(v) = image.as_rgba8() {
        v
    } else {
        // RAW/JPEG 等常见基底不是 RGBA8；to_rgba8 与 bake_dynamic 同路径。
        rgba8 = image.to_rgba8();
        &rgba8
    };
    check_dimensions([rgba.width(), rgba.height()])?;
    let scale = (limit.max(1) as f64 / rgba.width().max(rgba.height()) as f64).min(1.0);
    let width = (rgba.width() as f64 * scale).round().max(1.0) as u32;
    let height = (rgba.height() as f64 * scale).round().max(1.0) as u32;
    let mut pixels = Vec::new();
    pixels
        .try_reserve_exact(width as usize * height as usize * 4)
        .map_err(|_| "人像代理内存不足")?;
    let sx = rgba.width() as f64 / width as f64;
    let sy = rgba.height() as f64 / height as f64;
    for y in 0..height {
        check_current(current)?;
        let top = y as f64 * sy;
        let bottom = ((y + 1) as f64 * sy).min(rgba.height() as f64);
        for x in 0..width {
            let left = x as f64 * sx;
            let right = ((x + 1) as f64 * sx).min(rgba.width() as f64);
            let mut sum = [0.0; 4];
            for iy in top.floor() as u32..bottom.ceil() as u32 {
                let wy = (bottom.min(iy as f64 + 1.0) - top.max(iy as f64)).max(0.0);
                for ix in left.floor() as u32..right.ceil() as u32 {
                    let weight = wy * (right.min(ix as f64 + 1.0) - left.max(ix as f64)).max(0.0);
                    let p = rgba.get_pixel(ix, iy).0;
                    let alpha = p[3] as f64 / 255.0;
                    for c in 0..3 {
                        sum[c] += p[c] as f64 / 255.0 * alpha * weight;
                    }
                    sum[3] += alpha * weight;
                }
            }
            let alpha = (sum[3] / (sx * sy)).clamp(0.0, 1.0);
            for value in &sum[..3] {
                pixels.push(if sum[3] > 0.0 {
                    (value / sum[3]) as f32
                } else {
                    0.0
                });
            }
            pixels.push(alpha as f32);
        }
    }
    let image = image::Rgba32FImage::from_raw(width, height, pixels).ok_or("人像代理尺寸错误")?;
    Ok(DynamicImage::ImageRgba32F(image))
}

#[cfg(test)]
fn fit_rgba8_long_edge(image: &DynamicImage, limit: u32) -> RgbaImage {
    let limit = limit.max(1);
    let scale = (limit as f64 / image.width().max(image.height()).max(1) as f64).min(1.0);
    let width = (image.width() as f64 * scale).round().max(1.0) as u32;
    let height = (image.height() as f64 * scale).round().max(1.0) as u32;
    image::imageops::resize(
        &image.to_rgba8(),
        width,
        height,
        image::imageops::FilterType::Triangle,
    )
}

/// 连续仿射映射保留裁剪的亚像素边界，不能把代理裁剪再次四舍五入。
fn proxy_output_window(window: &OutputWindow, full: [u32; 2], proxy: [u32; 2]) -> OutputWindow {
    let mut output = window.clone();
    let sx = proxy[0] as f64 / full[0] as f64;
    let sy = proxy[1] as f64 / full[1] as f64;
    for v in &mut output.output_to_base[..3] {
        *v *= sx;
    }
    for v in &mut output.output_to_base[3..] {
        *v *= sy;
    }
    output
}

/// 渲染期安全回退对用户可见的状态；质量门（如 Unknown×0.85）不计入，避免正脸常亮提示。
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum ReshapeAttenuation {
    Limited,
    Identity,
}

fn log_reshape_diagnostics(scene: &SafeScene, field_identity: bool) {
    for (report, mesh) in scene.reports().zip(scene.meshes()) {
        let face_width = mesh.face_width();
        let max_disp = mesh.diagnostics.max_displacement;
        log::info!(
            target: "portrait_reshape",
            "reshape face={} requested={:?} actual={:?} quality={:.3} common_quality={:.3} safety_scale={:.3} max_disp_face={:.4} max_disp_px={:.2} bypassed={} field_identity={} reasons={:?}",
            report.face_id,
            report.requested.0,
            report.actual.0,
            report.quality.response_scale,
            report.common_quality_scale,
            report.safety_scale,
            max_disp,
            max_disp * face_width,
            report.bypassed,
            field_identity,
            report.reasons,
        );
    }
}

fn compute_reshape_attenuation(asset: &FrozenReshapeAsset) -> Option<ReshapeAttenuation> {
    let reports: Vec<_> = asset.scene().reports().collect();
    let any_nonzero = reports
        .iter()
        .any(|report| report.requested.0.iter().any(|value| *value > 0.0));
    if !any_nonzero {
        return None;
    }
    if asset.field().is_identity() || reports.iter().all(|report| report.bypassed) {
        return Some(ReshapeAttenuation::Identity);
    }
    let limited = reports.iter().any(|report| {
        report.safety_scale < 1.0 - 1e-12
            || report.reasons.contains(&LimitReason::SafetyReduced)
            || report.reasons.contains(&LimitReason::SafetyBypass)
    });
    limited.then_some(ReshapeAttenuation::Limited)
}

/// JSON 对象递归排序，数组保序；不得以 id 或数据长度代替像素内容。
pub(crate) fn content_key(value: &Value) -> String {
    fn ordered(value: &Value) -> Value {
        match value {
            Value::Object(object) => {
                let sorted: std::collections::BTreeMap<_, _> = object
                    .iter()
                    .map(|(key, value)| (key.clone(), ordered(value)))
                    .collect();
                Value::Object(sorted.into_iter().collect())
            }
            Value::Array(values) => Value::Array(values.iter().map(ordered).collect()),
            value => value.clone(),
        }
    }
    hex::encode(Sha256::digest(ordered(value).to_string().as_bytes()))
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct FileFingerprint {
    canonical: PathBuf,
    identity: [u64; 2],
    length: u64,
    modified: u128,
}

#[cfg(unix)]
fn file_identity(file: &File) -> Result<[u64; 2], String> {
    use std::os::unix::fs::MetadataExt;
    let metadata = file.metadata().map_err(|e| e.to_string())?;
    Ok([metadata.dev(), metadata.ino()])
}

#[cfg(windows)]
fn file_identity(file: &File) -> Result<[u64; 2], String> {
    use std::os::windows::io::AsRawHandle;
    #[repr(C)]
    #[derive(Default)]
    struct FileInformation {
        attributes: u32,
        creation: [u32; 2],
        access: [u32; 2],
        write: [u32; 2],
        volume: u32,
        size_high: u32,
        size_low: u32,
        links: u32,
        index_high: u32,
        index_low: u32,
    }
    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn GetFileInformationByHandle(
            handle: *mut std::ffi::c_void,
            info: *mut FileInformation,
        ) -> i32;
    }
    let mut information = FileInformation::default();
    // 句柄由 File 持有且在调用期有效；结构布局对应 BY_HANDLE_FILE_INFORMATION。
    if unsafe { GetFileInformationByHandle(file.as_raw_handle(), &mut information) } == 0 {
        return Err(std::io::Error::last_os_error().to_string());
    }
    Ok([
        information.volume as u64,
        ((information.index_high as u64) << 32) | information.index_low as u64,
    ])
}

#[cfg(not(any(unix, windows)))]
fn file_identity(_: &File) -> Result<[u64; 2], String> {
    Err("此平台尚未实现严格源文件身份".into())
}

impl FileFingerprint {
    /// 再次打开路径，避免只核对已被重命名移走的旧句柄。
    fn verify_path(&self, path: &Path) -> Result<(), String> {
        let file = File::open(path).map_err(|e| e.to_string())?;
        if Self::read(path, &file)? != *self {
            return Err("重塑源路径已指向另一文件，请重新加载".into());
        }
        Ok(())
    }

    fn read(path: &Path, file: &File) -> Result<Self, String> {
        let metadata = file.metadata().map_err(|e| e.to_string())?;
        if !metadata.is_file() {
            return Err("重塑源必须是普通文件".into());
        }
        Ok(Self {
            canonical: path.canonicalize().map_err(|e| e.to_string())?,
            identity: file_identity(file)?,
            length: metadata.len(),
            modified: metadata
                .modified()
                .map_err(|e| e.to_string())?
                .duration_since(UNIX_EPOCH)
                .map_err(|e| e.to_string())?
                .as_nanos(),
        })
    }
}

/// 读入内容为自有字节，防止 mmap 在解码期间观察到另一版本的文件。
fn read_snapshot_bytes(
    path: &Path,
    current: &dyn Fn() -> bool,
) -> Result<(FileFingerprint, Vec<u8>, String), String> {
    let mut file = File::open(path).map_err(|e| e.to_string())?;
    let before = FileFingerprint::read(path, &file)?;
    let length = usize::try_from(before.length).map_err(|_| "源文件长度超限")?;
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(length)
        .map_err(|_| "源文件快照内存不足")?;
    let mut digest = Sha256::new();
    let mut chunk = [0u8; 64 * 1024];
    loop {
        check_current(current)?;
        let count = file.read(&mut chunk).map_err(|e| e.to_string())?;
        if count == 0 {
            break;
        }
        if bytes.len().checked_add(count).is_none_or(|n| n > length) {
            return Err("源文件在读取期间增长".into());
        }
        bytes.extend_from_slice(&chunk[..count]);
        digest.update(&chunk[..count]);
    }
    if bytes.len() != length || FileFingerprint::read(path, &file)? != before {
        return Err("源文件在读取期间发生变化".into());
    }
    before.verify_path(path)?;
    Ok((before, bytes, hex::encode(digest.finalize())))
}

pub struct SourceSnapshot {
    virtual_path: String,
    fingerprint: FileFingerprint,
    content_hash: String,
    decode_settings: Value,
    source_key: String,
    revision: u64,
    image: Arc<DynamicImage>,
    is_raw: bool,
    /// 同快照生命周期内全文哈希只做一次；之后 verify 只核指纹，避免交互预览反复读盘被代次取消。
    content_verified: AtomicBool,
}

impl SourceSnapshot {
    fn decode_key(settings: &crate::AppSettings) -> Value {
        json!([
            settings.raw_highlight_compression,
            settings.linear_raw_mode,
            settings.raw_preprocessing_color_nr,
            settings.raw_preprocessing_sharpening,
            settings.apply_preprocessing_to_non_raws
        ])
    }

    pub fn load(
        path: &str,
        settings: &crate::AppSettings,
        revision: u64,
        current: &dyn Fn() -> bool,
    ) -> Result<Self, String> {
        check_current(current)?;
        let (source, _) = crate::file_management::parse_virtual_path(path);
        let (fingerprint, bytes, content_hash) = read_snapshot_bytes(&source, current)?;
        let source_str = source.to_str().ok_or("重塑源路径编码错误")?;
        let image = crate::image_loader::load_base_image_strict(&bytes, source_str, settings, None)
            .map_err(|e| e.to_string())?;
        check_dimensions([image.width(), image.height()])?;
        check_current(current)?;
        let decode_settings = Self::decode_key(settings);
        let source_key = content_key(&json!([
            "strict-source-v1",
            fingerprint.canonical.to_string_lossy(),
            path,
            fingerprint.identity,
            fingerprint.length,
            fingerprint.modified.to_string(),
            content_hash,
            decode_settings
        ]));
        let snapshot = Self {
            virtual_path: path.into(),
            fingerprint,
            content_hash,
            decode_settings,
            source_key,
            revision,
            image: Arc::new(image),
            is_raw: crate::formats::is_raw_file(source_str),
            content_verified: AtomicBool::new(false),
        };
        drop(bytes);
        snapshot.verify(current)?;
        Ok(snapshot)
    }

    /// 发布前重新读取内容，捕获同长度、同时间戳替换；tile 内只检查消费者代次。
    /// 同一 SourceSnapshot 上全文哈希只核验一次，后续仅核指纹（仍每次 check_current）。
    pub fn verify(&self, current: &dyn Fn() -> bool) -> Result<(), String> {
        check_current(current)?;
        let (source, _) = crate::file_management::parse_virtual_path(&self.virtual_path);
        let mut file = File::open(&source).map_err(|e| e.to_string())?;
        if FileFingerprint::read(&source, &file)? != self.fingerprint {
            return Err("重塑源快照已过期，请重新加载".into());
        }
        self.fingerprint.verify_path(&source)?;
        if self.content_verified.load(Ordering::Acquire) {
            return check_current(current);
        }
        let mut hash = Sha256::new();
        let mut chunk = [0u8; 64 * 1024];
        let mut count = 0u64;
        loop {
            check_current(current)?;
            let n = file.read(&mut chunk).map_err(|e| e.to_string())?;
            if n == 0 {
                break;
            }
            count = count.checked_add(n as u64).ok_or("源文件长度溢出")?;
            if count > self.fingerprint.length {
                return Err("重塑源文件已改变".into());
            }
            hash.update(&chunk[..n]);
        }
        if count != self.fingerprint.length
            || hex::encode(hash.finalize()) != self.content_hash
            || FileFingerprint::read(&source, &file)? != self.fingerprint
        {
            return Err("重塑源内容已改变，请重新加载".into());
        }
        self.content_verified.store(true, Ordering::Release);
        check_current(current)
    }

    pub fn source_key(&self) -> &str {
        &self.source_key
    }
    pub fn image(&self) -> &DynamicImage {
        &self.image
    }
    pub fn is_raw(&self) -> bool {
        self.is_raw
    }
    pub fn revision(&self) -> u64 {
        self.revision
    }
    pub fn path(&self) -> &str {
        &self.virtual_path
    }

    /// 目标即使是虚拟路径、硬链接或 Windows 大小写别名也不得覆盖源。
    pub fn reject_source_target(&self, target: &Path) -> Result<(), String> {
        let target_text = target.to_str().ok_or("导出路径编码错误")?;
        let (target, _) = crate::file_management::parse_virtual_path(target_text);
        match File::open(&target) {
            Ok(file) => {
                let identity = FileFingerprint::read(&target, &file)?;
                if identity.identity == self.fingerprint.identity
                    || identity.canonical == self.fingerprint.canonical
                {
                    return Err("禁止覆盖原始照片".into());
                }
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
            Err(e) => return Err(e.to_string()),
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RenderIntent {
    Bypass,
    Prepare,
}

/// GPUPixel 已旁路自研美肤场时，清零调色用的人像 GPU 参数，避免无 maps 仍要求分析。
pub(crate) fn suppress_native_portrait_when_gpupixel(
    gp_active: bool,
    params: &mut crate::image_processing::AllAdjustments,
) {
    if gp_active {
        params.portrait = Default::default();
    }
}

/// 未分析不能把空脸列表当作无脸；关闭和历史参数仍精确走旧路径。
/// GPUPixel 六项或身体塑形任一非零时进入 Prepare（整图 bake／identity 场）。
pub fn render_intent(adjustments: &Value) -> Result<RenderIntent, String> {
    let gp_active = crate::portrait_gpupixel::requested(adjustments);
    let body_active = crate::portrait_engine::body_requested(adjustments);

    // 产品默认 `gpupixel-runtime-v1`：瘦脸／大眼走运行时 bake，不再激活 UV 场。
    // 仍读取旧 UV 引擎记录时，若未迁到 runtime 且仅 UV 非零，保持 Prepare 以兼容。
    let parsed = read_parameters(&adjustments["portraitReshape"]);
    let runtime_engine = parsed.status == "supported"
        && parsed.value["engineVersion"]
            .as_str()
            .map(|e| {
                e == crate::portrait_gpupixel::ENGINE_VERSION
                    || e == crate::portrait_reshape::ENGINE_UV
            })
            .unwrap_or(true);
    let face_uv_active = match parsed.status {
        "legacy" => false,
        "supported" => {
            // runtime 引擎下 slim/bigEye 由 GPUPixel 负责，不单独拉 UV。
            if runtime_engine
                && parsed.value["engineVersion"].as_str()
                    == Some(crate::portrait_gpupixel::ENGINE_VERSION)
            {
                false
            } else {
                let reshape_vis = crate::portrait_reshape::reshape_section_visible(adjustments);
                let eyes_vis = crate::portrait_reshape::eyes_section_visible(adjustments);
                if parsed.value["enabled"] != true || (!reshape_vis && !eyes_vis) {
                    false
                } else {
                    let uv_slim_default =
                        crate::portrait_reshape::amount(&adjustments["portrait"]["uvSlimFace"])
                            .max(crate::portrait_reshape::amount(
                                &adjustments["portrait"]["slimFace"],
                            ));
                    let big_eye_default =
                        crate::portrait_reshape::amount(&adjustments["portrait"]["bigEye"]);
                    let faces = parsed.value["faces"].as_array().ok_or("缺少重塑脸列表")?;
                    let big_eye_nonzero = eyes_vis
                        && (big_eye_default > 0
                            || faces.iter().any(|face| {
                                crate::portrait_reshape::effective_big_eye(
                                    Some(face),
                                    big_eye_default,
                                    eyes_vis,
                                ) > 0
                            }));
                    let slim_nonzero = reshape_vis
                        && (uv_slim_default > 0
                            || faces.iter().any(|face| {
                                crate::portrait_reshape::effective_uv_slim(
                                    Some(face),
                                    uv_slim_default,
                                    reshape_vis,
                                ) > 0
                            }));
                    slim_nonzero || big_eye_nonzero
                }
            }
        }
        _ if gp_active || body_active => false,
        _ => return Err("重塑参数版本非法或不支持".into()),
    };

    if !face_uv_active && !gp_active && !body_active {
        return Ok(RenderIntent::Bypass);
    }
    if face_uv_active {
        validate_options(&parsed.value)?;
    }
    // 空间参数校验；GPUPixel／身体-only 亦不做人脸分析。
    analysis_base(adjustments)?;
    Ok(RenderIntent::Prepare)
}

fn validate_options(parameters: &Value) -> Result<(), String> {
    for face in parameters["faces"].as_array().ok_or("缺少重塑脸列表")? {
        let protection = face["protection"].as_object().ok_or("保护配置非法")?;
        if protection
            .keys()
            .any(|key| !matches!(key.as_str(), "hair" | "background" | "strokes"))
        {
            return Err("当前安全引擎不支持此保护选项，禁止忽略后渲染".into());
        }
        if face["protection"]["hair"] == false || face["protection"]["background"] == false {
            return Err("当前安全引擎不支持关闭头发或背景保护".into());
        }
        if face["protection"]["strokes"]
            .as_array()
            .is_some_and(|v| !v.is_empty())
        {
            return Err("源 UV 保护笔迹尚未投影，禁止忽略后渲染".into());
        }
    }
    Ok(())
}

fn collect_uv_face_strengths(
    analysis: &ReshapeAnalysis,
    adjustments: &Value,
    parameters: &Value,
    include_slim: bool,
    include_big_eye: bool,
) -> Result<Vec<(String, crate::portrait_uv_warp::UvWarpStrengths)>, String> {
    let summaries: Vec<_> = analysis.faces.iter().map(|f| f.summary.clone()).collect();
    let matched = match_faces(&summaries, parameters)?;
    if matched.len() != summaries.len()
        || matched.iter().any(|f| {
            matches!(
                f.reason,
                Some(crate::portrait_reshape::ReshapeReason::UnmatchedFace)
            )
        })
    {
        return Err("逐脸参数未唯一匹配当前源照片，请重新选择人脸".into());
    }
    let saved = parameters["faces"].as_array().ok_or("缺少重塑脸列表")?;
    let slim_default = if include_slim {
        f64::from(
            crate::portrait_reshape::amount(&adjustments["portrait"]["uvSlimFace"]).max(
                crate::portrait_reshape::amount(&adjustments["portrait"]["slimFace"]),
            ),
        )
    } else {
        0.0
    };
    let big_eye_default = if include_big_eye {
        f64::from(crate::portrait_reshape::amount(
            &adjustments["portrait"]["bigEye"],
        ))
    } else {
        0.0
    };
    let mut uv_faces = Vec::new();
    for (original, matched) in summaries.iter().zip(&matched) {
        if !original.processable {
            continue;
        }
        let face = saved.iter().find(|face| face["faceId"] == matched.face_id);
        let slim = if include_slim {
            f64::from(crate::portrait_reshape::effective_uv_slim(
                face,
                slim_default as u32,
                true,
            ))
        } else {
            0.0
        };
        let big_eye = if include_big_eye {
            f64::from(crate::portrait_reshape::effective_big_eye(
                face,
                big_eye_default as u32,
                true,
            ))
        } else {
            0.0
        };
        let strengths = crate::portrait_uv_warp::UvWarpStrengths::from_ui(slim, big_eye)
            .map_err(|_| "UV 重塑强度非法".to_string())?;
        if strengths.is_zero() {
            continue;
        }
        uv_faces.push((original.face_id.clone(), strengths));
    }
    uv_faces.sort_by(|a, b| a.0.cmp(&b.0));
    Ok(uv_faces)
}

pub struct FrozenReshapeAsset {
    analysis: Arc<ReshapeAnalysis>,
    scene: Arc<SafeScene>,
    // Arc 供未裁剪共轭场 F_U=R⁻¹∘F_B∘R 复用同一冻结 B 场，不重新分析。
    field: Arc<CanonicalField>,
    key: String,
    // 快照保存输入：参数哈希、逐脸固定集／软约束与重叠分组随资产冻结。
    parameter_hash: String,
    plans: BTreeMap<String, FrozenPlan>,
    groups: Vec<Vec<usize>>,
}
impl FrozenReshapeAsset {
    pub fn scene(&self) -> &Arc<SafeScene> {
        &self.scene
    }
    pub fn field(&self) -> &Arc<CanonicalField> {
        &self.field
    }
    pub fn key(&self) -> &str {
        &self.key
    }
    fn byte_size(&self) -> Option<usize> {
        // 分析由分析缓存计账；这里保守再次计入，pin 的页由独立场页预算计账。
        let plans = self.plans.iter().try_fold(0usize, |sum, (id, plan)| {
            sum.checked_add(id.capacity())?
                .checked_add(plan.fixed.capacity().div_ceil(8))?
                .checked_add(plan.weights.capacity().checked_mul(8)?)?
                .checked_add(96)
        })?;
        let groups = self.groups.iter().try_fold(0usize, |sum, group| {
            sum.checked_add(group.capacity().checked_mul(8)?)?
                .checked_add(96)
        })?;
        self.analysis
            .byte_size()
            .checked_add(self.scene.byte_size())?
            .checked_add(self.field.index_byte_size())?
            .checked_add(self.key.capacity() + 512)?
            .checked_add(self.parameter_hash.capacity())?
            .checked_add(plans)?
            .checked_add(groups)
    }
}

#[derive(Default)]
struct EditorRenderState {
    source: Option<Arc<SourceSnapshot>>,
    lease: Option<Arc<RenderLease>>,
    // 上一成功编辑帧显式 pin，直到切源或新帧提交；导出自己持有独立 Arc。
    asset: Option<Arc<FrozenReshapeAsset>>,
    committed: Option<RenderStamp>,
}

pub struct PortraitRenderState {
    pub session: String,
    assets: TaskCache<FrozenReshapeAsset>,
    pub(crate) masks: NativeMaskCache,
    portrait_bases: Mutex<PortraitBaseCache>,
    next_epoch: AtomicU64,
    editor: Mutex<EditorRenderState>,
    editor_revision: AtomicU64,
    editor_epoch: AtomicU64,
}
impl Default for PortraitRenderState {
    fn default() -> Self {
        Self {
            session: uuid::Uuid::new_v4().to_string(),
            assets: TaskCache::pinned(ASSET_BUDGET),
            masks: NativeMaskCache::default(),
            portrait_bases: Mutex::new(PortraitBaseCache::default()),
            next_epoch: AtomicU64::new(1),
            editor: Mutex::new(EditorRenderState::default()),
            editor_revision: AtomicU64::new(0),
            editor_epoch: AtomicU64::new(0),
        }
    }
}
impl PortraitRenderState {
    pub fn prepare(
        &self,
        app: &tauri::AppHandle,
        source: &SourceSnapshot,
        adjustments: &Value,
        current: &dyn Fn() -> bool,
    ) -> Result<Arc<FrozenReshapeAsset>, String> {
        if render_intent(adjustments)? != RenderIntent::Prepare {
            return Err("旁路请求不应创建冻结资产".into());
        }
        source.verify(current)?;
        let gp_active = crate::portrait_gpupixel::requested(adjustments);
        let body_active = crate::portrait_engine::body_requested(adjustments);
        let parameters = read_parameters(&adjustments["portraitReshape"]);
        let runtime_only = parameters.status == "supported"
            && parameters.value["engineVersion"].as_str()
                == Some(crate::portrait_gpupixel::ENGINE_VERSION);

        // GPUPixel／身体塑形-only：identity 场，不跑人脸分析。
        if (gp_active || body_active) && (runtime_only || parameters.status == "legacy") {
            return self.prepare_gpupixel_identity(source, adjustments, current);
        }

        let state = app.state::<crate::AppState>();
        let base = analysis_base(adjustments)?;
        let analysis: Arc<ReshapeAnalysis> =
            state
                .reshape_analysis
                .prepare_source(app, source, &base, current, &|_| {})?;
        // runtime 引擎：瘦脸／大眼由 bake 完成，不建 UV 场。
        let face_active = !runtime_only
            && parameters.status == "supported"
            && parameters.value["enabled"] == true
            && {
                let reshape_vis = crate::portrait_reshape::reshape_section_visible(adjustments);
                let eyes_vis = crate::portrait_reshape::eyes_section_visible(adjustments);
                if !reshape_vis && !eyes_vis {
                    false
                } else {
                    let uv_faces = collect_uv_face_strengths(
                        &analysis,
                        adjustments,
                        &parameters.value,
                        reshape_vis,
                        eyes_vis,
                    )?;
                    !uv_faces.is_empty()
                }
            };

        if face_active {
            return self.prepare_uv(source, &analysis, adjustments, &parameters.value, current);
        }
        if gp_active || body_active {
            return self.prepare_gpupixel_identity(source, adjustments, current);
        }
        Err("无可执行的面部重塑请求".into())
    }

    /// GPUPixel 整图 bake 用的 identity 冻结场（无 MediaPipe／无 UV）。
    fn prepare_gpupixel_identity(
        &self,
        source: &SourceSnapshot,
        adjustments: &Value,
        current: &dyn Fn() -> bool,
    ) -> Result<Arc<FrozenReshapeAsset>, String> {
        use crate::portrait_analysis::ReshapeAnalysis;
        use crate::portrait_coordinates::CoordinateMap;
        let coordinates = CoordinateMap::new(
            [source.image().width(), source.image().height()],
            adjustments,
        )?;
        let gp_hash = crate::portrait_gpupixel::parameter_hash(adjustments);
        let key = content_key(&json!([
            "frozen-scene-v1",
            crate::portrait_gpupixel::ENGINE_VERSION,
            source.source_key(),
            gp_hash,
            coordinates.base_dimensions,
        ]));
        let asset = self.assets.run_shared(
            &key,
            current,
            |shared_current| {
                let analysis = Arc::new(ReshapeAnalysis {
                    key: format!("gpupixel-runtime:{gp_hash}"),
                    source_base_hash: source.source_key().to_owned(),
                    model_hashes: BTreeMap::new(),
                    coordinates: coordinates.clone(),
                    coarse_dimensions: coordinates.base_dimensions,
                    faces: Vec::new(),
                    protection: None,
                    rejected_faces: 0,
                });
                let scene = Arc::new(SafeScene::from_uv_warps(
                    analysis.key.clone(),
                    coordinates.base_dimensions,
                    Vec::new(),
                    shared_current,
                )?);
                let field = Arc::new(CanonicalField::from_scene(&scene, shared_current)?);
                source.verify(shared_current)?;
                Ok(FrozenReshapeAsset {
                    analysis,
                    scene,
                    key: content_key(&json!([key, field.key()])),
                    field,
                    parameter_hash: gp_hash.clone(),
                    plans: BTreeMap::new(),
                    groups: Vec::new(),
                })
            },
            FrozenReshapeAsset::byte_size,
        )?;
        source.verify(current)?;
        Ok(asset)
    }

    /// GPUPixel UV 旁路：不走 ARAP，直接建稠密逆位移网格场。
    fn prepare_uv(
        &self,
        source: &SourceSnapshot,
        analysis: &Arc<ReshapeAnalysis>,
        adjustments: &Value,
        parameters: &Value,
        current: &dyn Fn() -> bool,
    ) -> Result<Arc<FrozenReshapeAsset>, String> {
        validate_options(parameters)?;
        let reshape_vis = crate::portrait_reshape::reshape_section_visible(adjustments);
        let eyes_vis = crate::portrait_reshape::eyes_section_visible(adjustments);
        let uv_faces =
            collect_uv_face_strengths(analysis, adjustments, parameters, reshape_vis, eyes_vis)?;
        if uv_faces.is_empty() {
            return Err("UV 重塑无非零瘦脸／大眼请求".into());
        }
        self.prepare_uv_faces(source, analysis, uv_faces, current)
    }

    fn prepare_uv_faces(
        &self,
        source: &SourceSnapshot,
        analysis: &Arc<ReshapeAnalysis>,
        uv_faces: Vec<(String, crate::portrait_uv_warp::UvWarpStrengths)>,
        current: &dyn Fn() -> bool,
    ) -> Result<Arc<FrozenReshapeAsset>, String> {
        let parameter_hash = crate::portrait_uv_warp::parameter_hash(&analysis.key, &uv_faces);
        let key = content_key(&json!([
            "frozen-scene-v1",
            "gpupixel-uv-v1",
            analysis.key,
            uv_faces
                .iter()
                .map(|(id, s)| json!([id, s.slim_face, s.big_eye]))
                .collect::<Vec<_>>(),
            crate::face_reshape::LANDMARK_VERSION,
            crate::portrait_uv_warp::ENGINE_VERSION,
            crate::portrait_tiles::FIELD_VERSION
        ]));
        let asset = self.assets.run_shared(
            &key,
            current,
            |shared_current| {
                let (scene, field) =
                    self.solve_uv_scene_field(analysis, &uv_faces, shared_current)?;
                let plans: BTreeMap<String, FrozenPlan> = scene
                    .reports()
                    .zip(scene.meshes())
                    .map(|(r, m)| {
                        let n = m.source_vertices().len();
                        (
                            r.face_id.clone(),
                            FrozenPlan {
                                fixed: vec![false; n],
                                weights: vec![0.0; n],
                            },
                        )
                    })
                    .collect();
                let groups: Vec<Vec<usize>> =
                    (0..scene.reports().count()).map(|i| vec![i]).collect();
                source.verify(shared_current)?;
                Ok(FrozenReshapeAsset {
                    analysis: Arc::clone(analysis),
                    scene,
                    key: content_key(&json!([key, field.key()])),
                    field,
                    parameter_hash: parameter_hash.clone(),
                    plans,
                    groups,
                })
            },
            FrozenReshapeAsset::byte_size,
        )?;
        source.verify(current)?;
        Ok(asset)
    }

    fn solve_uv_scene_field(
        &self,
        analysis: &Arc<ReshapeAnalysis>,
        uv_faces: &[(String, crate::portrait_uv_warp::UvWarpStrengths)],
        current: &dyn Fn() -> bool,
    ) -> Result<(Arc<SafeScene>, Arc<CanonicalField>), String> {
        let mut scale = 1.0_f64;
        let mut last_err = String::new();
        for attempt in 0..8 {
            current()
                .then_some(())
                .ok_or_else(|| crate::portrait_analysis::CANCELLED.to_string())?;
            let scaled: Vec<(String, crate::portrait_uv_warp::UvWarpStrengths)> = uv_faces
                .iter()
                .map(|(id, s)| {
                    (
                        id.clone(),
                        crate::portrait_uv_warp::UvWarpStrengths::from_ui(
                            (s.slim_face * scale).clamp(0.0, 100.0),
                            (s.big_eye * scale).clamp(0.0, 100.0),
                        )
                        .unwrap_or(*s),
                    )
                })
                .collect();
            let warps = match crate::portrait_uv_warp::solve_analysis(analysis, &scaled, current) {
                Ok(w) => w,
                Err(e) => {
                    last_err = e;
                    scale *= 0.7;
                    continue;
                }
            };
            let scene = match SafeScene::from_uv_warps(
                analysis.key.clone(),
                analysis.coordinates.base_dimensions,
                warps,
                current,
            ) {
                Ok(s) => s,
                Err(e) => {
                    last_err = e;
                    scale *= 0.7;
                    continue;
                }
            };
            match CanonicalField::from_scene(&scene, current) {
                Ok(field) => {
                    let field = Arc::new(field);
                    if attempt > 0 {
                        log::warn!("UV 重塑强度回退至 {:.0}% 后通过场校验", scale * 100.0);
                    }
                    log_reshape_diagnostics(&scene, field.is_identity());
                    return Ok((Arc::new(scene), field));
                }
                Err(e) if crate::portrait_tiles::lower_strength_retryable(&e) => {
                    last_err = e;
                    scale *= 0.7;
                }
                Err(e) => return Err(e),
            }
        }
        Err(format!("UV 重塑场校验失败：{last_err}"))
    }

    /// UV-only 管线不再写入网格快照。
    pub(crate) fn snapshot_for_save(&self, _path: &str, _adjustments: &Value) -> Option<Value> {
        None
    }
}

/// CPU 原生照片蒙版逐个生成；缓存持锁切片，不把整幅蒙版 pin 到所有 tile。
/// 128 MiB 约束保存的完整结果，生成器自己的临时滤波图另计 CPU 工作内存。
pub(crate) struct NativeMaskCache {
    inner: Mutex<NativeMaskCacheInner>,
    budget: usize,
}
#[derive(Default)]
struct NativeMaskCacheInner {
    entries: VecDeque<(String, image::GrayImage)>,
    bytes: usize,
}
impl Default for NativeMaskCache {
    fn default() -> Self {
        Self {
            inner: Mutex::new(NativeMaskCacheInner::default()),
            budget: 128 * 1024 * 1024,
        }
    }
}
impl NativeMaskCache {
    // 08 蒙版切片入口：键、定义、维度、窗口、变形图、输出缓冲与取消谓词各自独立；
    // 聚合成结构体会扩散到全部调用方与测试，保持边界签名。
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn append_slice(
        &self,
        base_key: &str,
        definition: &crate::mask_generation::MaskDefinition,
        dims: [u32; 2],
        window: crate::portrait_tiles::PixelRect,
        warped: Option<&DynamicImage>,
        output: &mut Vec<u8>,
        current: &dyn Fn() -> bool,
    ) -> Result<(), String> {
        self.append_resampled_slice(
            base_key, definition, dims, dims, window, warped, output, current,
        )
    }

    /// Alpha 在原基底生成后降采样；不改变笔刷、颜色范围或亮度范围的取样语义。
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn append_resampled_slice(
        &self,
        base_key: &str,
        definition: &crate::mask_generation::MaskDefinition,
        source_dims: [u32; 2],
        dims: [u32; 2],
        window: crate::portrait_tiles::PixelRect,
        warped: Option<&DynamicImage>,
        output: &mut Vec<u8>,
        current: &dyn Fn() -> bool,
    ) -> Result<(), String> {
        check_current(current)?;
        if !crate::portrait_tiles::PixelRect::full(dims)?.contains(window) {
            return Err("照片蒙版窗口缺少真实扩展域".into());
        }
        let length = (window.width as usize)
            .checked_mul(window.height as usize)
            .ok_or("蒙版切片溢出")?;
        output
            .try_reserve_exact(length)
            .map_err(|_| "蒙版切片内存不足")?;
        if !definition.visible || definition.sub_masks.is_empty() {
            output.resize(
                output.len().checked_add(length).ok_or("蒙版切片长度溢出")?,
                0,
            );
            return Ok(());
        }
        let mut value = serde_json::to_value(definition).map_err(|e| e.to_string())?;
        value["adjustments"] = Value::Null;
        let key = content_key(&json!([
            "native-photo-mask-v2",
            base_key,
            source_dims,
            dims,
            value
        ]));
        let mut cache = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        check_current(current)?;
        let entry = if let Some(i) = cache.entries.iter().position(|v| v.0 == key) {
            cache.entries.remove(i).ok_or("蒙版缓存索引错误")?
        } else {
            let bytes = (dims[0] as usize)
                .checked_mul(dims[1] as usize)
                .ok_or("蒙版尺寸溢出")?;
            if bytes > self.budget {
                return Err("完整照片蒙版超过 CPU 缓存预算".into());
            }
            while cache.bytes > self.budget - bytes {
                let (_, old) = cache.entries.pop_front().ok_or("蒙版预算计账错误")?;
                cache.bytes -= old.as_raw().len();
            }
            let source_bytes = (source_dims[0] as usize)
                .checked_mul(source_dims[1] as usize)
                .ok_or("原始蒙版尺寸溢出")?;
            if source_bytes > self.budget {
                return Err("完整照片蒙版超过 CPU 缓存预算".into());
            }
            let image = crate::mask_generation::generate_mask_bitmap_checked(
                definition,
                source_dims[0],
                source_dims[1],
                warped,
                current,
            )?
            .ok_or("可见非空照片蒙版没有生成结果")?;
            check_current(current)?;
            let image = if source_dims == dims {
                image
            } else {
                image::imageops::resize(
                    &image,
                    dims[0],
                    dims[1],
                    image::imageops::FilterType::Triangle,
                )
            };
            check_current(current)?;
            cache.bytes += image.as_raw().len();
            (key, image)
        };
        // 先恢复 LRU，再允许取消；避免取消切片造成 bytes 与 entries 不一致。
        cache.entries.push_back(entry);
        let image = &cache.entries.back().ok_or("蒙版缓存为空")?.1;
        for y in window.y as u32..window.bottom() as u32 {
            check_current(current)?;
            let offset = y as usize * dims[0] as usize + window.x as usize;
            output.extend_from_slice(&image.as_raw()[offset..offset + window.width as usize]);
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub enum RenderQuality {
    Interactive,
    HighQuality,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub enum RenderPurpose {
    Preview,
    Uncropped,
    Thumbnail,
    Export,
    Estimate,
    Mask,
}

/// 在输出排队前持有严格源和冻结场；字段不开放给 IPC 或其他模块改写。
pub(crate) struct FrozenPhotoInput {
    source: Arc<SourceSnapshot>,
    adjustments: Value,
    asset: Arc<FrozenReshapeAsset>,
    lut: Option<crate::lut_processing::Lut>,
    lut_key: String,
}
impl FrozenPhotoInput {
    pub(crate) fn source(&self) -> &Arc<SourceSnapshot> {
        &self.source
    }

    pub(crate) fn lut(&self) -> Option<crate::lut_processing::Lut> {
        self.lut.clone()
    }
}

fn freeze_lut(
    adjustments: &Value,
    enabled: bool,
    current: &dyn Fn() -> bool,
) -> Result<(Option<crate::lut_processing::Lut>, String), String> {
    if !enabled {
        return Ok((None, "disabled".into()));
    }
    let path = adjustments["lutPath"].as_str().ok_or("活动 LUT 缺少路径")?;
    crate::lut_processing::validate_lut_path(path).map_err(|e| e.to_string())?;
    let (_, bytes, key) = read_snapshot_bytes(Path::new(path), current)?;
    let lut = crate::lut_processing::parse_lut_bytes(path, &bytes).map_err(|e| e.to_string())?;
    Ok((Some(lut), key))
}

pub struct PhotoRenderRequest {
    pub source: Arc<SourceSnapshot>,
    pub adjustments: Value,
    pub window: OutputWindow,
    pub quality: RenderQuality,
    pub purpose: RenderPurpose,
    pub request_id: u64,
    pub sixteen_bit: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct RenderStamp {
    pub request_id: u64,
    pub source_revision: u64,
    pub epoch: u64,
    pub composite_key: String,
    pub reshape_key: String,
    pub window_key: String,
}

/// 所有主预览使用同一封包；旁路只保留旧像素语义，不保留旧裸字节协议。
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PreviewMetadata {
    pub path: String,
    pub stamp: RenderStamp,
    pub mode: PreviewMode,
    pub dimensions: [u32; 2],
    pub roi: crate::portrait_tiles::PixelRect,
    /// 非零请求被安全回退限幅或压到恒等时供面板提示；旁路／零强度为 None。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reshape_attenuation: Option<ReshapeAttenuation>,
    /// 身体塑形（瘦手臂／腰／腿）引擎旁路；与人脸重塑限幅隔离。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub body_attenuation: Option<ReshapeAttenuation>,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub enum PreviewMode {
    Native,
    Jpeg,
}

impl PreviewMetadata {
    pub fn encode(&self, body: &[u8]) -> Result<Vec<u8>, String> {
        let full = crate::portrait_tiles::PixelRect::full(self.dimensions)?;
        if !full.contains(self.roi)
            || self.path.is_empty()
            || (self.mode == PreviewMode::Native && (!body.is_empty() || self.roi != full))
            || (self.mode == PreviewMode::Jpeg && body.is_empty())
        {
            return Err("预览封包模式、窗口或图像体非法".into());
        }
        // JavaScript 以安全整数核对身份，不允许 u64 静默丢失低位。
        if [
            self.stamp.request_id,
            self.stamp.source_revision,
            self.stamp.epoch,
        ]
        .iter()
        .any(|v| *v == 0 || *v > 9_007_199_254_740_991)
        {
            return Err("预览身份超出安全整数范围".into());
        }
        let digest = |key: &str| {
            key.len() == 64
                && key
                    .bytes()
                    .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
        };
        if !digest(&self.stamp.composite_key)
            || !digest(&self.stamp.window_key)
            || !(self.stamp.reshape_key == "bypass" || digest(&self.stamp.reshape_key))
        {
            return Err("预览资产摘要非法".into());
        }
        let metadata = serde_json::to_vec(self).map_err(|e| e.to_string())?;
        if metadata.len() > 64 * 1024 {
            return Err("预览元数据超限".into());
        }
        let length = 8usize
            .checked_add(metadata.len())
            .and_then(|n| n.checked_add(body.len()))
            .ok_or("预览封包长度溢出")?;
        let mut bytes = Vec::new();
        bytes
            .try_reserve_exact(length)
            .map_err(|_| "预览封包内存不足")?;
        bytes.extend_from_slice(b"RRV2");
        bytes.extend_from_slice(&(metadata.len() as u32).to_le_bytes());
        bytes.extend_from_slice(&metadata);
        bytes.extend_from_slice(body);
        Ok(bytes)
    }
}

/// 预览完整输出先确定尺寸，ROI 只在该输出坐标中取整，不改变规范基底。
pub(crate) fn preview_window(
    dimensions: [u32; 2],
    crop: &Value,
    target: u32,
    roi: Option<(f32, f32, f32, f32)>,
) -> Result<OutputWindow, String> {
    let crop_rect = crate::portrait_tiles::effective_crop(dimensions, crop)?;
    if target == 0 || target > crate::portrait_tiles::MAX_SIDE {
        return Err("预览目标尺寸非法".into());
    }
    let scale = (target as f64 / crop_rect.width.max(crop_rect.height) as f64).min(1.0);
    let output = [
        (crop_rect.width as f64 * scale).round().max(1.0) as u32,
        (crop_rect.height as f64 * scale).round().max(1.0) as u32,
    ];
    let roi = normalized_preview_roi(output, roi)?;
    OutputWindow::new(dimensions, crop, output, roi)
}

pub(crate) fn normalized_preview_roi(
    dimensions: [u32; 2],
    roi: Option<(f32, f32, f32, f32)>,
) -> Result<Option<crate::portrait_tiles::PixelRect>, String> {
    use crate::portrait_tiles::PixelRect;
    let full = PixelRect::full(dimensions)?;
    let Some((x, y, w, h)) = roi else {
        return Ok(None);
    };
    if [x, y, w, h]
        .iter()
        .any(|v| !v.is_finite() || !(0.0..=1.0).contains(v))
        || w <= 0.0
        || h <= 0.0
        || x + w > 1.000001
        || y + h > 1.000001
    {
        return Err("预览 ROI 非法".into());
    }
    let rect = PixelRect::new(
        (x * dimensions[0] as f32).round() as i32,
        (y * dimensions[1] as f32).round() as i32,
        (w * dimensions[0] as f32).round() as u32,
        (h * dimensions[1] as f32).round() as u32,
    )?;
    rect.intersect(full)
        .map(Some)
        .ok_or_else(|| "预览 ROI 为空".into())
}

/// 独立消费者的取消与最终提交共用短临界区；不取消同键公共分析或其他导出。
pub struct RenderLease {
    epoch: u64,
    request_id: u64,
    source_revision: u64,
    source_key: String,
    cancelled: AtomicBool,
    committed: Mutex<Option<RenderStamp>>,
}
impl RenderLease {
    pub fn is_current(&self) -> bool {
        !self.cancelled.load(Ordering::SeqCst)
    }
    pub fn cancel(&self) {
        let _commit = self.committed.lock().unwrap_or_else(|e| e.into_inner());
        self.cancelled.store(true, Ordering::SeqCst);
    }
    fn matches(&self, source: &SourceSnapshot, request_id: u64) -> bool {
        self.is_current()
            && self.source_key == source.source_key()
            && self.source_revision == source.revision()
            && self.request_id == request_id
    }
    pub fn has_committed(&self, stamp: &RenderStamp) -> bool {
        let committed = self.committed.lock().unwrap_or_else(|e| e.into_inner());
        self.is_current() && committed.as_ref() == Some(stamp)
    }
    pub fn commit<T>(
        &self,
        source: &SourceSnapshot,
        stamp: &RenderStamp,
        current: &dyn Fn() -> bool,
        publish: impl FnOnce() -> Result<T, String>,
    ) -> Result<T, String> {
        let valid = || self.matches(source, stamp.request_id) && current();
        source.verify(&valid)?;
        self.commit_verified(source, stamp, current, publish)
    }

    /// 仅在本模块已核验完整源内容后调用；不得暴露为跳过源守卫的公共入口。
    fn commit_verified<T>(
        &self,
        source: &SourceSnapshot,
        stamp: &RenderStamp,
        current: &dyn Fn() -> bool,
        publish: impl FnOnce() -> Result<T, String>,
    ) -> Result<T, String> {
        let valid = || self.matches(source, stamp.request_id) && current();
        let mut committed = self.committed.lock().unwrap_or_else(|e| e.into_inner());
        if !valid() || stamp.epoch != self.epoch || stamp.source_revision != self.source_revision {
            return Err("渲染消费者已取消或源修订过期".into());
        }
        let result = publish()?;
        *committed = Some(stamp.clone());
        Ok(result)
    }
}

/// image=None 仅表示合法旁路，由入口继续原有像素路径；失败始终是 Err。
pub struct PhotoRenderResult {
    pub image: Option<DynamicImage>,
    pub stamp: RenderStamp,
    pub source: Arc<SourceSnapshot>,
    pub asset: Option<Arc<FrozenReshapeAsset>>,
    pub lease: Arc<RenderLease>,
    pub reshape_attenuation: Option<ReshapeAttenuation>,
    pub body_attenuation: Option<ReshapeAttenuation>,
}

impl PhotoRenderResult {
    /// 桌面单文件事务：完整编码和元数据字节先写入同目录临时文件，最终发布不覆盖。
    /// 多个蒙版附属文件分别提交，不承诺整组原子性；已提交文件不会因稍后取消删除。
    #[cfg(not(target_os = "android"))]
    pub fn publish_encoded(
        &self,
        target: &Path,
        bytes: &[u8],
        auto_number: bool,
        current: &dyn Fn() -> bool,
    ) -> Result<PathBuf, String> {
        self.publish_encoded_guarded(target, bytes, auto_number, current, &Mutex::new(()), |_| {
            Ok(())
        })
    }

    /// 外部任务的取消与最终重命名共用 gate；耗时写入、元数据与源核验不持有此锁。
    #[cfg(not(target_os = "android"))]
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn publish_encoded_guarded<T>(
        &self,
        target: &Path,
        bytes: &[u8],
        auto_number: bool,
        current: &dyn Fn() -> bool,
        gate: &Mutex<T>,
        prepare_file: impl FnOnce(&File) -> Result<(), String>,
    ) -> Result<PathBuf, String> {
        let valid = || self.lease.is_current() && current();
        check_current(&valid)?;
        self.source.reject_source_target(target)?;
        if bytes.is_empty() {
            return Err("禁止发布空编码文件".into());
        }
        let filename = target.file_name().ok_or("导出目标缺少文件名")?;
        let parent = target
            .parent()
            .filter(|p| !p.as_os_str().is_empty())
            .unwrap_or_else(|| Path::new("."));
        let parent = parent.canonicalize().map_err(|e| e.to_string())?;
        let target = parent.join(filename);
        self.source.reject_source_target(&target)?;
        let mut temporary = tempfile::Builder::new()
            .prefix(".rr-reshape-")
            .tempfile_in(&parent)
            .map_err(|e| format!("无法建立导出暂存文件：{e}"))?;
        for chunk in bytes.chunks(64 * 1024) {
            check_current(&valid)?;
            temporary
                .write_all(chunk)
                .map_err(|e| format!("导出暂存写入失败：{e}"))?;
        }
        temporary.flush().map_err(|e| e.to_string())?;
        prepare_file(temporary.as_file())?;
        temporary.as_file().sync_all().map_err(|e| e.to_string())?;
        check_current(&valid)?;
        let stem = target.file_stem().ok_or("导出目标缺少文件名")?;
        let extension = target.extension();
        let mut staged = Some(temporary);
        for suffix in 0..=10000 {
            let candidate = if suffix == 0 {
                target.clone()
            } else {
                let mut name = stem.to_os_string();
                name.push(format!(" ({suffix})"));
                if let Some(extension) = extension {
                    name.push(".");
                    name.push(extension);
                }
                parent.join(name)
            };
            // 每次碰撞重试前完整核验源；持取消锁的区间仅包含目标身份检查和文件发布。
            let result = self.lease.commit(&self.source, &self.stamp, current, || {
                let _publication = gate.lock().map_err(|_| "导出发布锁异常")?;
                check_current(&valid)?;
                self.source.reject_source_target(&candidate)?;
                let temporary = staged.take().ok_or("导出暂存状态错误")?;
                match temporary.persist_noclobber(&candidate) {
                    Ok(_) => Ok(Some(candidate.clone())),
                    Err(error)
                        if auto_number
                            && error.error.kind() == std::io::ErrorKind::AlreadyExists =>
                    {
                        staged = Some(error.file);
                        Err("export-target-collision".to_owned())
                    }
                    Err(error) => Err(format!("导出发布失败（不覆盖已有文件）：{}", error.error)),
                }
            });
            match result {
                Ok(Some(path)) => return Ok(path),
                Err(error) if error == "export-target-collision" => continue,
                Err(error) => return Err(error),
                Ok(None) => return Err("导出发布状态错误".into()),
            }
        }
        Err("导出目标序号已耗尽".into())
    }
}

impl PortraitRenderState {
    /// 调用方在 preview_epoch 锁内接受请求；锁顺序固定为 editor → lease → 显示发布。
    /// 原生 tile 的 current 只读取原子代次，不反向获取旧 preview_epoch／GPU 锁。
    pub fn advance_editor(&self, revision: u64, epoch: u64) -> Result<(), String> {
        let mut editor = self.editor.lock().unwrap_or_else(|e| e.into_inner());
        let previous_revision = self.editor_revision.load(Ordering::SeqCst);
        let previous_epoch = self.editor_epoch.load(Ordering::SeqCst);
        if revision < previous_revision || epoch <= previous_epoch {
            return Err("编辑器源修订或请求代次不能倒退".into());
        }
        if let Some(lease) = editor.lease.take() {
            lease.cancel();
        }
        editor.committed = None;
        if revision != previous_revision {
            editor.source = None;
            editor.asset = None;
        }
        self.editor_revision.store(revision, Ordering::SeqCst);
        self.editor_epoch.store(epoch, Ordering::SeqCst);
        if revision != previous_revision {
            // portrait_bases 的键包含源内容身份，可安全跨图片保留；预算 LRU 负责淘汰。
            crate::gpu_processing::clear_native_preview_resources();
        }
        Ok(())
    }

    pub fn editor_is_current(&self, revision: u64, epoch: u64) -> bool {
        self.editor_revision.load(Ordering::SeqCst) == revision
            && self.editor_epoch.load(Ordering::SeqCst) == epoch
    }

    /// 复用只限严格完整解码的同修订快照；旧快速 RAW／内嵌预览缓存不参与。
    pub fn editor_source(
        &self,
        path: &str,
        settings: &crate::AppSettings,
        revision: u64,
        current: &dyn Fn() -> bool,
    ) -> Result<Arc<SourceSnapshot>, String> {
        let valid = || current() && self.editor_revision.load(Ordering::SeqCst) == revision;
        check_current(&valid)?;
        let cached = self
            .editor
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .source
            .clone();
        if let Some(source) = cached.filter(|s| {
            s.path() == path
                && s.revision() == revision
                && s.decode_settings == SourceSnapshot::decode_key(settings)
        }) {
            // 改过的同路径文件必须报过期；不能静默重载并重新分析。
            source.verify(&valid)?;
            return Ok(source);
        }
        let source = Arc::new(SourceSnapshot::load(path, settings, revision, &valid)?);
        let mut editor = self.editor.lock().unwrap_or_else(|e| e.into_inner());
        check_current(&valid)?;
        editor.source = Some(Arc::clone(&source));
        Ok(source)
    }

    /// 独立输出可复用已冻结的编辑源，但不继承编辑器切图取消令牌。
    /// 同路径已有快照必须核验；文件改变时报错，不能静默换源重新分析。
    pub(crate) fn output_source(
        &self,
        path: &str,
        settings: &crate::AppSettings,
        current: &dyn Fn() -> bool,
    ) -> Result<Arc<SourceSnapshot>, String> {
        check_current(current)?;
        let cached = self
            .editor
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .source
            .clone();
        if let Some(source) = cached.filter(|source| source.path() == path) {
            // 即使解码设置改变，也不能把已被替换的编辑源悄悄当成新文件导出。
            source.verify(current)?;
            if source.decode_settings == SourceSnapshot::decode_key(settings) {
                return Ok(source);
            }
        }
        let revision = self
            .next_epoch
            .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |n| n.checked_add(1))
            .map_err(|_| "渲染源修订已耗尽")?;
        Ok(Arc::new(SourceSnapshot::load(
            path, settings, revision, current,
        )?))
    }

    pub fn begin_editor_consumer(
        &self,
        source: &SourceSnapshot,
        request_id: u64,
        epoch: u64,
    ) -> Result<Arc<RenderLease>, String> {
        let mut editor = self.editor.lock().unwrap_or_else(|e| e.into_inner());
        if !self.editor_is_current(source.revision(), epoch) {
            return Err("编辑器请求已被替换".into());
        }
        if let Some(previous) = editor.lease.take() {
            previous.cancel();
        }
        let lease = self.begin_consumer(source, request_id)?;
        editor.lease = Some(Arc::clone(&lease));
        Ok(lease)
    }

    /// 完整源 hash 在短临界区之前核验；新请求接受与提交互斥，失败保留上次 pin。
    pub fn publish_editor<T>(
        &self,
        output: &PhotoRenderResult,
        epoch: u64,
        current: &dyn Fn() -> bool,
        publish: impl FnOnce() -> Result<T, String>,
    ) -> Result<T, String> {
        let valid = || {
            output.lease.is_current()
                && self.editor_is_current(output.source.revision(), epoch)
                && current()
        };
        output.source.verify(&valid)?;
        let mut editor = self.editor.lock().unwrap_or_else(|e| e.into_inner());
        if editor
            .lease
            .as_ref()
            .is_none_or(|lease| !Arc::ptr_eq(lease, &output.lease))
        {
            return Err("编辑器消费者已被替换".into());
        }
        let value = output
            .lease
            .commit_verified(&output.source, &output.stamp, &valid, publish)?;
        editor.asset = output.asset.clone();
        editor.committed = Some(output.stamp.clone());
        Ok(value)
    }

    /// 旧像素路径也发布完整版本；不为关闭／零值请求创建严格源或形变资源。
    pub(crate) fn legacy_stamp(
        &self,
        request_id: u64,
        revision: u64,
        path: &str,
        adjustments: &Value,
        window: &Value,
    ) -> Result<RenderStamp, String> {
        let epoch = self
            .next_epoch
            .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |n| n.checked_add(1))
            .map_err(|_| "渲染代次已耗尽")?;
        Ok(RenderStamp {
            request_id,
            source_revision: revision,
            epoch,
            composite_key: content_key(&json!(["legacy-preview-v2", path, revision, adjustments])),
            reshape_key: "bypass".into(),
            window_key: content_key(window),
        })
    }

    pub(crate) fn publish_legacy_preview<T>(
        &self,
        stamp: &RenderStamp,
        epoch: u64,
        publish: impl FnOnce() -> Result<T, String>,
    ) -> Result<T, String> {
        let mut editor = self.editor.lock().unwrap_or_else(|e| e.into_inner());
        if !self.editor_is_current(stamp.source_revision, epoch) {
            return Err("Superseded or worker failed".into());
        }
        let result = publish()?;
        editor.committed = Some(stamp.clone());
        // 无脸 identity 结果仍保留有效分析 pin；纯关闭旁路在下一次切图释放。
        Ok(result)
    }

    pub(crate) fn with_committed_preview<T>(
        &self,
        stamp: &RenderStamp,
        epoch: u64,
        publish: impl FnOnce() -> T,
    ) -> Option<T> {
        let editor = self.editor.lock().unwrap_or_else(|e| e.into_inner());
        (self.editor_is_current(stamp.source_revision, epoch)
            && editor.committed.as_ref() == Some(stamp))
        .then(publish)
    }

    pub fn begin_consumer(
        &self,
        source: &SourceSnapshot,
        request_id: u64,
    ) -> Result<Arc<RenderLease>, String> {
        let epoch = self
            .next_epoch
            .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |n| n.checked_add(1))
            .map_err(|_| "渲染代次已耗尽，请重启会话")?;
        Ok(Arc::new(RenderLease {
            epoch,
            request_id,
            source_revision: source.revision(),
            source_key: source.source_key().to_owned(),
            cancelled: AtomicBool::new(false),
            committed: Mutex::new(None),
        }))
    }

    /// 统一原生照片协调器：严格源→前置基底→原生合成→冻结重塑→尾部→最终缩放／量化。
    /// 只返回完整结果；各入口必须通过 lease.commit 完成缓存、帧或文件发布。
    pub fn render_photo(
        &self,
        app: &tauri::AppHandle,
        request: PhotoRenderRequest,
        lease: Arc<RenderLease>,
        settings: &crate::AppSettings,
        current: &dyn Fn() -> bool,
    ) -> Result<PhotoRenderResult, String> {
        self.render_photo_inner(app, request, lease, settings, None, current)
    }

    pub(crate) fn freeze_output(
        &self,
        app: &tauri::AppHandle,
        path: &str,
        adjustments: &Value,
        settings: &crate::AppSettings,
        current: &dyn Fn() -> bool,
    ) -> Result<FrozenPhotoInput, String> {
        let source = self.output_source(path, settings, current)?;
        self.freeze_output_with_source(app, &source, adjustments, settings, current)
    }

    /// 已持有严格源时直接冻结输出；低清共享入口改写裁剪后复用同一冻结语义。
    pub(crate) fn freeze_output_with_source(
        &self,
        app: &tauri::AppHandle,
        source: &Arc<SourceSnapshot>,
        adjustments: &Value,
        settings: &crate::AppSettings,
        current: &dyn Fn() -> bool,
    ) -> Result<FrozenPhotoInput, String> {
        let params = crate::image_processing::get_all_adjustments_from_json(
            adjustments,
            source.is_raw(),
            crate::white_balance::as_shot_white_balance(source.path()),
            crate::image_processing::resolve_tonemapper_override(settings, source.is_raw()),
        );
        let (lut, lut_key) = freeze_lut(adjustments, params.global.has_lut != 0, current)?;
        let asset = self.prepare(app, source, adjustments, current)?;
        Ok(FrozenPhotoInput {
            source: Arc::clone(source),
            adjustments: adjustments.clone(),
            asset,
            lut,
            lut_key,
        })
    }

    /// 低清照片入口（缩略图／预设／社区／按路径预览）的共享渲染：
    /// 严格源→冻结场→原生合成→最终浮点缩放，不先缩小再分析或求解脸型。
    /// square_target 把中心方形裁剪并入照片最终输出窗口（社区 tile 等）；
    /// 裁剪不参与分析键，同一冻结场跨尺寸／跨入口共享。
    /// Ok(None)：旁路或有效 identity，入口继续旧像素路径。
    /// publish 在 lease 提交临界区内执行：编码／缓存写入前源已完整核验。
    #[allow(clippy::too_many_arguments)]
    pub fn render_reduced_photo<T>(
        &self,
        app: &tauri::AppHandle,
        path: &str,
        adjustments: &Value,
        target: u32,
        quality: RenderQuality,
        square_target: bool,
        settings: &crate::AppSettings,
        current: &dyn Fn() -> bool,
        publish: impl FnOnce(&DynamicImage) -> Result<T, String>,
    ) -> Result<Option<T>, String> {
        if render_intent(adjustments)? == RenderIntent::Bypass {
            return Ok(None);
        }
        let source = self.output_source(path, settings, current)?;
        let coordinates = crate::portrait_coordinates::CoordinateMap::new(
            [source.image().width(), source.image().height()],
            adjustments,
        )?;
        let mut rendered = adjustments.clone();
        if square_target {
            // 中心方形裁剪并入照片最终窗口；分析键不含 crop，冻结场与分析跨尺寸共享。
            let crop = crate::portrait_tiles::effective_crop(
                coordinates.base_dimensions,
                &rendered["crop"],
            )?;
            let size = crop.width.min(crop.height).max(1);
            rendered["crop"] = json!({
                "x": crop.x as i64 + (crop.width - size) as i64 / 2,
                "y": crop.y as i64 + (crop.height - size) as i64 / 2,
                "width": size,
                "height": size,
            });
        }
        let frozen = self.freeze_output_with_source(app, &source, &rendered, settings, current)?;
        let window = if square_target {
            crate::portrait_tiles::OutputWindow::new(
                coordinates.base_dimensions,
                &rendered["crop"],
                [target, target],
                None,
            )?
        } else {
            preview_window(coordinates.base_dimensions, &rendered["crop"], target, None)?
        };
        let request_id = source.revision();
        let lease = self.begin_consumer(&source, request_id)?;
        let result = self.render_frozen_photo(
            app,
            PhotoRenderRequest {
                source: Arc::clone(&source),
                adjustments: rendered,
                window,
                request_id,
                quality,
                purpose: RenderPurpose::Thumbnail,
                sixteen_bit: false,
            },
            lease,
            settings,
            &frozen,
            current,
        )?;
        let Some(image) = result.image.as_ref() else {
            // 有效 identity：无重塑效果；入口继续旧像素路径，不发布共享结果。
            return Ok(None);
        };
        let published = result
            .lease
            .commit(&result.source, &result.stamp, current, || publish(image))?;
        Ok(Some(published))
    }

    /// 未裁剪预览入口（裁剪面板）：U 基底＝定向／镜像后、自由旋转前的真实像素来源；
    /// 输出窗口＝U 全图缩放到 target，轴对齐，复用现有缩放管线。与主预览共享同一
    /// 冻结分析场（分析键含原 rotation），仅像素基底与位移场改为共轭 F_U=R⁻¹∘F_B∘R，
    /// 不清空 rotation 重新分析，也不反旋已截断的 B 图。Ok(None)：旁路或有效 identity，
    /// 入口继续旧像素路径。publish 在 lease 提交临界区内执行：编码前源已完整核验。
    // 08 入口边界：应用句柄、路径、调整、目标边长、设置、取消谓词与发布回调各自独立。
    #[allow(clippy::too_many_arguments)]
    pub fn render_uncropped_photo<T>(
        &self,
        app: &tauri::AppHandle,
        path: &str,
        adjustments: &Value,
        target: u32,
        settings: &crate::AppSettings,
        current: &dyn Fn() -> bool,
        publish: impl FnOnce(&DynamicImage) -> Result<T, String>,
    ) -> Result<Option<T>, String> {
        if render_intent(adjustments)? != RenderIntent::Prepare {
            return Ok(None);
        }
        let source = self.output_source(path, settings, current)?;
        let coordinates = crate::portrait_coordinates::CoordinateMap::new(
            [source.image().width(), source.image().height()],
            adjustments,
        )?;
        // U 与 B 同尺寸画布：自由旋转不改变画布尺寸，裁剪不参与基底尺寸。
        let window = preview_window(coordinates.base_dimensions, &Value::Null, target, None)?;
        let frozen =
            self.freeze_output_with_source(app, &source, adjustments, settings, current)?;
        let request_id = source.revision();
        let lease = self.begin_consumer(&source, request_id)?;
        let result = self.render_frozen_photo(
            app,
            PhotoRenderRequest {
                source: Arc::clone(&source),
                adjustments: adjustments.clone(),
                window,
                request_id,
                quality: RenderQuality::Interactive,
                purpose: RenderPurpose::Uncropped,
                sixteen_bit: false,
            },
            lease,
            settings,
            &frozen,
            current,
        )?;
        let Some(image) = result.image.as_ref() else {
            // 有效 identity：无重塑效果；入口继续旧像素路径，不发布共享结果。
            return Ok(None);
        };
        let published = result
            .lease
            .commit(&result.source, &result.stamp, current, || publish(image))?;
        Ok(Some(published))
    }

    pub(crate) fn render_frozen_photo(
        &self,
        app: &tauri::AppHandle,
        request: PhotoRenderRequest,
        lease: Arc<RenderLease>,
        settings: &crate::AppSettings,
        frozen: &FrozenPhotoInput,
        current: &dyn Fn() -> bool,
    ) -> Result<PhotoRenderResult, String> {
        if !Arc::ptr_eq(&request.source, &frozen.source)
            || request.adjustments != frozen.adjustments
        {
            return Err("输出请求不匹配启动时的冻结源或参数".into());
        }
        self.render_photo_inner(app, request, lease, settings, Some(frozen), current)
    }

    /// 编辑器预览优先代理尺寸 bake；导出仍原尺寸。超大图 full 常超预算无法缓存。
    fn prepare_portrait_base(
        &self,
        app: &tauri::AppHandle,
        request: &PhotoRenderRequest,
        spatial: &Value,
        proxy_limit: Option<u32>,
        settings: &crate::AppSettings,
        current: &dyn Fn() -> bool,
    ) -> Result<PortraitBasePixels, String> {
        let started = std::time::Instant::now();
        let stable_base = analysis_base(spatial)?;
        let stable_proxy_key = content_key(&json!([
            "portrait-gp-source-proxy-v1",
            request.source.source_key(),
            stable_base.clone(),
        ]));
        let key = content_key(&json!([
            portrait_base_key(request.source.source_key(), &request.adjustments)?,
            stable_base,
        ]));
        let cacheable = request.purpose == RenderPurpose::Preview;
        let cached = if cacheable {
            let mut cache = self
                .portrait_bases
                .lock()
                .unwrap_or_else(|e| e.into_inner());
            check_current(current)?;
            let hit = proxy_limit.and_then(|limit| cache.proxy(&key, limit));
            if let Some(hit) = hit {
                cache.hits += 1;
                log::debug!(
                    "[portrait-preview] proxy hit={} bakes={} elapsed={:?}",
                    cache.hits,
                    cache.bakes,
                    started.elapsed()
                );
                return Ok(hit);
            }
            cache.full(&key)
        } else {
            None
        };
        // 参数变化只使动态 GPUPixel 结果失效；几何基底代理按源图/几何复用。
        // 这样连续拖动不再重复整图几何变换与代理缩放。
        if cached.is_none()
            && let Some(limit) = proxy_limit
        {
            let stable_proxy = if cacheable {
                self.portrait_bases
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .proxy(&stable_proxy_key, limit)
            } else {
                None
            };
            if let Some(proxy_input) = stable_proxy {
                check_current(current)?;
                let bake_started = std::time::Instant::now();
                let baked = bake_gpupixel_base(
                    app,
                    request,
                    spatial,
                    proxy_input.image.as_ref(),
                    current,
                )?;
                check_current(current)?;
                let pixels = PortraitBasePixels {
                    original_dimensions: proxy_input.original_dimensions,
                    image: Arc::new(baked),
                };
                if cacheable {
                    let mut cache = self
                        .portrait_bases
                        .lock()
                        .unwrap_or_else(|e| e.into_inner());
                    cache.hits += 1;
                    cache.bakes += 1;
                    cache.insert_proxy(&key, limit, &pixels);
                }
                log::info!(
                    "[portrait-bench] stage=gpupixel-proxy source_cache=hit output={}x{} bake_ms={:.1}",
                    pixels.image.width(),
                    pixels.image.height(),
                    bake_started.elapsed().as_secs_f64() * 1000.0
                );
                return Ok(pixels);
            }
        }
        let full = if let Some(cached) = cached {
            log::debug!(
                "[portrait-preview] full base hit elapsed={:?}",
                started.elapsed()
            );
            cached
        } else {
            let base = analysis_base(spatial)?;
            let warped =
                crate::adjustment_utils::prepare_patched_warped(request.source.image(), &base)?;
            check_current(current)?;
            let (transformed, _) =
                crate::adjustment_utils::apply_spatial_transformations(warped.as_ref(), &base);
            log::debug!(
                "[portrait-preview] spatial preparation={:?}",
                started.elapsed()
            );
            let original_dimensions = [transformed.width(), transformed.height()];

            // 交互帧必须先缩到代理尺寸再进入 GPUPixel。旧路径先对原图 bake、再缩小，
            // 虽然最终缓存名为 proxy，却没有省掉最昂贵的整图人像处理。
            if let Some(limit) = proxy_limit {
                let proxy_started = std::time::Instant::now();
                let proxy_input = portrait_proxy(transformed.as_ref(), limit, current)?;
                let resize_ms = proxy_started.elapsed().as_secs_f64() * 1000.0;
                check_current(current)?;
                let bake_started = std::time::Instant::now();
                let baked = bake_gpupixel_base(app, request, spatial, &proxy_input, current)?;
                check_current(current)?;
                let pixels = PortraitBasePixels {
                    original_dimensions,
                    image: Arc::new(baked),
                };
                if cacheable {
                    let mut cache = self
                        .portrait_bases
                        .lock()
                        .unwrap_or_else(|e| e.into_inner());
                    check_current(current)?;
                    let stable_pixels = PortraitBasePixels {
                        original_dimensions,
                        image: Arc::new(proxy_input),
                    };
                    cache.insert_proxy(&stable_proxy_key, limit, &stable_pixels);
                    cache.bakes += 1;
                    cache.insert_proxy(&key, limit, &pixels);
                }
                log::info!(
                    "[portrait-bench] stage=gpupixel-proxy source_cache=miss input={:?} output={}x{} resize_ms={:.1} bake_ms={:.1}",
                    original_dimensions,
                    pixels.image.width(),
                    pixels.image.height(),
                    resize_ms,
                    bake_started.elapsed().as_secs_f64() * 1000.0
                );
                return Ok(pixels);
            }

            let bake_started = std::time::Instant::now();
            let baked =
                bake_gpupixel_base(app, request, spatial, transformed.as_ref(), current)?;
            check_current(current)?;
            let pixels = PortraitBasePixels {
                original_dimensions,
                image: Arc::new(baked),
            };
            log::info!(
                "[portrait-bench] stage=gpupixel-full output={:?} bake_ms={:.1}",
                pixels.original_dimensions,
                bake_started.elapsed().as_secs_f64() * 1000.0
            );
            if cacheable {
                let mut cache = self
                    .portrait_bases
                    .lock()
                    .unwrap_or_else(|e| e.into_inner());
                check_current(current)?;
                cache.bakes += 1;
                cache.insert_full(&key, &pixels, PORTRAIT_BASE_BUDGET);
            }
            pixels
        };
        // 无强制代理时仍预热一份，超大原图不入 full 缓存也能在后续调色命中。
        let warm_limit = cacheable
            .then(|| {
                portrait_preview_limit(settings.live_preview_quality.as_deref().unwrap_or("high"))
            })
            .flatten();
        if let Some(limit) = warm_limit {
            let _ = self.ensure_portrait_proxy(&key, limit, &full, current, started);
        }
        Ok(full)
    }

    fn ensure_portrait_proxy(
        &self,
        key: &str,
        limit: u32,
        full: &PortraitBasePixels,
        current: &dyn Fn() -> bool,
        started: std::time::Instant,
    ) -> Result<PortraitBasePixels, String> {
        let cached = self
            .portrait_bases
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .proxy(key, limit);
        if let Some(proxy) = cached {
            return Ok(proxy);
        }
        match portrait_proxy(&full.image, limit, current) {
            Ok(image) => {
                let pixels = PortraitBasePixels {
                    image: Arc::new(image),
                    original_dimensions: full.original_dimensions,
                };
                let mut cache = self
                    .portrait_bases
                    .lock()
                    .unwrap_or_else(|e| e.into_inner());
                check_current(current)?;
                cache.insert_proxy(key, limit, &pixels);
                log::debug!(
                    "[portrait-preview] proxy ready {}x{} total={:?}",
                    pixels.image.width(),
                    pixels.image.height(),
                    started.elapsed()
                );
                Ok(pixels)
            }
            Err(error) => {
                check_current(current)?;
                log::debug!("人像代理准备失败，保留原尺寸处理：{error}");
                Ok(full.clone())
            }
        }
    }

    fn render_photo_inner(
        &self,
        app: &tauri::AppHandle,
        request: PhotoRenderRequest,
        lease: Arc<RenderLease>,
        settings: &crate::AppSettings,
        frozen_input: Option<&FrozenPhotoInput>,
        current: &dyn Fn() -> bool,
    ) -> Result<PhotoRenderResult, String> {
        let valid = || lease.matches(&request.source, request.request_id) && current();
        check_current(&valid)?;
        if request.source.decode_settings != SourceSnapshot::decode_key(settings) {
            return Err("渲染解码设置与冻结源快照不一致".into());
        }
        request.source.verify(&valid)?;
        let intent = render_intent(&request.adjustments)?;
        let gp_params = crate::portrait_gpupixel::params_from_adjustments(&request.adjustments);
        let gp_active = gp_params.is_active();
        let body_params =
            crate::portrait_engine::body_params_from_adjustments(&request.adjustments);
        let body_active = body_params.is_active();
        let asset = if intent == RenderIntent::Prepare {
            Some(match frozen_input {
                Some(input) => Arc::clone(&input.asset),
                None => self.prepare(app, &request.source, &request.adjustments, &valid)?,
            })
        } else {
            None
        };
        let reshape_key = asset
            .as_ref()
            .map_or_else(|| "bypass".to_owned(), |a| a.key().to_owned());
        // identity 场在仅 GPUPixel／身体塑形时仍走原生管线，避免回落 legacy。
        let field_identity = asset.as_ref().is_none_or(|a| a.field().is_identity());
        let bypass = field_identity && !gp_active && !body_active;
        let mut stamp = RenderStamp {
            request_id: request.request_id,
            source_revision: request.source.revision(),
            epoch: lease.epoch,
            composite_key: String::new(),
            reshape_key,
            window_key: content_key(&json!([
                RENDER_VERSION,
                request.window,
                request.quality,
                request.purpose,
                request.sixteen_bit,
                request.adjustments["showClipping"],
                if gp_active {
                    crate::portrait_gpupixel::parameter_hash(&request.adjustments)
                } else {
                    String::new()
                },
            ])),
        };
        let result_image = if bypass {
            stamp.composite_key = composite_key(
                &request.source,
                &request.adjustments,
                "legacy-bypass",
                "legacy",
            );
            None
        } else {
            if request.purpose == RenderPurpose::Mask {
                return Err("蒙版 Alpha 同场重塑尚未接通，禁止输出颜色与 Alpha 错位的图层".into());
            }
            #[cfg(target_os = "android")]
            if matches!(request.purpose, RenderPurpose::Export | RenderPurpose::Mask) {
                return Err("Android 活动重塑图库事务尚未支持".into());
            }
            let base_parameters = analysis_base(&request.adjustments)?;
            let uncropped = request.purpose == RenderPurpose::Uncropped;
            // U 基底：保留定向／镜像，不应用自由旋转与裁剪；与旧未裁剪路径 crop=Null+rotation=0
            // 语义一致。分析键仍用原 rotation 的 base_parameters，与主预览共享同一冻结场。
            let spatial_parameters = if uncropped {
                let mut view = base_parameters.clone();
                view["crop"] = Value::Null;
                view["rotation"] = serde_json::json!(0);
                view
            } else {
                base_parameters.clone()
            };
            let definitions: Vec<crate::mask_generation::MaskDefinition> =
                match request.adjustments.get("masks") {
                    None | Some(Value::Null) => Vec::new(),
                    Some(value) => serde_json::from_value(value.clone())
                        .map_err(|e| format!("照片蒙版参数非法：{e}"))?,
                };
            let output_long_edge = request.window.output_dimensions[0]
                .max(request.window.output_dimensions[1]);
            let proxy_limit = interactive_portrait_proxy_limit(
                request.purpose,
                request.quality,
                field_identity,
                settings.live_preview_quality.as_deref().unwrap_or("high"),
                output_long_edge,
            );
            let gp_base = if gp_active {
                Some(self.prepare_portrait_base(
                    app,
                    &request,
                    &spatial_parameters,
                    proxy_limit,
                    settings,
                    &valid,
                )?)
            } else {
                None
            };
            let mask_base_key = content_key(&json!([
                "portrait-mask-base-v1",
                request.source.source_key(),
                spatial_parameters,
            ]));
            // 无活动蒙版时直接复用 bake；有蒙版时保留生成器需要的原始颜色基底。
            let needs_mask_source = definitions
                .iter()
                .any(|mask| mask.visible && !mask.sub_masks.is_empty());
            let warped = if !gp_active || needs_mask_source {
                Some(crate::adjustment_utils::prepare_patched_warped(
                    request.source.image(),
                    &base_parameters,
                )?)
            } else {
                None
            };
            check_current(&valid)?;
            let transformed = if gp_active {
                None
            } else {
                Some(
                    crate::adjustment_utils::apply_spatial_transformations(
                        warped.as_ref().ok_or("缺少原生空间基底")?.as_ref(),
                        &spatial_parameters,
                    )
                    .0,
                )
            };
            let dims = if let Some(base) = &gp_base {
                base.original_dimensions
            } else {
                let image = transformed.as_ref().ok_or("缺少变换基底")?;
                [image.width(), image.height()]
            };
            check_dimensions(dims)?;
            let expected = OutputWindow::new(
                dims,
                if uncropped {
                    &Value::Null
                } else {
                    &request.adjustments["crop"]
                },
                request.window.output_dimensions,
                Some(request.window.roi),
            )?;
            if expected != request.window {
                return Err("输出窗口不匹配当前规范基底及整数裁剪".into());
            }
            let state = app.state::<crate::AppState>();
            let mut params = crate::image_processing::get_all_adjustments_from_json(
                &request.adjustments,
                request.source.is_raw(),
                crate::white_balance::as_shot_white_balance(request.source.path()),
                crate::image_processing::resolve_tonemapper_override(
                    settings,
                    request.source.is_raw(),
                ),
            );
            if !matches!(
                request.purpose,
                RenderPurpose::Preview | RenderPurpose::Uncropped
            ) {
                params.global.show_clipping = 0;
            }
            let (lut, lut_key) = match frozen_input {
                Some(input) => (input.lut(), input.lut_key.clone()),
                None => freeze_lut(&request.adjustments, params.global.has_lut != 0, &valid)?,
            };
            // 代理已经是浮点图，借用 Arc 内存；高清只复用原尺寸 bake，不复用低清像素。
            let color_base = if let Some(base) = &gp_base {
                base.image.as_ref()
            } else {
                transformed.as_ref().ok_or("缺少调色基底")?.as_ref()
            };
            // GPUPixel 激活时美肤分析必须基于 bake 后像素（瘦脸／大眼已移动五官），
            // 且分析尺寸须与调色基底一致（可能为代理）。美肤层以差值叠加，键只含几何参数，
            // 拖动口红／腮红／磨皮时复用分析，不重跑模型。
            let analysis_key_base = if gp_active {
                json!([
                    "gpupixel-baked-v1",
                    spatial_parameters,
                    gp_params.thin_face.to_bits(),
                    gp_params.big_eye.to_bits(),
                    [color_base.width(), color_base.height()],
                ])
            } else {
                // U 视图美肤检测键用无旋转基底，与 B 主预览的严格美肤缓存隔离。
                spatial_parameters.clone()
            };
            let maps = crate::portrait_processing::prepare_native_render(
                app,
                &state,
                request.source.source_key(),
                &analysis_key_base,
                color_base,
                request.source.is_raw(),
                &request.adjustments,
                &valid,
            )?;
            // 无 maps（未请求原生美肤）时清零人像 GPU 强度，否则调色会报「旧美肤分析尚未就绪」。
            suppress_native_portrait_when_gpupixel(
                crate::portrait_gpupixel::face_requested(&request.adjustments) && maps.is_none(),
                &mut params,
            );
            let base_key = composite_key(
                &request.source,
                &request.adjustments,
                &lut_key,
                &params.global.tonemapper_mode.to_string(),
            );
            stamp.composite_key = content_key(&json!([
                base_key,
                maps.as_ref().map(|m| &m.analysis.summary.key),
                if gp_active {
                    crate::portrait_gpupixel::parameter_hash(&request.adjustments)
                } else {
                    String::new()
                },
                if body_active {
                    crate::portrait_engine::body_parameter_hash(&request.adjustments)
                } else {
                    String::new()
                },
            ]));
            check_current(&valid)?;
            let native = if gp_base.is_some()
                && let Some(rgba) = color_base.as_rgba32f()
            {
                std::borrow::Cow::Borrowed(rgba)
            } else {
                std::borrow::Cow::Owned(native_rgba(color_base, &valid)?)
            };
            let working_dims = [native.width(), native.height()];
            let mut body_attenuation = None;
            let body_field = if body_active {
                let (field, identity) = crate::portrait_engine::field_rgba(
                    &color_base.to_rgba8(),
                    body_params,
                    &valid,
                )?;
                if identity {
                    body_attenuation = Some(ReshapeAttenuation::Identity);
                }
                field.map(Arc::new)
            } else {
                None
            };
            let proxy_field = if working_dims != dims {
                if !field_identity {
                    return Err("非 identity 场不能使用人像代理".into());
                }
                // 代理尺寸下身体场已按 color_base 求解；此处仅补 identity 人脸场。
                let scene = SafeScene::from_uv_warps(
                    format!("portrait-proxy:{mask_base_key}"),
                    working_dims,
                    Vec::new(),
                    &valid,
                )?;
                Some(CanonicalField::from_scene(&scene, &valid)?)
            } else {
                None
            };
            let output_window = proxy_output_window(&request.window, dims, working_dims);
            let input_key = content_key(&json!([
                "portrait-native-input-v1",
                mask_base_key,
                working_dims,
                crate::portrait_gpupixel::parameter_hash(&request.adjustments),
                if body_active {
                    crate::portrait_engine::body_parameter_hash(&request.adjustments)
                } else {
                    String::new()
                },
            ]));
            stamp.window_key = content_key(&json!([stamp.window_key, working_dims]));
            let profile = if request.quality == RenderQuality::Interactive {
                crate::portrait_gpu::Profile::Preview
            } else {
                crate::portrait_gpu::Profile::HighQuality
            };
            let active = asset.as_ref().ok_or("活动重塑缺少冻结资产")?;
            // U 视图使用同一冻结 B 场的共轭 F_U=R⁻¹∘F_B∘R；B 视图直接用冻结场。
            // 共轭场与 B 场同尺寸画布（apply_rotation 同尺寸），蒙版／调色缓存键天然一致。
            let conjugated = if uncropped {
                let rotation = request.adjustments["rotation"].as_f64().unwrap_or(0.0);
                Some(CanonicalField::conjugate_uncropped(
                    active.field(),
                    rotation,
                    &valid,
                )?)
            } else {
                None
            };
            let reshape_field = match (proxy_field, conjugated) {
                (Some(proxy), _) => Arc::new(proxy),
                (_, Some(conjugated)) => Arc::new(conjugated),
                _ => Arc::clone(active.field()),
            };
            // GPUPixel 先完成整图 bake；人脸重塑场之后叠加身体逆位移，预览与导出共用组合场。
            let field = match body_field {
                Some(body) if reshape_field.is_identity() => body,
                Some(body) => Arc::new(CanonicalField::compose_inverse_chain(
                    &body,
                    &reshape_field,
                    &valid,
                )?),
                None => reshape_field,
            };
            let color_started = std::time::Instant::now();
            let rendered = crate::gpu_processing::render_native_photo_shared(
                app,
                &state,
                crate::gpu_processing::NativePhotoInput {
                    base: &native,
                    base_key: &input_key,
                    field: field.as_ref(),
                    adjustments: params,
                    masks: &definitions,
                    mask_cache: &self.masks,
                    mask_source: Some((dims, &mask_base_key)),
                    cache_preview_resources: gp_active && proxy_limit.is_some(),
                    warped: warped.as_ref().map(|image| image.as_ref()),
                    lut: lut.as_ref(),
                    portrait: maps.as_ref(),
                    other_resident_bytes: 0,
                },
                profile,
                &valid,
            )?;
            log::debug!(
                "[portrait-preview] color/readback {:?} elapsed={:?}",
                working_dims,
                color_started.elapsed()
            );
            drop(native);
            drop(transformed);
            drop(warped);
            let output = resize_native_output(&rendered, &output_window, request.quality, &valid)?;
            drop(rendered);
            Some((
                quantize_native_output(output, request.sixteen_bit, &valid)?,
                body_attenuation,
            ))
        };
        request.source.verify(&valid)?;
        let reshape_attenuation = asset.as_ref().and_then(|a| compute_reshape_attenuation(a));
        let (result_image, body_attenuation) = match result_image {
            Some((image, body)) => (Some(image), body),
            None => (None, None),
        };
        Ok(PhotoRenderResult {
            image: result_image,
            stamp,
            source: request.source,
            asset,
            lease,
            reshape_attenuation,
            body_attenuation,
        })
    }
}

/// LUT 调用方必须传实际内容摘要；完整 hydrated 参数包含补丁和蒙版字节。
pub fn composite_key(
    source: &SourceSnapshot,
    adjustments: &Value,
    lut_content_key: &str,
    tone_mapper: &str,
) -> String {
    let mut composite = adjustments.clone();
    if let Some(object) = composite.as_object_mut() {
        for key in ["portraitReshape", "crop", "showClipping"] {
            object.remove(key);
        }
        if let Some(portrait) = object.get_mut("portrait").and_then(Value::as_object_mut) {
            for key in KEYS {
                portrait.remove(key);
            }
        }
    }
    content_key(&json!([
        RENDER_VERSION,
        source.source_key(),
        composite,
        lut_content_key,
        tone_mapper
    ]))
}

/// 保留解码位深，逐行转换并显式处理 CPU 分配失败；输出仍为未预乘源颜色。
pub(crate) fn native_rgba(
    image: &DynamicImage,
    current: &dyn Fn() -> bool,
) -> Result<image::Rgba32FImage, String> {
    check_current(current)?;
    let [width, height] = [image.width(), image.height()];
    check_dimensions([width, height])?;
    let length = (width as usize)
        .checked_mul(height as usize)
        .and_then(|n| n.checked_mul(4))
        .ok_or("原生转换长度溢出")?;
    let mut values = Vec::new();
    values
        .try_reserve_exact(length)
        .map_err(|_| "原生转换 CPU 内存不足")?;
    macro_rules! convert {
        ($buffer:expr, $max:expr) => {{
            use image::Pixel;
            for row in $buffer.rows() {
                check_current(current)?;
                for pixel in row {
                    let rgba = pixel.to_rgba().0.map(|v| v as f32 / $max);
                    if rgba.iter().any(|v| !v.is_finite() || v.abs() > 65504.0)
                        || !(0.0..=1.0).contains(&rgba[3])
                    {
                        return Err("原生照片含非法浮点或 Alpha".into());
                    }
                    values.extend_from_slice(&if rgba[3] == 0.0 { [0.0; 4] } else { rgba });
                }
            }
        }};
    }
    match image {
        DynamicImage::ImageLuma8(v) => convert!(v, 255.0),
        DynamicImage::ImageLumaA8(v) => convert!(v, 255.0),
        DynamicImage::ImageRgb8(v) => convert!(v, 255.0),
        DynamicImage::ImageRgba8(v) => convert!(v, 255.0),
        DynamicImage::ImageLuma16(v) => convert!(v, 65535.0),
        DynamicImage::ImageLumaA16(v) => convert!(v, 65535.0),
        DynamicImage::ImageRgb16(v) => convert!(v, 65535.0),
        DynamicImage::ImageRgba16(v) => convert!(v, 65535.0),
        DynamicImage::ImageRgb32F(v) => convert!(v, 1.0),
        DynamicImage::ImageRgba32F(v) => convert!(v, 1.0),
        _ => return Err("原生照片像素类型不支持".into()),
    }
    check_current(current)?;
    image::Rgba32FImage::from_raw(width, height, values).ok_or_else(|| "原生转换长度不符".into())
}

fn output_axis_weights(
    start: i32,
    count: u32,
    scale: f64,
    offset: f64,
    limit: u32,
    quality: RenderQuality,
    current: &dyn Fn() -> bool,
) -> Result<Vec<Vec<(usize, f64)>>, String> {
    check_current(current)?;
    if count == 0
        || limit == 0
        || count > crate::portrait_tiles::MAX_SIDE
        || limit > crate::portrait_tiles::MAX_SIDE
        || !scale.is_finite()
        || scale.abs() < 1e-12
        || scale.abs() > 32768.0
        || !offset.is_finite()
        || offset.abs() > 65536.0
    {
        return Err("输出缩放矩阵非法".into());
    }
    let spread = scale.abs().max(1.0);
    let radius = if quality == RenderQuality::Interactive {
        1.0
    } else {
        3.0
    };
    let max_taps = ((2.0 * radius * spread).ceil() as usize + 2).min(limit as usize);
    let bytes = max_taps
        .checked_mul(std::mem::size_of::<(usize, f64)>())
        .and_then(|v| v.checked_add(std::mem::size_of::<Vec<(usize, f64)>>()))
        .and_then(|v| v.checked_mul(count as usize))
        .ok_or("输出滤波权重预算溢出")?;
    if bytes > 16 * 1024 * 1024 {
        return Err("输出滤波权重超过单轴 16 MiB 预算".into());
    }
    let sinc = |x: f64| {
        if x.abs() < 1e-12 {
            1.0
        } else {
            let p = x * std::f64::consts::PI;
            p.sin() / p
        }
    };
    let mut all = Vec::new();
    all.try_reserve_exact(count as usize)
        .map_err(|_| "输出滤波索引内存不足")?;
    for i in 0..count {
        check_current(current)?;
        let center = (f64::from(start) + f64::from(i) + 0.5) * scale + offset - 0.5;
        let first = (center - radius * spread).ceil() as i64;
        let last = (center + radius * spread).floor() as i64;
        let mut taps: Vec<(usize, f64)> = Vec::new();
        taps.try_reserve_exact(((last - first + 1).max(0) as usize).min(limit as usize))
            .map_err(|_| "输出滤波权重内存不足")?;
        let mut sum = 0.0;
        for at in first..=last {
            if (at - first) % 1024 == 0 {
                check_current(current)?;
            }
            let distance = (at as f64 - center) / spread;
            let weight = if distance.abs() >= radius {
                0.0
            } else if quality == RenderQuality::Interactive {
                1.0 - distance.abs()
            } else {
                sinc(distance) * sinc(distance / 3.0)
            };
            if weight == 0.0 {
                continue;
            }
            let index = at.clamp(0, i64::from(limit) - 1) as usize;
            if let Some(previous) = taps.last_mut().filter(|t| t.0 == index) {
                previous.1 += weight;
            } else {
                taps.push((index, weight));
            }
            sum += weight;
        }
        if !sum.is_finite() || sum.abs() < 1e-12 {
            return Err("输出滤波支持域无效".into());
        }
        for (_, weight) in &mut taps {
            *weight /= sum;
        }
        all.push(taps);
    }
    Ok(all)
}

/// 原生尾部之后才重建输出；核使用完整源边界，不把 crop／ROI 边界当 ClampToEdge。
/// 返回仍为预乘线性参考 RGBA，避免最终 U16 量化前经过 U8 或 JPEG。
pub(crate) fn resize_native_output(
    native: &image::Rgba32FImage,
    window: &OutputWindow,
    quality: RenderQuality,
    current: &dyn Fn() -> bool,
) -> Result<image::Rgba32FImage, String> {
    use crate::portrait_tiles::PixelRect;
    check_current(current)?;
    check_dimensions([native.width(), native.height()])?;
    if !PixelRect::full(window.output_dimensions)?.contains(window.roi) {
        return Err("输出 ROI 非法".into());
    }
    let [a, b, c, d, e, f] = window.output_to_base;
    if b != 0.0 || d != 0.0 {
        return Err("未裁剪共轭视图必须先完成带真实扩展域的视图重建".into());
    }
    let x_taps = output_axis_weights(
        window.roi.x,
        window.roi.width,
        a,
        c,
        native.width(),
        quality,
        current,
    )?;
    let y_taps = output_axis_weights(
        window.roi.y,
        window.roi.height,
        e,
        f,
        native.height(),
        quality,
        current,
    )?;
    let width = window.roi.width as usize;
    let length = width
        .checked_mul(window.roi.height as usize)
        .and_then(|v| v.checked_mul(4))
        .ok_or("输出像素长度溢出")?;
    let mut data = Vec::new();
    data.try_reserve_exact(length)
        .map_err(|_| "输出浮点图内存不足")?;
    // 最多保留 32 MiB 横向中间行；任意长宽比都不分配 out_width × source_height 巨图。
    let capacity = (32 * 1024 * 1024 / (width * std::mem::size_of::<[f64; 4]>())).max(1);
    let mut rows: VecDeque<(usize, Vec<[f64; 4]>)> = VecDeque::new();
    rows.try_reserve_exact(capacity.min(native.height() as usize))
        .map_err(|_| "输出行缓存内存不足")?;
    let allocate_row = || -> Result<Vec<[f64; 4]>, String> {
        let mut row = Vec::new();
        row.try_reserve_exact(width)
            .map_err(|_| "输出滤波行内存不足")?;
        row.resize(width, [0.0; 4]);
        Ok(row)
    };
    for taps in y_taps {
        check_current(current)?;
        let mut output = allocate_row()?;
        for (y, weight) in taps {
            check_current(current)?;
            let row = if let Some(i) = rows.iter().position(|r| r.0 == y) {
                rows.remove(i).ok_or("输出滤波行索引错误")?
            } else {
                if rows.len() >= capacity {
                    rows.pop_front();
                }
                let mut row = allocate_row()?;
                for (out, taps) in row.iter_mut().zip(&x_taps) {
                    for &(x, weight) in taps {
                        let sample = native.get_pixel(x as u32, y as u32).0;
                        if sample.iter().any(|v| !v.is_finite())
                            || !(0.0..=1.0).contains(&sample[3])
                            || (sample[3] == 0.0 && sample[..3].iter().any(|v| *v != 0.0))
                        {
                            return Err("缩放源包含非法浮点或 Alpha".into());
                        }
                        for channel in 0..4 {
                            out[channel] += f64::from(sample[channel]) * weight;
                        }
                    }
                }
                (y, row)
            };
            for (out, sample) in output.iter_mut().zip(&row.1) {
                for channel in 0..4 {
                    out[channel] += sample[channel] * weight;
                }
            }
            if rows.len() >= capacity {
                rows.pop_front();
            }
            rows.push_back(row);
        }
        for pixel in output {
            let alpha = pixel[3].clamp(0.0, 1.0) as f32;
            if alpha == 0.0 {
                data.extend_from_slice(&[0.0; 4]);
            } else {
                data.extend_from_slice(&[
                    pixel[0].clamp(0.0, alpha as f64) as f32,
                    pixel[1].clamp(0.0, alpha as f64) as f32,
                    pixel[2].clamp(0.0, alpha as f64) as f32,
                    alpha,
                ]);
            }
        }
    }
    check_current(current)?;
    image::Rgba32FImage::from_raw(window.roi.width, window.roi.height, data)
        .ok_or_else(|| "缩放输出长度错误".into())
}

pub(crate) fn quantize_native_output(
    linear: image::Rgba32FImage,
    sixteen_bit: bool,
    current: &dyn Fn() -> bool,
) -> Result<DynamicImage, String> {
    check_current(current)?;
    let (width, height) = linear.dimensions();
    check_dimensions([width, height])?;
    fn quantize<T>(
        linear: &image::Rgba32FImage,
        convert: impl Fn(f32) -> T,
        current: &dyn Fn() -> bool,
    ) -> Result<Vec<T>, String> {
        let length = (linear.width() as usize)
            .checked_mul(linear.height() as usize)
            .and_then(|v| v.checked_mul(4))
            .ok_or("量化输出长度溢出")?;
        let mut output = Vec::new();
        output
            .try_reserve_exact(length)
            .map_err(|_| "量化输出内存不足")?;
        for (i, pixel) in linear.pixels().enumerate() {
            if i % linear.width() as usize == 0 {
                check_current(current)?;
            }
            if pixel.0.iter().any(|v| !v.is_finite()) || !(0.0..=1.0).contains(&pixel[3]) {
                return Err("最终量化输入非法".into());
            }
            for channel in 0..4 {
                let v = if channel == 3 {
                    pixel[3]
                } else if pixel[3] == 0.0 {
                    0.0
                } else {
                    let v = (pixel[channel] / pixel[3]).clamp(0.0, 1.0);
                    if v <= 0.0031308 {
                        12.92 * v
                    } else {
                        1.055 * v.powf(1.0 / 2.4) - 0.055
                    }
                };
                output.push(convert(v.clamp(0.0, 1.0)));
            }
        }
        check_current(current)?;
        Ok(output)
    }
    if sixteen_bit {
        let words = quantize(&linear, |v| (v * 65535.0).round() as u16, current)?;
        Ok(DynamicImage::ImageRgba16(
            image::ImageBuffer::from_raw(width, height, words).ok_or("U16 输出长度错误")?,
        ))
    } else {
        let bytes = quantize(&linear, |v| (v * 255.0).round() as u8, current)?;
        Ok(DynamicImage::ImageRgba8(
            image::ImageBuffer::from_raw(width, height, bytes).ok_or("U8 输出长度错误")?,
        ))
    }
}

#[cfg(test)]
#[path = "portrait_render_tests.rs"]
mod tests;
