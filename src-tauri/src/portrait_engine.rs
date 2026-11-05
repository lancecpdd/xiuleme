//! 原生人像引擎（C++／OpenCV `portrait_engine` 动态库）：加载、分析／求解封装，
//! 以及把引擎输出的逆向采样网格转换为现有 wgpu 重塑通路使用的 `CanonicalField`。
//! C ABI 见 native/portrait-engine/include/portrait_engine.h，设计见
//! docs/portrait-reshape/17-native-portrait-engine.md。
#![allow(dead_code)]

use image::RgbaImage;
use libloading::Library;
use once_cell::sync::OnceCell;
use serde::Deserialize;
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::ffi::{CStr, CString, c_char, c_void};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use crate::face_reshape::{GeometrySolution, Strengths};
use crate::portrait_safety::SafeScene;
use crate::portrait_tiles::CanonicalField;
use crate::portrait_uv_warp::{UvFaceWarp, UvWarpStrengths};

pub const FEATURE_SLIM_ARMS: &str = "body.slim_arms";
pub const FEATURE_SLIM_WAIST: &str = "body.slim_waist";
pub const FEATURE_SLIM_LEGS: &str = "body.slim_legs";
/// 纳入 composite／参数哈希；换算法后旧缓存失效。
pub const BODY_PARAMETER_HASH_VERSION: &str = "portrait-engine-body-v1";

pub const ABI_VERSION: u32 = 1;
/// 与 `SafeScene::from_uv_warps` 的单场景上限一致。
const MAX_REGIONS: usize = 32;

const PE_OK: i32 = 0;
const PE_CANCELLED: i32 = 1;
const PE_FORMAT_RGBA8: u32 = 1;

#[repr(C)]
struct PeEngineDesc {
    struct_size: u32,
    threads: i32,
    resource_dir: *const c_char,
    onnxruntime_path: *const c_char,
}

#[repr(C)]
struct PeImage {
    struct_size: u32,
    format: u32,
    width: u32,
    height: u32,
    stride_bytes: u32,
    data: *const c_void,
}

#[repr(C)]
struct PeParam {
    feature_id: *const c_char,
    name: *const c_char,
    value: f64,
}

#[repr(C)]
struct PeWarpRegion {
    struct_size: u32,
    x0: f32,
    y0: f32,
    x1: f32,
    y1: f32,
    cells_x: u32,
    cells_y: u32,
    source_xy: *const f32,
}

type Handle = *mut c_void;
type CancelFn = unsafe extern "C" fn(*mut c_void) -> i32;

struct Api {
    _lib: Library,
    last_error: unsafe extern "C" fn() -> *const c_char,
    engine_create: unsafe extern "C" fn(*const PeEngineDesc, *mut Handle) -> i32,
    engine_destroy: unsafe extern "C" fn(Handle),
    engine_capabilities: unsafe extern "C" fn(Handle) -> *const c_char,
    analyze: unsafe extern "C" fn(
        Handle,
        *const c_char,
        *const PeImage,
        Option<CancelFn>,
        *mut c_void,
        *mut Handle,
    ) -> i32,
    analysis_feature: unsafe extern "C" fn(Handle) -> *const c_char,
    analysis_diagnostics: unsafe extern "C" fn(Handle) -> *const c_char,
    analysis_serialize: unsafe extern "C" fn(Handle, *mut *mut u8, *mut usize) -> i32,
    analysis_deserialize: unsafe extern "C" fn(Handle, *const u8, usize, *mut Handle) -> i32,
    analysis_release: unsafe extern "C" fn(Handle),
    buffer_free: unsafe extern "C" fn(*mut u8),
    solve_warp: unsafe extern "C" fn(
        Handle,
        *const Handle,
        usize,
        *const PeParam,
        usize,
        Option<CancelFn>,
        *mut c_void,
        *mut Handle,
    ) -> i32,
    warp_region_count: unsafe extern "C" fn(Handle) -> usize,
    warp_region: unsafe extern "C" fn(Handle, usize, *mut PeWarpRegion) -> i32,
    warp_diagnostics: unsafe extern "C" fn(Handle) -> *const c_char,
    warp_result_release: unsafe extern "C" fn(Handle),
}

fn library_name() -> &'static str {
    if cfg!(target_os = "windows") {
        "portrait_engine.dll"
    } else if cfg!(target_os = "macos") {
        "libportrait_engine.dylib"
    } else {
        "libportrait_engine.so"
    }
}

impl Api {
    fn load(path: &Path) -> Result<Self, String> {
        unsafe {
            let lib = Library::new(path)
                .map_err(|e| format!("加载原生人像引擎失败（{}）：{e}", path.display()))?;
            macro_rules! symbol {
                ($name:literal) => {
                    *lib.get(concat!($name, "\0").as_bytes())
                        .map_err(|e| format!("原生人像引擎缺少导出 {}：{e}", $name))?
                };
            }
            let abi_version: unsafe extern "C" fn() -> u32 = symbol!("pe_abi_version");
            let found = abi_version();
            if found != ABI_VERSION {
                return Err(format!(
                    "原生人像引擎 ABI 版本不匹配：需要 {ABI_VERSION}，实际 {found}"
                ));
            }
            Ok(Self {
                last_error: symbol!("pe_last_error"),
                engine_create: symbol!("pe_engine_create"),
                engine_destroy: symbol!("pe_engine_destroy"),
                engine_capabilities: symbol!("pe_engine_capabilities"),
                analyze: symbol!("pe_analyze"),
                analysis_feature: symbol!("pe_analysis_feature"),
                analysis_diagnostics: symbol!("pe_analysis_diagnostics"),
                analysis_serialize: symbol!("pe_analysis_serialize"),
                analysis_deserialize: symbol!("pe_analysis_deserialize"),
                analysis_release: symbol!("pe_analysis_release"),
                buffer_free: symbol!("pe_buffer_free"),
                solve_warp: symbol!("pe_solve_warp"),
                warp_region_count: symbol!("pe_warp_region_count"),
                warp_region: symbol!("pe_warp_region"),
                warp_diagnostics: symbol!("pe_warp_diagnostics"),
                warp_result_release: symbol!("pe_warp_result_release"),
                _lib: lib,
            })
        }
    }

    /// 非 OK 状态转为错误；取消统一使用分析通路的过期文案，便于上层按既有规则丢弃结果。
    fn check(&self, status: i32, context: &str) -> Result<(), String> {
        match status {
            PE_OK => Ok(()),
            PE_CANCELLED => Err(crate::portrait_analysis::CANCELLED.into()),
            _ => {
                let message = unsafe { owned_string((self.last_error)()) };
                Err(format!("原生人像引擎{context}失败（{status}）：{message}"))
            }
        }
    }
}

unsafe fn owned_string(pointer: *const c_char) -> String {
    if pointer.is_null() {
        return String::new();
    }
    unsafe { CStr::from_ptr(pointer) }
        .to_string_lossy()
        .into_owned()
}

/// `current()` 为假表示请求已过期；回调内 panic 也按取消处理，避免跨 FFI 展开。
unsafe extern "C" fn cancel_trampoline(user: *mut c_void) -> i32 {
    let current = unsafe { &*(user as *const &dyn Fn() -> bool) };
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| current())) {
        Ok(still_current) => i32::from(!still_current),
        Err(_) => 1,
    }
}

#[derive(Clone, Debug, Deserialize)]
pub struct Capabilities {
    pub abi: u32,
    pub engine: String,
    pub opencv: Option<String>,
    pub features: Vec<FeatureInfo>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct FeatureInfo {
    pub id: String,
    pub version: String,
    pub kind: String,
    pub params: Vec<String>,
    pub models: Vec<ModelInfo>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct ModelInfo {
    pub file: String,
    pub sha256: String,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Eq)]
pub struct AnalysisDiagnostics {
    pub status: String,
    #[serde(default)]
    pub reason: Option<String>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Eq)]
pub struct FeatureStatus {
    pub id: String,
    pub status: String,
    #[serde(default)]
    pub reason: Option<String>,
}

#[derive(Clone, Debug, Deserialize)]
struct SolveDiagnostics {
    features: Vec<FeatureStatus>,
}

/// 引擎输出的一块形变区域：`bounds` 内均匀铺 `(cells+1)²` 个目标节点，`source` 为行优先源坐标。
#[derive(Clone, Debug, PartialEq)]
pub struct WarpRegion {
    pub bounds: [f32; 4],
    pub cells: [u32; 2],
    pub source: Vec<[f32; 2]>,
}

#[derive(Clone, Debug)]
pub struct WarpOutput {
    pub regions: Vec<WarpRegion>,
    pub features: Vec<FeatureStatus>,
}

pub struct Engine {
    api: Arc<Api>,
    handle: Handle,
    capabilities: Capabilities,
    contract_hash: String,
}

// 引擎实例内部以互斥量串行执行；句柄创建后只读。
unsafe impl Send for Engine {}
unsafe impl Sync for Engine {}

impl Drop for Engine {
    fn drop(&mut self) {
        unsafe { (self.api.engine_destroy)(self.handle) };
    }
}

impl Engine {
    /// `resource_dir` 放模型与能力资源；动态库在 `resource_dir/portrait-engine/`。
    pub fn load(resource_dir: &Path, onnxruntime: Option<&Path>) -> Result<Arc<Self>, String> {
        let library_dir = resource_dir.join("portrait-engine");
        let api = Arc::new(Api::load(&library_dir.join(library_name()))?);
        let path_string = |path: &Path| {
            CString::new(path.to_string_lossy().as_bytes())
                .map_err(|_| format!("原生人像引擎路径含 NUL：{}", path.display()))
        };
        let resource_dir_c = path_string(resource_dir)?;
        let onnxruntime = onnxruntime.map(path_string).transpose()?;
        let desc = PeEngineDesc {
            struct_size: std::mem::size_of::<PeEngineDesc>() as u32,
            threads: 0,
            resource_dir: resource_dir_c.as_ptr(),
            onnxruntime_path: onnxruntime
                .as_ref()
                .map_or(std::ptr::null(), |path| path.as_ptr()),
        };
        let mut handle: Handle = std::ptr::null_mut();
        api.check(unsafe { (api.engine_create)(&desc, &mut handle) }, "创建")?;
        let raw = unsafe { owned_string((api.engine_capabilities)(handle)) };
        let parsed = serde_json::from_str::<Capabilities>(&raw)
            .map_err(|e| format!("原生人像引擎能力清单无法解析：{e}"))
            .and_then(|capabilities| {
                if capabilities.abi == ABI_VERSION {
                    Ok(capabilities)
                } else {
                    Err(format!(
                        "原生人像引擎能力清单 ABI 不匹配：{}",
                        capabilities.abi
                    ))
                }
            });
        let capabilities = match parsed {
            Ok(capabilities) => capabilities,
            Err(error) => {
                unsafe { (api.engine_destroy)(handle) };
                return Err(error);
            }
        };
        Ok(Arc::new(Self {
            capabilities,
            contract_hash: hex::encode(Sha256::digest(raw.as_bytes())),
            api,
            handle,
        }))
    }

    pub fn capabilities(&self) -> &Capabilities {
        &self.capabilities
    }

    /// 引擎版本、功能版本与模型哈希的摘要；纳入参数哈希，换引擎后旧缓存自然失效。
    pub fn contract_hash(&self) -> &str {
        &self.contract_hash
    }

    pub fn feature(&self, id: &str) -> Option<&FeatureInfo> {
        self.capabilities.features.iter().find(|f| f.id == id)
    }

    pub fn analyze(
        self: &Arc<Self>,
        feature: &str,
        image: &RgbaImage,
        current: &dyn Fn() -> bool,
    ) -> Result<Analysis, String> {
        let feature = CString::new(feature).map_err(|_| "功能标识含 NUL".to_string())?;
        let stride = image
            .width()
            .checked_mul(4)
            .ok_or("原生人像引擎输入宽度溢出")?;
        let descriptor = PeImage {
            struct_size: std::mem::size_of::<PeImage>() as u32,
            format: PE_FORMAT_RGBA8,
            width: image.width(),
            height: image.height(),
            stride_bytes: stride,
            data: image.as_raw().as_ptr().cast(),
        };
        let current_ref: &dyn Fn() -> bool = current;
        let mut handle: Handle = std::ptr::null_mut();
        let status = unsafe {
            (self.api.analyze)(
                self.handle,
                feature.as_ptr(),
                &descriptor,
                Some(cancel_trampoline),
                &current_ref as *const &dyn Fn() -> bool as *mut c_void,
                &mut handle,
            )
        };
        self.api.check(status, "分析")?;
        Ok(Analysis {
            engine: Arc::clone(self),
            handle,
        })
    }

    pub fn deserialize(self: &Arc<Self>, blob: &[u8]) -> Result<Analysis, String> {
        let mut handle: Handle = std::ptr::null_mut();
        let status = unsafe {
            (self.api.analysis_deserialize)(self.handle, blob.as_ptr(), blob.len(), &mut handle)
        };
        self.api.check(status, "恢复分析")?;
        Ok(Analysis {
            engine: Arc::clone(self),
            handle,
        })
    }

    /// 一次求解多个功能；`params` 为 `(功能, 参数名, 值)`。结果立即复制为 Rust 所有。
    pub fn solve_warp(
        &self,
        analyses: &[&Analysis],
        params: &[(&str, &str, f64)],
        current: &dyn Fn() -> bool,
    ) -> Result<WarpOutput, String> {
        if analyses
            .iter()
            .any(|analysis| !std::ptr::eq(analysis.engine.as_ref(), self))
        {
            return Err("分析结果不属于当前原生人像引擎".into());
        }
        let handles: Vec<Handle> = analyses.iter().map(|analysis| analysis.handle).collect();
        let names = params
            .iter()
            .map(|(feature, name, _)| -> Result<_, std::ffi::NulError> {
                Ok((CString::new(*feature)?, CString::new(*name)?))
            })
            .collect::<Result<Vec<_>, _>>()
            .map_err(|_| "参数名含 NUL".to_string())?;
        let raw_params: Vec<PeParam> = names
            .iter()
            .zip(params)
            .map(|((feature, name), (_, _, value))| PeParam {
                feature_id: feature.as_ptr(),
                name: name.as_ptr(),
                value: *value,
            })
            .collect();
        let current_ref: &dyn Fn() -> bool = current;
        let mut result: Handle = std::ptr::null_mut();
        let status = unsafe {
            (self.api.solve_warp)(
                self.handle,
                handles.as_ptr(),
                handles.len(),
                raw_params.as_ptr(),
                raw_params.len(),
                Some(cancel_trampoline),
                &current_ref as *const &dyn Fn() -> bool as *mut c_void,
                &mut result,
            )
        };
        self.api.check(status, "求解")?;
        let output = self.copy_result(result);
        unsafe { (self.api.warp_result_release)(result) };
        output
    }

    fn copy_result(&self, result: Handle) -> Result<WarpOutput, String> {
        let count = unsafe { (self.api.warp_region_count)(result) };
        if count > MAX_REGIONS {
            return Err(format!("原生人像引擎区域过多：{count}"));
        }
        let mut regions = Vec::with_capacity(count);
        for index in 0..count {
            let mut raw = PeWarpRegion {
                struct_size: std::mem::size_of::<PeWarpRegion>() as u32,
                x0: 0.0,
                y0: 0.0,
                x1: 0.0,
                y1: 0.0,
                cells_x: 0,
                cells_y: 0,
                source_xy: std::ptr::null(),
            };
            self.api.check(
                unsafe { (self.api.warp_region)(result, index, &mut raw) },
                "读取区域",
            )?;
            let nodes = (raw.cells_x as usize + 1) * (raw.cells_y as usize + 1);
            if raw.source_xy.is_null() || raw.cells_x == 0 || raw.cells_y == 0 {
                return Err("原生人像引擎返回了空区域".into());
            }
            let flat = unsafe { std::slice::from_raw_parts(raw.source_xy, nodes * 2) };
            regions.push(WarpRegion {
                bounds: [raw.x0, raw.y0, raw.x1, raw.y1],
                cells: [raw.cells_x, raw.cells_y],
                source: flat.chunks_exact(2).map(|p| [p[0], p[1]]).collect(),
            });
        }
        let raw = unsafe { owned_string((self.api.warp_diagnostics)(result)) };
        let diagnostics: SolveDiagnostics =
            serde_json::from_str(&raw).map_err(|e| format!("原生人像引擎求解诊断无法解析：{e}"))?;
        Ok(WarpOutput {
            regions,
            features: diagnostics.features,
        })
    }
}

/// 引擎分析结果；持有引擎引用，保证释放顺序。
pub struct Analysis {
    engine: Arc<Engine>,
    handle: Handle,
}

unsafe impl Send for Analysis {}
unsafe impl Sync for Analysis {}

impl Drop for Analysis {
    fn drop(&mut self) {
        unsafe { (self.engine.api.analysis_release)(self.handle) };
    }
}

impl Analysis {
    pub fn feature(&self) -> String {
        unsafe { owned_string((self.engine.api.analysis_feature)(self.handle)) }
    }

    pub fn diagnostics(&self) -> Result<AnalysisDiagnostics, String> {
        let raw = unsafe { owned_string((self.engine.api.analysis_diagnostics)(self.handle)) };
        serde_json::from_str(&raw).map_err(|e| format!("原生人像引擎分析诊断无法解析：{e}"))
    }

    /// 供快照保存；只能由同 ABI、同功能版本的引擎恢复。
    pub fn serialize(&self) -> Result<Vec<u8>, String> {
        let api = &self.engine.api;
        let mut data: *mut u8 = std::ptr::null_mut();
        let mut size = 0usize;
        api.check(
            unsafe { (api.analysis_serialize)(self.handle, &mut data, &mut size) },
            "序列化分析",
        )?;
        let blob = unsafe { std::slice::from_raw_parts(data, size) }.to_vec();
        unsafe { (api.buffer_free)(data) };
        Ok(blob)
    }
}

struct EngineConfig {
    /// 模型所在的 `resources/` 目录；DLL 在其子目录 `portrait-engine/`。
    resource_dir: PathBuf,
    onnxruntime: Option<PathBuf>,
}

static CONFIG: OnceCell<EngineConfig> = OnceCell::new();
static ENGINE: OnceCell<Result<Arc<Engine>, String>> = OnceCell::new();

/// 应用启动时登记资源目录与共用的 ONNX Runtime；首次使用时才加载动态库。
pub fn set_resource_directory(resource_dir: PathBuf, onnxruntime: Option<PathBuf>) {
    let _ = CONFIG.set(EngineConfig {
        resource_dir: crate::portrait_gpupixel::normalize_fs_dir(resource_dir),
        onnxruntime,
    });
}

pub fn engine() -> Result<Arc<Engine>, String> {
    let config = CONFIG
        .get()
        .ok_or("原生人像引擎目录未配置（应用启动时应 set_resource_directory）")?;
    ENGINE
        .get_or_init(|| Engine::load(&config.resource_dir, config.onnxruntime.as_deref()))
        .clone()
}

fn grid_triangles(cells_x: u32, cells_y: u32) -> Vec<[usize; 3]> {
    let stride = cells_x as usize + 1;
    let mut triangles = Vec::with_capacity(cells_x as usize * cells_y as usize * 2);
    for y in 0..cells_y as usize {
        for x in 0..cells_x as usize {
            let a = y * stride + x;
            let b = a + 1;
            let c = a + stride;
            let d = c + 1;
            triangles.push([a, b, d]);
            triangles.push([a, d, c]);
        }
    }
    triangles
}

/// 把引擎区域转换为规范逆位移场；全部区域恒等时返回 `None`。
/// `requested` 为该次求解的用户强度（0–100），仅用于安全报告。
pub fn field_from_regions(
    key: String,
    dimensions: [u32; 2],
    regions: &[WarpRegion],
    requested: f64,
    current: &dyn Fn() -> bool,
) -> Result<Option<CanonicalField>, String> {
    if regions.len() > MAX_REGIONS {
        return Err(format!("原生人像引擎区域过多：{}", regions.len()));
    }
    let strengths = UvWarpStrengths::from_ui(requested, 0.0)
        .map_err(|error| format!("原生人像引擎强度非法：{error:?}"))?;
    let mut warps = Vec::with_capacity(regions.len());
    for (index, region) in regions.iter().enumerate() {
        let [x0, y0, x1, y1] = region.bounds.map(f64::from);
        let [cells_x, cells_y] = region.cells;
        let columns = cells_x as usize + 1;
        let valid_bounds = [x0, y0, x1, y1].iter().all(|v| v.is_finite())
            && 0.0 <= x0
            && x0 < x1
            && x1 <= f64::from(dimensions[0])
            && 0.0 <= y0
            && y0 < y1
            && y1 <= f64::from(dimensions[1]);
        if !valid_bounds
            || cells_x == 0
            || cells_y == 0
            || region.source.len() != columns * (cells_y as usize + 1)
            || region.source.iter().flatten().any(|v| !v.is_finite())
        {
            return Err(format!("原生人像引擎区域 {index} 非法"));
        }
        // 安全层按顶点数区分稠密网格与人脸 ARAP 控制网，二者不能同数。
        if region.source.len() == crate::face_reshape::MESH_VERTICES {
            return Err(format!("原生人像引擎区域 {index} 节点数与人脸控制网冲突"));
        }
        let target: Vec<[f64; 2]> = (0..region.source.len())
            .map(|node| {
                let i = (node % columns) as f64;
                let j = (node / columns) as f64;
                [
                    x0 + (x1 - x0) * i / f64::from(cells_x),
                    y0 + (y1 - y0) * j / f64::from(cells_y),
                ]
            })
            .collect();
        let source: Vec<[f64; 2]> = region
            .source
            .iter()
            .map(|p| [f64::from(p[0]), f64::from(p[1])])
            .collect();
        if source == target {
            continue;
        }
        let geometry = GeometrySolution::from_dense_pixel_mesh(
            source,
            target,
            Arc::new(grid_triangles(cells_x, cells_y)),
            (x1 - x0).max(y1 - y0),
            Strengths([requested, 0.0, 0.0, 0.0]),
            [requested / 100.0, 0.0, 0.0, 0.0],
            Strengths([requested, 0.0, 0.0, 0.0]),
            false,
        )
        .map_err(|error| format!("原生人像引擎区域 {index} 网格失败：{error:?}"))?;
        warps.push(UvFaceWarp {
            face_id: format!("region-{index:02}"),
            strengths,
            geometry,
        });
    }
    if warps.is_empty() {
        return Ok(None);
    }
    let scene = SafeScene::from_uv_warps(key, dimensions, warps, current)?;
    CanonicalField::from_scene(&scene, current).map(Some)
}

/// 身体塑形 UI 0–100；分区关闭时全部为零。旧 sidecar 顶层 `slimArms` 不进入此结构。
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct BodyParams {
    pub slim_arms: f64,
    pub slim_waist: f64,
    pub slim_legs: f64,
}

impl BodyParams {
    pub fn is_active(self) -> bool {
        self.slim_arms > 0.0 || self.slim_waist > 0.0 || self.slim_legs > 0.0
    }

    pub fn max_amount(self) -> f64 {
        self.slim_arms.max(self.slim_waist).max(self.slim_legs)
    }

    fn active_features(self) -> Vec<(&'static str, f64)> {
        let mut features = Vec::new();
        if self.slim_arms > 0.0 {
            features.push((FEATURE_SLIM_ARMS, self.slim_arms));
        }
        if self.slim_waist > 0.0 {
            features.push((FEATURE_SLIM_WAIST, self.slim_waist));
        }
        if self.slim_legs > 0.0 {
            features.push((FEATURE_SLIM_LEGS, self.slim_legs));
        }
        features
    }
}

fn clamp_body_amount(value: &Value) -> f64 {
    match value {
        Value::Number(n) => n
            .as_f64()
            .filter(|v| v.is_finite())
            .map(|v| v.clamp(0.0, 100.0))
            .unwrap_or(0.0),
        _ => 0.0,
    }
}

pub fn body_params_from_adjustments(adjustments: &Value) -> BodyParams {
    // 缺省／true 为开组；显式 false 才旁路。
    if adjustments["sectionVisibility"]["portraitBody"] == false {
        return BodyParams::default();
    }
    let portrait = &adjustments["portrait"];
    BodyParams {
        slim_arms: clamp_body_amount(&portrait["bodySlimArms"]),
        slim_waist: clamp_body_amount(&portrait["bodySlimWaist"]),
        slim_legs: clamp_body_amount(&portrait["bodySlimLegs"]),
    }
}

pub fn body_requested(adjustments: &Value) -> bool {
    body_params_from_adjustments(adjustments).is_active()
}

/// 含引擎 contract_hash；引擎／模型变更后旧缓存自然失效。
pub fn body_parameter_hash(adjustments: &Value) -> String {
    let params = body_params_from_adjustments(adjustments);
    let contract = engine()
        .map(|loaded| loaded.contract_hash().to_owned())
        .unwrap_or_else(|_| "unavailable".into());
    let payload = format!(
        "{BODY_PARAMETER_HASH_VERSION}|{contract}|{:.6}|{:.6}|{:.6}",
        params.slim_arms, params.slim_waist, params.slim_legs
    );
    hex::encode(Sha256::digest(payload.as_bytes()))
}

/// 分析＋求解身体形变场。请求非零但引擎旁路／恒等时 `attenuation_identity` 为真。
pub fn field_rgba(
    image: &RgbaImage,
    params: BodyParams,
    current: &dyn Fn() -> bool,
) -> Result<(Option<CanonicalField>, bool), String> {
    if !params.is_active() {
        return Ok((None, false));
    }
    let engine = engine()?;
    let active = params.active_features();
    let mut analyses = Vec::with_capacity(active.len());
    for &(feature, _) in &active {
        current()
            .then_some(())
            .ok_or_else(|| crate::portrait_analysis::CANCELLED.to_string())?;
        analyses.push(engine.analyze(feature, image, current)?);
    }
    let refs: Vec<&Analysis> = analyses.iter().collect();
    let solve_params: Vec<(&str, &str, f64)> = active
        .iter()
        .map(|&(feature, amount)| (feature, "amount", amount))
        .collect();
    let output = engine.solve_warp(&refs, &solve_params, current)?;
    let requested = active.iter().any(|&(_, amount)| amount > 0.0);
    let all_soft = !output.features.is_empty()
        && output
            .features
            .iter()
            .all(|feature| feature.status == "bypass" || feature.status == "identity");
    let attenuation_identity = requested && (output.regions.is_empty() || all_soft);
    let key = format!(
        "portrait-engine-body:{}:{}",
        engine.contract_hash(),
        body_parameter_hash_from_params(params, engine.contract_hash())
    );
    let field = field_from_regions(
        key,
        [image.width(), image.height()],
        &output.regions,
        params.max_amount(),
        current,
    )?;
    let identity = attenuation_identity && field.is_none();
    Ok((field, identity))
}

fn body_parameter_hash_from_params(params: BodyParams, contract: &str) -> String {
    let payload = format!(
        "{BODY_PARAMETER_HASH_VERSION}|{contract}|{:.6}|{:.6}|{:.6}",
        params.slim_arms, params.slim_waist, params.slim_legs
    );
    hex::encode(Sha256::digest(payload.as_bytes()))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 与引擎 `debug.pinch` 相同的径向收缩，用于不依赖动态库的转换测试。
    fn pinch_region(dimensions: [u32; 2], strength: f64) -> WarpRegion {
        let cx = f64::from(dimensions[0]) * 0.5;
        let cy = f64::from(dimensions[1]) * 0.5;
        let radius = 0.25 * f64::from(dimensions[0].min(dimensions[1]));
        let cells = 16u32;
        let mut source = Vec::new();
        for j in 0..=cells {
            for i in 0..=cells {
                let px = cx - radius + 2.0 * radius * f64::from(i) / f64::from(cells);
                let py = cy - radius + 2.0 * radius * f64::from(j) / f64::from(cells);
                let t = (px - cx).hypot(py - cy) / radius;
                let scale = if t >= 1.0 {
                    1.0
                } else {
                    1.0 + strength * (1.0 - t) * (1.0 - t)
                };
                source.push([
                    (cx + (px - cx) * scale) as f32,
                    (cy + (py - cy) * scale) as f32,
                ]);
            }
        }
        WarpRegion {
            bounds: [
                (cx - radius) as f32,
                (cy - radius) as f32,
                (cx + radius) as f32,
                (cy + radius) as f32,
            ],
            cells: [cells, cells],
            source,
        }
    }

    #[test]
    fn grid_triangles_cover_every_cell() {
        let triangles = grid_triangles(3, 2);
        assert_eq!(triangles.len(), 12);
        assert!(triangles.iter().flatten().all(|&index| index < 4 * 3));
    }

    #[test]
    fn empty_or_identity_regions_yield_no_field() {
        let dims = [160, 120];
        assert!(
            field_from_regions("empty".into(), dims, &[], 50.0, &|| true)
                .unwrap()
                .is_none()
        );
        let identity = pinch_region(dims, 0.0);
        assert!(
            field_from_regions("identity".into(), dims, &[identity], 50.0, &|| true)
                .unwrap()
                .is_none()
        );
    }

    #[test]
    fn pinch_region_builds_non_identity_field() {
        let dims = [160, 120];
        let field = field_from_regions(
            "synthetic-pinch".into(),
            dims,
            &[pinch_region(dims, 0.2)],
            100.0,
            &|| true,
        )
        .unwrap()
        .expect("pinch must produce a field");
        assert!(!field.is_identity());
        assert_eq!(field.dimensions(), dims);
    }

    #[test]
    fn malformed_regions_are_rejected() {
        let dims = [160, 120];
        let mut short = pinch_region(dims, 0.2);
        short.source.pop();
        let mut non_finite = pinch_region(dims, 0.2);
        non_finite.source[5][0] = f32::NAN;
        let mut outside = pinch_region(dims, 0.2);
        outside.bounds[2] = 400.0;
        let mut flat = pinch_region(dims, 0.2);
        flat.cells = [0, 16];
        for region in [short, non_finite, outside, flat] {
            assert!(field_from_regions("bad".into(), dims, &[region], 50.0, &|| true).is_err());
        }
        assert!(
            field_from_regions(
                "bad".into(),
                dims,
                &[pinch_region(dims, 0.2)],
                120.0,
                &|| true
            )
            .is_err()
        );
    }

    fn resource_directory() -> PathBuf {
        std::env::var_os("PORTRAIT_RESOURCE_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|| Path::new(env!("CARGO_MANIFEST_DIR")).join("resources"))
    }

    #[test]
    #[ignore = "需要已构建的 portrait_engine 动态库（native/portrait-engine/scripts/build.ps1）"]
    fn native_engine_debug_features_round_trip() {
        let resources = resource_directory();
        let ort = resources.join(if cfg!(target_os = "windows") {
            "onnxruntime.dll"
        } else if cfg!(target_os = "macos") {
            "libonnxruntime.dylib"
        } else {
            "libonnxruntime.so"
        });
        let engine = Engine::load(&resources, ort.is_file().then_some(ort.as_path())).unwrap();
        assert!(engine.feature("debug.identity").is_some());
        assert!(engine.feature("body.slim_arms").is_some());
        assert!(engine.feature("body.slim_waist").is_some());
        assert!(engine.feature("body.slim_legs").is_some());
        assert_eq!(engine.contract_hash().len(), 64);
        let image = RgbaImage::from_pixel(200, 120, image::Rgba([90, 120, 150, 255]));

        let identity = engine.analyze("debug.identity", &image, &|| true).unwrap();
        assert_eq!(identity.feature(), "debug.identity");
        assert_eq!(identity.diagnostics().unwrap().status, "ready");
        let restored = engine.deserialize(&identity.serialize().unwrap()).unwrap();
        let output = engine
            .solve_warp(&[&restored], &[("debug.identity", "amount", 40.0)], &|| {
                true
            })
            .unwrap();
        assert!(output.regions.is_empty());
        assert_eq!(output.features[0].status, "identity");

        let pinch = engine.analyze("debug.pinch", &image, &|| true).unwrap();
        let output = engine
            .solve_warp(&[&pinch], &[("debug.pinch", "amount", 100.0)], &|| true)
            .unwrap();
        assert_eq!(output.regions.len(), 1);
        let field = field_from_regions(
            format!("native-pinch:{}", engine.contract_hash()),
            [200, 120],
            &output.regions,
            100.0,
            &|| true,
        )
        .unwrap()
        .expect("pinch must produce a field");
        assert!(!field.is_identity());

        assert_eq!(
            engine
                .solve_warp(&[&pinch], &[("debug.pinch", "amount", 100.0)], &|| false)
                .unwrap_err(),
            crate::portrait_analysis::CANCELLED
        );
        assert!(
            engine
                .solve_warp(&[&pinch], &[("debug.pinch", "strength", 1.0)], &|| true)
                .is_err()
        );
        assert!(engine.analyze("body.unknown", &image, &|| true).is_err());
    }
}
