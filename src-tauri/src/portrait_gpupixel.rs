//! GPUPixel 原版运行时（RGBA bake）：瘦脸／大眼／口红／腮红／提亮眼白／祛红血丝。
//! 经典磨皮／美白已下线；FFI 仍传 beauty/white 恒为 0。
//! 动态加载 `libgpupixel_rr`；关键点走上游 VNN，不注入 MediaPipe。
#![allow(dead_code)]

use image::{DynamicImage, RgbaImage};
use libloading::Library;
use once_cell::sync::OnceCell;
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::ffi::{CString, c_char, c_int, c_void};
use std::path::{Path, PathBuf};
use std::sync::mpsc;

pub const ENGINE_VERSION: &str = "gpupixel-runtime-v1";
/// v4：眼白提亮／祛红改走 GPUPixel bake（VNN 关键点 CPU 后处理）。
pub const PARAMETER_HASH_VERSION: &str = "gpupixel-runtime-params-v4";

/// UI 0–100；beauty/white 保留字段供 FFI，产品侧恒为 0。
/// 按上游 desktop 0–10 的 `/10` 约定映射。
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct BeautyParams {
    pub beauty: f32,
    pub white: f32,
    pub thin_face: f32,
    pub big_eye: f32,
    pub lipstick: f32,
    pub blusher: f32,
    pub eye_brighten: f32,
    pub eye_redness: f32,
}

impl BeautyParams {
    pub fn from_ui(
        beauty: f64,
        white: f64,
        thin_face: f64,
        big_eye: f64,
        lipstick: f64,
        blusher: f64,
        eye_brighten: f64,
        eye_redness: f64,
    ) -> Self {
        let map = |v: f64| -> f32 {
            if !v.is_finite() {
                return 0.0;
            }
            ((v.clamp(0.0, 100.0) / 10.0) as f32).clamp(0.0, 10.0)
        };
        Self {
            beauty: map(beauty),
            white: map(white),
            thin_face: map(thin_face),
            big_eye: map(big_eye),
            lipstick: map(lipstick),
            blusher: map(blusher),
            eye_brighten: map(eye_brighten),
            eye_redness: map(eye_redness),
        }
    }

    pub fn is_active(self) -> bool {
        self.thin_face > 0.0
            || self.big_eye > 0.0
            || self.lipstick > 0.0
            || self.blusher > 0.0
            || self.eye_brighten > 0.0
            || self.eye_redness > 0.0
    }

    fn face_active(self) -> bool {
        self.is_active()
    }
}

pub fn amount(value: &Value) -> f64 {
    match value {
        Value::Number(n) => n
            .as_f64()
            .filter(|v| v.is_finite())
            .map(|v| v.clamp(0.0, 100.0))
            .unwrap_or(0.0),
        _ => 0.0,
    }
}

fn section_visible(adjustments: &Value, key: &str) -> bool {
    adjustments["sectionVisibility"][key] != false
}

/// 整图 VNN 取全局与逐脸 override 的最大值（UI 有脸时写 `overrides.uvSlimFace`）。
fn max_face_override(adjustments: &Value, key: &str) -> f64 {
    let Some(faces) = adjustments["portraitReshape"]["faces"].as_array() else {
        return 0.0;
    };
    faces.iter().fold(0.0, |acc, face| {
        acc.max(amount(&face["overrides"][key]))
    })
}

/// 从 adjustments 读取瘦脸／大眼／口红／腮红／眼白；经典磨皮／美白恒为 0。
/// 瘦脸／大眼：`portrait` 暂存 ∪ 逐脸 `uvSlimFace`／`bigEye` override（取 max）。
pub fn params_from_adjustments(adjustments: &Value) -> BeautyParams {
    let portrait = &adjustments["portrait"];
    let reshape = section_visible(adjustments, "portraitReshape");
    let eyes = section_visible(adjustments, "portraitEyes");
    let makeup = section_visible(adjustments, "portraitMakeup");
    let slim = amount(&portrait["slimFace"])
        .max(amount(&portrait["uvSlimFace"]))
        .max(max_face_override(adjustments, "uvSlimFace"))
        .max(max_face_override(adjustments, "slimFace"));
    let big_eye = amount(&portrait["bigEye"]).max(max_face_override(adjustments, "bigEye"));
    BeautyParams::from_ui(
        0.0,
        0.0,
        if reshape { slim } else { 0.0 },
        if eyes { big_eye } else { 0.0 },
        if makeup {
            amount(&portrait["lipstick"])
        } else {
            0.0
        },
        if makeup {
            amount(&portrait["blusher"])
        } else {
            0.0
        },
        if eyes {
            amount(&portrait["eyeWhiteBrighten"])
        } else {
            0.0
        },
        if eyes {
            amount(&portrait["eyeRedness"])
        } else {
            0.0
        },
    )
}

pub fn requested(adjustments: &Value) -> bool {
    params_from_adjustments(adjustments).is_active()
}

pub fn face_requested(adjustments: &Value) -> bool {
    params_from_adjustments(adjustments).face_active()
}

fn validate_bake_output(input: &RgbaImage, output: &RgbaImage) -> Result<(), String> {
    if input.dimensions() != output.dimensions() {
        return Err("GPUPixel 输出尺寸与输入不一致".into());
    }
    let input_has_visible_color = input
        .pixels()
        .any(|p| p[3] != 0 && (p[0] != 0 || p[1] != 0 || p[2] != 0));
    if input_has_visible_color {
        let output_has_alpha = output.pixels().any(|p| p[3] != 0);
        let output_has_color = output
            .pixels()
            .any(|p| p[0] != 0 || p[1] != 0 || p[2] != 0);
        if !output_has_alpha || !output_has_color {
            return Err("GPUPixel 未产出有效当前帧，已拒绝发布黑屏预览".into());
        }
    }
    Ok(())
}

pub fn parameter_hash(adjustments: &Value) -> String {
    let p = params_from_adjustments(adjustments);
    let mut hasher = Sha256::new();
    hasher.update(PARAMETER_HASH_VERSION.as_bytes());
    hasher.update(ENGINE_VERSION.as_bytes());
    for v in [
        p.beauty,
        p.white,
        p.thin_face,
        p.big_eye,
        p.lipstick,
        p.blusher,
        p.eye_brighten,
        p.eye_redness,
    ] {
        hasher.update(v.to_bits().to_le_bytes());
    }
    hex::encode(hasher.finalize())
}

type CreateFn = unsafe extern "C" fn() -> *mut c_void;
type InitRrFn = unsafe extern "C" fn(*mut c_void, *const c_char) -> c_int;
type InitPyFn = unsafe extern "C" fn(*mut c_void) -> bool;
type VoidHandleFn = unsafe extern "C" fn(*mut c_void);
/// 新 ABI：8 个 float（含眼白）。
type SetParamsFn = unsafe extern "C" fn(
    *mut c_void,
    f32,
    f32,
    f32,
    f32,
    f32,
    f32,
    f32,
    f32,
);
/// 旧 ABI：6 个 float（无眼白）。
type SetParamsV1Fn = unsafe extern "C" fn(*mut c_void, f32, f32, f32, f32, f32, f32);
type RunFn = unsafe extern "C" fn(
    *mut c_void,
    *const u8,
    c_int,
    c_int,
    c_int,
) -> *mut u8;

enum InitKind {
    /// `gpupixel_rr_initialize(handle, resource_dir) -> int`
    Rr(InitRrFn),
    /// `GPUPixelWrapper_initialize(handle) -> bool`（资源靠 CWD）
    Py(InitPyFn),
}

enum SetParamsKind {
    /// `gpupixel_rr_set_parameters` 8 参（含眼白）。
    V2(SetParamsFn),
    /// 旧 6 参 / `gpupixel_rr_set_parameters_v1`（眼白不可用）。
    V1(SetParamsV1Fn),
}

struct Api {
    _lib: Library,
    create: CreateFn,
    initialize: InitKind,
    set_callbacks: VoidHandleFn,
    set_parameters: SetParamsKind,
    run: RunFn,
    release_output: VoidHandleFn,
    /// 可选：销毁会话句柄。pywrapper 未导出句柄析构，可为 None。
    destroy: Option<VoidHandleFn>,
}

/// 仅存活于 `gpupixel-gl` 常驻线程内，不跨线程传递。
struct Session {
    api: &'static Api,
    handle: *mut c_void,
}

struct BakeJob {
    image: RgbaImage,
    params: BeautyParams,
    reply: mpsc::Sender<Result<RgbaImage, String>>,
}

static API: OnceCell<Result<Api, String>> = OnceCell::new();
static WORKER: OnceCell<Result<mpsc::Sender<BakeJob>, String>> = OnceCell::new();
static RESOURCE_DIR: OnceCell<PathBuf> = OnceCell::new();

fn library_names() -> &'static [&'static str] {
    #[cfg(target_os = "windows")]
    {
        &[
            "libgpupixel_rr.dll",
            "gpupixel_rr.dll",
            // 已打包的社区 pywrapper（符号 GPUPixelWrapper_*）
            "libgpupixel_pywrapper.dll",
            "gpupixel_pywrapper.dll",
        ]
    }
    #[cfg(target_os = "macos")]
    {
        &[
            "libgpupixel_rr.dylib",
            "gpupixel_rr.dylib",
            "libgpupixel_pywrapper.dylib",
        ]
    }
    #[cfg(target_os = "android")]
    {
        &["libgpupixel_rr.so", "gpupixel_rr.so", "libgpupixel_pywrapper.so"]
    }
    #[cfg(not(any(target_os = "windows", target_os = "macos", target_os = "android")))]
    {
        &["libgpupixel_rr.so", "libgpupixel_pywrapper.so"]
    }
}

/// Tauri Resource 目录常带 `\\?\` 前缀；VNN/`_chdir`/CRT fopen 对扩展路径不友好。
pub(crate) fn normalize_fs_dir(dir: PathBuf) -> PathBuf {
    #[cfg(windows)]
    {
        let raw = dir.to_string_lossy();
        if let Some(stripped) = raw.strip_prefix(r"\\?\") {
            // 保留 UNC：`\\?\UNC\server\share` → `\\server\share`
            if let Some(unc) = stripped.strip_prefix("UNC\\") {
                return PathBuf::from(format!(r"\\{unc}"));
            }
            return PathBuf::from(stripped);
        }
    }
    dir
}

pub fn set_resource_directory(dir: PathBuf) {
    let _ = RESOURCE_DIR.set(normalize_fs_dir(dir));
}

pub fn resource_directory() -> Option<&'static Path> {
    RESOURCE_DIR.get().map(|p| p.as_path())
}

fn resolve_library_path(dir: &Path) -> Result<PathBuf, String> {
    for name in library_names() {
        let candidate = dir.join(name);
        if candidate.is_file() {
            return Ok(candidate);
        }
    }
    Err(format!(
        "GPUPixel 运行时未找到（{}）。请放入 libgpupixel_rr 或 libgpupixel_pywrapper。",
        dir.display()
    ))
}

fn load_api(dir: &Path) -> Result<Api, String> {
    let path = resolve_library_path(dir)?;
    #[cfg(target_os = "windows")]
    {
        // 依赖 DLL（libgpupixel、vnn、glfw）与资源同目录解析。
        let _ = set_dll_directory(dir);
    }
    unsafe {
        let lib = Library::new(&path).map_err(|e| format!("加载 GPUPixel 失败：{e}"))?;
        let name = path
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("")
            .to_ascii_lowercase();
        let prefer_rr = name.contains("gpupixel_rr")
            || lib.get::<CreateFn>(b"gpupixel_rr_create\0").is_ok();

        if prefer_rr {
            if let Ok(create) = lib.get::<CreateFn>(b"gpupixel_rr_create\0") {
                let create = *create;
                let initialize: InitRrFn = *lib
                    .get(b"gpupixel_rr_initialize\0")
                    .map_err(|e| format!("缺少 gpupixel_rr_initialize：{e}"))?;
                let set_callbacks: VoidHandleFn = *lib
                    .get(b"gpupixel_rr_set_callbacks\0")
                    .map_err(|e| format!("缺少 gpupixel_rr_set_callbacks：{e}"))?;
                // 新 DLL 同时导出 set_parameters(8) 与 set_parameters_v1(6)；旧 DLL 仅有 6 参同名。
                let set_parameters = if lib
                    .get::<SetParamsV1Fn>(b"gpupixel_rr_set_parameters_v1\0")
                    .is_ok()
                {
                    let f: SetParamsFn = *lib
                        .get(b"gpupixel_rr_set_parameters\0")
                        .map_err(|e| format!("缺少 gpupixel_rr_set_parameters：{e}"))?;
                    SetParamsKind::V2(f)
                } else {
                    let f: SetParamsV1Fn = *lib
                        .get(b"gpupixel_rr_set_parameters\0")
                        .map_err(|e| format!("缺少 gpupixel_rr_set_parameters：{e}"))?;
                    log::warn!(
                        "[portrait] GPUPixel rr 为旧 ABI（无眼白）；请重编 libgpupixel_rr"
                    );
                    SetParamsKind::V1(f)
                };
                let run: RunFn = *lib
                    .get(b"gpupixel_rr_run\0")
                    .map_err(|e| format!("缺少 gpupixel_rr_run：{e}"))?;
                let release_output: VoidHandleFn = *lib
                    .get(b"gpupixel_rr_release_output\0")
                    .map_err(|e| format!("缺少 gpupixel_rr_release_output：{e}"))?;
                let destroy: VoidHandleFn = *lib
                    .get(b"gpupixel_rr_destroy\0")
                    .map_err(|e| format!("缺少 gpupixel_rr_destroy：{e}"))?;
                log::info!("[portrait] GPUPixel backend=rr path={}", path.display());
                return Ok(Api {
                    _lib: lib,
                    create,
                    initialize: InitKind::Rr(initialize),
                    set_callbacks,
                    set_parameters,
                    run,
                    release_output,
                    destroy: Some(destroy),
                });
            }
        }

        let create: CreateFn = *lib
            .get(b"GPUPixelWrapper_create\0")
            .map_err(|e| {
                format!(
                    "既无 gpupixel_rr_* 也无 GPUPixelWrapper_*（{}）：{e}",
                    path.display()
                )
            })?;
        let initialize: InitPyFn = *lib
            .get(b"GPUPixelWrapper_initialize\0")
            .map_err(|e| format!("缺少 GPUPixelWrapper_initialize：{e}"))?;
        let set_callbacks: VoidHandleFn = *lib
            .get(b"GPUPixelWrapper_setCallbacks\0")
            .map_err(|e| format!("缺少 GPUPixelWrapper_setCallbacks：{e}"))?;
        let set_parameters_v1: SetParamsV1Fn = *lib
            .get(b"GPUPixelWrapper_setParameters\0")
            .map_err(|e| format!("缺少 GPUPixelWrapper_setParameters：{e}"))?;
        let run: RunFn = *lib
            .get(b"GPUPixelWrapper_run\0")
            .map_err(|e| format!("缺少 GPUPixelWrapper_run：{e}"))?;
        let release_output: VoidHandleFn = *lib
            .get(b"GPUPixelWrapper_release\0")
            .map_err(|e| format!("缺少 GPUPixelWrapper_release：{e}"))?;
        log::info!(
            "[portrait] GPUPixel backend=pywrapper path={}",
            path.display()
        );
        Ok(Api {
            _lib: lib,
            create,
            initialize: InitKind::Py(initialize),
            set_callbacks,
            set_parameters: SetParamsKind::V1(set_parameters_v1),
            run,
            release_output,
            destroy: None,
        })
    }
}

#[cfg(target_os = "windows")]
fn set_dll_directory(dir: &Path) -> Result<(), String> {
    use std::os::windows::ffi::OsStrExt;
    let wide: Vec<u16> = dir
        .as_os_str()
        .encode_wide()
        .chain(std::iter::once(0))
        .collect();
    // Prefer AddDllDirectory / SetDllDirectory via winapi-less LoadLibrary path:
    // prepend to PATH for dependent DLL resolution.
    let dir_str = dir
        .to_str()
        .ok_or_else(|| "GPUPixel 资源路径含非 UTF-8".to_string())?;
    let path = std::env::var_os("PATH").unwrap_or_default();
    let mut new_path = std::ffi::OsString::from(dir_str);
    new_path.push(";");
    new_path.push(path);
    unsafe {
        // SetDllDirectoryW keeps dependent DLL lookup scoped.
        #[link(name = "kernel32")]
        unsafe extern "system" {
            fn SetDllDirectoryW(path: *const u16) -> i32;
        }
        if SetDllDirectoryW(wide.as_ptr()) == 0 {
            std::env::set_var("PATH", new_path);
        }
    }
    Ok(())
}

fn api() -> Result<&'static Api, String> {
    let dir = resource_directory().ok_or_else(|| {
        "GPUPixel 资源目录未配置（应用启动时应 set_resource_directory）".to_string()
    })?;
    match API.get_or_init(|| load_api(dir)) {
        Ok(api) => Ok(api),
        Err(e) => Err(e.clone()),
    }
}

/// 上游 `Util::getResourcePath` 在 Windows 只返回文件名，VNN／lookup 按 **进程 CWD** 解析。
/// Python 会话整段保持 chdir；此处在持锁的 with_session 内同样保持，run 结束再恢复。
struct CwdGuard(Option<PathBuf>);

impl Drop for CwdGuard {
    fn drop(&mut self) {
        if let Some(prev) = self.0.take() {
            let _ = std::env::set_current_dir(prev);
        }
    }
}

fn enter_resource_cwd(dir: &Path) -> Result<CwdGuard, String> {
    let prev = std::env::current_dir().ok();
    std::env::set_current_dir(dir).map_err(|e| {
        format!(
            "GPUPixel chdir 失败（{}）：{e}；需能读到 face_pc[1.0.0].vnnmodel",
            dir.display()
        )
    })?;
    // 启动时校验模型存在，避免 VNN 失败后 native AV。
    let model = dir.join("face_pc[1.0.0].vnnmodel");
    if !model.is_file() {
        return Err(format!(
            "缺少 VNN 模型：{}（请重新 package-rr / 检查 resources/gpupixel）",
            model.display()
        ));
    }
    Ok(CwdGuard(prev))
}

/// GLFW 隐藏窗口（HWND）归属创建线程，线程退出即被系统销毁；tokio blocking 线程空闲
/// 约 10s 就回收，之后换线程 wglMakeCurrent 失败，GL 调用直接访问违例（0xc0000005）。
/// 因此加载、create、initialize 与每次 run 都固定在同一常驻线程，并天然串行化。
fn worker() -> Result<&'static mpsc::Sender<BakeJob>, String> {
    WORKER
        .get_or_init(|| {
            let (tx, rx) = mpsc::channel::<BakeJob>();
            std::thread::Builder::new()
                .name("gpupixel-gl".into())
                .stack_size(8 * 1024 * 1024)
                .spawn(move || {
                    let mut session: Option<Session> = None;
                    while let Ok(job) = rx.recv() {
                        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(
                            || bake_on_worker(&mut session, &job.image, job.params),
                        ))
                        .unwrap_or_else(|_| Err("GPUPixel bake 发生 panic".into()));
                        let _ = job.reply.send(result);
                    }
                })
                .map(|_| tx)
                .map_err(|e| format!("GPUPixel 线程启动失败：{e}"))
        })
        .as_ref()
        .map_err(Clone::clone)
}

fn with_session<R>(
    slot: &mut Option<Session>,
    f: impl FnOnce(&Session) -> Result<R, String>,
) -> Result<R, String> {
    let api = api()?;
    let dir = resource_directory().ok_or("GPUPixel 资源目录未配置")?;
    // 必须覆盖 initialize + run：face_pc 常在首帧 landmark 时才打开。
    let _cwd = enter_resource_cwd(dir)?;
    if slot.is_none() {
        let handle = unsafe { (api.create)() };
        if handle.is_null() {
            return Err("GPUPixel create 失败".into());
        }
        let ok = match api.initialize {
            InitKind::Rr(init) => {
                let c_dir = CString::new(dir.to_string_lossy().as_bytes())
                    .map_err(|_| "GPUPixel 资源路径含 NUL".to_string())?;
                unsafe { init(handle, c_dir.as_ptr()) != 0 }
            }
            InitKind::Py(init) => unsafe { init(handle) },
        };
        if !ok {
            if let Some(destroy) = api.destroy {
                unsafe { destroy(handle) };
            }
            return Err(
                "GPUPixel initialize 失败（OpenGL/EGL 上下文或资源缺失）".into(),
            );
        }
        unsafe { (api.set_callbacks)(handle) };
        *slot = Some(Session { api, handle });
        log::info!(
            "[portrait] GPUPixel session ready cwd={}",
            dir.display()
        );
    }
    let session = slot.as_ref().ok_or("GPUPixel 会话未就绪")?;
    f(session)
}

/// 对 RGBA8 整图 bake；强度全 0 时原样返回。
pub fn bake_rgba(image: &RgbaImage, params: BeautyParams) -> Result<RgbaImage, String> {
    if !params.is_active() {
        return Ok(image.clone());
    }
    bake_owned(image.clone(), params)
}

/// 投递到 `gpupixel-gl` 线程并阻塞等待结果。
fn bake_owned(image: RgbaImage, params: BeautyParams) -> Result<RgbaImage, String> {
    log::info!(
        "[portrait] bake {}x{} beauty={:.1} white={:.1} thin={:.1} eye={:.1} lip={:.1} blush={:.1}",
        image.width(),
        image.height(),
        params.beauty,
        params.white,
        params.thin_face,
        params.big_eye,
        params.lipstick,
        params.blusher,
    );
    let (reply, result) = mpsc::channel();
    worker()?
        .send(BakeJob {
            image,
            params,
            reply,
        })
        .map_err(|_| "GPUPixel 线程已退出".to_string())?;
    result
        .recv()
        .map_err(|_| "GPUPixel 线程未返回结果".to_string())?
}

fn bake_on_worker(
    slot: &mut Option<Session>,
    image: &RgbaImage,
    params: BeautyParams,
) -> Result<RgbaImage, String> {
    if params.face_active() {
        bake_face_on_worker(slot, image, params)
    } else {
        Ok(image.clone())
    }
}

/// 桌面 GPUPixel 读回在奇数宽/高上会写坏堆（I420 按偶数假设）；提交前裁成偶数。
fn snap_even_rgba(image: &RgbaImage) -> RgbaImage {
    let w = image.width();
    let h = image.height();
    let ew = (w & !1).max(2);
    let eh = (h & !1).max(2);
    if ew == w && eh == h {
        return image.clone();
    }
    if ew <= w && eh <= h {
        return image::imageops::crop_imm(image, 0, 0, ew, eh).to_image();
    }
    let mut out = RgbaImage::from_pixel(ew, eh, image::Rgba([0, 0, 0, 0]));
    for (x, y, p) in image.enumerate_pixels() {
        out.put_pixel(x, y, *p);
    }
    out
}

/// 将偶数尺寸 bake 结果扩回原始奇数尺寸（末行／列复制边缘）。
fn restore_rgba_dims(baked: &RgbaImage, orig_w: u32, orig_h: u32) -> RgbaImage {
    if baked.width() == orig_w && baked.height() == orig_h {
        return baked.clone();
    }
    let mut out = RgbaImage::new(orig_w, orig_h);
    let bw = baked.width().max(1);
    let bh = baked.height().max(1);
    for y in 0..orig_h {
        for x in 0..orig_w {
            out.put_pixel(x, y, *baked.get_pixel(x.min(bw - 1), y.min(bh - 1)));
        }
    }
    out
}

fn bake_face_on_worker(
    slot: &mut Option<Session>,
    image: &RgbaImage,
    params: BeautyParams,
) -> Result<RgbaImage, String> {
    let orig_w = image.width();
    let orig_h = image.height();
    let needs_snap = orig_w < 2 || orig_h < 2 || orig_w % 2 != 0 || orig_h % 2 != 0;
    let snapped = needs_snap.then(|| {
        log::warn!(
            "[portrait] GPUPixel 输入 {}x{} 含奇数边，已裁成偶数以避免堆损坏",
            orig_w,
            orig_h
        );
        snap_even_rgba(image)
    });
    let src = snapped.as_ref().unwrap_or(image);
    let width = src.width() as i32;
    let height = src.height() as i32;
    let channels = 4i32;
    let input = src.as_raw();
    with_session(slot, |session| unsafe {
        match session.api.set_parameters {
            SetParamsKind::V2(f) => f(
                session.handle,
                params.beauty,
                params.white,
                params.thin_face,
                params.big_eye,
                params.lipstick,
                params.blusher,
                params.eye_brighten,
                params.eye_redness,
            ),
            SetParamsKind::V1(f) => {
                if params.eye_brighten > 0.0 || params.eye_redness > 0.0 {
                    log::warn!(
                        "[portrait] 当前 GPUPixel DLL 不支持眼白；请执行 tools/gpupixel-ref/scripts/build.ps1 与 package-rr.ps1"
                    );
                }
                f(
                    session.handle,
                    params.beauty,
                    params.white,
                    params.thin_face,
                    params.big_eye,
                    params.lipstick,
                    params.blusher,
                );
            }
        };
        let ptr = (session.api.run)(
            session.handle,
            input.as_ptr(),
            width,
            height,
            channels,
        );
        if ptr.is_null() {
            return Err("GPUPixel run 返回空指针（可能未检出人脸或上下文失败）".into());
        }
        let len = (width as usize)
            .checked_mul(height as usize)
            .and_then(|n| n.checked_mul(4))
            .ok_or("GPUPixel 输出尺寸溢出")?;
        let slice = std::slice::from_raw_parts(ptr, len);
        let mut out = RgbaImage::new(src.width(), src.height());
        out.as_mut().copy_from_slice(slice);
        (session.api.release_output)(session.handle);
        validate_bake_output(src, &out)?;
        let out = if needs_snap {
            restore_rgba_dims(&out, orig_w, orig_h)
        } else {
            out
        };
        Ok(out)
    })
}

pub fn bake_dynamic(image: &DynamicImage, params: BeautyParams) -> Result<DynamicImage, String> {
    let rgba = image.to_rgba8();
    if !params.is_active() {
        return Ok(DynamicImage::ImageRgba8(rgba));
    }
    Ok(DynamicImage::ImageRgba8(bake_owned(rgba, params)?))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn params_inactive_when_zero() {
        let a = json!({"portrait":{},"sectionVisibility":{}});
        assert!(!requested(&a));
    }

    #[test]
    fn classic_beauty_white_ignored() {
        let a = json!({
            "portrait": {"beauty": 100, "white": 50},
            "sectionVisibility": {}
        });
        let p = params_from_adjustments(&a);
        assert!(!p.is_active());
        assert_eq!(p.beauty, 0.0);
        assert_eq!(p.white, 0.0);
    }

    #[test]
    fn params_map_ui_100_to_level_10() {
        let a = json!({
            "portrait": {
                "beauty": 100, "white": 50, "slimFace": 80, "bigEye": 20,
                "lipstick": 10, "blusher": 0, "eyeWhiteBrighten": 50, "eyeRedness": 30
            },
            "sectionVisibility": {}
        });
        let p = params_from_adjustments(&a);
        assert!(p.is_active());
        assert_eq!(p.beauty, 0.0);
        assert_eq!(p.white, 0.0);
        assert!((p.thin_face - 8.0).abs() < 1e-5);
        assert!((p.big_eye - 2.0).abs() < 1e-5);
        assert!((p.lipstick - 1.0).abs() < 1e-5);
        assert!((p.eye_brighten - 5.0).abs() < 1e-5);
        assert!((p.eye_redness - 3.0).abs() < 1e-5);
    }

    #[test]
    fn eye_white_alone_requests_gpupixel() {
        let a = json!({
            "portrait": {"eyeWhiteBrighten": 40, "eyeRedness": 0},
            "sectionVisibility": {}
        });
        assert!(requested(&a));
        let hidden = json!({
            "portrait": {"eyeWhiteBrighten": 100, "eyeRedness": 100},
            "sectionVisibility": {"portraitEyes": false}
        });
        assert!(!requested(&hidden));
    }

    #[test]
    fn section_visibility_gates_params() {
        let a = json!({
            "portrait": {"beauty": 100, "slimFace": 100, "lipstick": 100},
            "sectionVisibility": {"portraitSkin": false, "portraitReshape": false, "portraitMakeup": true}
        });
        let p = params_from_adjustments(&a);
        assert_eq!(p.beauty, 0.0);
        assert_eq!(p.thin_face, 0.0);
        assert!((p.lipstick - 10.0).abs() < 1e-5);
    }

    #[test]
    fn face_override_uv_slim_drives_runtime_thin_face() {
        let a = json!({
            "portrait": {"slimFace": 0, "bigEye": 0},
            "portraitReshape": {
                "schemaVersion": 1,
                "engineVersion": "gpupixel-runtime-v1",
                "enabled": true,
                "faces": [{
                    "faceId": "f1",
                    "overrides": {"uvSlimFace": 80, "bigEye": 40}
                }]
            },
            "sectionVisibility": {}
        });
        let p = params_from_adjustments(&a);
        assert!(p.is_active());
        assert!((p.thin_face - 8.0).abs() < 1e-5);
        assert!((p.big_eye - 4.0).abs() < 1e-5);
    }

    #[test]
    fn rejects_empty_or_black_runtime_frame_for_visible_input() {
        let input = RgbaImage::from_pixel(2, 1, image::Rgba([80, 40, 20, 255]));
        let transparent = RgbaImage::from_pixel(2, 1, image::Rgba([0, 0, 0, 0]));
        let opaque_black = RgbaImage::from_pixel(2, 1, image::Rgba([0, 0, 0, 255]));
        assert!(validate_bake_output(&input, &transparent).is_err());
        assert!(validate_bake_output(&input, &opaque_black).is_err());
        assert!(validate_bake_output(&input, &input).is_ok());
    }

    #[test]
    fn allows_legitimate_black_or_transparent_input() {
        for input in [
            RgbaImage::from_pixel(1, 1, image::Rgba([0, 0, 0, 255])),
            RgbaImage::from_pixel(1, 1, image::Rgba([0, 0, 0, 0])),
        ] {
            assert!(validate_bake_output(&input, &input).is_ok());
        }
    }

    #[test]
    fn snap_even_rgba_crops_odd_edges() {
        let input = RgbaImage::from_pixel(960, 639, image::Rgba([10, 20, 30, 255]));
        let snapped = snap_even_rgba(&input);
        assert_eq!(snapped.dimensions(), (960, 638));
        let restored = restore_rgba_dims(&snapped, 960, 639);
        assert_eq!(restored.dimensions(), (960, 639));
        assert_eq!(*restored.get_pixel(0, 638), image::Rgba([10, 20, 30, 255]));
    }

    #[test]
    fn snap_even_rgba_keeps_already_even() {
        let input = RgbaImage::from_pixel(640, 480, image::Rgba([1, 2, 3, 255]));
        let snapped = snap_even_rgba(&input);
        assert_eq!(snapped.dimensions(), (640, 480));
    }
}
