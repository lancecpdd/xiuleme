use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::time::UNIX_EPOCH;

use ndarray::{Array3, Array4, ArrayD, Ix4};
use ort::{session::Session, value::Tensor};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use tauri::Manager;

const MANIFEST: &str = include_str!("../resources/portrait-models.json");

#[derive(Deserialize)]
struct Manifest {
    version: u32,
    models: Vec<ModelSpec>,
    #[serde(default)]
    reshape: Option<serde_json::Value>,
}

#[derive(Deserialize)]
struct ModelSpec {
    id: String,
    filename: String,
    aliases: Vec<String>,
    sha256: String,
}

/// 只读取安装包内置资源；不复用其他 AI 功能的下载器。
pub fn resource_directory(app: &tauri::AppHandle) -> Result<PathBuf, String> {
    let packaged = app
        .path()
        .resolve("resources", tauri::path::BaseDirectory::Resource)
        .map_err(|e| format!("无法解析内置人像模型目录：{e}"))?;
    if packaged.is_dir() {
        return Ok(packaged);
    }
    #[cfg(debug_assertions)]
    {
        let development = Path::new(env!("CARGO_MANIFEST_DIR")).join("resources");
        if development.is_dir() {
            return Ok(development);
        }
    }
    Err(format!("缺少内置人像模型目录：{}", packaged.display()))
}

#[derive(Serialize, Deserialize)]
struct IntegrityStamp {
    len: u64,
    modified_secs: u64,
    sha256: String,
}

fn integrity_stamp_path(cache_directory: &Path, spec: &ModelSpec) -> PathBuf {
    let hash = spec.sha256.get(..16).unwrap_or(&spec.sha256);
    cache_directory.join(format!("{}-{}.integrity", spec.id, hash))
}

fn model_file_fingerprint(path: &Path) -> Result<(u64, u64), String> {
    let metadata = std::fs::metadata(path)
        .map_err(|e| format!("无法读取人像模型元数据 {}：{e}", path.display()))?;
    let modified_secs = metadata
        .modified()
        .ok()
        .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
        .map(|d| d.as_secs())
        .unwrap_or(0);
    Ok((metadata.len(), modified_secs))
}

fn verify_model(
    directory: &Path,
    spec: &ModelSpec,
    stamp_directory: Option<&Path>,
) -> Result<PathBuf, String> {
    let path = std::iter::once(&spec.filename)
        .chain(&spec.aliases)
        .map(|name| directory.join(name))
        .find(|path| path.is_file())
        .ok_or_else(|| format!("缺少内置人像模型：{}", spec.filename))?;
    let (len, modified_secs) = model_file_fingerprint(&path)?;
    if let Some(stamp_directory) = stamp_directory {
        let stamp_path = integrity_stamp_path(stamp_directory, spec);
        if let Ok(raw) = std::fs::read_to_string(&stamp_path)
            && let Ok(stamp) = serde_json::from_str::<IntegrityStamp>(&raw)
            && stamp.len == len
            && stamp.modified_secs == modified_secs
            && stamp.sha256 == spec.sha256
        {
            return Ok(path);
        }
    }
    let mut file =
        File::open(&path).map_err(|e| format!("无法读取人像模型 {}：{e}", path.display()))?;
    let mut hash = Sha256::new();
    let mut buffer = [0u8; 65536];
    loop {
        let count = file
            .read(&mut buffer)
            .map_err(|e| format!("读取人像模型 {} 失败：{e}", path.display()))?;
        if count == 0 {
            break;
        }
        hash.update(&buffer[..count]);
    }
    let digest = hex::encode(hash.finalize());
    if digest != spec.sha256 {
        return Err(format!("内置人像模型损坏或版本不匹配：{}", path.display()));
    }
    if let Some(stamp_directory) = stamp_directory {
        let _ = std::fs::create_dir_all(stamp_directory);
        let stamp = IntegrityStamp {
            len,
            modified_secs,
            sha256: digest,
        };
        if let Ok(raw) = serde_json::to_string(&stamp) {
            let _ = std::fs::write(integrity_stamp_path(stamp_directory, spec), raw);
        }
    }
    Ok(path)
}

/// 推理线程随 CPU 并行度自适应：人像分析串行执行，给缩略图与界面预览留出余量。
fn inference_threads() -> usize {
    std::thread::available_parallelism()
        .map(|value| value.get().saturating_sub(1).clamp(2, 6))
        .unwrap_or(2)
}

/// ORT 离线图优化缓存目录；不可用时返回 None，加载自动降级为无缓存。
pub fn model_cache_directory(app: &tauri::AppHandle) -> Option<PathBuf> {
    let directory = app.path().app_cache_dir().ok()?.join("portrait-models");
    std::fs::create_dir_all(&directory).ok()?;
    Some(directory)
}

/// 按 id 顺序从清单提取模型条目，缺失时报错。
fn manifest_models<'m>(manifest: &'m Manifest, ids: &[&str]) -> Result<Vec<&'m ModelSpec>, String> {
    ids.iter()
        .map(|id| {
            manifest
                .models
                .iter()
                .find(|spec| spec.id == *id)
                .ok_or_else(|| format!("人像模型清单缺少 {id}"))
        })
        .collect()
}

/// 并行校验模型文件完整性；保持严格 SHA256 语义，仅把串行哈希改为并行。
/// `stamp_directory` 命中时跳过全量哈希（仍核对长度／mtime／清单 hash）。
fn verify_models(
    directory: &Path,
    specs: &[&ModelSpec],
    stamp_directory: Option<&Path>,
) -> Result<Vec<PathBuf>, String> {
    specs
        .par_iter()
        .copied()
        .map(|spec| {
            let start = std::time::Instant::now();
            let path = verify_model(directory, spec, stamp_directory)?;
            log::info!(
                "[portrait] {} 校验 {:.1}ms（{}）",
                spec.id,
                start.elapsed().as_secs_f64() * 1000.0,
                spec.filename
            );
            Ok(path)
        })
        .collect()
}

/// 优化缓存文件名：含清单哈希前 16 位，模型更新后自动失效。
fn cache_filename(spec: &ModelSpec) -> String {
    let hash = spec.sha256.get(..16).unwrap_or(&spec.sha256);
    format!("{}-{}.onnx", spec.id, hash)
}

/// 删除属于给定清单但哈希已过期的旧优化缓存，避免模型更新后缓存堆积。
fn clean_stale_caches(directory: &Path, specs: &[&ModelSpec]) {
    let Ok(entries) = std::fs::read_dir(directory) else {
        return;
    };
    for entry in entries.flatten() {
        let name = entry.file_name();
        let Some(name) = name.to_str() else {
            continue;
        };
        let stale = specs.iter().any(|spec| {
            name.starts_with(&format!("{}-", spec.id))
                && name.ends_with(".onnx")
                && name != cache_filename(spec).as_str()
        });
        if stale {
            if let Err(e) = std::fs::remove_file(entry.path()) {
                log::warn!("[portrait] 清理旧优化缓存失败：{e}");
            }
        }
    }
}

/// 创建推理会话：优先读 ORT 离线优化缓存；未命中时生成缓存；缓存不可用时回退原始模型。
fn load_session(
    spec: &ModelSpec,
    path: &Path,
    cache_directory: Option<&Path>,
    threads: usize,
) -> Result<Session, String> {
    let start = std::time::Instant::now();
    if let Some(cache) = cache_directory.map(|directory| directory.join(cache_filename(spec))) {
        if cache.is_file() {
            match Session::builder()
                .and_then(|builder| builder.with_intra_threads(threads))
                .and_then(|builder| builder.commit_from_file(&cache))
            {
                Ok(session) => {
                    log::info!(
                        "[portrait] {} 会话 {:.1}ms（缓存命中）",
                        spec.id,
                        start.elapsed().as_secs_f64() * 1000.0
                    );
                    return Ok(session);
                }
                Err(e) => {
                    log::warn!(
                        "[portrait] {} 优化缓存不可用（{e}），删除后重新生成",
                        spec.id
                    );
                    let _ = std::fs::remove_file(&cache);
                }
            }
        }
        match Session::builder()
            .and_then(|builder| builder.with_intra_threads(threads))
            .and_then(|builder| builder.with_optimized_model_path(&cache))
            .and_then(|builder| builder.commit_from_file(path))
        {
            Ok(session) => {
                log::info!(
                    "[portrait] {} 会话 {:.1}ms（生成优化缓存）",
                    spec.id,
                    start.elapsed().as_secs_f64() * 1000.0
                );
                return Ok(session);
            }
            Err(e) => {
                log::warn!(
                    "[portrait] {} 生成优化缓存失败（{e}），回退原始模型",
                    spec.id
                );
                let _ = std::fs::remove_file(&cache);
            }
        }
    }
    let session = Session::builder()
        .and_then(|builder| builder.with_intra_threads(threads))
        .and_then(|builder| builder.commit_from_file(path))
        .map_err(|e| format!("无法加载内置人像模型 {}：{e}", path.display()))?;
    log::info!(
        "[portrait] {} 会话 {:.1}ms（无缓存）",
        spec.id,
        start.elapsed().as_secs_f64() * 1000.0
    );
    Ok(session)
}

/// 并行创建多个推理会话；结果顺序与 specs 一致，任一失败即整体失败。
fn load_sessions_parallel(
    specs: &[&ModelSpec],
    paths: &[PathBuf],
    cache_directory: Option<&Path>,
    threads: usize,
) -> Result<Vec<Session>, String> {
    std::thread::scope(|scope| {
        let handles: Vec<_> = specs
            .iter()
            .copied()
            .zip(paths)
            .map(|(spec, path)| {
                scope.spawn(move || load_session(spec, path, cache_directory, threads))
            })
            .collect();
        handles
            .into_iter()
            .map(|handle| {
                handle
                    .join()
                    .unwrap_or_else(|_| Err("人像模型加载线程异常".into()))
            })
            .collect()
    })
}

/// 后台预热被前台任务打断时返回的错误；调用方据此区分“让出”与真实失败。
pub const PREWARM_YIELDED: &str = "人像预热已让出";

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
pub enum ReshapeCapability {
    Detection,
    Landmarks,
    Protection,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct ReshapeModelManifest {
    version: u32,
    approved_models: Vec<serde_json::Value>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct ReshapeModelBinding {
    capability: ReshapeCapability,
    model_id: String,
    adapter: String,
    approval_status: String,
    reviewer: String,
    evidence: String,
}

const RESHAPE_LANDMARK_HASH: &str =
    "111795f8703cdeb6d0c68a9f3cc966a0f23f8786bb00f4577a11f461fc4276ac";
const RESHAPE_PROTECTION_HASH: &str =
    "534fd56e6c826e810a0e65d2e9a1d7bea4566c71ca354f3b2663e3a470acd738";
const RESHAPE_DETECTION_HASH: &str =
    "410b1c028c4c2b1c46909b6b212a2a4540276462285339c61bfc2a8c6f9ab317";

fn reshape_model_spec(
    manifest: &Manifest,
    capability: ReshapeCapability,
) -> Result<&ModelSpec, String> {
    if manifest.version != 1 {
        return Err("不支持的人像模型清单版本".into());
    }
    let raw = manifest.reshape.as_ref().ok_or("缺少重塑能力清单")?;
    let reshape: ReshapeModelManifest =
        serde_json::from_value(raw.clone()).map_err(|e| format!("重塑能力清单非法：{e}"))?;
    if reshape.version != 1 {
        return Err("不支持的重塑模型清单版本".into());
    }
    let key = match capability {
        ReshapeCapability::Detection => "detection",
        ReshapeCapability::Landmarks => "landmarks",
        ReshapeCapability::Protection => "protection",
    };
    let bindings: Vec<_> = reshape
        .approved_models
        .iter()
        .filter(|b| b["capability"] == key)
        .collect();
    if bindings.len() != 1 {
        return Err(format!("{capability:?} 缺少唯一、已核准的重塑模型"));
    }
    // 只解析被请求的能力；其他能力的无效绑定不能拖累当前会话或旧美肤加载。
    let binding: ReshapeModelBinding = serde_json::from_value(bindings[0].clone())
        .map_err(|e| format!("{capability:?} 绑定非法：{e}"))?;
    if binding.capability != capability {
        return Err("重塑能力绑定不一致".into());
    }
    if binding.approval_status != "approved"
        || binding.reviewer.trim().is_empty()
        || binding.evidence.trim().is_empty()
    {
        return Err(format!("{capability:?} 模型缺少维护者授权审批证据"));
    }
    let (adapter, expected_hash) = match capability {
        ReshapeCapability::Landmarks => ("face-mesh-478-v1", RESHAPE_LANDMARK_HASH),
        ReshapeCapability::Protection => ("selfie-multiclass-logits-v1", RESHAPE_PROTECTION_HASH),
        ReshapeCapability::Detection => ("blazeface-192-anchors-v1", RESHAPE_DETECTION_HASH),
    };
    let specs = manifest_models(manifest, &[binding.model_id.as_str()])?;
    let spec = specs[0];
    if binding.adapter != adapter || spec.sha256 != expected_hash {
        return Err(format!("{capability:?} 模型哈希与推理适配版本不匹配"));
    }
    // 新通路只接受资源目录中的单个文件名；不能借审批条目读取任意路径。
    for name in std::iter::once(&spec.filename).chain(&spec.aliases) {
        if name.is_empty() || name.contains(['/', '\\', ':']) || name == "." || name == ".." {
            return Err(format!("{capability:?} 模型文件名非法"));
        }
    }
    Ok(spec)
}

struct LazyReshapeSessions<T> {
    detection: Option<T>,
    landmarks: Option<T>,
    protection: Option<T>,
}

impl<T> LazyReshapeSessions<T> {
    fn new() -> Self {
        Self {
            detection: None,
            landmarks: None,
            protection: None,
        }
    }

    fn get_or_try_init(
        &mut self,
        capability: ReshapeCapability,
        load: impl FnOnce() -> Result<T, String>,
    ) -> Result<&mut T, String> {
        let slot = match capability {
            ReshapeCapability::Detection => &mut self.detection,
            ReshapeCapability::Landmarks => &mut self.landmarks,
            ReshapeCapability::Protection => &mut self.protection,
        };
        if slot.is_none() {
            *slot = Some(load().map_err(|e| format!("{capability:?}：{e}"))?)
        }
        slot.as_mut()
            .ok_or_else(|| format!("{capability:?} 会话未就绪"))
    }
}

/// 构造不读取任何权重；每项能力独立加载。不会实例化 PortraitModels 或加载 parsing/retouch。
pub struct ReshapeModels {
    directory: PathBuf,
    cache_directory: Option<PathBuf>,
    sessions: LazyReshapeSessions<Session>,
}

impl ReshapeModels {
    /// 仅核对已审批能力与适配绑定，不访问文件或创建 Session。
    pub fn approved_model_hashes() -> Result<std::collections::BTreeMap<String, String>, String> {
        let manifest: Manifest = serde_json::from_str(MANIFEST).map_err(|e| e.to_string())?;
        [
            ("detection", ReshapeCapability::Detection),
            ("landmarks", ReshapeCapability::Landmarks),
            ("protection", ReshapeCapability::Protection),
        ]
        .into_iter()
        .map(|(name, capability)| {
            let spec = reshape_model_spec(&manifest, capability)?;
            Ok((name.to_owned(), spec.sha256.clone()))
        })
        .collect()
    }

    pub fn new(directory: &Path, cache_directory: Option<&Path>) -> Self {
        Self {
            directory: directory.to_owned(),
            cache_directory: cache_directory.map(Path::to_owned),
            sessions: LazyReshapeSessions::new(),
        }
    }

    fn session(&mut self, capability: ReshapeCapability) -> Result<&mut Session, String> {
        let directory = &self.directory;
        let cache_directory = self.cache_directory.as_deref();
        self.sessions.get_or_try_init(capability, || {
            let manifest: Manifest = serde_json::from_str(MANIFEST).map_err(|e| e.to_string())?;
            let spec = reshape_model_spec(&manifest, capability)?;
            let path = verify_model(directory, spec, cache_directory)?;
            let session = load_session(spec, &path, cache_directory, inference_threads())?;
            let (input, outputs): (&str, &[&str]) = match capability {
                ReshapeCapability::Landmarks => ("input", &["landmarks", "score"]),
                ReshapeCapability::Protection => ("input_29", &["Identity"]),
                ReshapeCapability::Detection => (
                    "input",
                    &["reshaped_regressor_face_4", "reshaped_classifier_face_4"],
                ),
            };
            if session.inputs.len() != 1
                || session.inputs[0].name != input
                || session.outputs.len() != outputs.len()
                || outputs
                    .iter()
                    .any(|name| !session.outputs.iter().any(|output| output.name == *name))
            {
                return Err("ONNX 输入／输出名称或数量不符合适配契约".into());
            }
            Ok(session)
        })
    }

    pub fn landmarks(
        &mut self,
        image: &image::Rgb32FImage,
        roi: FaceRoi,
    ) -> Result<crate::portrait_adapters::LandmarkSet, String> {
        use crate::portrait_adapters::{RgbInput, decode_landmarks, prepare_rgb};
        let input = prepare_rgb(image, Some(roi), 256, RgbInput::FaceMesh478)?;
        let output = infer(
            self.session(ReshapeCapability::Landmarks)?,
            input,
            &[("landmarks", &[1, 478, 3]), ("score", &[1, 1])],
            "重塑关键点",
        )?;
        decode_landmarks(&output[0], &output[1], RESHAPE_LANDMARK_HASH)
    }

    pub fn protection(
        &mut self,
        image: &image::Rgb32FImage,
    ) -> Result<crate::portrait_adapters::ProtectionMaps, String> {
        use crate::portrait_adapters::{RgbInput, ScoreDomain, decode_protection, prepare_rgb};
        let input = prepare_rgb(image, None, 256, RgbInput::SelfieMulticlass)?;
        let output = infer(
            self.session(ReshapeCapability::Protection)?,
            input,
            &[("Identity", &[1, 256, 256, 6])],
            "重塑保护分割",
        )?;
        decode_protection(&output[0], ScoreDomain::Logits)
    }
}

impl crate::portrait_adapters::ReshapeDetector for ReshapeModels {
    fn detect_tile(
        &mut self,
        image: &image::Rgb32FImage,
        tile: crate::portrait_adapters::DetectionTile,
    ) -> Result<Vec<crate::portrait_adapters::ReshapeDetection>, String> {
        use crate::portrait_adapters::{
            BLAZEFACE_ANCHOR_GRID, BLAZEFACE_INPUT_EDGE, BLAZEFACE_MIN_SCORE,
            BlazeFaceDecodeContract, ScoreDomain, blazeface_anchors, blazeface_tile_from_model,
            decode_blazeface, prepare_blazeface_tile,
        };
        // 预处理几何使用模型固有的 192 输入；tile.input_edge 只服务其他检测器。
        let input = prepare_blazeface_tile(image, tile)?;
        let output = infer(
            self.session(ReshapeCapability::Detection)?,
            input,
            &[
                (
                    "reshaped_regressor_face_4",
                    &[1, BLAZEFACE_ANCHOR_GRID * BLAZEFACE_ANCHOR_GRID, 16],
                ),
                (
                    "reshaped_classifier_face_4",
                    &[1, BLAZEFACE_ANCHOR_GRID * BLAZEFACE_ANCHOR_GRID, 1],
                ),
            ],
            "重塑人脸检测",
        )?;
        let contract = BlazeFaceDecodeContract {
            anchors: blazeface_anchors().to_vec(),
            scales: [BLAZEFACE_INPUT_EDGE as f32; 4],
            keypoint_count: 6,
            score_domain: ScoreDomain::Logits,
        };
        let detections = decode_blazeface(&output[0], &output[1], &contract, BLAZEFACE_MIN_SCORE)?;
        // 反 letterbox：模型归一化空间 → 块归一化（detect_tiles 负责全图映射与 NMS）。
        Ok(detections
            .into_iter()
            .map(|mut detection| {
                let [x0, y0] =
                    blazeface_tile_from_model(tile, detection.bounds[0], detection.bounds[1]);
                let [x1, y1] =
                    blazeface_tile_from_model(tile, detection.bounds[2], detection.bounds[3]);
                detection.bounds = [x0, y0, x1, y1];
                for point in &mut detection.keypoints {
                    *point = blazeface_tile_from_model(tile, point[0], point[1]);
                }
                detection
            })
            .collect())
    }
}

#[cfg(test)]
mod reshape_loading_tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn independent_lazy_sessions_and_retry_without_beauty_assets() {
        let mut sessions = LazyReshapeSessions::<usize>::new();
        let mut calls = 0;
        assert!(
            sessions
                .get_or_try_init(ReshapeCapability::Detection, || Err("缺少模型".into()))
                .is_err()
        );
        assert_eq!(
            *sessions
                .get_or_try_init(ReshapeCapability::Landmarks, || {
                    calls += 1;
                    Ok(478)
                })
                .unwrap(),
            478
        );
        assert_eq!(
            *sessions
                .get_or_try_init(ReshapeCapability::Landmarks, || {
                    calls += 1;
                    Ok(0)
                })
                .unwrap(),
            478
        );
        assert_eq!(calls, 1);
        assert!(sessions.protection.is_none());
        assert!(sessions.detection.is_none());
        assert_eq!(
            *sessions
                .get_or_try_init(ReshapeCapability::Protection, || Ok(6))
                .unwrap(),
            6
        );
        assert_eq!(
            *sessions
                .get_or_try_init(ReshapeCapability::Detection, || Ok(1))
                .unwrap(),
            1
        );
    }

    #[test]
    fn unapproved_and_version_mismatch_fail_before_file_access() {
        let manifest: Manifest = serde_json::from_str(MANIFEST).unwrap();
        // 三条维护者审批（landmarks／protection／detection）只做清单解析，不访问文件。
        for capability in [
            ReshapeCapability::Detection,
            ReshapeCapability::Landmarks,
            ReshapeCapability::Protection,
        ] {
            assert!(reshape_model_spec(&manifest, capability).is_ok());
        }
        // 适配器名与代码常量不一致：Detection 拒绝，其余能力不受影响。
        let mut raw: serde_json::Value = serde_json::from_str(MANIFEST).unwrap();
        for binding in raw["reshape"]["approvedModels"].as_array_mut().unwrap() {
            if binding["capability"] == "detection" {
                binding["adapter"] = json!("blazeface-wrong-v0");
            }
        }
        let wrong_adapter: Manifest = serde_json::from_value(raw).unwrap();
        assert!(reshape_model_spec(&wrong_adapter, ReshapeCapability::Detection).is_err());
        assert!(reshape_model_spec(&wrong_adapter, ReshapeCapability::Landmarks).is_ok());
        // 删除 detection 审批条目：Detection 拒绝，其余能力不受影响。
        let mut raw: serde_json::Value = serde_json::from_str(MANIFEST).unwrap();
        raw["reshape"]["approvedModels"]
            .as_array_mut()
            .unwrap()
            .retain(|binding| binding["capability"] != "detection");
        let without_detection: Manifest = serde_json::from_value(raw).unwrap();
        assert!(reshape_model_spec(&without_detection, ReshapeCapability::Detection).is_err());
        assert!(reshape_model_spec(&without_detection, ReshapeCapability::Landmarks).is_ok());
        let mut raw: serde_json::Value = serde_json::from_str(MANIFEST).unwrap();
        raw["reshape"]["version"] = json!(2);
        let invalid: Manifest = serde_json::from_value(raw).unwrap();
        assert!(reshape_model_spec(&invalid, ReshapeCapability::Landmarks).is_err());
        let mut raw: serde_json::Value = serde_json::from_str(MANIFEST).unwrap();
        raw["reshape"]["approvedModels"] = json!([{
            "capability":"landmarks","modelId":"landmarker","adapter":"face-mesh-478-v1",
            "approvalStatus":"pending-review","reviewer":"","evidence":""
        }]);
        let invalid: Manifest = serde_json::from_value(raw.clone()).unwrap();
        assert!(reshape_model_spec(&invalid, ReshapeCapability::Landmarks).is_err());
        raw["reshape"]["approvedModels"][0]["approvalStatus"] = json!("approved");
        raw["reshape"]["approvedModels"][0]["reviewer"] = json!("synthetic-test");
        raw["reshape"]["approvedModels"][0]["evidence"] = json!("synthetic-test-only");
        raw["reshape"]["approvedModels"]
            .as_array_mut()
            .unwrap()
            .push(json!({"capability":"protection","adapter":false}));
        let independent: Manifest = serde_json::from_value(raw.clone()).unwrap();
        assert!(reshape_model_spec(&independent, ReshapeCapability::Landmarks).is_ok());
        assert!(reshape_model_spec(&independent, ReshapeCapability::Protection).is_err());
        raw["models"][1]["sha256"] = json!("0".repeat(64));
        let invalid: Manifest = serde_json::from_value(raw).unwrap();
        assert!(reshape_model_spec(&invalid, ReshapeCapability::Landmarks).is_err());
    }

    #[test]
    #[ignore = "需要内置 BlazeFace ONNX 与 ORT_DYLIB_PATH；显式运行离线集成测试"]
    fn reshape_detection_runtime_contract() {
        use crate::portrait_adapters::{DetectionTile, detect_tiles};
        let directory = std::env::var_os("PORTRAIT_RESOURCE_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|| Path::new(env!("CARGO_MANIFEST_DIR")).join("resources"));
        let mut models = ReshapeModels::new(&directory, None);
        let image = image::Rgb32FImage::from_pixel(320, 240, image::Rgb([0.0; 3]));
        let started = std::time::Instant::now();
        let detections = detect_tiles(
            &mut models,
            &image,
            &[DetectionTile {
                rect: [0.0, 0.0, 320.0, 240.0],
                input_edge: 192,
            }],
            0.35,
        )
        .unwrap();
        eprintln!(
            "BlazeFace 检测链路完成：{:?}，候选 {} 个（纯色图不保证检出）",
            started.elapsed(),
            detections.len()
        );
    }

    #[test]
    fn missing_model_and_wrong_hash_do_not_initialize_ort() {
        let directory = Path::new(env!("CARGO_MANIFEST_DIR"));
        let mut spec = ModelSpec {
            id: "test".into(),
            filename: "missing-portrait-contract-fixture.onnx".into(),
            aliases: vec![],
            sha256: "0".repeat(64),
        };
        assert!(verify_model(directory, &spec, None).is_err());
        spec.filename = "Cargo.toml".into();
        assert!(
            verify_model(directory, &spec, None)
                .unwrap_err()
                .contains("版本不匹配")
        );
    }
}

/// Session 由独立的人像分析锁串行使用，避免同时推理大量人脸耗尽内存。
pub struct PortraitModels {
    pub detector: Session,
    pub landmarker: Session,
    pub parsing: Session,
    pub skin: Session,
    /// AI 美肤模型随 AI 功能懒加载，避免基础分析等待约 200MB 的额外模型。
    pub retouch_detection: Option<Session>,
    pub retouch_inpainting: Option<Session>,
    directory: PathBuf,
    cache_directory: Option<PathBuf>,
    generator: Option<Session>,
    retouch_skin: Option<Session>,
}

impl PortraitModels {
    /// 加载基础分析模型；祛痘/美肤模型由 load_retouch_models / load_beauty_models 懒加载。
    pub fn load(directory: &Path, cache_directory: Option<&Path>) -> Result<Self, String> {
        let manifest: Manifest = serde_json::from_str(MANIFEST).map_err(|e| e.to_string())?;
        if manifest.version != 1 {
            return Err("不支持的人像模型清单版本".into());
        }
        const MODEL_IDS: [&str; 4] = ["detector", "landmarker", "parsing", "skin"];
        let specs = manifest_models(&manifest, &MODEL_IDS)?;
        if let Some(cache) = cache_directory {
            clean_stale_caches(cache, &specs);
        }
        let threads = inference_threads();
        let load_start = std::time::Instant::now();
        log::info!(
            "[portrait] 基础模型加载开始（{} 个，推理线程 {threads}）",
            specs.len()
        );
        let verify_start = std::time::Instant::now();
        let paths = verify_models(directory, &specs, cache_directory)?;
        let verify_ms = verify_start.elapsed().as_secs_f64() * 1000.0;
        let mut sessions =
            load_sessions_parallel(&specs, &paths, cache_directory, threads)?.into_iter();
        let detector = sessions.next().ok_or("人像模型顺序错误")?;
        let landmarker = sessions.next().ok_or("人像模型顺序错误")?;
        let parsing = sessions.next().ok_or("人像模型顺序错误")?;
        let skin = sessions.next().ok_or("人像模型顺序错误")?;
        log::info!(
            "[portrait] 基础模型加载完成：校验合计 {verify_ms:.1}ms（并行），总耗时 {:.1}ms（会话并行）",
            load_start.elapsed().as_secs_f64() * 1000.0
        );
        Ok(Self {
            detector,
            landmarker,
            parsing,
            skin,
            retouch_detection: None,
            retouch_inpainting: None,
            directory: directory.to_owned(),
            cache_directory: cache_directory.map(|path| path.to_owned()),
            generator: None,
            retouch_skin: None,
        })
    }
}

/// 同时检查真实运行时张量形状和数值；ONNX 的静态符号维度不足以验证契约。
pub fn infer(
    session: &mut Session,
    input: Array4<f32>,
    expected: &[(&str, &[usize])],
    model: &str,
) -> Result<Vec<ArrayD<f32>>, String> {
    crate::portrait_adapters::validate_tensor(input.view().into_dyn(), &[0, 0, 0, 0], model)?;
    let tensor = Tensor::from_array(input).map_err(|e| format!("{model} 输入失败：{e}"))?;
    let outputs = session
        .run(ort::inputs![tensor])
        .map_err(|e| format!("{model} 推理失败：{e}"))?;
    let mut result = Vec::with_capacity(expected.len());
    for (name, shape) in expected {
        let value = outputs
            .get(*name)
            .ok_or_else(|| format!("{model} 缺少输出 {name}"))?;
        let array = value
            .try_extract_array::<f32>()
            .map_err(|e| format!("{model} 输出 {name} 不是 float32：{e}"))?;
        if array.is_empty()
            || array.ndim() != shape.len()
            || array
                .shape()
                .iter()
                .zip(*shape)
                .any(|(actual, expected)| *expected != 0 && actual != expected)
        {
            return Err(format!(
                "{model} 输出 {name} 形状异常：{:?}，预期 {shape:?}",
                array.shape()
            ));
        }
        if array.iter().any(|v| !v.is_finite()) {
            return Err(format!("{model} 输出 {name} 包含非法数值"));
        }
        result.push(array.to_owned());
    }
    Ok(result)
}

#[derive(Clone, Copy, Debug)]
pub struct FaceRoi {
    pub center: [f32; 2],
    pub side: f32,
    pub angle: f32,
}

impl FaceRoi {
    pub fn project(&self, u: f32, v: f32) -> [f32; 2] {
        let (s, c) = self.angle.sin_cos();
        let x = (u - 0.5) * self.side;
        let y = (v - 0.5) * self.side;
        [
            self.center[0] + c * x - s * y,
            self.center[1] + s * x + c * y,
        ]
    }

    pub fn unproject(&self, x: f32, y: f32) -> [f32; 2] {
        let (s, c) = self.angle.sin_cos();
        let x = x - self.center[0];
        let y = y - self.center[1];
        [
            (c * x + s * y) / self.side + 0.5,
            (-s * x + c * y) / self.side + 0.5,
        ]
    }
}

#[derive(Clone)]
struct Detection {
    bounds: [f32; 4],
    eyes: [[f32; 2]; 2],
    confidence: f32,
}

impl Detection {
    fn roi(&self) -> FaceRoi {
        FaceRoi {
            center: [
                (self.bounds[0] + self.bounds[2]) * 0.5,
                (self.bounds[1] + self.bounds[3]) * 0.5,
            ],
            side: (self.bounds[2] - self.bounds[0]).max(self.bounds[3] - self.bounds[1]) * 1.5,
            angle: (self.eyes[1][1] - self.eyes[0][1]).atan2(self.eyes[1][0] - self.eyes[0][0]),
        }
    }
}

pub struct FaceAnalysis {
    pub roi: FaceRoi,
    /// 检测阶段在分析图中的轴对齐人脸框 [x0,y0,x1,y1]。
    pub bounds: [f32; 4],
    /// 归一化到旋正裁剪，包含 478 个三维点。
    pub points: Vec<[f32; 3]>,
    /// 512×512；只保留皮肤、鼻子、耳部、颈部，其他类别为零。
    pub skin: Vec<f32>,
    pub contour_reliable: bool,
}

pub struct Segmentation {
    /// 256×256 身体皮肤概率，不将脸部皮肤用作身体修容。
    pub body: Vec<f32>,
    pub faces: Vec<FaceAnalysis>,
    pub skipped_faces: usize,
    pub rejected_faces: Vec<FaceRoi>,
}

pub fn sample_rgb(image: &image::Rgb32FImage, x: f32, y: f32) -> [f32; 3] {
    if x < 0.0 || y < 0.0 || x >= image.width() as f32 || y >= image.height() as f32 {
        return [0.0; 3];
    }
    let x = (x - 0.5).max(0.0);
    let y = (y - 0.5).max(0.0);
    let x0 = x.floor() as u32;
    let y0 = y.floor() as u32;
    let x1 = (x0 + 1).min(image.width() - 1);
    let y1 = (y0 + 1).min(image.height() - 1);
    let fx = x.fract();
    let fy = y.fract();
    std::array::from_fn(|c| {
        let top = image.get_pixel(x0, y0)[c] * (1.0 - fx) + image.get_pixel(x1, y0)[c] * fx;
        let bottom = image.get_pixel(x0, y1)[c] * (1.0 - fx) + image.get_pixel(x1, y1)[c] * fx;
        top * (1.0 - fy) + bottom * fy
    })
}

/// 并行构造 NCHW 三通道输入；map 决定每个通道取值方式。
fn nchw_input(
    width: usize,
    height: usize,
    model: &str,
    sample: impl Fn(usize, usize) -> [f32; 3] + Sync,
    map: impl Fn(usize, [f32; 3]) -> f32 + Sync,
) -> Result<Array4<f32>, String> {
    let plane = width * height;
    let mut data = vec![0f32; 3 * plane];
    let (r_plane, rest) = data.split_at_mut(plane);
    let (g_plane, b_plane) = rest.split_at_mut(plane);
    r_plane
        .par_chunks_mut(width)
        .zip(g_plane.par_chunks_mut(width))
        .zip(b_plane.par_chunks_mut(width))
        .enumerate()
        .for_each(|(y, ((r_row, g_row), b_row))| {
            for x in 0..width {
                let rgb = sample(x, y);
                r_row[x] = map(0, rgb);
                g_row[x] = map(1, rgb);
                b_row[x] = map(2, rgb);
            }
        });
    Array4::from_shape_vec((1, 3, height, width), data)
        .map_err(|e| format!("{model} 输入构建失败：{e}"))
}

/// RetinaFace 输入：按尺度缩放到 32 倍数画布，BGR 通道减均值。
fn detector_input(
    image: &image::Rgb32FImage,
    scale: f32,
    width: usize,
    height: usize,
) -> Result<Array4<f32>, String> {
    nchw_input(
        width,
        height,
        "RetinaFace",
        |x, y| sample_rgb(image, (x as f32 + 0.5) / scale, (y as f32 + 0.5) / scale),
        |c, rgb| rgb[2 - c] * 255.0 - [104.0, 117.0, 123.0][c],
    )
}

/// 人脸 ROI 输入：旋正裁剪到 side×side 画布。
fn roi_nchw_input(
    image: &image::Rgb32FImage,
    roi: &FaceRoi,
    side: usize,
    map: impl Fn(usize, [f32; 3]) -> f32 + Sync,
) -> Result<Array4<f32>, String> {
    nchw_input(
        side,
        side,
        "Face ROI",
        |x, y| {
            let xy = roi.project(
                (x as f32 + 0.5) / side as f32,
                (y as f32 + 0.5) / side as f32,
            );
            sample_rgb(image, xy[0], xy[1])
        },
        map,
    )
}

/// selfie_multiclass 输入：256×256 NHWC。
fn selfie_input(image: &image::Rgb32FImage) -> Result<Array4<f32>, String> {
    let mut data = vec![0f32; 256 * 256 * 3];
    data.par_chunks_mut(256 * 3)
        .enumerate()
        .for_each(|(y, row)| {
            for x in 0..256 {
                let rgb = sample_rgb(
                    image,
                    (x as f32 + 0.5) * image.width() as f32 / 256.0,
                    (y as f32 + 0.5) * image.height() as f32 / 256.0,
                );
                row[x * 3] = rgb[0];
                row[x * 3 + 1] = rgb[1];
                row[x * 3 + 2] = rgb[2];
            }
        });
    Array4::from_shape_vec((1, 256, 256, 3), data)
        .map_err(|e| format!("Selfie Segmentation 输入构建失败：{e}"))
}

fn probability(values: impl Iterator<Item = f32> + Clone, selected: impl Fn(usize) -> bool) -> f32 {
    let max = values.clone().fold(f32::NEG_INFINITY, f32::max);
    let mut numerator = 0.0;
    let mut denominator = 0.0;
    for (i, value) in values.enumerate() {
        let e = (value - max).exp();
        denominator += e;
        if selected(i) {
            numerator += e;
        }
    }
    numerator / denominator.max(1e-20)
}

fn overlap(a: &Detection, b: &Detection) -> f32 {
    let width = (a.bounds[2].min(b.bounds[2]) - a.bounds[0].max(b.bounds[0])).max(0.0);
    let height = (a.bounds[3].min(b.bounds[3]) - a.bounds[1].max(b.bounds[1])).max(0.0);
    let area = |d: &Detection| (d.bounds[2] - d.bounds[0]) * (d.bounds[3] - d.bounds[1]);
    width * height / (area(a) + area(b) - width * height).max(1e-6)
}

/// cv_unet_skin-retouching 局部修复参数（与官方 skin_retouching_pipeline 一致）。
const RETOUCH_DET_INPUT: usize = 768;
const RETOUCH_PATCH: usize = 512;
const RETOUCH_PADDING: usize = 32;
const RETOUCH_WINDOW: usize = RETOUCH_PATCH + 2 * RETOUCH_PADDING;
const RETOUCH_MASK_LOW: f32 = 0.35;
const RETOUCH_MASK_HIGH: f32 = 0.5;

/// 对齐 torch F.interpolate(mode='bilinear', align_corners=True)。
fn resize_bilinear_align_corners(src: &Array4<f32>, h: usize, w: usize) -> Array4<f32> {
    let (_, channels, ih, iw) = src.dim();
    let mut out = Array4::zeros((1, channels, h, w));
    let scale_y = if h > 1 {
        (ih - 1) as f32 / (h - 1) as f32
    } else {
        0.0
    };
    let scale_x = if w > 1 {
        (iw - 1) as f32 / (w - 1) as f32
    } else {
        0.0
    };
    for y in 0..h {
        let sy = y as f32 * scale_y;
        let y0 = sy.floor() as usize;
        let fy = sy - y0 as f32;
        let y1 = (y0 + 1).min(ih - 1);
        for px in 0..w {
            let sx = px as f32 * scale_x;
            let x0 = sx.floor() as usize;
            let fx = sx - x0 as f32;
            let x1 = (x0 + 1).min(iw - 1);
            for c in 0..channels {
                let top = src[[0, c, y0, x0]] * (1.0 - fx) + src[[0, c, y0, x1]] * fx;
                let bottom = src[[0, c, y1, x0]] * (1.0 - fx) + src[[0, c, y1, x1]] * fx;
                out[[0, c, y, px]] = top * (1.0 - fy) + bottom * fy;
            }
        }
    }
    out
}

/// 对齐 torch F.interpolate(mode='nearest')。
fn resize_nearest(src: &Array4<f32>, h: usize, w: usize) -> Array4<f32> {
    let (_, channels, ih, iw) = src.dim();
    let mut out = Array4::zeros((1, channels, h, w));
    for y in 0..h {
        let sy = y * ih / h;
        for px in 0..w {
            let sx = px * iw / w;
            for c in 0..channels {
                out[[0, c, y, px]] = src[[0, c, sy, sx]];
            }
        }
    }
    out
}

/// PyTorch bilinear 的默认半像素中心语义（align_corners=false）。
fn resize_bilinear_half_pixel(src: &Array4<f32>, h: usize, w: usize) -> Array4<f32> {
    let (_, channels, ih, iw) = src.dim();
    Array4::from_shape_fn((1, channels, h, w), |(_, c, y, x)| {
        let sy = ((y as f32 + 0.5) * ih as f32 / h as f32 - 0.5).max(0.0);
        let sx = ((x as f32 + 0.5) * iw as f32 / w as f32 - 0.5).max(0.0);
        let (y0, x0) = ((sy as usize).min(ih - 1), (sx as usize).min(iw - 1));
        let (y1, x1) = ((y0 + 1).min(ih - 1), (x0 + 1).min(iw - 1));
        let top = src[[0, c, y0, x0]] * (1.0 - sx.fract()) + src[[0, c, y0, x1]] * sx.fract();
        let bottom = src[[0, c, y1, x0]] * (1.0 - sx.fract()) + src[[0, c, y1, x1]] * sx.fract();
        top * (1.0 - sy.fract()) + bottom * sy.fract()
    })
}

fn blemish_keep(probability: f32) -> f32 {
    if probability < RETOUCH_MASK_LOW {
        1.0
    } else if probability >= RETOUCH_MASK_HIGH {
        0.0
    } else {
        1.0 - probability
    }
}

fn retouch_rect(bounds: [f32; 4], width: u32, height: u32) -> (u32, u32, u32, u32) {
    let cx = (bounds[0] + bounds[2]) * 0.5;
    let cy = (bounds[1] + bounds[3]) * 0.5;
    let side = ((bounds[2] - bounds[0]).max(bounds[3] - bounds[1]) * 1.5).trunc();
    (
        (cx - side * 0.5).trunc().clamp(0.0, width as f32) as u32,
        (cy - side * 0.5).trunc().clamp(0.0, height as f32) as u32,
        (cx + side * 0.5).trunc().clamp(0.0, width as f32) as u32,
        (cy + side * 0.5).trunc().clamp(0.0, height as f32) as u32,
    )
}

/// 美白脸区椭圆羽化相对半径（对齐 OVAL 量级，略宽以压颈根硬边）。
const WHITENING_FACE_FEATHER: f32 = 0.08;
/// 脸权外扩到邻接身体皮肤的相对边长比例。
const WHITENING_FACE_EXPAND: f32 = 0.03;

fn smoothstep_f32(low: f32, high: f32, value: f32) -> f32 {
    let t = ((value - low) / (high - low)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// 椭圆软边脸权：ROI 内中心≈1、贴边≈0；ROI 外 `expand` 带内衰减正权（供身体皮肤衔接）。
fn whitening_face_weight(
    px: f32,
    py: f32,
    x0: u32,
    y0: u32,
    x1: u32,
    y1: u32,
) -> f32 {
    let rw = (x1 - x0) as f32;
    let rh = (y1 - y0) as f32;
    if rw < 1.0 || rh < 1.0 {
        return 0.0;
    }
    let cx = x0 as f32 + rw * 0.5;
    let cy = y0 as f32 + rh * 0.5;
    let rx = rw * 0.5;
    let ry = rh * 0.5;
    let nx = (px - cx) / rx;
    let ny = (py - cy) / ry;
    let r = (nx * nx + ny * ny).sqrt();
    if r <= 1.0 {
        return 1.0 - smoothstep_f32(1.0 - WHITENING_FACE_FEATHER, 1.0, r);
    }
    let expand = rw.min(rh) * WHITENING_FACE_EXPAND;
    if expand <= 1e-3 {
        return 0.0;
    }
    let closest_x = px.clamp(x0 as f32, (x1 - 1) as f32 + 0.999);
    let closest_y = py.clamp(y0 as f32, (y1 - 1) as f32 + 0.999);
    let dx = px - closest_x;
    let dy = py - closest_y;
    let dist = (dx * dx + dy * dy).sqrt();
    if dist >= expand {
        return 0.0;
    }
    let falloff = 1.0 - smoothstep_f32(0.0, expand, dist);
    // 外扩峰值压低，衔接 ROI 椭圆外缘（约 0），避免颈根外扩带过亮。
    falloff * 0.2
}

/// 按脸 ROI 软边(+)与身体蒙版(-)写美白分区场（不跑模型，便于单测）。
fn assign_whitening_regions(
    width: usize,
    height: usize,
    image_width: u32,
    image_height: u32,
    faces: &[[f32; 4]],
    body: &[f32],
) -> Result<Vec<f32>, String> {
    if body.len() != width * height {
        return Err("身体蒙版尺寸与分析图不一致".into());
    }
    let mut region = vec![0.0f32; width * height];
    for &bounds in faces {
        if bounds.iter().any(|v| !v.is_finite()) {
            continue;
        }
        let (x0, y0, x1, y1) = retouch_rect(bounds, image_width, image_height);
        if x1 <= x0 || y1 <= y0 {
            continue;
        }
        let expand = ((x1 - x0).min(y1 - y0) as f32 * WHITENING_FACE_EXPAND)
            .ceil()
            .max(1.0) as u32;
        let sx = x0.saturating_sub(expand);
        let sy = y0.saturating_sub(expand);
        let ex = (x1 + expand).min(width as u32);
        let ey = (y1 + expand).min(height as u32);
        for y in sy..ey {
            for x in sx..ex {
                let i = y as usize * width + x as usize;
                let w = whitening_face_weight(x as f32 + 0.5, y as f32 + 0.5, x0, y0, x1, y1);
                if w <= 0.0 {
                    continue;
                }
                // 外扩带仅落在身体皮肤上，避免背景被当成脸区。
                let outside = x < x0 || x >= x1 || y < y0 || y >= y1;
                if outside && body[i] <= 0.05 {
                    continue;
                }
                if w > region[i] {
                    region[i] = w;
                }
            }
        }
    }
    for (i, &weight) in body.iter().enumerate() {
        if region[i] >= 0.05 || weight <= 0.05 {
            continue;
        }
        region[i] = -weight.clamp(0.0, 1.0);
    }
    Ok(region)
}

fn blend_curve(rgb: f32, mask: f32) -> f32 {
    (1.0 - 2.0 * mask) * rgb * rgb + 2.0 * mask * rgb
}

fn diffuse_mask(h: usize, w: usize) -> Array4<f32> {
    let small = Array4::from_shape_fn((1, 1, 500, 500), |(_, _, y, x)| {
        (y.min(500 - y).min(x.min(500 - x)) as f32 / 20.0).min(1.0)
    });
    resize_bilinear_half_pixel(&small, h, w)
}

/// OpenCV 椭圆结构元、默认锚点与中性常量边界。每行滑动极值避免 O(k²) 扫描。
fn morph_ellipse(src: &[u8], w: usize, h: usize, kernel: usize, dilate: bool) -> Vec<u8> {
    use std::collections::VecDeque;
    let radius = kernel / 2;
    let neutral = if dilate { 0 } else { 255 };
    let mut out = vec![neutral; w * h];
    for ky in 0..kernel {
        let dy = ky as isize - radius as isize;
        let dx = (radius as f64 * (1.0 - (dy as f64 / radius as f64).powi(2)).max(0.0).sqrt())
            .round_ties_even() as usize;
        let left = dx;
        let right = dx.min(kernel - 1 - radius);
        for y in 0..h {
            let sy = y as isize + dy;
            if !(0..h as isize).contains(&sy) {
                continue;
            }
            let row = &src[sy as usize * w..(sy as usize + 1) * w];
            let mut queue: VecDeque<usize> = VecDeque::new();
            let mut next = 0;
            for x in 0..w {
                let end = (x + right + 1).min(w);
                while next < end {
                    while queue.back().is_some_and(|&i| {
                        if dilate {
                            row[i] <= row[next]
                        } else {
                            row[i] >= row[next]
                        }
                    }) {
                        queue.pop_back();
                    }
                    queue.push_back(next);
                    next += 1;
                }
                while queue.front().is_some_and(|&i| i < x.saturating_sub(left)) {
                    queue.pop_front();
                }
                let value = row[*queue.front().unwrap()];
                let pixel = &mut out[y * w + x];
                *pixel = if dilate {
                    (*pixel).max(value)
                } else {
                    (*pixel).min(value)
                };
            }
        }
    }
    out
}

fn reflect101(mut p: isize, size: usize) -> usize {
    if size == 1 {
        return 0;
    }
    while p < 0 || p >= size as isize {
        p = if p < 0 { -p } else { 2 * size as isize - p - 2 };
    }
    p as usize
}

/// OpenCV blur(20,20)：偶数核锚点为 10，默认 BORDER_REFLECT_101，输出仍为 uint8。
fn smooth_skin_mask(mask: &[u8], w: usize, h: usize, big: bool) -> Array4<f32> {
    let kernel = if big { 80 } else { 30 };
    let closed = morph_ellipse(
        &morph_ellipse(mask, w, h, kernel, true),
        w,
        h,
        kernel,
        false,
    );
    let mut horizontal = vec![0u32; w * h];
    for y in 0..h {
        for x in 0..w {
            horizontal[y * w + x] = (-10..10)
                .map(|dx| closed[y * w + reflect101(x as isize + dx, w)] as u32)
                .sum();
        }
    }
    Array4::from_shape_fn((1, 1, h, w), |(_, _, y, x)| {
        let sum: u32 = (-10..10)
            .map(|dy| horizontal[reflect101(y as isize + dy, h) * w + x])
            .sum();
        // OpenCV 的 uint8 均值滤波使用最近偶数舍入，不能一律在 .5 时向上取整。
        (sum as f64 / 400.0).round_ties_even() as f32 / 255.0
    })
}

/// pad 右、下到 divisor 的倍数（constant 0），与官方 pad_multiples 一致。
fn pad_to_multiple(src: &Array4<f32>, divisor: usize) -> Array4<f32> {
    let (_, channels, h, w) = src.dim();
    let ph = (divisor - h % divisor) % divisor;
    let pw = (divisor - w % divisor) % divisor;
    let mut out = Array4::zeros((1, channels, h + ph, w + pw));
    for y in 0..h {
        for px in 0..w {
            for c in 0..channels {
                out[[0, c, y, px]] = src[[0, c, y, px]];
            }
        }
    }
    out
}

/// AI 修复补丁：rect 为分析图中的轴对齐裁剪 [x0,y0,x1,y1]，与修复图逐像素对齐。
pub struct RetouchPatch {
    pub rect: (u32, u32, u32, u32),
    pub repaired: image::Rgb32FImage,
    pub weights: Vec<f32>,
}

impl PortraitModels {
    /// 懒加载局部祛痘模型（瑕疵检测 + 修复）；普通滑块与完整美肤共用。
    fn load_retouch_models(&mut self) -> Result<(), String> {
        if self.retouch_detection.is_some() && self.retouch_inpainting.is_some() {
            return Ok(());
        }
        let manifest: Manifest = serde_json::from_str(MANIFEST).map_err(|e| e.to_string())?;
        const MODEL_IDS: [&str; 2] = ["retouch_detection", "retouch_inpainting"];
        let specs = manifest_models(&manifest, &MODEL_IDS)?;
        if let Some(cache) = self.cache_directory.as_deref() {
            clean_stale_caches(cache, &specs);
        }
        let threads = inference_threads();
        let load_start = std::time::Instant::now();
        log::info!(
            "[portrait] 祛痘模型加载开始（{} 个，推理线程 {threads}）",
            specs.len()
        );
        let verify_start = std::time::Instant::now();
        let paths = verify_models(&self.directory, &specs, self.cache_directory.as_deref())?;
        let verify_ms = verify_start.elapsed().as_secs_f64() * 1000.0;
        let mut sessions =
            load_sessions_parallel(&specs, &paths, self.cache_directory.as_deref(), threads)?
                .into_iter();
        self.retouch_detection = Some(sessions.next().ok_or("祛痘模型顺序错误")?);
        self.retouch_inpainting = Some(sessions.next().ok_or("祛痘模型顺序错误")?);
        log::info!(
            "[portrait] 祛痘模型加载完成：校验合计 {verify_ms:.1}ms（并行），总耗时 {:.1}ms",
            load_start.elapsed().as_secs_f64() * 1000.0
        );
        Ok(())
    }

    /// 仅预热祛痘模型会话，不执行图像推理；供编辑器空闲期隐藏首次启用的模型加载延迟。
    pub fn prewarm_retouch_models(&mut self) -> Result<(), String> {
        self.load_retouch_models()
    }

    /// 懒加载匀肤 generator；均匀肤色滑块与完整美肤共用，不拉祛痘/美白。
    pub fn load_generator_only(&mut self) -> Result<(), String> {
        if self.generator.is_some() {
            return Ok(());
        }
        let manifest: Manifest = serde_json::from_str(MANIFEST).map_err(|e| e.to_string())?;
        const MODEL_IDS: [&str; 1] = ["retouch_generator"];
        let specs = manifest_models(&manifest, &MODEL_IDS)?;
        if let Some(cache) = self.cache_directory.as_deref() {
            clean_stale_caches(cache, &specs);
        }
        let threads = inference_threads();
        let load_start = std::time::Instant::now();
        log::info!(
            "[portrait] 匀肤模型加载开始（{} 个，推理线程 {threads}）",
            specs.len()
        );
        let verify_start = std::time::Instant::now();
        let paths = verify_models(&self.directory, &specs, self.cache_directory.as_deref())?;
        let verify_ms = verify_start.elapsed().as_secs_f64() * 1000.0;
        let mut sessions = Vec::with_capacity(specs.len());
        for (spec, path) in specs.iter().copied().zip(&paths) {
            sessions.push(load_session(
                spec,
                path,
                self.cache_directory.as_deref(),
                threads,
            )?);
        }
        self.generator = Some(sessions.into_iter().next().ok_or("匀肤模型顺序错误")?);
        log::info!(
            "[portrait] 匀肤模型加载完成：校验合计 {verify_ms:.1}ms（并行），总耗时 {:.1}ms",
            load_start.elapsed().as_secs_f64() * 1000.0
        );
        Ok(())
    }

    /// 懒加载美白分割模型；独立美白滑块与完整美肤共用，不拉祛痘/匀肤。
    pub fn load_whitening_model(&mut self) -> Result<(), String> {
        if self.retouch_skin.is_some() {
            return Ok(());
        }
        let manifest: Manifest = serde_json::from_str(MANIFEST).map_err(|e| e.to_string())?;
        const MODEL_IDS: [&str; 1] = ["retouch_skin"];
        let specs = manifest_models(&manifest, &MODEL_IDS)?;
        if let Some(cache) = self.cache_directory.as_deref() {
            clean_stale_caches(cache, &specs);
        }
        let threads = inference_threads();
        let load_start = std::time::Instant::now();
        log::info!(
            "[portrait] 美白分割模型加载开始（{} 个，推理线程 {threads}）",
            specs.len()
        );
        let verify_start = std::time::Instant::now();
        let paths = verify_models(&self.directory, &specs, self.cache_directory.as_deref())?;
        let verify_ms = verify_start.elapsed().as_secs_f64() * 1000.0;
        let mut sessions = Vec::with_capacity(specs.len());
        for (spec, path) in specs.iter().copied().zip(&paths) {
            sessions.push(load_session(
                spec,
                path,
                self.cache_directory.as_deref(),
                threads,
            )?);
        }
        self.retouch_skin = Some(sessions.into_iter().next().ok_or("美肤模型顺序错误")?);
        log::info!(
            "[portrait] 美白分割模型加载完成：校验合计 {verify_ms:.1}ms（并行），总耗时 {:.1}ms",
            load_start.elapsed().as_secs_f64() * 1000.0
        );
        Ok(())
    }

    /// 懒加载完整 AI 美肤模型（祛痘 + 匀肤 + 美白分割）；运行时由 beautify_interruptible 分步加载。
    #[cfg(test)]
    fn load_beauty_models(&mut self) -> Result<(), String> {
        self.load_retouch_models()?;
        self.load_generator_only()?;
        self.load_whitening_model()
    }

    /// 按 AI 一键美肤同源曲线对 `target` 做皮肤美白；蒙版始终取自 `mask_source`。
    pub fn apply_whitening_rgb(
        &mut self,
        mask_source: &image::Rgb32FImage,
        target: &mut image::Rgb32FImage,
        big: bool,
    ) -> Result<(), String> {
        if mask_source.dimensions() != target.dimensions() {
            return Err("美白蒙版源与目标尺寸不一致".into());
        }
        self.load_whitening_model()?;
        let mask = self.whitening_mask(mask_source, big)?;
        for (x, y, pixel) in target.enumerate_pixels_mut() {
            let weight = mask[[0, 0, y as usize, x as usize]];
            if weight == 0.0 {
                continue;
            }
            for c in 0..3 {
                let mg = 0.5 + weight * if c == 0 { 0.16 } else { 0.12 };
                pixel[c] = (blend_curve(pixel[c], mg).clamp(0.0, 1.0) * 255.0).floor() / 255.0;
            }
        }
        Ok(())
    }

    /// 独立美白：整图美白后按脸 ROI(+)/身体蒙版(-) 软边分区，供 GPU 分强度缩放。
    pub fn whitening_compose(
        &mut self,
        image: &image::Rgb32FImage,
        faces: &[[f32; 4]],
        body: &[f32],
    ) -> Result<(image::Rgb32FImage, Vec<f32>), String> {
        let (width, height) = (image.width() as usize, image.height() as usize);
        if body.len() != width * height {
            return Err("身体蒙版尺寸与分析图不一致".into());
        }
        let mut big = false;
        for &bounds in faces {
            if bounds.iter().any(|v| !v.is_finite()) {
                continue;
            }
            let (_, y0, _, y1) = retouch_rect(bounds, image.width(), image.height());
            big |= (y1 - y0) as f32 > image.height() as f32 * 0.4;
        }
        let mut output = image.clone();
        self.apply_whitening_rgb(image, &mut output, big)?;
        let region = assign_whitening_regions(
            width,
            height,
            image.width(),
            image.height(),
            faces,
            body,
        )?;
        Ok((output, region))
    }

    pub(crate) fn even_roi(
        &mut self,
        image: &image::Rgb32FImage,
    ) -> Result<image::Rgb32FImage, String> {
        let (w, h) = (image.width() as usize, image.height() as usize);
        let tensor = Array4::from_shape_fn((1, 3, h, w), |(_, c, y, x)| {
            image.get_pixel(x as u32, y as u32)[c] * 2.0 - 1.0
        });
        let input = resize_bilinear_half_pixel(&tensor, 512, 512);
        let output = infer(
            self.generator.as_mut().ok_or("匀肤模型未加载")?,
            input,
            &[("output", &[1, 3, 512, 512])],
            "Skin Retouch Generator",
        )?;
        let mg = output[0]
            .clone()
            .into_dimensionality::<Ix4>()
            .map_err(|e| e.to_string())?;
        if mg.iter().any(|&v| !(0.0..=1.0).contains(&v)) {
            return Err("匀肤蒙版不在 0–1 范围内".into());
        }
        let mg = resize_bilinear_half_pixel(&mg, h, w);
        let diffuse = diffuse_mask(h, w);
        Ok(image::Rgb32FImage::from_fn(w as u32, h as u32, |x, y| {
            image::Rgb(std::array::from_fn(|c| {
                let m = (mg[[0, c, y as usize, x as usize]] - 0.5)
                    * diffuse[[0, 0, y as usize, x as usize]]
                    + 0.5;
                // 官方在匀肤完成后转 uint8，局部修复与 generator 之间不量化。
                (blend_curve(image.get_pixel(x, y)[c], m).clamp(0.0, 1.0) * 255.0).floor() / 255.0
            }))
        }))
    }

    /// 均匀肤色：人脸 ROI 与身体皮肤连通域分别 `even_roi` 贴回。
    /// 返回 (匀肤后 sRGB 图, 区域符号场：脸 +1、身 -1、未改 0)。
    pub fn even_skin_compose(
        &mut self,
        image: &image::Rgb32FImage,
        faces: &[[f32; 4]],
        body: &[f32],
    ) -> Result<(image::Rgb32FImage, Vec<f32>), String> {
        let (width, height) = (image.width() as usize, image.height() as usize);
        if body.len() != width * height {
            return Err("身体蒙版尺寸与分析图不一致".into());
        }
        self.load_generator_only()?;
        let mut output = image.clone();
        let mut region = vec![0.0f32; width * height];

        for &bounds in faces {
            if bounds.iter().any(|v| !v.is_finite()) {
                continue;
            }
            let (x0, y0, x1, y1) = retouch_rect(bounds, image.width(), image.height());
            if x1 <= x0 || y1 <= y0 || x1 - x0 < 32 || y1 - y0 < 32 {
                continue;
            }
            let roi = image::imageops::crop_imm(image, x0, y0, x1 - x0, y1 - y0).to_image();
            let evened = self.even_roi(&roi)?;
            image::imageops::replace(&mut output, &evened, x0 as i64, y0 as i64);
            for y in y0..y1 {
                for x in x0..x1 {
                    region[y as usize * width + x as usize] = 1.0;
                }
            }
        }

        let mut visited = vec![false; width * height];
        for seed in 0..body.len() {
            if visited[seed] || body[seed] <= 0.05 || region[seed] > 0.0 {
                continue;
            }
            let mut indices = vec![seed];
            visited[seed] = true;
            let mut cursor = 0;
            let (mut x0, mut y0, mut x1, mut y1) = (width, height, 0usize, 0usize);
            while cursor < indices.len() {
                let i = indices[cursor];
                cursor += 1;
                let x = i % width;
                let y = i / width;
                x0 = x0.min(x);
                y0 = y0.min(y);
                x1 = x1.max(x + 1);
                y1 = y1.max(y + 1);
                for (nx, ny) in [
                    (x.wrapping_sub(1), y),
                    (x + 1, y),
                    (x, y.wrapping_sub(1)),
                    (x, y + 1),
                ] {
                    if nx < width && ny < height {
                        let j = ny * width + nx;
                        if !visited[j] && body[j] > 0.05 && region[j] <= 0.0 {
                            visited[j] = true;
                            indices.push(j);
                        }
                    }
                }
            }
            if indices.len() < 64 || x1 <= x0 || y1 <= y0 {
                continue;
            }
            let rw = (x1 - x0) as u32;
            let rh = (y1 - y0) as u32;
            if rw < 32 || rh < 32 {
                continue;
            }
            let roi = image::imageops::crop_imm(image, x0 as u32, y0 as u32, rw, rh).to_image();
            let evened = self.even_roi(&roi)?;
            for &i in &indices {
                let lx = i % width - x0;
                let ly = i / width - y0;
                let weight = body[i].clamp(0.0, 1.0);
                if weight <= 0.0 {
                    continue;
                }
                let before = image.get_pixel((i % width) as u32, (i / width) as u32).0;
                let after = evened.get_pixel(lx as u32, ly as u32).0;
                let blended = [
                    before[0] * (1.0 - weight) + after[0] * weight,
                    before[1] * (1.0 - weight) + after[1] * weight,
                    before[2] * (1.0 - weight) + after[2] * weight,
                ];
                output.put_pixel((i % width) as u32, (i / width) as u32, image::Rgb(blended));
                region[i] = -weight;
            }
        }

        Ok((output, region))
    }

    fn whitening_mask(
        &mut self,
        image: &image::Rgb32FImage,
        big: bool,
    ) -> Result<Array4<f32>, String> {
        let scale = 800.0 / image.width().max(image.height()) as f64;
        let w = (image.width() as f64 * scale) as usize;
        let h = (image.height() as f64 * scale) as usize;
        let (w, h) = (w.max(1), h.max(1));
        let mut input = Array3::zeros((h, w, 3));
        for y in 0..h {
            for x in 0..w {
                let rgb = sample_rgb(
                    image,
                    (x as f32 + 0.5) * image.width() as f32 / w as f32,
                    (y as f32 + 0.5) * image.height() as f32 / h as f32,
                );
                for c in 0..3 {
                    input[[y, x, c]] = (rgb[c].clamp(0.0, 1.0) * 255.0).round();
                }
            }
        }
        let session = self.retouch_skin.as_mut().ok_or("美白分割模型未加载")?;
        let outputs = session.run(ort::inputs!["input_image:0" => Tensor::from_array(input).map_err(|e| e.to_string())?])
            .map_err(|e| format!("美白皮肤分割失败：{e}"))?;
        let mask = outputs
            .get("output_png:0")
            .ok_or("美白分割缺少 output_png:0")?
            .try_extract_array::<u8>()
            .map_err(|e| format!("美白蒙版必须是 uint8：{e}"))?;
        if mask.shape() != [h, w] {
            return Err(format!("美白蒙版形状异常：{:?}", mask.shape()));
        }
        let values = mask.as_slice().ok_or("美白蒙版内存不连续")?;
        let smoothed = smooth_skin_mask(values, w, h, big);
        Ok(resize_bilinear_half_pixel(
            &smoothed,
            image.height() as usize,
            image.width() as usize,
        ))
    }

    /// 仅局部祛痘（检测 + 修复贴回），不含匀肤/美白；供普通祛斑祛痘滑块使用。
    #[cfg(test)]
    pub fn retouch_blemishes_only(
        &mut self,
        image: &image::Rgb32FImage,
        faces: &[[f32; 4]],
    ) -> Result<image::Rgb32FImage, String> {
        self.retouch_blemishes_only_interruptible(image, faces, &|| false)
    }

    pub fn retouch_blemishes_only_interruptible(
        &mut self,
        image: &image::Rgb32FImage,
        faces: &[[f32; 4]],
        should_stop: &dyn Fn() -> bool,
    ) -> Result<image::Rgb32FImage, String> {
        if faces.is_empty() {
            return Ok(image.clone());
        }
        self.load_retouch_models()?;
        let (output, _, _) =
            self.map_retouch_faces(image, faces, should_stop, |_, patch| Ok(patch.repaired))?;
        Ok(output)
    }

    /// 对人脸框逐张执行局部修复，再经 `map` 变换后贴回；返回 (输出图, 是否大脸, 是否处理过)。
    /// `should_stop` 在每张脸前及修复与 `map` 之间检查，返回 true 时以 PREWARM_YIELDED 中止。
    fn map_retouch_faces(
        &mut self,
        image: &image::Rgb32FImage,
        faces: &[[f32; 4]],
        should_stop: &dyn Fn() -> bool,
        mut map: impl FnMut(&mut Self, RetouchPatch) -> Result<image::Rgb32FImage, String>,
    ) -> Result<(image::Rgb32FImage, bool, bool), String> {
        let total_started = std::time::Instant::now();
        let mut output = image.clone();
        let mut big = false;
        let mut processed = false;
        let mut processed_faces = 0usize;
        for (face_index, &bounds) in faces.iter().enumerate() {
            if should_stop() {
                return Err(PREWARM_YIELDED.into());
            }
            let face_started = std::time::Instant::now();
            let Some(patch) = self.retouch_face_interruptible(image, bounds, should_stop)? else {
                continue;
            };
            if should_stop() {
                return Err(PREWARM_YIELDED.into());
            }
            log::debug!(
                "AI 局部修复覆盖像素：{} / {}",
                patch.weights.iter().filter(|&&v| v > 0.0).count(),
                patch.weights.len()
            );
            big |= patch.repaired.height() as f32 > image.height() as f32 * 0.4;
            let (x0, y0) = (patch.rect.0, patch.rect.1);
            let mapped = map(self, patch)?;
            image::imageops::replace(&mut output, &mapped, x0 as i64, y0 as i64);
            processed = true;
            processed_faces += 1;
            log::info!(
                "[portrait-bench] stage=retouch-face index={} total_faces={} elapsed_ms={:.1}",
                face_index + 1,
                faces.len(),
                face_started.elapsed().as_secs_f64() * 1000.0
            );
        }
        log::info!(
            "[portrait-bench] stage=retouch-all requested_faces={} processed_faces={} elapsed_ms={:.1}",
            faces.len(),
            processed_faces,
            total_started.elapsed().as_secs_f64() * 1000.0
        );
        Ok((output, big, processed))
    }

    /// 同一基底上的局部祛痘 → 匀肤回贴 → 一次皮肤美白；不在重叠人脸上累计效果。
    #[cfg(test)]
    pub fn beautify(
        &mut self,
        image: &image::Rgb32FImage,
        faces: &[[f32; 4]],
    ) -> Result<image::Rgb32FImage, String> {
        self.beautify_interruptible(image, faces, &|| false)
    }

    /// 可中断的 `beautify`：模型加载之间、逐脸修复前后与美白前检查 `should_stop`，
    /// 返回 true 时以 PREWARM_YIELDED 中止，已加载的模型保留供下次复用。
    pub fn beautify_interruptible(
        &mut self,
        image: &image::Rgb32FImage,
        faces: &[[f32; 4]],
        should_stop: &dyn Fn() -> bool,
    ) -> Result<image::Rgb32FImage, String> {
        if faces.is_empty() {
            return Ok(image.clone());
        }
        let check = || {
            if should_stop() {
                Err(String::from(PREWARM_YIELDED))
            } else {
                Ok(())
            }
        };
        check()?;
        self.load_retouch_models()?;
        check()?;
        self.load_generator_only()?;
        check()?;
        self.load_whitening_model()?;
        let (mut output, big, processed) =
            self.map_retouch_faces(image, faces, should_stop, |this, patch| {
                this.even_roi(&patch.repaired)
            })?;
        if !processed {
            return Ok(output);
        }
        check()?;
        self.apply_whitening_rgb(image, &mut output, big)?;
        Ok(output)
    }

    /// 粗检已有足够大的可用人脸时跳过 1280，避免人像半身／大头照多付约一整档推理。
    const COARSE_FACE_MIN_SIDE: f32 = 80.0;

    fn detect_at_edge(
        &mut self,
        image: &image::Rgb32FImage,
        edge: u32,
    ) -> Result<Vec<Detection>, String> {
        let scale = edge as f32 / image.width().max(image.height()) as f32;
        let width = ((image.width() as f32 * scale).ceil() as usize).div_ceil(32) * 32;
        let height = ((image.height() as f32 * scale).ceil() as usize).div_ceil(32) * 32;
        let sampling_start = std::time::Instant::now();
        let input = detector_input(image, scale, width, height)?;
        let sampling_ms = sampling_start.elapsed().as_secs_f64() * 1000.0;
        let count: usize = [8, 16, 32]
            .iter()
            .map(|s| (height / s) * (width / s) * 2)
            .sum();
        let infer_start = std::time::Instant::now();
        let output = infer(
            &mut self.detector,
            input,
            &[
                ("loc", &[1, count, 4]),
                ("conf", &[1, count, 2]),
                ("landmarks", &[1, count, 10]),
            ],
            "RetinaFace",
        )?;
        log::info!(
            "[portrait] RetinaFace@{edge}（{width}x{height}）采样 {sampling_ms:.1}ms，推理 {:.1}ms",
            infer_start.elapsed().as_secs_f64() * 1000.0
        );
        let loc = output[0].as_slice().ok_or("RetinaFace 输出不连续")?;
        let conf = output[1].as_slice().ok_or("RetinaFace 输出不连续")?;
        let landmarks = output[2].as_slice().ok_or("RetinaFace 输出不连续")?;
        let mut candidates = Vec::new();
        let mut index = 0;
        for (level, step) in [8, 16, 32].into_iter().enumerate() {
            for y in 0..height / step {
                for x in 0..width / step {
                    for size in [[16.0, 32.0], [64.0, 128.0], [256.0, 512.0]][level] {
                        let i = index;
                        index += 1;
                        if conf[i * 2 + 1] < 0.8 {
                            continue;
                        }
                        let cx = (x as f32 + 0.5) * step as f32;
                        let cy = (y as f32 + 0.5) * step as f32;
                        let bx = cx + loc[i * 4] * 0.1 * size;
                        let by = cy + loc[i * 4 + 1] * 0.1 * size;
                        let bw = (loc[i * 4 + 2] * 0.2).exp() * size;
                        let bh = (loc[i * 4 + 3] * 0.2).exp() * size;
                        if !bw.is_finite() || !bh.is_finite() || bw.min(bh) / scale < 24.0 {
                            continue;
                        }
                        candidates.push(Detection {
                            bounds: [
                                (bx - bw * 0.5) / scale,
                                (by - bh * 0.5) / scale,
                                (bx + bw * 0.5) / scale,
                                (by + bh * 0.5) / scale,
                            ],
                            eyes: std::array::from_fn(|eye| {
                                [
                                    (cx + landmarks[i * 10 + eye * 2] * 0.1 * size) / scale,
                                    (cy + landmarks[i * 10 + eye * 2 + 1] * 0.1 * size) / scale,
                                ]
                            }),
                            confidence: conf[i * 2 + 1],
                        });
                    }
                }
            }
        }
        Ok(candidates)
    }

    fn select_detections(mut candidates: Vec<Detection>) -> Vec<Detection> {
        candidates.sort_by(|a, b| b.confidence.total_cmp(&a.confidence));
        let mut selected: Vec<Detection> = Vec::new();
        for candidate in candidates {
            if selected.iter().all(|d| overlap(d, &candidate) < 0.4) {
                selected.push(candidate);
            }
        }
        selected
    }

    fn needs_fine_detection(faces: &[Detection]) -> bool {
        faces.is_empty()
            || faces.iter().all(|face| {
                (face.bounds[2] - face.bounds[0]).min(face.bounds[3] - face.bounds[1])
                    < Self::COARSE_FACE_MIN_SIDE
            })
    }

    fn detect(&mut self, image: &image::Rgb32FImage) -> Result<Vec<Detection>, String> {
        let mut candidates = self.detect_at_edge(image, 640)?;
        let coarse = Self::select_detections(candidates.clone());
        if Self::needs_fine_detection(&coarse) {
            log::info!(
                "[portrait] RetinaFace@1280：粗检无人脸或仅有小人脸（{}），补跑",
                coarse.len()
            );
            candidates.extend(self.detect_at_edge(image, 1280)?);
            Ok(Self::select_detections(candidates))
        } else {
            log::info!(
                "[portrait] RetinaFace@1280：跳过（粗检已检出 {} 张可用人脸）",
                coarse.len()
            );
            Ok(coarse)
        }
    }

    pub fn analyze(&mut self, image: &image::Rgb32FImage) -> Result<Segmentation, String> {
        if image.width() == 0 || image.height() == 0 {
            return Err("人像分析图像为空".into());
        }
        let skin_start = std::time::Instant::now();
        let input = selfie_input(image)?;
        let output = infer(
            &mut self.skin,
            input,
            &[("Identity", &[1, 256, 256, 6])],
            "Selfie Segmentation",
        )?;
        let values = output[0].as_slice().ok_or("Selfie 输出不连续")?;
        let body = values
            .chunks_exact(6)
            .map(|v| probability(v.iter().copied(), |c| c == 2))
            .collect();
        let skin_ms = skin_start.elapsed().as_secs_f64() * 1000.0;
        let detect_start = std::time::Instant::now();
        let detections = self.detect(image)?;
        log::info!(
            "[portrait] selfie 分割 {skin_ms:.1}ms，人脸检测 {:.1}ms（筛选出 {} 张脸）",
            detect_start.elapsed().as_secs_f64() * 1000.0,
            detections.len()
        );
        let mut faces = Vec::new();
        let mut skipped_faces = 0;
        let mut rejected_faces = Vec::new();
        for detection in detections {
            let roi = detection.roi();
            if !roi.side.is_finite() || roi.side < 32.0 {
                skipped_faces += 1;
                continue;
            }
            let landmark_start = std::time::Instant::now();
            let input = roi_nchw_input(image, &roi, 256, |c, rgb| rgb[c])?;
            let output = infer(
                &mut self.landmarker,
                input,
                &[("landmarks", &[1, 478, 3]), ("score", &[1, 1])],
                "Face Landmarker",
            )?;
            log::info!(
                "[portrait] Face Landmarker {:.1}ms",
                landmark_start.elapsed().as_secs_f64() * 1000.0
            );
            let score = output[1]
                .iter()
                .next()
                .copied()
                .unwrap_or(f32::NEG_INFINITY);
            if score < 2.197_224_6 {
                skipped_faces += 1;
                rejected_faces.push(roi);
                continue;
            }
            let points: Vec<[f32; 3]> = output[0]
                .as_slice()
                .ok_or("关键点输出不连续")?
                .chunks_exact(3)
                .map(|v| [v[0] / 256.0, v[1] / 256.0, v[2] / 256.0])
                .collect();
            if points[..468]
                .iter()
                .any(|p| p[0] < -0.1 || p[0] > 1.1 || p[1] < -0.1 || p[1] > 1.1)
            {
                skipped_faces += 1;
                rejected_faces.push(roi);
                continue;
            }
            let eye_width = (points[263][0] - points[33][0]).abs();
            let nose_ratio = (points[1][0] - points[33][0]) / eye_width.max(1e-5);
            let contour_reliable = eye_width > 0.1
                && (0.2..0.8).contains(&nose_ratio)
                && (points[33][2] - points[263][2]).abs() < eye_width * 0.65;
            let parsing_start = std::time::Instant::now();
            let input = roi_nchw_input(image, &roi, 512, |c, rgb| {
                (rgb[c] - [0.485, 0.456, 0.406][c]) / [0.229, 0.224, 0.225][c]
            })?;
            let output = infer(
                &mut self.parsing,
                input,
                &[("output", &[1, 19, 512, 512])],
                "BiSeNet",
            )?;
            log::info!(
                "[portrait] BiSeNet {:.1}ms",
                parsing_start.elapsed().as_secs_f64() * 1000.0
            );
            let logits = output[0].as_slice().ok_or("BiSeNet 输出不连续")?;
            let plane = 512 * 512;
            let skin = (0..plane)
                .map(|i| {
                    probability((0..19).map(|c| logits[c * plane + i]), |c| {
                        matches!(c, 1 | 7 | 8 | 10 | 14)
                    })
                })
                .collect();
            faces.push(FaceAnalysis {
                roi,
                bounds: detection.bounds,
                points,
                skin,
                contour_reliable,
            });
        }
        Ok(Segmentation {
            body,
            faces,
            skipped_faces,
            rejected_faces,
        })
    }

    /// 官方 cv_unet_skin-retouching 局部修复：768 双线性检测 + 576 窗口门控卷积修复。
    /// 输入 sRGB 0-1 裁剪，输出修复后 sRGB 与逐像素修改权重 (1-keep)。
    fn retouch_local(
        &mut self,
        roi: &image::Rgb32FImage,
        should_stop: &dyn Fn() -> bool,
    ) -> Result<(image::Rgb32FImage, Vec<f32>), String> {
        let total_started = std::time::Instant::now();
        let h = roi.height() as usize;
        let w = roi.width() as usize;
        if h == 0 || w == 0 {
            return Err("AI 祛痘裁剪为空".into());
        }
        // 只有检测输入缩至 768；局部修复保留分析图 ROI 的像素尺度。
        let (wh, ww) = (h, w);
        let mut tensor = Array4::zeros((1, 3, wh, ww));
        for y in 0..wh {
            for x in 0..ww {
                let rgb = roi.get_pixel(x as u32, y as u32).0;
                for c in 0..3 {
                    tensor[[0, c, y, x]] = (rgb[c] - 0.5) * 2.0;
                }
            }
        }
        // 1) 瑕疵检测：插值 768、sigmoid、双阈值 → keep 掩码（1=保留）。
        let detection_started = std::time::Instant::now();
        let det_input =
            resize_bilinear_align_corners(&tensor, RETOUCH_DET_INPUT, RETOUCH_DET_INPUT);
        let logits = infer(
            self.retouch_detection
                .as_mut()
                .ok_or("瑕疵检测模型未加载")?,
            det_input,
            &[("output", &[1, 1, RETOUCH_DET_INPUT, RETOUCH_DET_INPUT])],
            "Skin Retouch Detection",
        )?;
        let logits: Array4<f32> = logits[0]
            .clone()
            .into_dimensionality::<Ix4>()
            .map_err(|e| format!("Skin Retouch Detection 输出维度异常：{e}"))?;
        let prob = resize_nearest(&logits.mapv(|v| 1.0 / (1.0 + (-v).exp())), wh, ww);
        let mut keep = Array4::zeros((1, 1, wh, ww));
        for y in 0..wh {
            for x in 0..ww {
                let p = prob[[0, 0, y, x]];
                keep[[0, 0, y, x]] = blemish_keep(p);
            }
        }
        let detection_ms = detection_started.elapsed().as_secs_f64() * 1000.0;
        // 2) 修复：pad 到 512 倍数 → 576 窗口（32 上下文）→ 逐窗推理 → 聚合。
        let inpainting_started = std::time::Instant::now();
        let tensor_pad = pad_to_multiple(&tensor, RETOUCH_PATCH);
        let keep_pad = pad_to_multiple(&keep, RETOUCH_PATCH);
        let (ph, pw) = (tensor_pad.dim().2, tensor_pad.dim().3);
        let (gh, gw) = (ph / RETOUCH_PATCH, pw / RETOUCH_PATCH);
        let mut canvas = Array4::zeros((1, 3, gh * RETOUCH_PATCH, gw * RETOUCH_PATCH));
        // 所有窗口形状相同，复用宿主缓冲区，避免每个 576 窗口重复分配两块 ndarray。
        let mut keep_win = Array4::zeros((1, 1, RETOUCH_WINDOW, RETOUCH_WINDOW));
        let mut masked = Array4::zeros((1, 3, RETOUCH_WINDOW, RETOUCH_WINDOW));
        for i in 0..gh {
            for j in 0..gw {
                if should_stop() {
                    return Err(PREWARM_YIELDED.into());
                }
                let y0 = i * RETOUCH_PATCH;
                let x0 = j * RETOUCH_PATCH;
                keep_win.fill(0.0);
                masked.fill(0.0);
                for y in 0..RETOUCH_WINDOW {
                    for x in 0..RETOUCH_WINDOW {
                        let py = y0 as isize + y as isize - RETOUCH_PADDING as isize;
                        let px = x0 as isize + x as isize - RETOUCH_PADDING as isize;
                        let inside = py >= 0 && px >= 0 && (py as usize) < ph && (px as usize) < pw;
                        let k = if inside {
                            keep_pad[[0, 0, py as usize, px as usize]]
                        } else {
                            0.0
                        };
                        keep_win[[0, 0, y, x]] = k;
                        for c in 0..3 {
                            let v = if inside {
                                tensor_pad[[0, c, py as usize, px as usize]]
                            } else {
                                0.0
                            };
                            masked[[0, c, y, x]] = v * k;
                        }
                    }
                }
                let masked_tensor = Tensor::from_array(masked.clone())
                    .map_err(|e| format!("Skin Retouch Inpainting 输入失败：{e}"))?;
                let keep_tensor = Tensor::from_array(keep_win.clone())
                    .map_err(|e| format!("Skin Retouch Inpainting 掩码失败：{e}"))?;
                let outputs = self
                    .retouch_inpainting
                    .as_mut()
                    .ok_or("瑕疵修复模型未加载")?
                    .run(ort::inputs!["input" => masked_tensor, "mask" => keep_tensor])
                    .map_err(|e| format!("Skin Retouch Inpainting 推理失败：{e}"))?;
                let value = outputs
                    .get("output")
                    .ok_or("Skin Retouch Inpainting 缺少输出 output")?;
                let repaired = value
                    .try_extract_array::<f32>()
                    .map_err(|e| format!("Skin Retouch Inpainting 输出不是 float32：{e}"))?;
                if repaired.shape() != &[1, 3, RETOUCH_WINDOW, RETOUCH_WINDOW] {
                    return Err(format!(
                        "Skin Retouch Inpainting 输出形状异常：{:?}",
                        repaired.shape()
                    ));
                }
                if repaired.iter().any(|v| !v.is_finite()) {
                    return Err("Skin Retouch Inpainting 输出包含非法数值".into());
                }
                for y in RETOUCH_PADDING..RETOUCH_PADDING + RETOUCH_PATCH {
                    for x in RETOUCH_PADDING..RETOUCH_PADDING + RETOUCH_PATCH {
                        let k = keep_win[[0, 0, y, x]];
                        for c in 0..3 {
                            canvas[[
                                0,
                                c,
                                i * RETOUCH_PATCH + y - RETOUCH_PADDING,
                                j * RETOUCH_PATCH + x - RETOUCH_PADDING,
                            ]] = masked[[0, c, y, x]] + (1.0 - k) * repaired[[0, c, y, x]];
                        }
                    }
                }
            }
        }
        // 去掉窗口上下文并裁回 ROI；不再缩小或放大修复结果。
        let mut crop = Array4::zeros((1, 3, wh, ww));
        for y in 0..wh {
            for x in 0..ww {
                for c in 0..3 {
                    crop[[0, c, y, x]] = canvas[[0, c, y, x]];
                }
            }
        }
        let repaired = crop;
        let keep_resized = keep;
        let mut image_out = image::Rgb32FImage::new(w as u32, h as u32);
        let mut weights = vec![0.0; h * w];
        for y in 0..h {
            for x in 0..w {
                let rgb = [
                    (repaired[[0, 0, y, x]] * 0.5 + 0.5).clamp(0.0, 1.0),
                    (repaired[[0, 1, y, x]] * 0.5 + 0.5).clamp(0.0, 1.0),
                    (repaired[[0, 2, y, x]] * 0.5 + 0.5).clamp(0.0, 1.0),
                ];
                image_out.put_pixel(x as u32, y as u32, image::Rgb(rgb));
                weights[y * w + x] = (1.0 - keep_resized[[0, 0, y, x]]).clamp(0.0, 1.0);
            }
        }
        log::info!(
            "[portrait-bench] stage=retouch-roi roi={}x{} windows={} detection_ms={:.1} inpainting_ms={:.1} total_ms={:.1}",
            w,
            h,
            gh * gw,
            detection_ms,
            inpainting_started.elapsed().as_secs_f64() * 1000.0,
            total_started.elapsed().as_secs_f64() * 1000.0
        );
        Ok((image_out, weights))
    }

    /// 对一张检测人脸框执行 AI 局部修复：1.5 倍轴对齐外扩（官方 get_crop_bbox）。
    #[cfg(test)]
    pub fn retouch_face(
        &mut self,
        srgb: &image::Rgb32FImage,
        bounds: [f32; 4],
    ) -> Result<Option<RetouchPatch>, String> {
        self.retouch_face_interruptible(srgb, bounds, &|| false)
    }

    fn retouch_face_interruptible(
        &mut self,
        srgb: &image::Rgb32FImage,
        bounds: [f32; 4],
        should_stop: &dyn Fn() -> bool,
    ) -> Result<Option<RetouchPatch>, String> {
        if bounds.iter().any(|v| !v.is_finite()) {
            return Err("人脸框包含非有限值".into());
        }
        let (x0, y0, x1, y1) = retouch_rect(bounds, srgb.width(), srgb.height());
        if x1 <= x0 || y1 <= y0 || x1 - x0 < 32 || y1 - y0 < 32 {
            return Ok(None);
        }
        let roi = image::imageops::crop_imm(srgb, x0, y0, x1 - x0, y1 - y0).to_image();
        let (repaired, weights) = self.retouch_local(&roi, should_stop)?;
        Ok(Some(RetouchPatch {
            rect: (x0, y0, x1, y1),
            repaired,
            weights,
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn whitening_regions_feather_face_and_expand_onto_body() {
        let (width, height) = (64usize, 64usize);
        // 检测框经 ×1.5 后约落在图中部，便于取中心与底边样本。
        let faces = [[16.0_f32, 16.0, 48.0, 48.0]];
        let (x0, y0, x1, y1) = retouch_rect(faces[0], width as u32, height as u32);
        assert!(x1 > x0 && y1 > y0);
        let mut body = vec![0.0f32; width * height];
        for y in y1..height as u32 {
            for x in x0..x1 {
                body[y as usize * width + x as usize] = 0.9;
            }
        }
        let region = assign_whitening_regions(
            width,
            height,
            width as u32,
            height as u32,
            &faces,
            &body,
        )
        .unwrap();
        let cx = ((x0 + x1) / 2) as usize;
        let cy = ((y0 + y1) / 2) as usize;
        let center = region[cy * width + cx];
        assert!(
            center > 0.95,
            "ROI 中心应接近满脸权，got {center}"
        );
        let edge_y = (y1 - 1) as usize;
        let edge = region[edge_y * width + cx];
        assert!(
            edge > 0.0 && edge < 0.85,
            "ROI 底边应处于羽化带，got {edge}"
        );
        let expand = ((x1 - x0).min(y1 - y0) as f32 * WHITENING_FACE_EXPAND)
            .ceil()
            .max(1.0) as u32;
        assert!(expand >= 1);
        let out_y = (y1 as usize).min(height - 1);
        let expanded = region[out_y * width + cx];
        assert!(
            expanded > 0.0 && expanded < 0.25 && expanded <= edge + 0.05,
            "外扩带应为衰减正权，edge={edge} expanded={expanded}"
        );
        assert!(
            edge < center,
            "底边羽化应低于中心，center={center} edge={edge}"
        );
        // 外扩带之外的身体应为负权。
        let far_y = ((y1 + expand + 2).min(height as u32 - 1)) as usize;
        if far_y * width + cx < region.len() && body[far_y * width + cx] > 0.05 {
            assert!(
                region[far_y * width + cx] < 0.0,
                "外扩外身体应为身区负权"
            );
        }
    }

    #[test]
    fn whitening_face_weight_center_full_edge_soft() {
        let (x0, y0, x1, y1) = (10u32, 10, 50, 50);
        let cx = 30.5_f32;
        let cy = 30.5_f32;
        assert!((whitening_face_weight(cx, cy, x0, y0, x1, y1) - 1.0).abs() < 1e-5);
        let bottom = whitening_face_weight(cx, y1 as f32 - 0.5, x0, y0, x1, y1);
        assert!(bottom > 0.0 && bottom < 0.5, "底边羽化 got {bottom}");
        let outside = whitening_face_weight(cx, y1 as f32 + 0.5, x0, y0, x1, y1);
        assert!(
            outside > 0.0 && outside < 0.25,
            "外扩带衰减正权 got {outside}"
        );
        let far = whitening_face_weight(cx, y1 as f32 + 20.0, x0, y0, x1, y1);
        assert_eq!(far, 0.0);
    }

    #[test]
    fn retouch_interpolation_thresholds_and_crop() {
        let input = Array4::from_shape_vec((1, 1, 1, 3), vec![0.0, 1.0, 2.0]).unwrap();
        assert_eq!(
            resize_nearest(&input, 1, 5).as_slice().unwrap(),
            &[0., 0., 1., 1., 2.]
        );
        assert_eq!(resize_nearest(&input, 1, 2).as_slice().unwrap(), &[0., 1.]);
        assert_eq!(
            resize_bilinear_align_corners(&input, 1, 5)
                .as_slice()
                .unwrap(),
            &[0., 0.5, 1., 1.5, 2.]
        );
        let half = resize_bilinear_half_pixel(&input, 1, 5);
        for (a, b) in half.iter().zip([0., 0.4, 1., 1.6, 2.]) {
            assert!((a - b).abs() < 1e-6);
        }
        let one = Array4::from_elem((1, 1, 1, 1), 0.7);
        assert!(resize_nearest(&one, 3, 4).iter().all(|&v| v == 0.7));
        assert!(
            resize_bilinear_half_pixel(&one, 3, 4)
                .iter()
                .all(|&v| (v - 0.7).abs() < 1e-6)
        );
        assert_eq!(
            resize_bilinear_align_corners(&input, 1, 1)[[0, 0, 0, 0]],
            0.
        );
        assert_eq!(resize_bilinear_half_pixel(&input, 1, 1)[[0, 0, 0, 0]], 1.);
        for (p, expected) in [
            (0., 1.),
            (0.34999, 1.),
            (0.35, 0.65),
            (0.499, 0.501),
            (0.5, 0.),
            (1., 0.),
        ] {
            assert!((blemish_keep(p) - expected).abs() < 1e-6);
        }
        assert_eq!(
            retouch_rect([10.2, 20.8, 50.7, 81.1], 100, 100),
            (0, 5, 75, 95)
        );
        assert_eq!(retouch_rect([-10., -10., 30., 30.], 20, 20), (0, 0, 20, 20));
        let padded = pad_to_multiple(&input, 4);
        assert_eq!(padded.dim(), (1, 1, 4, 4));
        assert_eq!(padded[[0, 0, 0, 2]], 2.);
        assert_eq!(padded[[0, 0, 0, 3]], 0.);
        assert_eq!(padded[[0, 0, 3, 2]], 0.);
    }

    #[test]
    fn beauty_curve_diffuse_and_opencv_mask_contract() {
        for x in [0., 0.2, 0.5, 0.8, 1.] {
            assert_eq!(blend_curve(x, 0.5), x);
            assert!(blend_curve(x, 0.66) >= x);
        }
        let diffuse = diffuse_mask(500, 500);
        assert_eq!(diffuse[[0, 0, 0, 250]], 0.);
        assert_eq!(diffuse[[0, 0, 250, 250]], 1.);
        assert_eq!(diffuse[[0, 0, 499, 499]], 0.05);
        // 由 validate_retouch.py --fixtures-only 的独立 OpenCV 实现生成。
        let (w, h) = (97, 83);
        let input: Vec<u8> = (0..w * h)
            .map(|i| {
                let (x, y) = (i % w, i / w);
                if (23..51).contains(&y) && (35..61).contains(&x) {
                    177
                } else if (10..70).contains(&y) && (12..80).contains(&x) {
                    0
                } else {
                    ((x * 17 + y * 23) % 256) as u8
                }
            })
            .collect();
        for (kernel, closed_hash, blur_hash) in [
            (
                30,
                "1f422406582504d5dbcaeefeef38eb31f4dfbb1fba64817bad24252b068f6538",
                "ff487e3a0cc2cdcd734b7ed5402e95f57c5664eb9d8f4ad416a7445bfb4c845d",
            ),
            (
                80,
                "31131272a18181d688ab676a26277134763ccf639ee859cd730e1ec695abc551",
                "31131272a18181d688ab676a26277134763ccf639ee859cd730e1ec695abc551",
            ),
        ] {
            let closed = morph_ellipse(
                &morph_ellipse(&input, w, h, kernel, true),
                w,
                h,
                kernel,
                false,
            );
            assert_eq!(
                hex::encode(Sha256::digest(&closed)),
                closed_hash,
                "闭运算 {kernel}"
            );
            let blurred: Vec<u8> = smooth_skin_mask(&input, w, h, kernel == 80)
                .iter()
                .map(|v| (v * 255.).round() as u8)
                .collect();
            assert_eq!(
                hex::encode(Sha256::digest(&blurred)),
                blur_hash,
                "平滑 {kernel}"
            );
        }
        assert_eq!(smooth_skin_mask(&[0], 1, 1, false)[[0, 0, 0, 0]], 0.);
        assert_eq!(smooth_skin_mask(&[255], 1, 1, true)[[0, 0, 0, 0]], 1.);
    }

    #[test]
    #[ignore = "需要 validate_retouch.py 生成的固定 RGB/ROI 参考和本地八模型"]
    fn beauty_reference_images() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap();
        let directory = std::env::var_os("RETOUCH_REFERENCE_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|| root.join(".research/retouch/full-validation"));
        let mut models = PortraitModels::load(&root.join("src-tauri/resources"), None).unwrap();
        models.load_beauty_models().unwrap();
        let selected = std::env::var("RETOUCH_CASE").ok();
        let mut tested = 0;
        for entry in std::fs::read_dir(&directory).unwrap() {
            let path = entry.unwrap().path();
            let filename = path.file_name().unwrap().to_string_lossy();
            let Some(name) = filename
                .strip_prefix("onnx_")
                .and_then(|n| n.strip_suffix("_report.json"))
            else {
                continue;
            };
            if selected.as_ref().is_some_and(|s| s != name) {
                continue;
            }
            let report: serde_json::Value =
                serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
            let bounds: Vec<[f32; 4]> = serde_json::from_value(report["bounds"].clone()).unwrap();
            let input = image::open(directory.join(format!("onnx_{name}_00_input.png")))
                .unwrap()
                .to_rgb32f();
            for (i, &face) in bounds.iter().enumerate() {
                let Some(patch) = models.retouch_face(&input, face).unwrap() else {
                    continue;
                };
                let expected: Vec<f32> =
                    std::fs::read(directory.join(format!("torch_{name}_face{i}_local.f32")))
                        .unwrap()
                        .chunks_exact(4)
                        .map(|v| f32::from_le_bytes(v.try_into().unwrap()))
                        .collect();
                assert_eq!(patch.repaired.as_raw().len(), expected.len());
                let mut errors: Vec<f32> = patch
                    .repaired
                    .as_raw()
                    .iter()
                    .zip(expected)
                    .map(|(a, b)| (a - b).abs() * 255.)
                    .collect();
                errors.sort_by(f32::total_cmp);
                let mean = errors.iter().sum::<f32>() / errors.len() as f32;
                let p99 = errors[errors.len() * 99 / 100];
                eprintln!(
                    "{name} 局部 face{i}: mean={mean}, p99={p99}, max={}",
                    errors.last().unwrap()
                );
                assert!(mean <= 1. && p99 <= 2.);
                assert_eq!(
                    serde_json::json!([patch.rect.0, patch.rect.1, patch.rect.2, patch.rect.3]),
                    report["faces"][i]["bbox"]
                );
            }
            let started = std::time::Instant::now();
            let output = models.beautify(&input, &bounds).unwrap();
            eprintln!("{name} Rust 完整美肤 {:?}", started.elapsed());
            let encoded = image::RgbImage::from_fn(output.width(), output.height(), |x, y| {
                image::Rgb(output.get_pixel(x, y).0.map(|v| (v * 255.).round() as u8))
            });
            encoded
                .save(directory.join(format!("rust_{name}_08_full.png")))
                .unwrap();
            let expected = image::open(directory.join(format!("torch_{name}_08_full.png")))
                .unwrap()
                .to_rgb8();
            let mut errors: Vec<u8> = encoded
                .as_raw()
                .iter()
                .zip(expected.as_raw())
                .map(|(a, b)| a.abs_diff(*b))
                .collect();
            errors.sort_unstable();
            let mean = errors.iter().map(|&v| v as f64).sum::<f64>() / errors.len() as f64;
            let p99 = errors[errors.len() * 99 / 100];
            eprintln!(
                "{name} 完整结果: mean={mean}, p99={p99}, max={}",
                errors.last().unwrap()
            );
            assert!(mean <= 1. && p99 <= 2, "{name} 完整结果超限");
            tested += 1;
        }
        assert!(tested > 0, "没有参考样片，请先运行 Python 验证脚本");
    }

    #[test]
    fn missing_and_corrupt_models_are_rejected() {
        let directory = tempfile::tempdir().unwrap();
        let spec = ModelSpec {
            id: "test".into(),
            filename: "test.onnx".into(),
            aliases: vec!["alias.onnx".into()],
            sha256: "0".repeat(64),
        };
        assert!(
            verify_model(directory.path(), &spec, None)
                .unwrap_err()
                .contains("缺少")
        );
        std::fs::write(directory.path().join("alias.onnx"), b"not an onnx model").unwrap();
        assert!(
            verify_model(directory.path(), &spec, None)
                .unwrap_err()
                .contains("损坏")
        );
    }

    #[test]
    fn integrity_stamp_skips_full_hash_when_fingerprint_matches() {
        let directory = tempfile::tempdir().unwrap();
        let stamps = tempfile::tempdir().unwrap();
        let bytes = b"portrait-integrity-stamp-fixture";
        let digest = hex::encode(Sha256::digest(bytes));
        std::fs::write(directory.path().join("stamp.onnx"), bytes).unwrap();
        let spec = ModelSpec {
            id: "stamp_test".into(),
            filename: "stamp.onnx".into(),
            aliases: vec![],
            sha256: digest,
        };
        assert!(verify_model(directory.path(), &spec, Some(stamps.path())).is_ok());
        assert!(integrity_stamp_path(stamps.path(), &spec).is_file());
        // 二次校验应直接命中戳，行为与首次一致。
        assert_eq!(
            verify_model(directory.path(), &spec, Some(stamps.path())).unwrap(),
            directory.path().join("stamp.onnx")
        );
        // 清单 hash 变化后必须重新校验并拒绝。
        let mut wrong = spec;
        wrong.sha256 = "1".repeat(64);
        assert!(
            verify_model(directory.path(), &wrong, Some(stamps.path()))
                .unwrap_err()
                .contains("损坏")
        );
    }

    #[test]
    #[ignore = "需要本地八个模型及 ORT_DYLIB_PATH；显式运行离线集成测试"]
    fn bundled_models_runtime_contract() {
        let directory = std::env::var_os("PORTRAIT_RESOURCE_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|| Path::new(env!("CARGO_MANIFEST_DIR")).join("resources"));
        let started = std::time::Instant::now();
        let mut models = PortraitModels::load(&directory, None).unwrap();
        let outputs = infer(
            &mut models.detector,
            Array4::zeros((1, 3, 320, 320)),
            &[
                ("loc", &[1, 4200, 4]),
                ("conf", &[1, 4200, 2]),
                ("landmarks", &[1, 4200, 10]),
            ],
            "RetinaFace",
        )
        .unwrap();
        assert!(outputs[1].iter().all(|v| (0.0..=1.0).contains(v)));
        eprintln!("RetinaFace：{:?}", started.elapsed());
        let outputs = infer(
            &mut models.landmarker,
            Array4::zeros((1, 3, 256, 256)),
            &[("landmarks", &[1, 478, 3]), ("score", &[1, 1])],
            "Face Landmarker",
        )
        .unwrap();
        eprintln!("Landmarker score logit：{:?}", outputs[1]);
        infer(
            &mut models.parsing,
            Array4::zeros((1, 3, 512, 512)),
            &[
                ("output", &[1, 19, 512, 512]),
                ("566", &[1, 19, 512, 512]),
                ("576", &[1, 19, 512, 512]),
            ],
            "BiSeNet",
        )
        .unwrap();
        let outputs = infer(
            &mut models.skin,
            Array4::zeros((1, 256, 256, 3)),
            &[("Identity", &[1, 256, 256, 6])],
            "Selfie Segmentation",
        )
        .unwrap();
        let values = outputs[0].as_slice().unwrap();
        let range = values
            .iter()
            .fold((f32::INFINITY, f32::NEG_INFINITY), |(lo, hi), &v| {
                (lo.min(v), hi.max(v))
            });
        eprintln!("Selfie 范围：{range:?}，首像素：{:?}", &values[..6]);
        assert!(
            models.generator.is_none()
                && models.retouch_skin.is_none()
                && models.retouch_detection.is_none()
                && models.retouch_inpainting.is_none()
        );
        let empty = image::Rgb32FImage::from_pixel(32, 32, image::Rgb([0.2; 3]));
        assert_eq!(models.retouch_blemishes_only(&empty, &[]).unwrap(), empty);
        assert!(
            models.retouch_detection.is_none() && models.retouch_inpainting.is_none(),
            "空人脸不应加载祛痘模型"
        );
        assert_eq!(models.beautify(&empty, &[]).unwrap(), empty);
        assert!(models.generator.is_none() && models.retouch_skin.is_none());
        models.load_generator_only().unwrap();
        assert!(models.generator.is_some());
        assert!(
            models.retouch_detection.is_none()
                && models.retouch_inpainting.is_none()
                && models.retouch_skin.is_none(),
            "仅匀肤加载不应拉起祛痘/美白"
        );
        models.load_retouch_models().unwrap();
        assert!(models.retouch_detection.is_some() && models.retouch_inpainting.is_some());
        assert!(models.retouch_skin.is_none(), "仅祛痘加载不应拉起美白");
        models.load_beauty_models().unwrap();
        models.even_roi(&empty).unwrap();
        let mask = models.whitening_mask(&empty, false).unwrap();
        assert_eq!(mask.dim(), (1, 1, 32, 32));
        assert!(
            mask.iter()
                .all(|v| v.is_finite() && (0.0..=1.0).contains(v))
        );
        infer(
            models.retouch_detection.as_mut().unwrap(),
            Array4::zeros((1, 3, 768, 768)),
            &[("output", &[1, 1, 768, 768])],
            "Skin Retouch Detection",
        )
        .unwrap();
        {
            let outputs = models
                .retouch_inpainting
                .as_mut()
                .unwrap()
                .run(ort::inputs![
                    "input" => Tensor::from_array(Array4::<f32>::zeros((1, 3, 576, 576))).unwrap(),
                    "mask" => Tensor::from_array(Array4::<f32>::zeros((1, 1, 576, 576))).unwrap(),
                ])
                .unwrap();
            let repaired = outputs
                .get("output")
                .unwrap()
                .try_extract_array::<f32>()
                .unwrap();
            assert_eq!(repaired.shape(), &[1, 3, 576, 576]);
            assert!(
                repaired
                    .iter()
                    .all(|v| v.is_finite() && (-1.0..=1.0).contains(v))
            );
        }
        eprintln!("八模型离线验证完成：{:?}", started.elapsed());
        let blank = image::Rgb32FImage::from_pixel(320, 240, image::Rgb([0.0; 3]));
        let segmentation = models.analyze(&blank).unwrap();
        assert!(segmentation.faces.is_empty());
        assert!(segmentation.body.iter().all(|v| v.is_finite() && *v < 0.7));
        eprintln!("空图完整识别验证完成：{:?}", started.elapsed());
    }
}
