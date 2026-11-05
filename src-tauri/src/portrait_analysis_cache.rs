//! 重塑任务与缓存独立于美肤；只缓存不可变分析结果，不缓存照片像素或参数副本。
use crate::portrait_analysis::{
    ANALYSIS_VERSION, CANCELLED, ReshapeAnalysis, analyze_base, check_current, match_faces,
};
use crate::portrait_coordinates::CoordinateMap;
use crate::portrait_models::{ReshapeModels, model_cache_directory, resource_directory};
use crate::portrait_reshape::{ReshapeFaceSummary, ReshapeReason};
use image::{GenericImageView, Rgb32FImage};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::collections::{HashMap, VecDeque};
use std::sync::{
    Arc, Condvar, Mutex,
    atomic::{AtomicBool, AtomicUsize, Ordering},
};
use std::time::Duration;
use tauri::Manager;

const CACHE_BUDGET: usize = 64 * 1024 * 1024;
const MAX_JOBS: usize = 4;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AnalysisRequestId {
    pub path: String,
    pub source: String,
    pub generation: u64,
}

#[derive(Clone)]
struct Ticket {
    id: AnalysisRequestId,
    cancelled: Arc<AtomicBool>,
}

#[derive(Default)]
struct RequestGate(Mutex<Option<Ticket>>);

impl RequestGate {
    fn begin(&self, id: AnalysisRequestId) -> Result<Ticket, String> {
        let mut active = self.0.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(old) = active.as_ref() {
            if id.generation < old.id.generation
                || (id.generation == old.id.generation && id != old.id)
            {
                return Err(CANCELLED.into());
            }
            if id == old.id {
                return if old.cancelled.load(Ordering::SeqCst) {
                    Err(CANCELLED.into())
                } else {
                    Ok(old.clone())
                };
            }
            old.cancelled.store(true, Ordering::SeqCst);
        }
        let ticket = Ticket {
            id,
            cancelled: Arc::new(AtomicBool::new(false)),
        };
        *active = Some(ticket.clone());
        Ok(ticket)
    }

    fn cancel(&self, id: AnalysisRequestId) {
        let mut active = self.0.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(old) = active.as_ref() {
            if old.id == id {
                old.cancelled.store(true, Ordering::SeqCst);
            }
            if old.id.generation >= id.generation {
                return;
            }
            old.cancelled.store(true, Ordering::SeqCst);
        }
        // 保留墓碑：取消命令先到时，迟到的 prepare 不能复活同一代次。
        *active = Some(Ticket {
            id,
            cancelled: Arc::new(AtomicBool::new(true)),
        });
    }
}

type JobResult<T> = Result<Arc<T>, String>;
struct Job<T> {
    result: Mutex<Option<JobResult<T>>>,
    ready: Condvar,
    waiters: AtomicUsize,
}

struct WaitingConsumer<'a>(&'a AtomicUsize);
impl Drop for WaitingConsumer<'_> {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::SeqCst);
    }
}
struct CacheInner<T> {
    entries: VecDeque<(String, usize, Arc<T>)>,
    bytes: usize,
    jobs: HashMap<String, Arc<Job<T>>>,
    hits: u64,
    computations: u64,
}
pub(crate) struct TaskCache<T> {
    inner: Mutex<CacheInner<T>>,
    budget: usize,
    pinned: bool,
}

impl<T> TaskCache<T> {
    pub(crate) fn new(budget: usize) -> Self {
        Self {
            inner: Mutex::new(CacheInner {
                entries: VecDeque::new(),
                bytes: 0,
                jobs: HashMap::new(),
                hits: 0,
                computations: 0,
            }),
            budget,
            pinned: false,
        }
    }

    pub(crate) fn pinned(budget: usize) -> Self {
        Self {
            pinned: true,
            ..Self::new(budget)
        }
    }

    // 04 缓存语义的测试入口（成功／取消／partial／失败分支）；生产一律走 run_shared。
    #[cfg_attr(not(test), allow(dead_code))]
    fn run(
        &self,
        key: &str,
        current: &dyn Fn() -> bool,
        compute: impl Fn() -> Result<T, String>,
        byte_size: impl Fn(&T) -> Option<usize>,
    ) -> JobResult<T> {
        self.run_shared(key, current, |_| compute(), byte_size)
    }

    /// 公共工作只在 owner 和全部等待消费者均取消后终止；每位消费者独立发布。
    pub(crate) fn run_shared(
        &self,
        key: &str,
        current: &dyn Fn() -> bool,
        compute: impl Fn(&dyn Fn() -> bool) -> Result<T, String>,
        byte_size: impl Fn(&T) -> Option<usize>,
    ) -> JobResult<T> {
        loop {
            check_current(current)?;
            let (job, owner) = {
                let mut inner = self.inner.lock().unwrap_or_else(|e| e.into_inner());
                if let Some(i) = inner.entries.iter().position(|e| e.0 == key) {
                    let entry = inner.entries.remove(i).ok_or("重塑缓存索引错误")?;
                    let value = Arc::clone(&entry.2);
                    inner.entries.push_back(entry);
                    inner.hits += 1;
                    return Ok(value);
                }
                if let Some(job) = inner.jobs.get(key) {
                    job.waiters.fetch_add(1, Ordering::SeqCst);
                    (Arc::clone(job), false)
                } else {
                    if inner.jobs.len() >= MAX_JOBS {
                        return Err("重塑分析队列已满，请稍后重试".into());
                    }
                    let job = Arc::new(Job {
                        result: Mutex::new(None),
                        ready: Condvar::new(),
                        waiters: AtomicUsize::new(0),
                    });
                    inner.jobs.insert(key.into(), Arc::clone(&job));
                    inner.computations += 1;
                    (job, true)
                }
            };
            if !owner {
                let _consumer = WaitingConsumer(&job.waiters);
                let mut result = job.result.lock().unwrap_or_else(|e| e.into_inner());
                while result.is_none() {
                    check_current(current)?;
                    result = job
                        .ready
                        .wait_timeout(result, Duration::from_millis(20))
                        .unwrap_or_else(|e| e.into_inner())
                        .0;
                }
                check_current(current)?;
                let result = result.as_ref().ok_or("重塑任务结果缺失")?.clone();
                // 同键旧 owner 已取消时，新代次可重新成为 owner，不能继承旧任务的取消结果。
                if matches!(&result,Err(error) if error == CANCELLED) {
                    continue;
                }
                return result;
            }
            let shared_current = || current() || job.waiters.load(Ordering::SeqCst) > 0;
            let mut result: JobResult<T> =
                std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    check_current(&shared_current)?;
                    let value = compute(&shared_current)?;
                    check_current(&shared_current)?;
                    Ok(Arc::new(value))
                }))
                .unwrap_or_else(|_| Err("重塑分析内部错误".into()));
            {
                let mut inner = self.inner.lock().unwrap_or_else(|e| e.into_inner());
                if let Ok(value) = &result {
                    let bytes = byte_size(value).and_then(|size| size.checked_add(key.len() + 128));
                    if let Some(bytes) = bytes.filter(|bytes| *bytes <= self.budget) {
                        while inner.bytes > self.budget - bytes || inner.entries.len() >= 8 {
                            let removable = inner
                                .entries
                                .iter()
                                .position(|entry| !self.pinned || Arc::strong_count(&entry.2) == 1);
                            let Some(index) = removable else { break };
                            if let Some((_, size, _)) = inner.entries.remove(index) {
                                inner.bytes -= size;
                            }
                        }
                        if inner.bytes <= self.budget - bytes && inner.entries.len() < 8 {
                            inner
                                .entries
                                .push_back((key.into(), bytes, Arc::clone(value)));
                            inner.bytes += bytes;
                        } else if self.pinned {
                            result = Err("冻结资产预算已被活跃消费者占用，请释放后重新准备".into());
                        }
                    } else if self.pinned {
                        result = Err("冻结资产不完整或超过字节预算，禁止无缓存发布".into());
                    }
                }
                *job.result.lock().unwrap_or_else(|e| e.into_inner()) = Some(result.clone());
                inner.jobs.remove(key);
                job.ready.notify_all();
            }
            check_current(current)?;
            return result;
        }
    }

    pub(crate) fn stats(&self) -> CacheStats {
        let inner = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        CacheStats {
            bytes: inner.bytes,
            entries: inner.entries.len(),
            in_flight: inner.jobs.len(),
            hits: inner.hits,
            computations: inner.computations,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CacheStats {
    pub bytes: usize,
    pub entries: usize,
    pub in_flight: usize,
    pub hits: u64,
    pub computations: u64,
}

pub struct ReshapeAnalysisState {
    gate: RequestGate,
    cache: TaskCache<ReshapeAnalysis>,
    // 取锁后才解码源图；任何时刻最多一组完整基底与模型工作集。
    models: Mutex<Option<ReshapeModels>>,
}
impl Default for ReshapeAnalysisState {
    fn default() -> Self {
        Self {
            gate: RequestGate::default(),
            cache: TaskCache::pinned(CACHE_BUDGET),
            models: Mutex::new(None),
        }
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum AnalysisPhase {
    Loading,
    Analyzing,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AnalysisProgress {
    pub request: AnalysisRequestId,
    pub phase: AnalysisPhase,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AnalysisSummary {
    pub request: AnalysisRequestId,
    pub key: String,
    pub faces: Vec<ReshapeFaceSummary>,
    pub rejected_faces: usize,
    pub reason: Option<ReshapeReason>,
    pub cache: CacheStats,
    // 11 修复通道：true 表示摘要由有效快照重建（未做全量检测）；false 时前端移除与摘要
    // 失配的旧快照（深损坏与绑定失配回退），重新分析并保存即覆盖修复。
    pub snapshot_reused: bool,
}

/// 与旧美肤共享当前基底字段白名单，但不共享缓存、分析结果或模型会话。
pub fn analysis_base(adjustments: &Value) -> Result<Value, String> {
    if !adjustments.is_object() {
        return Err("重塑分析参数必须为对象".into());
    }
    // 当前工程没有前置液化映射；预留字段存在非空内容时拒绝，禁止忽略后错配坐标。
    if adjustments.get("liquify").is_some_and(|v| !v.is_null()) {
        return Err("前置液化坐标映射尚未接入重塑分析".into());
    }
    let mut base = crate::portrait_processing::base_adjustments(adjustments);
    if let Some(patches) = base["aiPatches"].as_array() {
        let active: Vec<_> = patches
            .iter()
            .filter(|p| p["visible"] != false && p["isLoading"] != true)
            .cloned()
            .collect();
        if active.iter().any(|p| p["patchData"].is_null()) {
            return Err("重塑基底缺少有效补丁内容".into());
        }
        base["aiPatches"] = Value::Array(active);
    }
    // 变换元数据本身有界；大像素补丁在源图预处理后立即释放，不放入分析结构。
    let geometry: serde_json::Map<String, Value> = base
        .as_object()
        .ok_or("重塑基底参数非法")?
        .iter()
        .filter(|(key, _)| key.as_str() != "aiPatches" && key.as_str() != "lensBlurDepthMap")
        .map(|(key, value)| (key.clone(), value.clone()))
        .collect();
    if Value::Object(geometry).to_string().len() > 12 * 1024 {
        return Err("重塑几何元数据超出安全上限".into());
    }
    Ok(base)
}

pub fn analysis_key(source_base_hash: &str, base: &Value, model_hashes: &Value) -> String {
    hex::encode(Sha256::digest(
        json!([ANALYSIS_VERSION, source_base_hash, base, model_hashes])
            .to_string()
            .as_bytes(),
    ))
}

pub(crate) fn prepare_encoded_base(
    source: &crate::portrait_render::SourceSnapshot,
    base: &Value,
    current: &dyn Fn() -> bool,
) -> Result<Rgb32FImage, String> {
    check_current(current)?;
    let warped = crate::adjustment_utils::prepare_patched_warped(source.image(), base)?;
    check_current(current)?;
    let (transformed, _) = crate::adjustment_utils::apply_spatial_transformations(warped, base);
    crate::portrait_tiles::check_dimensions([transformed.width(), transformed.height()])?;
    encoded_base(&transformed, source.is_raw())
}

impl ReshapeAnalysisState {
    /// 编辑器、缩略图和导出共用内部服务；消费者取消不绑定全局切图令牌。
    pub(crate) fn prepare_source(
        &self,
        app: &tauri::AppHandle,
        source: &crate::portrait_render::SourceSnapshot,
        base: &Value,
        current: &dyn Fn() -> bool,
        progress: &dyn Fn(AnalysisPhase),
    ) -> Result<Arc<ReshapeAnalysis>, String> {
        check_current(current)?;
        let hashes = ReshapeModels::approved_model_hashes()?;
        let key = analysis_key(source.source_key(), base, &json!(hashes));
        let analysis = self.cache.run_shared(
            &key,
            current,
            |shared_current| {
                let mut models = loop {
                    check_current(shared_current)?;
                    match self.models.try_lock() {
                        Ok(lock) => break lock,
                        Err(std::sync::TryLockError::Poisoned(e)) => break e.into_inner(),
                        Err(std::sync::TryLockError::WouldBlock) => {
                            std::thread::sleep(Duration::from_millis(10))
                        }
                    }
                };
                if models.is_none() {
                    let cache_dir = model_cache_directory(app);
                    *models = Some(ReshapeModels::new(
                        &resource_directory(app)?,
                        cache_dir.as_deref(),
                    ));
                }
                let dimensions = source.image().dimensions();
                let coordinates = CoordinateMap::new([dimensions.0, dimensions.1], base)?;
                let encoded = prepare_encoded_base(source, base, shared_current)?;
                progress(AnalysisPhase::Analyzing);
                let result = analyze_base(
                    &encoded,
                    coordinates,
                    source.source_key().into(),
                    hashes.clone(),
                    models.as_mut().ok_or("重塑模型容器缺失")?,
                    shared_current,
                )?;
                source.verify(shared_current)?;
                // 11：和保持量化的保护图随分析冻结；渲染与保存共用同一输入，保证保存前后效果一致。
                let mut result = crate::portrait_snapshot::quantized_analysis(result)
                    .ok_or("重塑保护图无法量化，请重新分析并保存")?;
                result.key = key.clone();
                Ok(result)
            },
            ReshapeAnalysis::cache_size,
        )?;
        source.verify(current)?;
        Ok(analysis)
    }
}

fn encoded_base(image: &image::DynamicImage, raw: bool) -> Result<Rgb32FImage, String> {
    // Alpha 仅用于分析时在黑底合成；从不改动原始图像或输出 Alpha。
    let mut rgba = image.to_rgba32f();
    if rgba.as_raw().iter().any(|v| !v.is_finite()) {
        return Err("重塑基底包含非有限像素".into());
    }
    for pixel in rgba.pixels_mut() {
        for c in 0..3 {
            let value = pixel[c].max(0.0);
            pixel[c] = (if raw {
                if value <= 0.0031308 {
                    value * 12.92
                } else {
                    1.055 * value.powf(1.0 / 2.4) - 0.055
                }
            } else {
                value
            })
            .clamp(0.0, 1.0)
                * pixel[3].clamp(0.0, 1.0);
        }
    }
    Ok(image::DynamicImage::ImageRgba32F(rgba).to_rgb32f())
}

fn validate_request(request: &AnalysisRequestId) -> Result<(), String> {
    if request.path.is_empty()
        || request.source.len() != 64
        || !request
            .source
            .bytes()
            .all(|c| c.is_ascii_digit() || (b'a'..=b'f').contains(&c))
        || request.generation == 0
        || request.generation > 9_007_199_254_740_991
    {
        return Err("重塑请求标识非法".into());
    }
    Ok(())
}

#[tauri::command]
pub fn cancel_portrait_reshape_analysis(
    state: tauri::State<crate::AppState>,
    request: AnalysisRequestId,
) -> Result<(), String> {
    validate_request(&request)?;
    state.reshape_analysis.gate.cancel(request);
    Ok(())
}

#[tauri::command]
pub async fn prepare_portrait_reshape_analysis(
    app: tauri::AppHandle,
    request: AnalysisRequestId,
    mut js_adjustments: Value,
    on_progress: tauri::ipc::Channel<AnalysisProgress>,
) -> Result<AnalysisSummary, String> {
    validate_request(&request)?;
    if !cfg!(debug_assertions) {
        return Err("重塑分析当前仅供开发构建内部验证".into());
    }
    if crate::portrait_reshape::read_parameters(&js_adjustments["portraitReshape"]).status
        != "supported"
    {
        return Err("重塑分析参数版本不可执行".into());
    }
    let state = app.state::<crate::AppState>();
    let generation = state.load_image_generation.load(Ordering::SeqCst);
    if state
        .original_image
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .as_ref()
        .is_none_or(|i| i.path != request.path)
    {
        return Err(CANCELLED.into());
    }
    let ticket = state.reshape_analysis.gate.begin(request.clone())?;
    // tauri::State 不实现 Drop：借用在末次使用处自然结束，无需显式 drop 释放。
    tokio::task::spawn_blocking(move || {
        let state = app.state::<crate::AppState>();
        let current = || {
            !ticket.cancelled.load(Ordering::SeqCst)
                && state.load_image_generation.load(Ordering::SeqCst) == generation
                && state
                    .original_image
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .as_ref()
                    .is_some_and(|i| i.path == request.path)
        };
        let progress = |phase: AnalysisPhase| {
            if current() {
                let _ = on_progress.send(AnalysisProgress {
                    request: request.clone(),
                    phase,
                });
            }
        };
        check_current(&current)?;
        crate::adjustment_utils::hydrate_adjustments(&state, &mut js_adjustments);
        let base = analysis_base(&js_adjustments)?;
        // 网格快照已移除；始终走完整检测。
        progress(AnalysisPhase::Loading);
        ReshapeModels::approved_model_hashes()?;
        let settings = crate::load_settings(app.clone()).map_err(|e| e.to_string())?;
        let source = state.portrait_render.editor_source(
            &request.path,
            &settings,
            generation as u64,
            &current,
        )?;
        let snapshot_reused = false;
        let analysis = state
            .reshape_analysis
            .prepare_source(&app, &source, &base, &current, &progress)?;
        check_current(&current)?;
        let summaries: Vec<_> = analysis.faces.iter().map(|f| f.summary.clone()).collect();
        let faces = match_faces(&summaries, &js_adjustments["portraitReshape"])?;
        let reason = if analysis.rejected_faces > 0 {
            Some(ReshapeReason::ProtectionLimited)
        } else if faces.is_empty() {
            Some(ReshapeReason::NoFace)
        } else {
            None
        };
        check_current(&current)?;
        Ok(AnalysisSummary {
            request,
            key: analysis.key.clone(),
            faces,
            rejected_faces: analysis.rejected_faces,
            reason,
            cache: state.reshape_analysis.cache.stats(),
            snapshot_reused,
        })
    })
    .await
    .map_err(|e| format!("重塑任务失败：{e}"))?
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicUsize;

    #[test]
    fn analysis_keys_ignore_sliders_and_selection_but_track_pixels_and_geometry() {
        let original =
            json!({"portrait":{"slimFace":0},"portraitReshape":{"enabled":false},"exposure":0});
        let key = analysis_key(
            "source",
            &analysis_base(&original).unwrap(),
            &json!({"landmarks":"a"}),
        );
        for amount in 0..100 {
            let mut changed = original.clone();
            changed["portrait"]["slimFace"] = json!(amount);
            changed["portraitReshape"] =
                json!({"faces":[{"protection":{"strokes":[1,2]}}],"selectedFaceId":"b"});
            changed["exposure"] = json!(amount);
            changed["crop"] = json!({"x":amount});
            assert_eq!(
                key,
                analysis_key(
                    "source",
                    &analysis_base(&changed).unwrap(),
                    &json!({"landmarks":"a"})
                )
            );
        }
        for changed in [
            json!({"transformDistortion":2}),
            json!({"rotation":30}),
            json!({"aiPatches":[{"id":"same","patchData":{"color":"x"}}]}),
        ] {
            assert_ne!(
                key,
                analysis_key(
                    "source",
                    &analysis_base(&changed).unwrap(),
                    &json!({"landmarks":"a"})
                )
            );
        }
        assert_ne!(
            key,
            analysis_key(
                "source",
                &analysis_base(&original).unwrap(),
                &json!({"landmarks":"b"})
            )
        );
        assert!(analysis_base(&json!({"liquify":[1]})).is_err());
    }

    #[test]
    fn hundred_cache_hits_byte_eviction_and_rebuild() {
        let cache = TaskCache::<usize>::new(300);
        let count = AtomicUsize::new(0);
        let compute = || {
            count.fetch_add(1, Ordering::SeqCst);
            Ok(1)
        };
        for _ in 0..100 {
            assert_eq!(
                *cache.run("a", &|| true, compute, |_| Some(100)).unwrap(),
                1
            );
        }
        assert_eq!(count.load(Ordering::SeqCst), 1);
        assert_eq!(cache.stats().hits, 99);
        cache.run("b", &|| true, compute, |_| Some(100)).unwrap();
        cache.run("a", &|| true, compute, |_| Some(100)).unwrap();
        assert_eq!(count.load(Ordering::SeqCst), 3);
        assert!(cache.stats().bytes <= 300);
        assert_eq!(cache.stats().in_flight, 0);
    }

    #[test]
    fn same_job_deduplicates_and_cancelled_owner_can_retry() {
        let cache = Arc::new(TaskCache::<usize>::new(1000));
        let count = Arc::new(AtomicUsize::new(0));
        std::thread::scope(|scope| {
            for _ in 0..4 {
                let cache = Arc::clone(&cache);
                let count = Arc::clone(&count);
                scope.spawn(move || {
                    cache
                        .run(
                            "same",
                            &|| true,
                            || {
                                count.fetch_add(1, Ordering::SeqCst);
                                std::thread::sleep(Duration::from_millis(30));
                                Ok(7)
                            },
                            |_| Some(100),
                        )
                        .unwrap()
                });
            }
        });
        assert_eq!(count.load(Ordering::SeqCst), 1);
        assert!(
            cache
                .run("cancel", &|| true, || Err(CANCELLED.into()), |_| Some(100))
                .is_err()
        );
        assert_eq!(
            *cache
                .run("cancel", &|| true, || Ok(8), |_| Some(100))
                .unwrap(),
            8
        );
    }

    #[test]
    fn uncached_partial_result_and_failed_job_can_retry() {
        let cache = TaskCache::<usize>::new(1000);
        let calls = AtomicUsize::new(0);
        let compute = || {
            calls.fetch_add(1, Ordering::SeqCst);
            Ok(4)
        };
        assert_eq!(
            *cache.run("partial", &|| true, compute, |_| None).unwrap(),
            4
        );
        assert_eq!(cache.stats().entries, 0);
        cache
            .run("partial", &|| true, compute, |_| Some(100))
            .unwrap();
        assert_eq!(calls.load(Ordering::SeqCst), 2);
        assert!(
            cache
                .run("failed", &|| true, || Err("合成失败".into()), |_| Some(100))
                .is_err()
        );
        assert_eq!(
            *cache
                .run("failed", &|| true, || Ok(9), |_| Some(100))
                .unwrap(),
            9
        );
        assert_eq!(cache.stats().in_flight, 0);
    }

    #[test]
    fn shared_work_survives_owner_cancel_while_export_waits() {
        let cache = TaskCache::<usize>::pinned(1000);
        let owner_active = AtomicBool::new(true);
        let calls = AtomicUsize::new(0);
        let (started_tx, started_rx) = std::sync::mpsc::channel();
        std::thread::scope(|scope| {
            let owner = scope.spawn(|| {
                cache.run_shared(
                    "shared",
                    &|| owner_active.load(Ordering::SeqCst),
                    |shared_current| {
                        calls.fetch_add(1, Ordering::SeqCst);
                        started_tx.send(()).unwrap();
                        let deadline = std::time::Instant::now() + Duration::from_secs(5);
                        loop {
                            let subscribed = cache
                                .inner
                                .lock()
                                .unwrap_or_else(|e| e.into_inner())
                                .jobs["shared"]
                                .waiters
                                .load(Ordering::SeqCst)
                                > 0;
                            if subscribed {
                                break;
                            }
                            assert!(std::time::Instant::now() < deadline);
                            std::thread::yield_now();
                        }
                        owner_active.store(false, Ordering::SeqCst);
                        assert!(shared_current());
                        Ok(11)
                    },
                    |_| Some(100),
                )
            });
            started_rx.recv_timeout(Duration::from_secs(5)).unwrap();
            let export = cache
                .run_shared(
                    "shared",
                    &|| true,
                    |_| panic!("导出不能因编辑器 owner 取消而重新计算"),
                    |_| Some(100),
                )
                .unwrap();
            assert_eq!(*export, 11);
            assert!(owner.join().unwrap().is_err());
        });
        assert_eq!(calls.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn pinned_cache_refuses_to_evict_active_assets() {
        let cache = TaskCache::<usize>::pinned(300);
        let held = cache
            .run_shared("a", &|| true, |_| Ok(1), |_| Some(100))
            .unwrap();
        assert!(
            cache
                .run_shared("b", &|| true, |_| Ok(2), |_| Some(100))
                .is_err()
        );
        assert_eq!(*held, 1);
        drop(held);
        assert_eq!(
            *cache
                .run_shared("b", &|| true, |_| Ok(2), |_| Some(100))
                .unwrap(),
            2
        );
        assert!(cache.stats().bytes <= 300);
    }

    #[test]
    fn generation_switch_and_cancel_before_prepare() {
        let gate = RequestGate::default();
        let a = AnalysisRequestId {
            path: "A".into(),
            source: "a".repeat(64),
            generation: 1,
        };
        let b = AnalysisRequestId {
            path: "B".into(),
            source: "b".repeat(64),
            generation: 2,
        };
        let ticket = gate.begin(a.clone()).unwrap();
        let next = gate.begin(b.clone()).unwrap();
        assert!(ticket.cancelled.load(Ordering::SeqCst));
        gate.cancel(a.clone());
        assert!(!next.cancelled.load(Ordering::SeqCst));
        assert!(gate.begin(a).is_err());
        let c = AnalysisRequestId { generation: 3, ..b };
        gate.cancel(c.clone());
        assert!(gate.begin(c).is_err());
    }
}
