use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, LockResult, Mutex, MutexGuard, OnceLock};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use image::{DynamicImage, GenericImageView, Rgb, Rgb32FImage};
use rayon::prelude::*;
use serde::Serialize;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use tauri::Manager;

use crate::AppState;
use crate::portrait_analysis_cache::TaskCache;
use crate::portrait_models::{
    FaceAnalysis, PREWARM_YIELDED, PortraitModels, Segmentation, model_cache_directory,
    resource_directory,
};

const ANALYSIS_EDGE: u32 = 2048;
/// 原生（严格）美肤分析边长：10 层 atlas 须同时放进 `CACHE_BUDGET` 与 256MB 显存上传峰值。
const NATIVE_ANALYSIS_EDGE: u32 = 1280;
const CACHE_BUDGET: usize = 384 * 1024 * 1024;
/// 旧路径基础分析缓存：2048 边长 10 层场单张约 450–670MB，须容纳两张以支持来回切图。
const LEGACY_ANALYSIS_CACHE_BUDGET: usize = 1024 * 1024 * 1024;
/// AI 美肤层缓存预算：约容纳一至两组（修复场 + 归零质感场）。
const AI_LAYER_BUDGET: usize = 160 * 1024 * 1024;
/// 旧路径 AI／匀肤／美白层：2048 边长单张约 90MB，须容纳两张。
const LEGACY_LAYER_BUDGET: usize = 256 * 1024 * 1024;
/// 基础分析中间产物预算：约容纳两到三组（线性分析图 + 人脸框）。
const INTERMEDIATE_BUDGET: usize = 120 * 1024 * 1024;
/// 交互预览神经美肤 bake 基底预算（预览分辨率，约数张）。
const NEURAL_PREVIEW_BAKE_BUDGET: usize = 128 * 1024 * 1024;
const NEURAL_PREVIEW_BAKE_VERSION: &str = "neural-preview-bake-v2";
pub(crate) const ALGORITHM_VERSION: &str = "portrait-retouch-20";
/// 低置信度烘焙／渲染默认最大亮度削减比例（约 1.0 EV）。
const SHINE_MAX_REDUCTION_LOW: f32 = 0.5;
/// 高置信度允许的最大亮度削减比例（强油光可修得更充分）。
const SHINE_MAX_REDUCTION_HIGH: f32 = 0.7;
/// 渲染侧单通道相对变化上限。
const SHINE_APPLY_REL_CAP: f32 = 0.55;
/// 祛油光五官边界羽化（略宽于频分，减轻眼下／唇周硬边）。
const SHINE_FEATURE_FEATHER: f32 = 0.035;
/// 鼻孔椭圆外缘羽化（归一化半径空间）。
const SHINE_NOSTRIL_INNER: f32 = 0.55;
const SHINE_NOSTRIL_OUTER: f32 = 1.2;
/// T 区外保底权重：侧光区域仍可祛油，但弱于额／鼻／颧。
const SHINE_T_ZONE_FLOOR: f32 = 0.45;
/// 供体颜色相似度（R−Y / B−Y 空间）衰减尺度。
const SHINE_CHROMA_SIGMA: f32 = 0.045;
const BLEMISH_SCALES: [f32; 3] = [0.006, 0.012, 0.022];
const BLEMISH_MIN_PIXELS: usize = 3;
const BLEMISH_MAX_DIAMETER: f32 = 0.08;
const BLEMISH_MAX_ASPECT: f32 = 3.5;
const BLEMISH_DARK_RANGE: [f32; 2] = [0.10, 0.28];
const BLEMISH_RED_RANGE: [f32; 2] = [0.025, 0.10];
const BLEMISH_CANDIDATE_SCORE: f32 = 0.12;
const BLEMISH_PEAK_SCORE: f32 = 0.5;
/// AI 美肤后台预热：轮询间隔、前台空闲门槛、最长等待与最多让出次数。
const AI_PREWARM_POLL: Duration = Duration::from_millis(500);
const AI_MODEL_PREWARM_IDLE: Duration = Duration::from_millis(750);
const AI_PREWARM_IDLE: Duration = Duration::from_millis(2500);
const AI_PREWARM_MAX_WAIT: Duration = Duration::from_secs(120);
const AI_PREWARM_MAX_YIELDS: usize = 5;

type AnalysisResult = Result<Arc<PortraitAnalysis>, String>;

#[derive(Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct PortraitSummary {
    pub key: String,
    pub face_count: usize,
    pub skipped_faces: usize,
    pub has_skin: bool,
}

pub struct PortraitAnalysis {
    pub summary: PortraitSummary,
    pub width: u32,
    pub height: u32,
    pub full_dimensions: (u32, u32),
    /// 已乘皮肤保护权重的平整 EV、五官 EV、轮廓 EV、线性亮度。
    pub pixels: Arc<[[f32; 4]]>,
    /// 已乘保护权重的线性 RGB 增量（低频修正）；A 为正表示脸部，为负表示身体。
    pub frequency_pixels: Arc<[[f32; 4]]>,
    /// 高低频分离的高频残差 source−low；A 与 frequency_pixels 同号区域标记。
    pub frequency_detail_pixels: Arc<[[f32; 4]]>,
    /// 保边柔润的线性 RGB 增量；A 为正表示脸部，为负表示身体。
    pub hydrated_pixels: Arc<[[f32; 4]]>,
    /// generator 匀肤线性 RGB 增量；A 为正表示脸部，为负表示身体。
    pub even_pixels: Arc<[[f32; 4]]>,
    /// AI 同源美白线性 RGB 增量；A 为正表示脸部，为负表示身体。
    pub whitening_pixels: Arc<[[f32; 4]]>,
    /// 中性灰平整色度中频线性 RGB 增量；A 为皮肤覆盖权重（与 strengths.x 相乘）。
    pub evenness_chroma_pixels: Arc<[[f32; 4]]>,
    /// 已合成的线性 RGB 增量；A 标记覆盖范围，AI 模式含匀肤及身体皮肤美白。
    pub blemish_pixels: Arc<[[f32; 4]]>,
    /// 来自清洁供体的高频 RGB 增量，不包含原始瑕疵残差。
    pub texture_pixels: Arc<[[f32; 4]]>,
    /// 低频油光保色线性 RGB 增量（分析置信已烘焙，不含滑块）；A 为覆盖／采样门控。
    pub shine_pixels: Arc<[[f32; 4]]>,
    /// 巩膜场：RGB = 单位强度祛红线性增量（已乘血丝概率×置信度）；A = 提亮用巩膜权重（不含血丝）。
    pub eye_white_pixels: Arc<[[f32; 4]]>,
}

#[derive(Clone)]
pub struct PortraitRenderMaps {
    pub analysis: Arc<PortraitAnalysis>,
    /// 已裁剪渲染基底对应的未裁剪分析图归一化矩形。
    pub uv_rect: [f32; 4],
}

impl PortraitAnalysis {
    fn byte_size(&self) -> usize {
        (self.pixels.len()
            + self.frequency_pixels.len()
            + self.frequency_detail_pixels.len()
            + self.hydrated_pixels.len()
            + self.even_pixels.len()
            + self.whitening_pixels.len()
            + self.evenness_chroma_pixels.len()
            + self.blemish_pixels.len()
            + self.texture_pixels.len()
            + self.shine_pixels.len()
            + self.eye_white_pixels.len())
            * std::mem::size_of::<[f32; 4]>()
    }

    pub fn for_render(self: &Arc<Self>, adjustments: &Value) -> PortraitRenderMaps {
        let (width, height) = self.full_dimensions;
        let mut rect = [0.0, 0.0, 1.0, 1.0];
        if let Ok(crop) =
            serde_json::from_value::<crate::image_processing::Crop>(adjustments["crop"].clone())
        {
            let x = crop.x.round() as u32;
            let y = crop.y.round() as u32;
            let w = crop.width.round() as u32;
            let h = crop.height.round() as u32;
            if x < width && y < height && w > 0 && h > 0 {
                rect = [
                    x as f32 / width as f32,
                    y as f32 / height as f32,
                    w.min(width - x) as f32 / width as f32,
                    h.min(height - y) as f32 / height as f32,
                ];
            }
        }
        PortraitRenderMaps {
            analysis: Arc::clone(self),
            uv_rect: rect,
        }
    }
}

#[cfg(test)]
#[derive(Default)]
struct PortraitCache {
    entries: VecDeque<(String, Arc<PortraitAnalysis>)>,
    bytes: usize,
}

#[cfg(test)]
impl PortraitCache {
    fn get(&mut self, key: &str) -> Option<Arc<PortraitAnalysis>> {
        let index = self.entries.iter().position(|(k, _)| k == key)?;
        let entry = self.entries.remove(index)?;
        let result = Arc::clone(&entry.1);
        self.entries.push_back(entry);
        Some(result)
    }

    fn insert(&mut self, key: String, analysis: Arc<PortraitAnalysis>) {
        let bytes = analysis.byte_size();
        if bytes > LEGACY_ANALYSIS_CACHE_BUDGET {
            return;
        }
        while self.bytes + bytes > LEGACY_ANALYSIS_CACHE_BUDGET {
            if let Some((_, old)) = self.entries.pop_front() {
                self.bytes -= old.byte_size();
            } else {
                break;
            }
        }
        self.bytes += bytes;
        self.entries.push_back((key, analysis));
    }
}

struct AnalysisJob<T> {
    result: Mutex<Option<Result<Arc<T>, String>>>,
    ready: Condvar,
    /// 前台已加入等待：后台预热不再让出该任务。
    promoted: AtomicBool,
}

impl<T> Default for AnalysisJob<T> {
    fn default() -> Self {
        Self {
            result: Mutex::new(None),
            ready: Condvar::new(),
            promoted: AtomicBool::new(false),
        }
    }
}

/// AI 美肤修复层：与基础分析场独立缓存，开关 AI 不再触发人脸重新检测。
struct AiBlemishLayer {
    /// AI 修复的线性 RGB 增量场。
    blemish: Arc<[[f32; 4]]>,
    /// 归零的质感场；AI 模式下质感强度恒为零。
    zeros: Arc<[[f32; 4]]>,
}

impl AiBlemishLayer {
    fn byte_size(&self) -> usize {
        (self.blemish.len() + self.zeros.len()) * std::mem::size_of::<[f32; 4]>()
    }
}

/// 均匀肤色层：generator 匀肤增量，与瑕疵场独立缓存。
struct EvenSkinLayer {
    even: Arc<[[f32; 4]]>,
}

impl EvenSkinLayer {
    fn byte_size(&self) -> usize {
        self.even.len() * std::mem::size_of::<[f32; 4]>()
    }
}

/// 独立美白层：与 AI 一键美白同源算法，脸/身分号增量场。
struct WhiteningLayer {
    whitening: Arc<[[f32; 4]]>,
}

impl WhiteningLayer {
    fn byte_size(&self) -> usize {
        self.whitening.len() * std::mem::size_of::<[f32; 4]>()
    }
}

/// 基础分析的中间产物：线性分析图、人脸框与身体蒙版，供 AI／匀肤／美白层按需计算。
#[derive(Clone)]
struct AnalysisIntermediates {
    linear: Rgb32FImage,
    faces: Vec<[f32; 4]>,
    body: Vec<f32>,
}

impl AnalysisIntermediates {
    fn byte_size(&self) -> usize {
        self.linear.width() as usize
            * self.linear.height() as usize
            * std::mem::size_of::<[f32; 3]>()
            + self.faces.len() * std::mem::size_of::<[f32; 4]>()
            + self.body.len() * std::mem::size_of::<f32>()
    }
}

/// 小容量按字节预算的 LRU，供 AI 层与中间产物缓存共用。
struct LruList<T> {
    entries: VecDeque<(String, usize, T)>,
    bytes: usize,
}

impl<T> Default for LruList<T> {
    fn default() -> Self {
        Self {
            entries: VecDeque::new(),
            bytes: 0,
        }
    }
}

impl<T> LruList<T> {
    fn get(&mut self, key: &str) -> Option<&T> {
        let index = self.entries.iter().position(|(k, _, _)| k == key)?;
        let entry = self.entries.remove(index)?;
        self.entries.push_back(entry);
        self.entries.back().map(|(_, _, value)| value)
    }

    fn insert(&mut self, key: String, bytes: usize, value: T, budget: usize) {
        if bytes > budget {
            return;
        }
        if let Some(index) = self.entries.iter().position(|(k, _, _)| *k == key) {
            if let Some((_, old_bytes, _)) = self.entries.remove(index) {
                self.bytes -= old_bytes;
            }
        }
        while self.bytes + bytes > budget {
            match self.entries.pop_front() {
                Some((_, old_bytes, _)) => self.bytes -= old_bytes,
                None => break,
            }
        }
        self.bytes += bytes;
        self.entries.push_back((key, bytes, value));
    }

    #[cfg(test)]
    fn remove(&mut self, key: &str) -> Option<T> {
        let index = self.entries.iter().position(|(k, _, _)| k == key)?;
        let (_, bytes, value) = self.entries.remove(index)?;
        self.bytes -= bytes;
        Some(value)
    }
}

#[derive(Default)]
pub struct PortraitState {
    models: Mutex<Option<PortraitModels>>,
    /// 旧预览与导出共用的基础分析缓存；与严格路径采用同一 TaskCache 去重/预算/取消语义。
    base_tasks: OnceLock<TaskCache<PortraitAnalysis>>,
    /// AI 美肤层缓存：键为基础分析键，与基础变体独立淘汰。
    ai_layers: Mutex<LruList<Arc<AiBlemishLayer>>>,
    /// AI 美肤层计算任务去重表：键为 AI 变体键。
    ai_jobs: Mutex<HashMap<String, Arc<AnalysisJob<AiBlemishLayer>>>>,
    /// 均匀肤色层缓存：键为 `{base}:even`。
    even_layers: Mutex<LruList<Arc<EvenSkinLayer>>>,
    /// 均匀肤色层任务去重。
    even_jobs: Mutex<HashMap<String, Arc<AnalysisJob<EvenSkinLayer>>>>,
    /// 独立美白层缓存：键为 `{base}:whiten`。
    whitening_layers: Mutex<LruList<Arc<WhiteningLayer>>>,
    /// 独立美白层任务去重。
    whitening_jobs: Mutex<HashMap<String, Arc<AnalysisJob<WhiteningLayer>>>>,
    /// 基础分析中间产物：键为基础分析键，AI 层算出后清除。
    intermediates: Mutex<LruList<AnalysisIntermediates>>,
    /// 交互预览：已 bake 人像场的颜色基底，调色帧不再绑定场纹理。
    neural_preview_bases: Mutex<LruList<Arc<DynamicImage>>>,
    neural_preview_bake_hits: AtomicU64,
    neural_preview_bake_misses: AtomicU64,
    strict: StrictPortraitState,
    /// 正在等待模型锁的前台任务数。
    foreground_waiters: AtomicUsize,
    /// 正在执行的前台任务数（预览、分析命令、导出）。
    foreground_active: AtomicUsize,
    /// 最近一次前台活动的时间（Unix 毫秒）。
    last_foreground_ms: AtomicU64,
    /// 当前排队中的 AI 美肤预热基础分析键；新键取代旧键时旧预热线程退出。
    ai_prewarm_key: Mutex<Option<String>>,
    /// 美肤分析请求代次；切图或新请求会使旧任务在最近检查点停止。
    analysis_epoch: AtomicU64,
}

/// 后台预热是否应让出：前台正等锁或正执行时让出，前台已在等该任务时不让出。
fn should_yield(waiters: usize, active: usize, promoted: bool) -> bool {
    !promoted && (waiters > 0 || active > 0)
}

fn unix_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis() as u64)
}

/// 前台任务标记：存活期间后台 AI 预热不启动，已启动的在下一个检查点让出。
pub(crate) struct ForegroundGuard<'a> {
    state: &'a PortraitState,
}

impl Drop for ForegroundGuard<'_> {
    fn drop(&mut self) {
        self.state.touch_foreground();
        self.state.foreground_active.fetch_sub(1, Ordering::SeqCst);
    }
}

impl PortraitState {
    fn begin_analysis(&self, generation: u64) -> Result<u64, String> {
        if generation == 0 || generation > 9_007_199_254_740_991 {
            return Err("人像分析请求代次非法".into());
        }
        let previous = self.analysis_epoch.fetch_max(generation, Ordering::SeqCst);
        if previous > generation {
            Err("Superseded or worker failed".into())
        } else {
            Ok(generation)
        }
    }

    fn cancel_analysis(&self, generation: u64) {
        let _ = self.analysis_epoch.compare_exchange(
            generation,
            generation.saturating_add(1),
            Ordering::SeqCst,
            Ordering::SeqCst,
        );
    }

    fn analysis_current(&self, epoch: u64) -> bool {
        self.analysis_epoch.load(Ordering::SeqCst) == epoch
    }

    pub(crate) fn foreground(&self) -> ForegroundGuard<'_> {
        self.foreground_active.fetch_add(1, Ordering::SeqCst);
        self.touch_foreground();
        ForegroundGuard { state: self }
    }

    fn touch_foreground(&self) {
        self.last_foreground_ms.store(unix_ms(), Ordering::SeqCst);
    }

    /// 前台取模型锁：等待期间计入 waiters，让持锁的后台预热尽快让出。
    fn lock_models_foreground(&self) -> LockResult<MutexGuard<'_, Option<PortraitModels>>> {
        self.foreground_waiters.fetch_add(1, Ordering::SeqCst);
        self.touch_foreground();
        let guard = self.models.lock();
        self.foreground_waiters.fetch_sub(1, Ordering::SeqCst);
        guard
    }

    fn background_should_yield(&self, promoted: &AtomicBool) -> bool {
        should_yield(
            self.foreground_waiters.load(Ordering::SeqCst),
            self.foreground_active.load(Ordering::SeqCst),
            promoted.load(Ordering::SeqCst),
        )
    }

    fn foreground_idle_for(&self, idle: Duration) -> bool {
        self.foreground_active.load(Ordering::SeqCst) == 0
            && self.foreground_waiters.load(Ordering::SeqCst) == 0
            && unix_ms().saturating_sub(self.last_foreground_ms.load(Ordering::SeqCst))
                >= idle.as_millis() as u64
    }
}

/// AI 美肤层的调用方：前台为用户请求，后台为空闲预热（可被前台打断）。
#[derive(Clone, Copy, PartialEq, Eq)]
enum LayerMode {
    Foreground,
    Background,
}

struct StrictPortraitBase {
    analysis: Arc<PortraitAnalysis>,
    intermediates: AnalysisIntermediates,
}
impl StrictPortraitBase {
    fn byte_size(&self) -> Option<usize> {
        self.analysis
            .byte_size()
            .checked_add(self.intermediates.byte_size())
    }
}
struct StrictPortraitState {
    base: crate::portrait_analysis_cache::TaskCache<StrictPortraitBase>,
    ai: crate::portrait_analysis_cache::TaskCache<AiBlemishLayer>,
    even: crate::portrait_analysis_cache::TaskCache<EvenSkinLayer>,
    whitening: crate::portrait_analysis_cache::TaskCache<WhiteningLayer>,
}
impl Default for StrictPortraitState {
    fn default() -> Self {
        Self {
            base: crate::portrait_analysis_cache::TaskCache::pinned(CACHE_BUDGET),
            ai: crate::portrait_analysis_cache::TaskCache::pinned(AI_LAYER_BUDGET),
            even: crate::portrait_analysis_cache::TaskCache::pinned(AI_LAYER_BUDGET),
            whitening: crate::portrait_analysis_cache::TaskCache::pinned(AI_LAYER_BUDGET),
        }
    }
}

/// 08 使用调用方已经准备的严格原生基底，禁止另读路径或命中元数据版缓存。
/// 基础检测与 AI 美肤分层；新重塑本身不会触发任何旧美肤模型。
// 08 入口边界：应用、状态、源键、基底、变换图、RAW 标记、调整与取消谓词各自独立。
#[allow(clippy::too_many_arguments)]
pub(crate) fn prepare_native_render(
    app: &tauri::AppHandle,
    state: &tauri::State<AppState>,
    source_key: &str,
    base: &Value,
    transformed: &DynamicImage,
    is_raw: bool,
    adjustments: &Value,
    current: &dyn Fn() -> bool,
) -> Result<Option<PortraitRenderMaps>, String> {
    use crate::portrait_analysis::check_current;
    if !enabled(adjustments) {
        return Ok(None);
    }
    check_current(current)?;
    crate::portrait_tiles::check_dimensions([transformed.width(), transformed.height()])?;
    let key = crate::portrait_render::content_key(&json!([
        "strict-beauty-base-v1",
        NATIVE_ANALYSIS_EDGE,
        ALGORITHM_VERSION,
        source_key,
        base,
        hex::encode(Sha256::digest(include_bytes!(
            "../resources/portrait-models.json"
        )))
    ]));
    let portrait = &state.portrait_state;
    let _foreground = portrait.foreground();
    let prepared = portrait.strict.base.run_shared(
        &key,
        current,
        |shared_current| {
            let mut models = portrait
                .lock_models_foreground()
                .map_err(|_| "人像模型锁异常")?;
            check_current(shared_current)?;
            if models.is_none() {
                let cache = model_cache_directory(app);
                *models = Some(PortraitModels::load(
                    &resource_directory(app)?,
                    cache.as_deref(),
                )?);
            }
            let (encoded, linear, luminance) =
                analysis_image_with_edge(transformed, is_raw, NATIVE_ANALYSIS_EDGE);
            check_current(shared_current)?;
            let segmentation = models
                .as_mut()
                .ok_or("人像模型初始化失败")?
                .analyze(&encoded)?;
            check_current(shared_current)?;
            let faces = segmentation.faces.iter().map(|f| f.bounds).collect();
            let (mut analysis, body) = build_fields(&linear, &luminance, segmentation);
            analysis.summary.key = key.clone();
            analysis.full_dimensions = transformed.dimensions();
            check_current(shared_current)?;
            Ok(StrictPortraitBase {
                analysis: Arc::new(analysis),
                intermediates: AnalysisIntermediates {
                    linear,
                    faces,
                    body,
                },
            })
        },
        StrictPortraitBase::byte_size,
    )?;
    let mut analysis =
        if ai_blemish_enabled(adjustments) && prepared.analysis.summary.face_count > 0 {
            let layer = portrait.strict.ai.run_shared(
                &ai_variant_key(&key),
                current,
                |shared_current| {
                    let mut models = portrait
                        .lock_models_foreground()
                        .map_err(|_| "人像模型锁异常")?;
                    check_current(shared_current)?;
                    let intermediate = &prepared.intermediates;
                    let encoded = encode_linear(&intermediate.linear);
                    let blemish = apply_ai_blemishes(
                        models.as_mut().ok_or("人像模型初始化失败")?,
                        &encoded,
                        &intermediate.linear,
                        &intermediate.faces,
                        &|| false,
                    )?;
                    check_current(shared_current)?;
                    let mut zeros = Vec::new();
                    zeros
                        .try_reserve_exact(blemish.len())
                        .map_err(|_| "AI 美肤零场内存不足")?;
                    zeros.resize(blemish.len(), [0.0; 4]);
                    Ok(AiBlemishLayer {
                        blemish: Arc::from(blemish),
                        zeros: Arc::from(zeros),
                    })
                },
                |layer| Some(layer.byte_size()),
            )?;
            assemble_neural_variant(
                &prepared.analysis,
                Some(&layer),
                &ai_variant_key(&prepared.analysis.summary.key),
            )
        } else if ordinary_neural_blemish_enabled(adjustments)
            && prepared.analysis.summary.face_count > 0
        {
            let layer = portrait.strict.ai.run_shared(
                &retouch_variant_key(&key),
                current,
                |shared_current| {
                    let mut models = portrait
                        .lock_models_foreground()
                        .map_err(|_| "人像模型锁异常")?;
                    check_current(shared_current)?;
                    let intermediate = &prepared.intermediates;
                    let encoded = encode_linear(&intermediate.linear);
                    let blemish = apply_retouch_blemishes(
                        models.as_mut().ok_or("人像模型初始化失败")?,
                        &encoded,
                        &intermediate.linear,
                        &intermediate.faces,
                        &|| !shared_current(),
                    )?;
                    check_current(shared_current)?;
                    let mut zeros = Vec::new();
                    zeros
                        .try_reserve_exact(blemish.len())
                        .map_err(|_| "神经祛痘零场内存不足")?;
                    zeros.resize(blemish.len(), [0.0; 4]);
                    Ok(AiBlemishLayer {
                        blemish: Arc::from(blemish),
                        zeros: Arc::from(zeros),
                    })
                },
                |layer| Some(layer.byte_size()),
            )?;
            assemble_neural_variant(
                &prepared.analysis,
                Some(&layer),
                &retouch_variant_key(&prepared.analysis.summary.key),
            )
        } else {
            Arc::clone(&prepared.analysis)
        };
    if even_skin_enabled(adjustments) {
        let layer = portrait.strict.even.run_shared(
            &even_variant_key(&key),
            current,
            |shared_current| {
                let mut models = portrait
                    .lock_models_foreground()
                    .map_err(|_| "人像模型锁异常")?;
                check_current(shared_current)?;
                let intermediate = &prepared.intermediates;
                let encoded = encode_linear(&intermediate.linear);
                let even = apply_even_skin(
                    models.as_mut().ok_or("人像模型初始化失败")?,
                    &encoded,
                    &intermediate.linear,
                    &intermediate.faces,
                    &intermediate.body,
                )?;
                check_current(shared_current)?;
                Ok(EvenSkinLayer {
                    even: Arc::from(even),
                })
            },
            |layer| Some(layer.byte_size()),
        )?;
        analysis = attach_even_pixels(&analysis, &layer);
    }
    if whitening_enabled(adjustments) {
        let layer = portrait.strict.whitening.run_shared(
            &whitening_variant_key(&key),
            current,
            |shared_current| {
                let mut models = portrait
                    .lock_models_foreground()
                    .map_err(|_| "人像模型锁异常")?;
                check_current(shared_current)?;
                let intermediate = &prepared.intermediates;
                let encoded = encode_linear(&intermediate.linear);
                let whitening = apply_skin_whitening(
                    models.as_mut().ok_or("人像模型初始化失败")?,
                    &encoded,
                    &intermediate.linear,
                    &intermediate.faces,
                    &intermediate.body,
                )?;
                check_current(shared_current)?;
                Ok(WhiteningLayer {
                    whitening: Arc::from(whitening),
                })
            },
            |layer| Some(layer.byte_size()),
        )?;
        analysis = attach_whitening_pixels(&analysis, &layer);
    }
    check_current(current)?;
    Ok(Some(PortraitRenderMaps {
        analysis,
        uv_rect: [0.0, 0.0, 1.0, 1.0],
    }))
}

/// 等待同键任务完成并返回其结果；用于合并并发的相同分析请求。
/// 仅前台调用：加入等待即提升任务，后台预热不再让出它。
fn wait_for_job_result<T>(job: &Arc<AnalysisJob<T>>) -> Result<Arc<T>, String> {
    job.promoted.store(true, Ordering::SeqCst);
    let mut result = job.result.lock().map_err(|_| "人像任务结果锁异常")?;
    while result.is_none() {
        result = job.ready.wait(result).map_err(|_| "等待人像任务失败")?;
    }
    result.as_ref().ok_or("人像任务结果缺失")?.clone()
}

pub fn strengths(adjustments: &Value) -> [f32; 4] {
    let skin_visible = adjustments["sectionVisibility"]["portraitSkin"]
        .as_bool()
        .unwrap_or(true);
    if !skin_visible {
        return [0.0; 4];
    }
    [
        "neutralGrayEvenness",
        "neutralGrayContour",
        "frequencySeparationFace",
        "frequencySeparationBody",
    ]
    .map(|key| {
        adjustments["portrait"][key]
            .as_f64()
            .filter(|v| v.is_finite())
            .map(|v| v.round().clamp(0.0, 100.0) as f32 / 100.0)
            .unwrap_or(0.0)
    })
}

/// 水润磨皮强度：前两项依次为脸部与身体，后两项为布局保留。
pub fn hydrated_strengths(adjustments: &Value) -> [f32; 4] {
    let skin_visible = adjustments["sectionVisibility"]["portraitSkin"]
        .as_bool()
        .unwrap_or(true);
    if !skin_visible {
        return [0.0; 4];
    }
    let amount = |key: &str| {
        adjustments["portrait"][key]
            .as_f64()
            .filter(|v| v.is_finite())
            .map(|v| v.round().clamp(0.0, 100.0) as f32 / 100.0)
            .unwrap_or(0.0)
    };
    [
        amount("hydratedSmoothingFace"),
        amount("hydratedSmoothingBody"),
        0.0,
        0.0,
    ]
}

/// 均匀肤色强度：[脸, 身, 0, 0]。
pub fn even_strengths(adjustments: &Value) -> [f32; 4] {
    let skin_visible = adjustments["sectionVisibility"]["portraitSkin"]
        .as_bool()
        .unwrap_or(true);
    if !skin_visible {
        return [0.0; 4];
    }
    let amount = |key: &str| {
        adjustments["portrait"][key]
            .as_f64()
            .filter(|v| v.is_finite())
            .map(|v| v.round().clamp(0.0, 100.0) as f32 / 100.0)
            .unwrap_or(0.0)
    };
    [amount("evenSkinFace"), amount("evenSkinBody"), 0.0, 0.0]
}

/// 均匀肤色滑块启用（皮肤分区可见且脸或身强度 > 0）。
pub fn even_skin_enabled(adjustments: &Value) -> bool {
    let s = even_strengths(adjustments);
    s[0] > 0.0 || s[1] > 0.0
}

/// 美白强度：[脸, 身, 0, 0]。
pub fn whitening_strengths(adjustments: &Value) -> [f32; 4] {
    let skin_visible = adjustments["sectionVisibility"]["portraitSkin"]
        .as_bool()
        .unwrap_or(true);
    if !skin_visible {
        return [0.0; 4];
    }
    let amount = |key: &str| {
        adjustments["portrait"][key]
            .as_f64()
            .filter(|v| v.is_finite())
            .map(|v| v.round().clamp(0.0, 100.0) as f32 / 100.0)
            .unwrap_or(0.0)
    };
    [amount("whiteningFace"), amount("whiteningBody"), 0.0, 0.0]
}

/// 美白滑块启用（皮肤分区可见且脸或身强度 > 0）。
pub fn whitening_enabled(adjustments: &Value) -> bool {
    let s = whitening_strengths(adjustments);
    s[0] > 0.0 || s[1] > 0.0
}

/// 脸区正权是脸/身混合系数：中心偏脸、贴边交给身体。
/// 两边滑块相同时羽化带保持满强度，避免交界发黑；仅脸部拉高时仍向 0 淡出。
fn whitening_strength_for_region(region: f32, face: f32, body: f32) -> f32 {
    if region > 0.0 {
        face * region + body * (1.0 - region)
    } else {
        body * region.abs()
    }
}

/// 眼白提亮／祛红已迁至 GPUPixel bake（VNN 关键点 CPU 后处理）。
/// Atlas／WGSL 路径恒为 0，避免双路径叠加；请走 `portrait_gpupixel::params_from_adjustments`。
pub fn eye_white_strengths(_adjustments: &Value) -> [f32; 4] {
    [0.0; 4]
}

/// 眼白滑块是否由 GPUPixel 处理（不再启用 Rust atlas 眼白场）。
pub fn eye_white_enabled(adjustments: &Value) -> bool {
    let p = crate::portrait_gpupixel::params_from_adjustments(adjustments);
    p.eye_brighten > 0.0 || p.eye_redness > 0.0
}

/// 中性灰立体子带：[五官, 轮廓, 0, 0]；缺省均为 0.8（与 UI 默认 80 对齐）。
pub fn contour_band_strengths(adjustments: &Value) -> [f32; 4] {
    let amount = |key: &str| {
        adjustments["portrait"][key]
            .as_f64()
            .filter(|v| v.is_finite())
            .map(|v| v.round().clamp(0.0, 100.0) as f32 / 100.0)
            .unwrap_or(0.8)
    };
    [
        amount("neutralGrayContourFeatures"),
        amount("neutralGrayContourOutline"),
        0.0,
        0.0,
    ]
}

/// 高低频子带：[脸低频, 脸高频 amt, 身低频, 身高频 amt]；缺省均为 1.0（满低频 / 高频中性）。
/// 低频 UI 0..100 → 0..1；高频 UI -100..100 → high_amt = 1 - ui/100（+100→0 满柔化，0→1 保纹理，-100→2 加强）。
/// 兼容旧共享键 `frequencySeparationLow`/`High`（新区键缺失时回退；信任前端已做旧值迁移）。
pub fn frequency_band_strengths(adjustments: &Value) -> [f32; 4] {
    let low = |region_key: &str, legacy_key: &str| {
        adjustments["portrait"][region_key]
            .as_f64()
            .or_else(|| adjustments["portrait"][legacy_key].as_f64())
            .filter(|v| v.is_finite())
            .map(|v| v.round().clamp(0.0, 100.0) as f32 / 100.0)
            .unwrap_or(1.0)
    };
    let high_amt = |region_key: &str, legacy_key: &str| {
        adjustments["portrait"][region_key]
            .as_f64()
            .or_else(|| adjustments["portrait"][legacy_key].as_f64())
            .filter(|v| v.is_finite())
            .map(|v| 1.0 - (v.round().clamp(-100.0, 100.0) as f32) / 100.0)
            .unwrap_or(1.0)
    };
    [
        low("frequencySeparationFaceLow", "frequencySeparationLow"),
        high_amt("frequencySeparationFaceHigh", "frequencySeparationHigh"),
        low("frequencySeparationBodyLow", "frequencySeparationLow"),
        high_amt("frequencySeparationBodyHigh", "frequencySeparationHigh"),
    ]
}

pub fn blemish_strengths(adjustments: &Value) -> [f32; 4] {
    if adjustments["sectionVisibility"]["portraitBlemishes"].as_bool() == Some(false) {
        return [0.0; 4];
    }
    let amount = |key: &str| {
        adjustments["portrait"][key]
            .as_f64()
            .filter(|v| v.is_finite())
            .map(|v| v.round().clamp(0.0, 100.0) as f32 / 100.0)
            .unwrap_or(0.0)
    };
    let removal = amount("blemishRemoval");
    let ai = ai_blemish_amount(adjustments);
    [
        removal.max(ai),
        // 神经祛痘场无独立质感回加；质感保留恒为 0（与 AI 一键一致）。
        0.0,
        amount("faceShineRemoval"),
        0.0,
    ]
}

/// AI 美肤强度：0-100 归一化到 0-1；兼容旧照片中的布尔开关（true→1.0）。
fn ai_blemish_amount(adjustments: &Value) -> f32 {
    match adjustments["portrait"]["aiBlemishRemoval"] {
        Value::Bool(true) => 1.0,
        _ => adjustments["portrait"]["aiBlemishRemoval"]
            .as_f64()
            .filter(|v| v.is_finite())
            .map(|v| v.round().clamp(0.0, 100.0) as f32 / 100.0)
            .unwrap_or(0.0),
    }
}

/// AI 美肤启用：强度大于 0 即启用，脸部瑕疵修复由 cv_unet_skin-retouching 接管。
pub fn ai_blemish_enabled(adjustments: &Value) -> bool {
    adjustments["sectionVisibility"]["portraitBlemishes"].as_bool() != Some(false)
        && ai_blemish_amount(adjustments) > 0.0
}

/// 普通祛斑祛痘滑块启用神经局部修复（不含匀肤/美白）；与 AI 一键互斥。
pub fn ordinary_neural_blemish_enabled(adjustments: &Value) -> bool {
    if ai_blemish_enabled(adjustments) {
        return false;
    }
    if adjustments["sectionVisibility"]["portraitBlemishes"].as_bool() == Some(false) {
        return false;
    }
    adjustments["portrait"]["blemishRemoval"]
        .as_f64()
        .filter(|v| v.is_finite())
        .map(|v| v.round().clamp(0.0, 100.0) > 0.0)
        .unwrap_or(false)
}

/// 仅参数请求契约，尚未连接重塑分析和渲染；不得用于开启旧模型通路。
#[allow(dead_code)]
pub fn reshape_requested(adjustments: &Value, processable_face_ids: &[String]) -> bool {
    crate::portrait_reshape::request_enabled(adjustments, processable_face_ids)
}

pub fn enabled(adjustments: &Value) -> bool {
    let blemishes = blemish_strengths(adjustments);
    let hydrated = hydrated_strengths(adjustments);
    let even = even_strengths(adjustments);
    let whitening = whitening_strengths(adjustments);
    let eyes = eye_white_strengths(adjustments);
    strengths(adjustments).iter().any(|&v| v > 0.0)
        || hydrated[0] > 0.0
        || hydrated[1] > 0.0
        || even[0] > 0.0
        || even[1] > 0.0
        || whitening[0] > 0.0
        || whitening[1] > 0.0
        || eyes[0] > 0.0
        || eyes[1] > 0.0
        || blemishes[0] > 0.0
        || blemishes[2] > 0.0
}

pub fn base_adjustments(adjustments: &Value) -> Value {
    let mut result = serde_json::Map::new();
    for key in crate::cache_utils::GEOMETRY_KEYS.iter().copied().chain([
        "orientationSteps",
        "rotation",
        "flipHorizontal",
        "flipVertical",
        "aiPatches",
        "lensBlurEnabled",
        "lensBlurAmount",
        "lensBlurDiffusion",
        "lensBlurShape",
        "lensBlurMinDepth",
        "lensBlurMaxDepth",
        "lensBlurMinFade",
        "lensBlurMaxFade",
        "lensBlurDepthMap",
    ]) {
        if let Some(value) = adjustments.get(key) {
            result.insert(key.to_owned(), value.clone());
        }
    }
    result.insert(
        "sectionVisibility".into(),
        json!({"effects": adjustments["sectionVisibility"]["effects"].as_bool().unwrap_or(true)}),
    );
    // AI 美肤开关不进入基础键：人脸检测与基础修正场共用同一份缓存，AI 层独立分层计算。
    Value::Object(result)
}

/// AI 美肤变体的缓存键：基础键加后缀，让 GPU 纹理缓存区分基础与 AI 两个变体。
fn ai_variant_key(base_key: &str) -> String {
    format!("{base_key}:ai")
}

/// 普通神经祛痘变体键；与完整 AI 美肤层隔离缓存。
fn retouch_variant_key(base_key: &str) -> String {
    format!("{base_key}:retouch")
}

/// 均匀肤色变体键。
fn even_variant_key(base_key: &str) -> String {
    format!("{base_key}:even")
}

/// 独立美白变体键。
fn whitening_variant_key(base_key: &str) -> String {
    format!("{base_key}:whiten")
}

/// 从线性分析图重建 sRGB 编码图；与 analysis_image 的编码规则一致。
fn encode_linear(linear: &Rgb32FImage) -> Rgb32FImage {
    let mut encoded = linear.clone();
    for pixel in encoded.pixels_mut() {
        *pixel = Rgb(pixel.0.map(|v| linear_to_srgb(v).clamp(0.0, 1.0)));
    }
    encoded
}

fn analysis_key(path: &str, base: &Value, settings: &crate::AppSettings) -> Result<String, String> {
    let (source, _) = crate::file_management::parse_virtual_path(path);
    let metadata = std::fs::metadata(&source).map_err(|e| format!("人像源文件不可用：{e}"))?;
    let identity = source.canonicalize().map_err(|e| e.to_string())?;
    let modified = metadata
        .modified()
        .ok()
        .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
        .map(|d| d.as_nanos());
    let mut hash = Sha256::new();
    hash.update(ALGORITHM_VERSION);
    hash.update(include_bytes!("../resources/portrait-models.json"));
    hash.update(identity.to_string_lossy().as_bytes());
    hash.update(path.as_bytes());
    hash.update(metadata.len().to_le_bytes());
    hash.update(modified.unwrap_or(0).to_le_bytes());
    hash.update(base.to_string().as_bytes());
    hash.update(
        json!([
            settings.raw_highlight_compression,
            settings.linear_raw_mode,
            settings.raw_preprocessing_color_nr,
            settings.raw_preprocessing_sharpening,
            settings.apply_preprocessing_to_non_raws
        ])
        .to_string()
        .as_bytes(),
    );
    Ok(hex::encode(hash.finalize()))
}

/// 所有真实照片入口共用；仅缓存未裁剪基底，强度与输出尺寸不进入分析键。
pub fn prepare(
    app: &tauri::AppHandle,
    state: &tauri::State<AppState>,
    path: &str,
    adjustments: &Value,
) -> AnalysisResult {
    prepare_with_progress(app, state, path, adjustments, &|_| {}, false, &|| true)
}

fn check_analysis_current(current: &dyn Fn() -> bool) -> Result<(), String> {
    if current() {
        Ok(())
    } else {
        Err("Superseded or worker failed".into())
    }
}

fn prepare_with_progress(
    app: &tauri::AppHandle,
    state: &tauri::State<AppState>,
    path: &str,
    adjustments: &Value,
    progress: &dyn Fn(&str),
    prewarm_ai: bool,
    current: &dyn Fn() -> bool,
) -> AnalysisResult {
    check_analysis_current(current)?;
    let base = base_adjustments(adjustments);
    if let Some(patches) = base["aiPatches"].as_array() {
        if patches.iter().any(|p| {
            p["visible"].as_bool().unwrap_or(true)
                && !p["isLoading"].as_bool().unwrap_or(false)
                && p["patchData"].is_null()
        }) {
            return Err("人像基底缺少补丁内容，请重新打开照片后重试".into());
        }
    }
    let settings = crate::load_settings(app.clone()).unwrap_or_default();
    let key = analysis_key(path, &base, &settings)?;
    let base_analysis =
        ensure_base_analysis(app, state, path, &base, &key, &settings, progress, current)?;
    check_analysis_current(current)?;
    let mut analysis = if ai_blemish_enabled(adjustments) {
        let layer = if base_analysis.summary.face_count == 0 {
            None
        } else {
            Some(ensure_ai_layer(
                app,
                state,
                path,
                &base,
                &key,
                &settings,
                progress,
                LayerMode::Foreground,
                current,
            )?)
        };
        assemble_neural_variant(
            &base_analysis,
            layer.as_ref(),
            &ai_variant_key(&base_analysis.summary.key),
        )
    } else if ordinary_neural_blemish_enabled(adjustments) {
        let layer = if base_analysis.summary.face_count == 0 {
            None
        } else {
            Some(ensure_retouch_layer(
                app, state, path, &base, &key, &settings, progress, current,
            )?)
        };
        assemble_neural_variant(
            &base_analysis,
            layer.as_ref(),
            &retouch_variant_key(&base_analysis.summary.key),
        )
    } else {
        if prewarm_ai {
            prewarm_ai_layer(app, path, &base, &key, &base_analysis);
        }
        Arc::clone(&base_analysis)
    };
    if even_skin_enabled(adjustments) {
        let layer = ensure_even_layer(app, state, path, &base, &key, &settings, progress, current)?;
        analysis = attach_even_pixels(&analysis, &layer);
    }
    if whitening_enabled(adjustments) {
        let layer =
            ensure_whitening_layer(app, state, path, &base, &key, &settings, progress, current)?;
        analysis = attach_whitening_pixels(&analysis, &layer);
    }
    check_analysis_current(current)?;
    Ok(analysis)
}

/// 基础分析（人脸检测与修正场）的缓存与任务去重；仅在缓存未命中时执行完整管线。
fn ensure_base_analysis(
    app: &tauri::AppHandle,
    state: &tauri::State<AppState>,
    path: &str,
    base: &Value,
    key: &str,
    settings: &crate::AppSettings,
    progress: &dyn Fn(&str),
    current: &dyn Fn() -> bool,
) -> AnalysisResult {
    check_analysis_current(current)?;
    let portrait = &state.portrait_state;
    let cache = portrait
        .base_tasks
        .get_or_init(|| TaskCache::new(LEGACY_ANALYSIS_CACHE_BUDGET));
    let before = cache.stats();
    let result = cache.run_shared(
        key,
        current,
        |shared_current| {
            let (mut analysis, intermediates) = run_base_pipeline(
                app,
                state.inner(),
                settings,
                path,
                base,
                progress,
                "base",
                key,
                shared_current,
            )?;
            check_analysis_current(shared_current)?;
            analysis.summary.key = key.to_owned();
            if analysis_key(path, base, settings)? != key {
                return Err("人像源文件在分析期间已变化，请重新分析".into());
            }
            portrait
                .intermediates
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .insert(
                    key.to_owned(),
                    intermediates.byte_size(),
                    intermediates,
                    INTERMEDIATE_BUDGET,
                );
            Ok(analysis)
        },
        |analysis| Some(analysis.byte_size()),
    );
    let after = cache.stats();
    log::info!(
        "[portrait-cache] base={} entries={} bytes={} hits={} computations={}",
        if after.hits > before.hits {
            "hit"
        } else {
            "miss"
        },
        after.entries,
        after.bytes,
        after.hits,
        after.computations
    );
    match result {
        Err(error) if error == crate::portrait_analysis::CANCELLED => {
            Err("Superseded or worker failed".into())
        }
        other => other,
    }
}

/// 优先复用编辑器已解码图；否则按分析策略加载（全质量 demosaic，跳过 enhance）。
fn load_analysis_base_image(
    state: &AppState,
    path: &str,
    settings: &crate::AppSettings,
) -> Result<(Arc<DynamicImage>, bool, &'static str), String> {
    {
        let guard = state
            .original_image
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        if let Some(loaded) = guard.as_ref()
            && loaded.path == path
        {
            return Ok((Arc::clone(&loaded.image), loaded.is_raw, "reused"));
        }
    }
    let (source, _) = crate::file_management::parse_virtual_path(path);
    let source_str = source.to_str().ok_or("人像源路径编码错误")?;
    let bytes = crate::file_management::read_file_mapped(&source).map_err(|e| e.to_string())?;
    // 分析不依赖 RAW enhance；关闭 NR／锐化以省 demosaic 后处理，尺寸仍与全质量 demosaic 一致。
    let mut analysis_settings = settings.clone();
    analysis_settings.raw_preprocessing_color_nr = Some(0.0);
    analysis_settings.raw_preprocessing_sharpening = Some(0.0);
    analysis_settings.apply_preprocessing_to_non_raws = Some(false);
    let image = crate::image_loader::load_base_image_from_bytes(
        &bytes,
        source_str,
        false,
        &analysis_settings,
        None,
    )
    .map_err(|e| format!("加载人像基底失败：{e}"))?;
    Ok((
        Arc::new(image),
        crate::formats::is_raw_file(source_str),
        "decoded",
    ))
}

/// 基础分析管线：加载、几何变换、人脸检测与修正场构建；AI 美肤层不在此处执行。
/// `origin` 与 `key` 仅用于日志，定位重复分析的触发方。
#[allow(clippy::too_many_arguments)]
fn run_base_pipeline(
    app: &tauri::AppHandle,
    state: &AppState,
    settings: &crate::AppSettings,
    path: &str,
    base: &Value,
    progress: &dyn Fn(&str),
    origin: &'static str,
    key: &str,
    current: &dyn Fn() -> bool,
) -> Result<(PortraitAnalysis, AnalysisIntermediates), String> {
    check_analysis_current(current)?;
    let portrait = &state.portrait_state;
    // 在加载整图前取得独立推理锁，将峰值分析内存限制为一个任务。
    let mut models = portrait
        .lock_models_foreground()
        .map_err(|_| "人像模型锁异常，请重启应用")?;
    if models.is_none() {
        progress("loading");
        let load_start = std::time::Instant::now();
        let cache_directory = model_cache_directory(app);
        *models = Some(PortraitModels::load(
            &resource_directory(app)?,
            cache_directory.as_deref(),
        )?);
        log::info!(
            "[portrait] 模型加载 {:.1}ms",
            load_start.elapsed().as_secs_f64() * 1000.0
        );
    }
    check_analysis_current(current)?;
    progress("analyzing");
    let decode_start = std::time::Instant::now();
    let (image, is_raw, decode_source) = load_analysis_base_image(state, path, settings)?;
    check_analysis_current(current)?;
    let decode_ms = decode_start.elapsed().as_secs_f64() * 1000.0;
    let geometry_start = std::time::Instant::now();
    let warped = crate::adjustment_utils::prepare_patched_warped(image.as_ref(), base)?;
    let (transformed, _) = crate::adjustment_utils::apply_spatial_transformations(warped, base);
    let full_dimensions = transformed.dimensions();
    let geometry_ms = geometry_start.elapsed().as_secs_f64() * 1000.0;
    let preprocess_start = std::time::Instant::now();
    let (encoded, linear, luminance) = analysis_image(&transformed, is_raw);
    check_analysis_current(current)?;
    let preprocess_ms = preprocess_start.elapsed().as_secs_f64() * 1000.0;
    drop(transformed);
    let analyze_start = std::time::Instant::now();
    let segmentation = models
        .as_mut()
        .ok_or("人像模型初始化失败")?
        .analyze(&encoded)?;
    check_analysis_current(current)?;
    let analyze_ms = analyze_start.elapsed().as_secs_f64() * 1000.0;
    let faces: Vec<[f32; 4]> = segmentation.faces.iter().map(|f| f.bounds).collect();
    let fields_start = std::time::Instant::now();
    let (mut analysis, body) = build_fields(&linear, &luminance, segmentation);
    check_analysis_current(current)?;
    log::info!(
        "[portrait] 基底解码 {decode_ms:.1}ms（{decode_source}），几何变换 {geometry_ms:.1}ms，分析预处理 {preprocess_ms:.1}ms，人脸分析 {analyze_ms:.1}ms，修正场构建 {:.1}ms，来源={origin} 键={}",
        fields_start.elapsed().as_secs_f64() * 1000.0,
        key.get(..8).unwrap_or(key)
    );
    analysis.full_dimensions = full_dimensions;
    Ok((
        analysis,
        AnalysisIntermediates {
            linear,
            faces,
            body,
        },
    ))
}

/// 打开图库或图片时触发的后台预热：提前加载基础分析模型，让人像首次分析免等待。
pub fn prewarm_models(app: &tauri::AppHandle) {
    static PREWARM_STARTED: std::sync::atomic::AtomicBool =
        std::sync::atomic::AtomicBool::new(false);
    if PREWARM_STARTED.swap(true, std::sync::atomic::Ordering::SeqCst) {
        return;
    }
    let app = app.clone();
    let spawned = std::thread::Builder::new()
        .name("portrait-prewarm".into())
        .spawn(move || {
            let state = app.state::<AppState>();
            let portrait = &state.portrait_state;
            // 等待真实前台空闲窗口，而不是固定延时后与解码/缩略图争抢。
            let wait_started = Instant::now();
            std::thread::sleep(AI_PREWARM_POLL);
            while !portrait.foreground_idle_for(AI_MODEL_PREWARM_IDLE) {
                if wait_started.elapsed() >= Duration::from_secs(15) {
                    log::info!("[portrait] 基础模型预加载放弃：前台持续繁忙");
                    return;
                }
                std::thread::sleep(AI_PREWARM_POLL);
            }
            let mut models = match portrait.models.lock() {
                Ok(guard) => guard,
                Err(_) => {
                    log::warn!("[portrait] 预加载跳过：模型锁异常");
                    return;
                }
            };
            // 正式分析已开始加载时直接退出，避免重复工作。
            if models.is_some() {
                return;
            }
            let started = std::time::Instant::now();
            log::info!("[portrait] 预加载开始（后台）");
            let cache_directory = model_cache_directory(&app);
            let result = resource_directory(&app)
                .and_then(|directory| PortraitModels::load(&directory, cache_directory.as_deref()));
            match result {
                Ok(loaded) => {
                    *models = Some(loaded);
                    log::info!(
                        "[portrait] 预加载完成 {:.1}ms",
                        started.elapsed().as_secs_f64() * 1000.0
                    );
                }
                Err(e) => log::warn!("[portrait] 预加载失败：{e}"),
            }
        });
    if let Err(e) = spawned {
        log::warn!("[portrait] 预加载线程启动失败：{e}");
    }
}

/// AI 美肤层的缓存与任务去重：优先命中缓存，其次复用基础分析中间产物，仅前台兜底时重跑检测。
/// 后台模式在前台活动时以 PREWARM_YIELDED 让出，结果不写缓存；前台等到让出则改为自己计算。
#[allow(clippy::too_many_arguments)]
fn ensure_ai_layer(
    app: &tauri::AppHandle,
    state: &tauri::State<AppState>,
    path: &str,
    base: &Value,
    base_key: &str,
    settings: &crate::AppSettings,
    progress: &dyn Fn(&str),
    mode: LayerMode,
    current: &dyn Fn() -> bool,
) -> Result<Arc<AiBlemishLayer>, String> {
    check_analysis_current(current)?;
    let portrait = &state.portrait_state;
    let variant_key = ai_variant_key(base_key);
    let job = loop {
        if let Some(layer) = portrait
            .ai_layers
            .lock()
            .map_err(|_| "AI 美肤层缓存锁异常")?
            .get(base_key)
            .cloned()
        {
            return Ok(layer);
        }
        let (job, owner) = {
            let mut jobs = portrait.ai_jobs.lock().map_err(|_| "AI 美肤任务锁异常")?;
            if let Some(job) = jobs.get(&variant_key) {
                (Arc::clone(job), false)
            } else {
                let job = Arc::new(AnalysisJob::default());
                jobs.insert(variant_key.clone(), Arc::clone(&job));
                (job, true)
            }
        };
        if owner {
            break job;
        }
        // 后台不等待前台任务，也不提升它；稍后重试时会命中缓存。
        if mode == LayerMode::Background {
            return Err(PREWARM_YIELDED.into());
        }
        match wait_for_job_result(&job) {
            Err(e) if e == PREWARM_YIELDED => continue,
            result => return result,
        }
    };
    let should_stop = || {
        !current()
            || (mode == LayerMode::Background && portrait.background_should_yield(&job.promoted))
    };
    let mut result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(
        || -> Result<Arc<AiBlemishLayer>, String> {
            // 注册任务期间其他线程可能已完成计算，先复查缓存。
            if let Some(layer) = portrait
                .ai_layers
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .get(base_key)
                .cloned()
            {
                return Ok(layer);
            }
            progress("analyzing");
            // 克隆保留中间产物，供后续匀肤/美白层复用，避免重跑基础管线。
            let intermediates = portrait
                .intermediates
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .get(base_key)
                .cloned();
            let intermediates = match intermediates {
                Some(intermediates) => intermediates,
                // 后台预热不为此重跑整套基础分析。
                None if mode == LayerMode::Background => {
                    return Err("基础分析中间产物已逐出，跳过预热".into());
                }
                // 中间产物已被逐出（如切图后切回）：重跑基础管线重建，行为等同旧版全量分析。
                None => {
                    let (_, intermediates) = run_base_pipeline(
                        app,
                        state.inner(),
                        settings,
                        path,
                        base,
                        progress,
                        "ai-fallback",
                        base_key,
                        current,
                    )?;
                    intermediates
                }
            };
            if should_stop() {
                return Err(PREWARM_YIELDED.into());
            }
            let mut models = match mode {
                LayerMode::Foreground => portrait.lock_models_foreground(),
                LayerMode::Background => portrait.models.lock(),
            }
            .map_err(|_| "人像模型锁异常，请重启应用")?;
            if should_stop() {
                return Err(PREWARM_YIELDED.into());
            }
            let models = models.as_mut().ok_or("人像模型初始化失败")?;
            let started = Instant::now();
            let encoded = encode_linear(&intermediates.linear);
            let blemish = apply_ai_blemishes(
                models,
                &encoded,
                &intermediates.linear,
                &intermediates.faces,
                &should_stop,
            )?;
            check_analysis_current(current)?;
            log::info!(
                "[portrait] AI 美肤层 {:.1}ms（{}）",
                started.elapsed().as_secs_f64() * 1000.0,
                match mode {
                    LayerMode::Foreground => "前台",
                    LayerMode::Background => "后台预热",
                }
            );
            let pixel_count =
                intermediates.linear.width() as usize * intermediates.linear.height() as usize;
            Ok(Arc::new(AiBlemishLayer {
                blemish: Arc::from(blemish),
                zeros: Arc::from(vec![[0.0; 4]; pixel_count]),
            }))
        },
    ))
    .unwrap_or_else(|_| Err("AI 美肤分析发生内部错误，请重试或重启应用".into()));
    if !current() {
        result = Err("Superseded or worker failed".into());
    }
    if let Ok(layer) = &result {
        portrait
            .ai_layers
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .insert(
                base_key.to_owned(),
                layer.byte_size(),
                Arc::clone(layer),
                LEGACY_LAYER_BUDGET,
            );
    }
    // 先移出任务表再发布结果：收到让出的前台重试时不会再加入这个已结束的任务。
    portrait
        .ai_jobs
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .remove(&variant_key);
    *job.result.lock().unwrap_or_else(|e| e.into_inner()) = Some(result.clone());
    job.ready.notify_all();
    result
}

/// 普通神经祛痘层：仅检测+修复，缓存键与完整 AI 美肤隔离。
fn ensure_retouch_layer(
    app: &tauri::AppHandle,
    state: &tauri::State<AppState>,
    path: &str,
    base: &Value,
    base_key: &str,
    settings: &crate::AppSettings,
    progress: &dyn Fn(&str),
    current: &dyn Fn() -> bool,
) -> Result<Arc<AiBlemishLayer>, String> {
    check_analysis_current(current)?;
    let portrait = &state.portrait_state;
    let cache_key = retouch_variant_key(base_key);
    if let Some(layer) = portrait
        .ai_layers
        .lock()
        .map_err(|_| "神经祛痘层缓存锁异常")?
        .get(&cache_key)
        .cloned()
    {
        return Ok(layer);
    }
    let (job, owner) = {
        let mut jobs = portrait.ai_jobs.lock().map_err(|_| "神经祛痘任务锁异常")?;
        if let Some(job) = jobs.get(&cache_key) {
            (Arc::clone(job), false)
        } else {
            let job = Arc::new(AnalysisJob::default());
            jobs.insert(cache_key.clone(), Arc::clone(&job));
            (job, true)
        }
    };
    if !owner {
        return wait_for_job_result(&job);
    }
    let mut result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(
        || -> Result<Arc<AiBlemishLayer>, String> {
            if let Some(layer) = portrait
                .ai_layers
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .get(&cache_key)
                .cloned()
            {
                return Ok(layer);
            }
            progress("analyzing");
            // 克隆保留中间产物，供后续匀肤/美白层复用，避免重跑基础管线。
            let intermediates = portrait
                .intermediates
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .get(base_key)
                .cloned();
            let intermediates = match intermediates {
                Some(intermediates) => intermediates,
                None => {
                    let (_, intermediates) = run_base_pipeline(
                        app,
                        state.inner(),
                        settings,
                        path,
                        base,
                        progress,
                        "retouch-fallback",
                        base_key,
                        current,
                    )?;
                    intermediates
                }
            };
            let mut models = portrait
                .lock_models_foreground()
                .map_err(|_| "人像模型锁异常，请重启应用")?;
            let models = models.as_mut().ok_or("人像模型初始化失败")?;
            let encoded = encode_linear(&intermediates.linear);
            let blemish = apply_retouch_blemishes(
                models,
                &encoded,
                &intermediates.linear,
                &intermediates.faces,
                &|| !current(),
            )?;
            check_analysis_current(current)?;
            let pixel_count =
                intermediates.linear.width() as usize * intermediates.linear.height() as usize;
            Ok(Arc::new(AiBlemishLayer {
                blemish: Arc::from(blemish),
                zeros: Arc::from(vec![[0.0; 4]; pixel_count]),
            }))
        },
    ))
    .unwrap_or_else(|_| Err("神经祛痘分析发生内部错误，请重试或重启应用".into()));
    if !current() {
        result = Err("Superseded or worker failed".into());
    }
    if let Ok(layer) = &result {
        portrait
            .ai_layers
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .insert(
                cache_key.clone(),
                layer.byte_size(),
                Arc::clone(layer),
                LEGACY_LAYER_BUDGET,
            );
    }
    *job.result.lock().unwrap_or_else(|e| e.into_inner()) = Some(result.clone());
    job.ready.notify_all();
    portrait
        .ai_jobs
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .remove(&cache_key);
    result
}

/// 均匀肤色层：generator 匀肤增量场。
fn ensure_even_layer(
    app: &tauri::AppHandle,
    state: &tauri::State<AppState>,
    path: &str,
    base: &Value,
    base_key: &str,
    settings: &crate::AppSettings,
    progress: &dyn Fn(&str),
    current: &dyn Fn() -> bool,
) -> Result<Arc<EvenSkinLayer>, String> {
    check_analysis_current(current)?;
    let portrait = &state.portrait_state;
    let cache_key = even_variant_key(base_key);
    if let Some(layer) = portrait
        .even_layers
        .lock()
        .map_err(|_| "均匀肤色层缓存锁异常")?
        .get(&cache_key)
        .cloned()
    {
        return Ok(layer);
    }
    let (job, owner) = {
        let mut jobs = portrait
            .even_jobs
            .lock()
            .map_err(|_| "均匀肤色任务锁异常")?;
        if let Some(job) = jobs.get(&cache_key) {
            (Arc::clone(job), false)
        } else {
            let job = Arc::new(AnalysisJob::default());
            jobs.insert(cache_key.clone(), Arc::clone(&job));
            (job, true)
        }
    };
    if !owner {
        return wait_for_job_result(&job);
    }
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(
        || -> Result<Arc<EvenSkinLayer>, String> {
            if let Some(layer) = portrait
                .even_layers
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .get(&cache_key)
                .cloned()
            {
                return Ok(layer);
            }
            progress("analyzing");
            // 克隆保留中间产物，供后续美白层复用，避免重跑基础管线。
            let intermediates = portrait
                .intermediates
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .get(base_key)
                .cloned();
            let intermediates = match intermediates {
                Some(intermediates) => intermediates,
                None => {
                    let (_, intermediates) = run_base_pipeline(
                        app,
                        state.inner(),
                        settings,
                        path,
                        base,
                        progress,
                        "even-fallback",
                        base_key,
                        current,
                    )?;
                    intermediates
                }
            };
            let mut models = portrait
                .lock_models_foreground()
                .map_err(|_| "人像模型锁异常，请重启应用")?;
            let models = models.as_mut().ok_or("人像模型初始化失败")?;
            let encoded = encode_linear(&intermediates.linear);
            let even = apply_even_skin(
                models,
                &encoded,
                &intermediates.linear,
                &intermediates.faces,
                &intermediates.body,
            )?;
            check_analysis_current(current)?;
            Ok(Arc::new(EvenSkinLayer {
                even: Arc::from(even),
            }))
        },
    ))
    .unwrap_or_else(|_| Err("均匀肤色分析发生内部错误，请重试或重启应用".into()));
    if let Ok(layer) = &result {
        portrait
            .even_layers
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .insert(
                cache_key.clone(),
                layer.byte_size(),
                Arc::clone(layer),
                LEGACY_LAYER_BUDGET,
            );
    }
    *job.result.lock().unwrap_or_else(|e| e.into_inner()) = Some(result.clone());
    job.ready.notify_all();
    portrait
        .even_jobs
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .remove(&cache_key);
    result
}

/// 独立美白层：同源美白增量场。
fn ensure_whitening_layer(
    app: &tauri::AppHandle,
    state: &tauri::State<AppState>,
    path: &str,
    base: &Value,
    base_key: &str,
    settings: &crate::AppSettings,
    progress: &dyn Fn(&str),
    current: &dyn Fn() -> bool,
) -> Result<Arc<WhiteningLayer>, String> {
    check_analysis_current(current)?;
    let portrait = &state.portrait_state;
    let cache_key = whitening_variant_key(base_key);
    if let Some(layer) = portrait
        .whitening_layers
        .lock()
        .map_err(|_| "美白层缓存锁异常")?
        .get(&cache_key)
        .cloned()
    {
        return Ok(layer);
    }
    let (job, owner) = {
        let mut jobs = portrait
            .whitening_jobs
            .lock()
            .map_err(|_| "美白任务锁异常")?;
        if let Some(job) = jobs.get(&cache_key) {
            (Arc::clone(job), false)
        } else {
            let job = Arc::new(AnalysisJob::default());
            jobs.insert(cache_key.clone(), Arc::clone(&job));
            (job, true)
        }
    };
    if !owner {
        return wait_for_job_result(&job);
    }
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(
        || -> Result<Arc<WhiteningLayer>, String> {
            if let Some(layer) = portrait
                .whitening_layers
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .get(&cache_key)
                .cloned()
            {
                return Ok(layer);
            }
            progress("analyzing");
            let intermediates = portrait
                .intermediates
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .get(base_key)
                .cloned();
            let intermediates = match intermediates {
                Some(intermediates) => intermediates,
                None => {
                    let (_, intermediates) = run_base_pipeline(
                        app,
                        state.inner(),
                        settings,
                        path,
                        base,
                        progress,
                        "whiten-fallback",
                        base_key,
                        current,
                    )?;
                    intermediates
                }
            };
            let mut models = portrait
                .lock_models_foreground()
                .map_err(|_| "人像模型锁异常，请重启应用")?;
            let models = models.as_mut().ok_or("人像模型初始化失败")?;
            let encoded = encode_linear(&intermediates.linear);
            let whitening = apply_skin_whitening(
                models,
                &encoded,
                &intermediates.linear,
                &intermediates.faces,
                &intermediates.body,
            )?;
            check_analysis_current(current)?;
            Ok(Arc::new(WhiteningLayer {
                whitening: Arc::from(whitening),
            }))
        },
    ))
    .unwrap_or_else(|_| Err("美白分析发生内部错误，请重试或重启应用".into()));
    if let Ok(layer) = &result {
        portrait
            .whitening_layers
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .insert(
                cache_key.clone(),
                layer.byte_size(),
                Arc::clone(layer),
                LEGACY_LAYER_BUDGET,
            );
    }
    *job.result.lock().unwrap_or_else(|e| e.into_inner()) = Some(result.clone());
    job.ready.notify_all();
    portrait
        .whitening_jobs
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .remove(&cache_key);
    result
}

/// 基础分析完成后排队 AI 美肤层预热：前台空闲满 AI_PREWARM_IDLE 才开始，
/// 前台一旦等锁或执行就在检查点让出，避免未开启 AI 的首帧被数十秒的推理堵住。
fn prewarm_ai_layer(
    app: &tauri::AppHandle,
    path: &str,
    base: &Value,
    base_key: &str,
    base_analysis: &Arc<PortraitAnalysis>,
) {
    // 无人脸时 AI 变体直接复用基础零场，无需预热。
    if base_analysis.summary.face_count == 0 {
        return;
    }
    let state = app.state::<AppState>();
    let portrait = &state.portrait_state;
    if portrait
        .ai_layers
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .get(base_key)
        .is_some()
    {
        return;
    }
    {
        let mut pending = portrait
            .ai_prewarm_key
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        if pending.as_deref() == Some(base_key) {
            return;
        }
        *pending = Some(base_key.to_owned());
    }
    let app = app.clone();
    let path = path.to_owned();
    let base = base.clone();
    let base_key = base_key.to_owned();
    let thread_key = base_key.clone();
    let spawned = std::thread::Builder::new()
        .name("portrait-ai-prewarm".into())
        .spawn(move || {
            let state = app.state::<AppState>();
            let portrait = &state.portrait_state;
            let superseded = || {
                portrait
                    .ai_prewarm_key
                    .lock()
                    .map(|pending| pending.as_deref() != Some(base_key.as_str()))
                    .unwrap_or(true)
            };
            // 图像已切换则放弃预热，避免与新的分析争抢推理锁。
            let stale = || {
                state
                    .original_image
                    .lock()
                    .map(|image| image.as_ref().is_none_or(|image| image.path != path))
                    .unwrap_or(true)
            };
            let settings = crate::load_settings(app.clone()).unwrap_or_default();
            let started = Instant::now();
            let mut yields = 0;
            let mut retouch_models_warmed = false;
            loop {
                std::thread::sleep(AI_PREWARM_POLL);
                if superseded() || stale() {
                    break;
                }
                if started.elapsed() > AI_PREWARM_MAX_WAIT {
                    log::info!("[portrait] AI 美肤预热放弃：等待前台空闲超时");
                    break;
                }
                // 先只加载祛痘会话，隐藏首次启用的约 200MB 模型初始化；不提前跑逐脸修复。
                // try_lock 保证任何前台分析到来时都不会在这里排队争抢。
                if !retouch_models_warmed
                    && portrait.foreground_idle_for(AI_MODEL_PREWARM_IDLE)
                    && let Ok(mut models) = portrait.models.try_lock()
                    && let Some(models) = models.as_mut()
                {
                    let model_started = Instant::now();
                    match models.prewarm_retouch_models() {
                        Ok(()) => {
                            retouch_models_warmed = true;
                            log::info!(
                                "[portrait-bench] stage=retouch-model-prewarm elapsed_ms={:.1}",
                                model_started.elapsed().as_secs_f64() * 1000.0
                            );
                        }
                        Err(error) => {
                            log::warn!("AI 祛痘模型预热失败：{error}");
                            break;
                        }
                    }
                }
                if !portrait.foreground_idle_for(AI_PREWARM_IDLE) {
                    continue;
                }
                match ensure_ai_layer(
                    &app,
                    &state,
                    &path,
                    &base,
                    &base_key,
                    &settings,
                    &|_| {},
                    LayerMode::Background,
                    &|| true,
                ) {
                    Ok(_) => break,
                    Err(e) if e == PREWARM_YIELDED => {
                        yields += 1;
                        log::info!("[portrait] AI 美肤预热让出（第 {yields} 次）");
                        if yields >= AI_PREWARM_MAX_YIELDS {
                            break;
                        }
                    }
                    Err(e) => {
                        log::warn!("AI 美肤层预热失败：{e}");
                        break;
                    }
                }
            }
            let mut pending = portrait
                .ai_prewarm_key
                .lock()
                .unwrap_or_else(|e| e.into_inner());
            if pending.as_deref() == Some(base_key.as_str()) {
                *pending = None;
            }
        });
    if let Err(e) = spawned {
        log::warn!("[portrait] AI 美肤预热线程启动失败：{e}");
        let mut pending = portrait
            .ai_prewarm_key
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        if pending.as_deref() == Some(thread_key.as_str()) {
            *pending = None;
        }
    }
}

/// 由基础变体与神经祛痘/美肤层装配变体：仅克隆 Arc，不复制像素数据。
fn assemble_neural_variant(
    base: &Arc<PortraitAnalysis>,
    layer: Option<&Arc<AiBlemishLayer>>,
    variant_key: &str,
) -> Arc<PortraitAnalysis> {
    let (blemish_pixels, texture_pixels) = match layer {
        Some(layer) => (Arc::clone(&layer.blemish), Arc::clone(&layer.zeros)),
        // 无人脸：基础瑕疵场与质感场本就为零，直接共享。
        None => (
            Arc::clone(&base.blemish_pixels),
            Arc::clone(&base.texture_pixels),
        ),
    };
    Arc::new(PortraitAnalysis {
        summary: PortraitSummary {
            key: variant_key.to_owned(),
            face_count: base.summary.face_count,
            skipped_faces: base.summary.skipped_faces,
            has_skin: base.summary.has_skin,
        },
        width: base.width,
        height: base.height,
        full_dimensions: base.full_dimensions,
        pixels: Arc::clone(&base.pixels),
        frequency_pixels: Arc::clone(&base.frequency_pixels),
        frequency_detail_pixels: Arc::clone(&base.frequency_detail_pixels),
        hydrated_pixels: Arc::clone(&base.hydrated_pixels),
        even_pixels: Arc::clone(&base.even_pixels),
        whitening_pixels: Arc::clone(&base.whitening_pixels),
        evenness_chroma_pixels: Arc::clone(&base.evenness_chroma_pixels),
        blemish_pixels,
        texture_pixels,
        shine_pixels: Arc::clone(&base.shine_pixels),
        eye_white_pixels: Arc::clone(&base.eye_white_pixels),
    })
}

#[tauri::command]
pub async fn prepare_portrait_analysis(
    app: tauri::AppHandle,
    path: String,
    generation: u64,
    mut js_adjustments: Value,
    on_progress: tauri::ipc::Channel<String>,
) -> Result<PortraitSummary, String> {
    let request_epoch = app
        .state::<AppState>()
        .portrait_state
        .begin_analysis(generation)?;
    tokio::task::spawn_blocking(move || {
        let state = app.state::<AppState>();
        let _foreground = state.portrait_state.foreground();
        let current = || state.portrait_state.analysis_current(request_epoch);
        check_analysis_current(&current)?;
        let generation = {
            let mut epoch = state.lock_preview_epoch();
            if state
                .original_image
                .lock()
                .unwrap()
                .as_ref()
                .is_none_or(|image| image.path != path)
            {
                return Err("Superseded or worker failed".into());
            }
            *epoch += 1;
            crate::adjustment_utils::hydrate_adjustments(&state, &mut js_adjustments);
            state
                .load_image_generation
                .load(std::sync::atomic::Ordering::SeqCst)
        };
        let analysis = prepare_with_progress(
            &app,
            &state,
            &path,
            &js_adjustments,
            &|phase| {
                let _ = on_progress.send(phase.to_owned());
            },
            true,
            &current,
        )?;
        check_analysis_current(&current)?;
        if generation
            != state
                .load_image_generation
                .load(std::sync::atomic::Ordering::SeqCst)
        {
            return Err("Superseded or worker failed".into());
        }
        Ok(analysis.summary.clone())
    })
    .await
    .map_err(|e| format!("人像分析任务失败：{e}"))?
}

#[tauri::command]
pub fn cancel_portrait_analysis(state: tauri::State<AppState>, generation: u64) {
    state.portrait_state.cancel_analysis(generation);
}

pub fn prepare_render(
    app: &tauri::AppHandle,
    state: &tauri::State<AppState>,
    path: &str,
    adjustments: &Value,
) -> Result<Option<PortraitRenderMaps>, String> {
    if !enabled(adjustments) {
        return Ok(None);
    }
    // 旧版预览与导出共用入口：执行期间压住后台 AI 预热。
    let _foreground = state.portrait_state.foreground();
    Ok(Some(
        prepare(app, state, path, adjustments)?.for_render(adjustments),
    ))
}

/// 交互预览 bake 键：只含几何、分析场版本与人像场强度，不含曝光/曲线等调色参数。
pub fn neural_preview_bake_key(
    path: &str,
    adjustments: &Value,
    analysis_key: &str,
    width: u32,
    height: u32,
    transform_hash: u64,
) -> String {
    crate::portrait_render::content_key(&json!([
        NEURAL_PREVIEW_BAKE_VERSION,
        path,
        transform_hash,
        analysis_key,
        width,
        height,
        strengths(adjustments),
        hydrated_strengths(adjustments),
        even_strengths(adjustments),
        whitening_strengths(adjustments),
        eye_white_strengths(adjustments),
        blemish_strengths(adjustments),
        frequency_band_strengths(adjustments),
        adjustments["sectionVisibility"]["portraitSkin"]
            .as_bool()
            .unwrap_or(true),
        adjustments["sectionVisibility"]["portraitBlemishes"]
            .as_bool()
            .unwrap_or(true),
        adjustments["sectionVisibility"]["portraitEyes"]
            .as_bool()
            .unwrap_or(true),
    ]))
}

/// 命中则复用已 bake 基底；未命中则在预览分辨率把人像场写入**场景线性**颜色基底。
///
/// 编辑器预览底图在加载时已是线性（含非 RAW 的 `apply_srgb_to_linear`），此处只做与
/// WGSL 人像段一致的线性域叠加，不再做 sRGB↔线性往返，避免全强度瑕疵增量被烤成白斑。
pub fn ensure_neural_preview_base(
    state: &tauri::State<AppState>,
    key: &str,
    source: &DynamicImage,
    maps: &PortraitRenderMaps,
    adjustments: &Value,
) -> Result<Arc<DynamicImage>, String> {
    let portrait = &state.portrait_state;
    if let Some(hit) = portrait
        .neural_preview_bases
        .lock()
        .map_err(|_| "神经预览 bake 缓存锁异常")?
        .get(key)
        .cloned()
    {
        portrait
            .neural_preview_bake_hits
            .fetch_add(1, Ordering::Relaxed);
        log::debug!(
            "[portrait-preview] neural bake hit hits={} misses={} size={}x{}",
            portrait.neural_preview_bake_hits.load(Ordering::Relaxed),
            portrait.neural_preview_bake_misses.load(Ordering::Relaxed),
            source.width(),
            source.height()
        );
        return Ok(hit);
    }
    let started = Instant::now();
    let baked = bake_neural_preview_onto(source, maps, adjustments)?;
    let bytes = baked.as_bytes().len();
    let arc = Arc::new(baked);
    portrait
        .neural_preview_bases
        .lock()
        .map_err(|_| "神经预览 bake 缓存锁异常")?
        .insert(
            key.to_owned(),
            bytes,
            Arc::clone(&arc),
            NEURAL_PREVIEW_BAKE_BUDGET,
        );
    portrait
        .neural_preview_bake_misses
        .fetch_add(1, Ordering::Relaxed);
    log::debug!(
        "[portrait-preview] neural bake miss hits={} misses={} size={}x{} elapsed={:?}",
        portrait.neural_preview_bake_hits.load(Ordering::Relaxed),
        portrait.neural_preview_bake_misses.load(Ordering::Relaxed),
        source.width(),
        source.height(),
        started.elapsed()
    );
    Ok(arc)
}

fn bake_neural_preview_onto(
    source: &DynamicImage,
    maps: &PortraitRenderMaps,
    adjustments: &Value,
) -> Result<DynamicImage, String> {
    let analysis = &maps.analysis;
    let aw = analysis.width;
    let ah = analysis.height;
    if aw == 0 || ah == 0 {
        return Err("美肤场尺寸无效".into());
    }
    let mut strengths = strengths(adjustments);
    let mut hydrated = hydrated_strengths(adjustments);
    let mut even = even_strengths(adjustments);
    let mut whitening = whitening_strengths(adjustments);
    let mut eyes = eye_white_strengths(adjustments);
    let mut blemishes = blemish_strengths(adjustments);
    let bands = frequency_band_strengths(adjustments);
    let contour_bands = contour_band_strengths(adjustments);
    if analysis.summary.face_count == 0 {
        strengths[2] = 0.0;
        strengths[3] = 0.0;
        hydrated[0] = 0.0;
        even[0] = 0.0;
        whitening[0] = 0.0;
        eyes = [0.0; 4];
        blemishes = [0.0; 4];
    }
    let need_freq = strengths[2] > 0.0 || strengths[3] > 0.0;
    let need_hydrated = hydrated[0] > 0.0 || hydrated[1] > 0.0;
    let need_even = even[0] > 0.0 || even[1] > 0.0;
    let need_whitening = whitening[0] > 0.0 || whitening[1] > 0.0;
    let need_eyes = eyes[0] > 0.0 || eyes[1] > 0.0;
    let need_blemish = blemishes[0] > 0.0;
    let need_shine = blemishes[2] > 0.0;
    let need_ev = strengths[0] > 0.0 || strengths[1] > 0.0;
    if !(need_freq
        || need_hydrated
        || need_even
        || need_whitening
        || need_eyes
        || need_blemish
        || need_shine
        || need_ev)
    {
        return Ok(source.clone());
    }

    let rgba = source.to_rgba32f();
    let width = rgba.width();
    let height = rgba.height();
    let uv_rect = maps.uv_rect;
    let full_w = width as f32;
    let full_h = height as f32;
    let pixels = rgba.into_raw();
    let out: Vec<f32> = (0..height)
        .into_par_iter()
        .flat_map_iter(|y| {
            let mut row = Vec::with_capacity(width as usize * 4);
            for x in 0..width {
                let i = ((y * width + x) * 4) as usize;
                // 预览基底已是场景线性，直接叠加场增量（与 shader 人像段一致）。
                let mut r = pixels[i];
                let mut g = pixels[i + 1];
                let mut b = pixels[i + 2];
                let a = pixels[i + 3];
                let luma = 0.2126 * r + 0.7152 * g + 0.0722 * b;
                let uv = [
                    uv_rect[0] + (x as f32 + 0.5) / full_w * uv_rect[2],
                    uv_rect[1] + (y as f32 + 0.5) / full_h * uv_rect[3],
                ];
                if need_freq {
                    let delta = sample_region_weighted_rgb(
                        &analysis.frequency_pixels,
                        Some(&analysis.frequency_detail_pixels),
                        &analysis.pixels,
                        aw,
                        ah,
                        uv,
                        luma,
                        |region| {
                            if region > 0.0 {
                                (strengths[2], bands[0], bands[1])
                            } else {
                                (strengths[3], bands[2], bands[3])
                            }
                        },
                        true,
                    );
                    r = (r + delta[0]).max(0.0);
                    g = (g + delta[1]).max(0.0);
                    b = (b + delta[2]).max(0.0);
                }
                if need_hydrated {
                    let delta = sample_region_weighted_rgb(
                        &analysis.hydrated_pixels,
                        None,
                        &analysis.pixels,
                        aw,
                        ah,
                        uv,
                        luma,
                        |region| {
                            let s = if region > 0.0 {
                                hydrated[0]
                            } else {
                                hydrated[1]
                            };
                            (s, 1.0, 1.0)
                        },
                        false,
                    );
                    r = (r + delta[0]).max(0.0);
                    g = (g + delta[1]).max(0.0);
                    b = (b + delta[2]).max(0.0);
                }
                if need_even {
                    let delta = sample_region_weighted_rgb(
                        &analysis.even_pixels,
                        None,
                        &analysis.pixels,
                        aw,
                        ah,
                        uv,
                        luma,
                        |region| {
                            let s = if region > 0.0 { even[0] } else { even[1] };
                            (s, 1.0, 1.0)
                        },
                        false,
                    );
                    r = (r + delta[0]).max(0.0);
                    g = (g + delta[1]).max(0.0);
                    b = (b + delta[2]).max(0.0);
                }
                if need_whitening {
                    let delta = sample_whitening_delta(
                        &analysis.whitening_pixels,
                        &analysis.pixels,
                        aw,
                        ah,
                        uv,
                        luma,
                        whitening[0],
                        whitening[1],
                    );
                    r = (r + delta[0]).max(0.0);
                    g = (g + delta[1]).max(0.0);
                    b = (b + delta[2]).max(0.0);
                }
                if need_blemish {
                    let delta = sample_blemish_delta(
                        &analysis.blemish_pixels,
                        &analysis.texture_pixels,
                        aw,
                        ah,
                        uv,
                        blemishes[0],
                        blemishes[1],
                    );
                    r = (r + delta[0]).max(0.0);
                    g = (g + delta[1]).max(0.0);
                    b = (b + delta[2]).max(0.0);
                }
                if need_shine {
                    let delta = sample_weighted_rgb(
                        &analysis.shine_pixels,
                        &analysis.pixels,
                        aw,
                        ah,
                        uv,
                        luma,
                        blemishes[2],
                    );
                    if delta.iter().any(|v| v.abs() > 0.0) {
                        let current = [r.max(0.0), g.max(0.0), b.max(0.0)];
                        let delta = apply_shine_safety(current, delta);
                        r = (current[0] + delta[0]).max(0.0);
                        g = (current[1] + delta[1]).max(0.0);
                        b = (current[2] + delta[2]).max(0.0);
                    }
                }
                if need_eyes {
                    let delta = sample_eye_white_delta(
                        &analysis.eye_white_pixels,
                        aw,
                        ah,
                        uv,
                        luma,
                        eyes[0],
                        eyes[1],
                        [r, g, b],
                    );
                    r = (r + delta[0]).max(0.0);
                    g = (g + delta[1]).max(0.0);
                    b = (b + delta[2]).max(0.0);
                }
                if need_ev {
                    let ev = sample_portrait_ev(
                        &analysis.pixels,
                        aw,
                        ah,
                        uv,
                        luma,
                        strengths,
                        contour_bands,
                    );
                    let gain = 2.0_f32.powf(ev);
                    r *= gain;
                    g *= gain;
                    b *= gain;
                    // 先 EV 再色度，避免亮度增益放大色度脏边。
                    if strengths[0] > 0.0 {
                        let chroma_delta = sample_weighted_rgb(
                            &analysis.evenness_chroma_pixels,
                            &analysis.pixels,
                            aw,
                            ah,
                            uv,
                            luma,
                            strengths[0],
                        );
                        r = (r + chroma_delta[0]).max(0.0);
                        g = (g + chroma_delta[1]).max(0.0);
                        b = (b + chroma_delta[2]).max(0.0);
                    }
                }
                row.extend_from_slice(&[r, g, b, a]);
            }
            row
        })
        .collect();
    let buffer = image::Rgba32FImage::from_raw(width, height, out)
        .ok_or("神经预览 bake 像素缓冲长度错误")?;
    Ok(DynamicImage::ImageRgba32F(buffer))
}

#[inline]
fn field_index(width: u32, height: u32, x: i32, y: i32) -> usize {
    let x = x.clamp(0, width.saturating_sub(1) as i32) as u32;
    let y = y.clamp(0, height.saturating_sub(1) as i32) as u32;
    (y * width + x) as usize
}

fn sample_blemish_delta(
    blemish: &[[f32; 4]],
    texture: &[[f32; 4]],
    width: u32,
    height: u32,
    uv: [f32; 2],
    amount: f32,
    texture_amount: f32,
) -> [f32; 3] {
    if amount <= 0.0 || blemish.len() != (width * height) as usize {
        return [0.0; 3];
    }
    let position = [uv[0] * width as f32 - 0.5, uv[1] * height as f32 - 0.5];
    let origin = [position[0].floor() as i32, position[1].floor() as i32];
    let fraction = [
        position[0] - origin[0] as f32,
        position[1] - origin[1] as f32,
    ];
    let nearest = [
        (position[0] + 0.5).floor() as i32,
        (position[1] + 0.5).floor() as i32,
    ];
    let nearest_i = field_index(width, height, nearest[0], nearest[1]);
    if blemish[nearest_i][3] <= 0.001 {
        return [0.0; 3];
    }
    let mut correction = [0.0_f32; 3];
    for y in 0..2 {
        for x in 0..2 {
            let coord = [origin[0] + x, origin[1] + y];
            let i = field_index(width, height, coord[0], coord[1]);
            let repair = blemish[i];
            let detail = texture.get(i).copied().unwrap_or([0.0; 4]);
            let wx = if x == 1 {
                fraction[0]
            } else {
                1.0 - fraction[0]
            };
            let wy = if y == 1 {
                fraction[1]
            } else {
                1.0 - fraction[1]
            };
            let w = wx * wy;
            correction[0] += (repair[0] + texture_amount * detail[0]) * w;
            correction[1] += (repair[1] + texture_amount * detail[1]) * w;
            correction[2] += (repair[2] + texture_amount * detail[2]) * w;
        }
    }
    [
        correction[0] * amount,
        correction[1] * amount,
        correction[2] * amount,
    ]
}

fn sample_eye_white_delta(
    field: &[[f32; 4]],
    width: u32,
    height: u32,
    uv: [f32; 2],
    luminance: f32,
    brighten: f32,
    redness: f32,
    current: [f32; 3],
) -> [f32; 3] {
    if (brighten <= 0.0 && redness <= 0.0) || field.len() != (width * height) as usize {
        return [0.0; 3];
    }
    let position = [uv[0] * width as f32 - 0.5, uv[1] * height as f32 - 0.5];
    let origin = [position[0].floor() as i32, position[1].floor() as i32];
    let fraction = [
        position[0] - origin[0] as f32,
        position[1] - origin[1] as f32,
    ];
    let nearest = [
        (position[0] + 0.5).floor() as i32,
        (position[1] + 0.5).floor() as i32,
    ];
    let nearest_i = field_index(width, height, nearest[0], nearest[1]);
    // 允许仅有祛红增量而 A 很弱（血丝在边缘），或仅有提亮 A。
    let nearest_sample = field[nearest_i];
    let nearest_energy = nearest_sample[3]
        .max(nearest_sample[0].abs())
        .max(nearest_sample[1].abs())
        .max(nearest_sample[2].abs());
    if nearest_energy <= 0.001 {
        return [0.0; 3];
    }
    let mut rgb = [0.0_f32; 3];
    let mut mask = 0.0_f32;
    let mut weight_sum = 0.0_f32;
    for y in 0..2 {
        for x in 0..2 {
            let coord = [origin[0] + x, origin[1] + y];
            let i = field_index(width, height, coord[0], coord[1]);
            let sample = field[i];
            let wx = if x == 1 {
                fraction[0]
            } else {
                1.0 - fraction[0]
            };
            let wy = if y == 1 {
                fraction[1]
            } else {
                1.0 - fraction[1]
            };
            let weight = wx * wy;
            rgb[0] += sample[0] * weight;
            rgb[1] += sample[1] * weight;
            rgb[2] += sample[2] * weight;
            mask += sample[3] * weight;
            weight_sum += weight;
        }
    }
    let inv = 1.0 / weight_sum.max(0.000001);
    let mask = (mask * inv).clamp(0.0, 1.0);
    // 提亮：目标亮度软上限，与祛红解耦；默认不做色度拉动。
    let soft_ceiling = EYE_WHITE_SOFT_CEILING;
    let gap = (soft_ceiling - luminance).max(0.0);
    let gap_curve = gap * (0.40 + 0.60 * (gap / soft_ceiling.max(1e-3)).clamp(0.0, 1.0));
    let shadow_protect = smoothstep(0.04, 0.16, luminance);
    let highlight_protect = 1.0 - smoothstep(soft_ceiling - 0.06, soft_ceiling, luminance);
    let lift = mask
        * brighten
        * EYE_WHITE_BRIGHTEN_GAIN
        * gap_curve
        * shadow_protect
        * highlight_protect;
    let chroma_w = mask * brighten * EYE_WHITE_CHROMA_GAIN;
    let neutral = [luminance, luminance, luminance];
    [
        rgb[0] * inv * redness + lift + (neutral[0] - current[0]) * chroma_w,
        rgb[1] * inv * redness + lift + (neutral[1] - current[1]) * chroma_w,
        rgb[2] * inv * redness + lift + (neutral[2] - current[2]) * chroma_w,
    ]
}

fn sample_weighted_rgb(
    field: &[[f32; 4]],
    reference: &[[f32; 4]],
    width: u32,
    height: u32,
    uv: [f32; 2],
    luminance: f32,
    strength: f32,
) -> [f32; 3] {
    if strength <= 0.0 || field.len() != (width * height) as usize {
        return [0.0; 3];
    }
    let position = [uv[0] * width as f32 - 0.5, uv[1] * height as f32 - 0.5];
    let origin = [position[0].floor() as i32, position[1].floor() as i32];
    let fraction = [
        position[0] - origin[0] as f32,
        position[1] - origin[1] as f32,
    ];
    let nearest = [
        (position[0] + 0.5).floor() as i32,
        (position[1] + 0.5).floor() as i32,
    ];
    let nearest_i = field_index(width, height, nearest[0], nearest[1]);
    if field[nearest_i][3] <= 0.001 {
        return [0.0; 3];
    }
    let mut correction = [0.0_f32; 3];
    let mut weight_sum = 0.0_f32;
    for y in 0..2 {
        for x in 0..2 {
            let coord = [origin[0] + x, origin[1] + y];
            let i = field_index(width, height, coord[0], coord[1]);
            let sample = field[i];
            let ref_luma = reference.get(i).map(|p| p[3]).unwrap_or(0.0).max(0.000001);
            let wx = if x == 1 {
                fraction[0]
            } else {
                1.0 - fraction[0]
            };
            let wy = if y == 1 {
                fraction[1]
            } else {
                1.0 - fraction[1]
            };
            let difference = (luminance.max(0.000001).log2() - ref_luma.log2()).abs();
            let weight = wx * wy * (-difference * 2.0).exp();
            correction[0] += sample[0] * weight;
            correction[1] += sample[1] * weight;
            correction[2] += sample[2] * weight;
            weight_sum += weight;
        }
    }
    let inv = strength / weight_sum.max(0.000001);
    [
        correction[0] * inv,
        correction[1] * inv,
        correction[2] * inv,
    ]
}

/// 美白专用采样：每个邻点用自己的 region 算强度，异号邻点也参与。
/// 避免交界处只累加同侧、分母却含另一侧，把过渡带压暗。
fn sample_whitening_delta(
    field: &[[f32; 4]],
    reference: &[[f32; 4]],
    width: u32,
    height: u32,
    uv: [f32; 2],
    luminance: f32,
    face: f32,
    body: f32,
) -> [f32; 3] {
    if (face <= 0.0 && body <= 0.0) || field.len() != (width * height) as usize {
        return [0.0; 3];
    }
    let position = [uv[0] * width as f32 - 0.5, uv[1] * height as f32 - 0.5];
    let origin = [position[0].floor() as i32, position[1].floor() as i32];
    let fraction = [
        position[0] - origin[0] as f32,
        position[1] - origin[1] as f32,
    ];
    let mut correction = [0.0_f32; 3];
    let mut weight_sum = 0.0_f32;
    for y in 0..2 {
        for x in 0..2 {
            let coord = [origin[0] + x, origin[1] + y];
            let i = field_index(width, height, coord[0], coord[1]);
            let sample = field[i];
            let strength = whitening_strength_for_region(sample[3], face, body);
            if strength <= 0.0 {
                continue;
            }
            let ref_luma = reference.get(i).map(|p| p[3]).unwrap_or(0.0).max(0.000001);
            let wx = if x == 1 {
                fraction[0]
            } else {
                1.0 - fraction[0]
            };
            let wy = if y == 1 {
                fraction[1]
            } else {
                1.0 - fraction[1]
            };
            let difference = (luminance.max(0.000001).log2() - ref_luma.log2()).abs();
            let weight = wx * wy * (-difference * 2.0).exp();
            correction[0] += sample[0] * weight * strength;
            correction[1] += sample[1] * weight * strength;
            correction[2] += sample[2] * weight * strength;
            weight_sum += weight;
        }
    }
    if weight_sum <= 1e-6 {
        return [0.0; 3];
    }
    let inv = 1.0 / weight_sum;
    [
        correction[0] * inv,
        correction[1] * inv,
        correction[2] * inv,
    ]
}

/// region 选择器返回 (master, low_amt, high_amt)；frequency 模式用 detail。
fn sample_region_weighted_rgb(
    field: &[[f32; 4]],
    detail: Option<&[[f32; 4]]>,
    reference: &[[f32; 4]],
    width: u32,
    height: u32,
    uv: [f32; 2],
    luminance: f32,
    select: impl Fn(f32) -> (f32, f32, f32),
    frequency_mode: bool,
) -> [f32; 3] {
    if field.len() != (width * height) as usize {
        return [0.0; 3];
    }
    let position = [uv[0] * width as f32 - 0.5, uv[1] * height as f32 - 0.5];
    let origin = [position[0].floor() as i32, position[1].floor() as i32];
    let fraction = [
        position[0] - origin[0] as f32,
        position[1] - origin[1] as f32,
    ];
    let nearest = [
        (position[0] + 0.5).floor() as i32,
        (position[1] + 0.5).floor() as i32,
    ];
    let nearest_i = field_index(width, height, nearest[0], nearest[1]);
    let region = field[nearest_i][3];
    if region.abs() <= 0.001 {
        return [0.0; 3];
    }
    let (master, low_amt, high_amt) = select(region);
    if master <= 0.0 {
        return [0.0; 3];
    }
    let mut correction = [0.0_f32; 3];
    let mut weight_sum = 0.0_f32;
    for y in 0..2 {
        for x in 0..2 {
            let coord = [origin[0] + x, origin[1] + y];
            let i = field_index(width, height, coord[0], coord[1]);
            let sample = field[i];
            let ref_luma = reference.get(i).map(|p| p[3]).unwrap_or(0.0).max(0.000001);
            let wx = if x == 1 {
                fraction[0]
            } else {
                1.0 - fraction[0]
            };
            let wy = if y == 1 {
                fraction[1]
            } else {
                1.0 - fraction[1]
            };
            let difference = (luminance.max(0.000001).log2() - ref_luma.log2()).abs();
            let weight = wx * wy * (-difference * 2.0).exp();
            let same_region = sample[3] * region > 0.0;
            if same_region {
                if frequency_mode {
                    let d = detail.and_then(|d| d.get(i)).copied().unwrap_or([0.0; 4]);
                    correction[0] += (low_amt * sample[0] + (high_amt - 1.0) * d[0]) * weight;
                    correction[1] += (low_amt * sample[1] + (high_amt - 1.0) * d[1]) * weight;
                    correction[2] += (low_amt * sample[2] + (high_amt - 1.0) * d[2]) * weight;
                    // 频分：异区邻点不参与归一化，避免身/脸交界被身体滑块牵动。
                    weight_sum += weight;
                } else {
                    correction[0] += sample[0] * weight;
                    correction[1] += sample[1] * weight;
                    correction[2] += sample[2] * weight;
                }
            }
            // 非频分仍让保护区参与归一化，避免边界修正被放大。
            if !frequency_mode {
                weight_sum += weight;
            }
        }
    }
    let inv = master / weight_sum.max(0.000001);
    [
        correction[0] * inv,
        correction[1] * inv,
        correction[2] * inv,
    ]
}

fn sample_portrait_ev(
    field: &[[f32; 4]],
    width: u32,
    height: u32,
    uv: [f32; 2],
    luminance: f32,
    strengths: [f32; 4],
    contour_bands: [f32; 4],
) -> f32 {
    if field.len() != (width * height) as usize {
        return 0.0;
    }
    let position = [uv[0] * width as f32 - 0.5, uv[1] * height as f32 - 0.5];
    let origin = [position[0].floor() as i32, position[1].floor() as i32];
    let fraction = [
        position[0] - origin[0] as f32,
        position[1] - origin[1] as f32,
    ];
    let nearest = [
        (position[0] + 0.5).floor() as i32,
        (position[1] + 0.5).floor() as i32,
    ];
    let nearest_i = field_index(width, height, nearest[0], nearest[1]);
    let nearest = field[nearest_i];
    if nearest[0].abs().max(nearest[1].abs()).max(nearest[2].abs()) <= 0.001 {
        return 0.0;
    }
    let mut correction = [0.0_f32; 3];
    let mut weight_sum = 0.0_f32;
    for y in 0..2 {
        for x in 0..2 {
            let coord = [origin[0] + x, origin[1] + y];
            let i = field_index(width, height, coord[0], coord[1]);
            let sample = field[i];
            let wx = if x == 1 {
                fraction[0]
            } else {
                1.0 - fraction[0]
            };
            let wy = if y == 1 {
                fraction[1]
            } else {
                1.0 - fraction[1]
            };
            let ref_luma = sample[3].max(0.000001);
            let difference = (luminance.max(0.000001).log2() - ref_luma.log2()).abs();
            let weight = wx * wy * (-difference * 2.0).exp();
            correction[0] += sample[0] * weight;
            correction[1] += sample[1] * weight;
            correction[2] += sample[2] * weight;
            weight_sum += weight;
        }
    }
    let inv = 1.0 / weight_sum.max(0.000001);
    let contour =
        correction[1] * inv * contour_bands[0] + correction[2] * inv * contour_bands[1];
    let ev = correction[0] * inv * strengths[0] + contour * strengths[1];
    ev.clamp(-0.85, 0.85)
}

/// 缩略图等低分辨率入口也先完成全尺寸变换与裁剪，再统一缩放。
/// 不使用 RAW 内嵌预览或先缩图再旋转，确保分析坐标与导出一致。
pub fn render_reduced(
    app: &tauri::AppHandle,
    state: &tauri::State<AppState>,
    path: &str,
    adjustments: &Value,
    max_edge: u32,
) -> Result<DynamicImage, String> {
    let settings = crate::load_settings(app.clone()).unwrap_or_default();
    let (source, _) = crate::file_management::parse_virtual_path(path);
    let source_str = source.to_str().ok_or("人像源路径编码错误")?;
    let is_raw = crate::formats::is_raw_file(source_str);
    let (preview, scale, offset) = {
        let bytes = crate::file_management::read_file_mapped(&source).map_err(|e| e.to_string())?;
        let image = crate::image_loader::load_base_image_from_bytes(
            &bytes, source_str, false, &settings, None,
        )
        .map_err(|e| e.to_string())?;
        let warped = crate::adjustment_utils::prepare_patched_warped(&image, adjustments)?;
        let (transformed, offset) =
            crate::adjustment_utils::apply_spatial_transformations(warped, adjustments);
        let preview =
            crate::image_processing::downscale_f32_image(&transformed, max_edge, max_edge);
        let scale = preview.width() as f32 / transformed.width() as f32;
        (preview, scale, offset)
    };
    let definitions: Vec<crate::mask_generation::MaskDefinition> =
        serde_json::from_value(adjustments["masks"].clone()).unwrap_or_default();
    let masks: Vec<_> = definitions
        .iter()
        .filter_map(|def| {
            crate::get_cached_or_generate_mask(
                state,
                def,
                preview.width(),
                preview.height(),
                scale,
                (offset.0 * scale, offset.1 * scale),
                adjustments,
            )
        })
        .collect();
    let tm = crate::image_processing::resolve_tonemapper_override(&settings, is_raw);
    let mut params = crate::image_processing::get_all_adjustments_from_json(
        adjustments,
        is_raw,
        crate::white_balance::as_shot_white_balance(path),
        tm,
    );
    params.global.show_clipping = 0;
    let lut = adjustments["lutPath"]
        .as_str()
        .and_then(|p| crate::lut_processing::get_or_load_lut(state, p).ok());
    let maps = prepare_render(app, state, path, adjustments)?;
    let context = crate::gpu_processing::get_or_init_gpu_context(state, app)?;
    crate::gpu_processing::process_and_get_dynamic_image(
        &context,
        state,
        &preview,
        crate::cache_utils::calculate_full_job_hash(path, adjustments),
        crate::gpu_processing::RenderRequest {
            adjustments: params,
            mask_bitmaps: &masks,
            lut,
            roi: None,
            portrait_maps: maps,
        },
        "portrait_reduced_preview",
    )
}

fn srgb_to_linear(value: f32) -> f32 {
    if value <= 0.04045 {
        value / 12.92
    } else {
        ((value + 0.055) / 1.055).powf(2.4)
    }
}

fn linear_to_srgb(value: f32) -> f32 {
    if value <= 0.0031308 {
        value * 12.92
    } else {
        1.055 * value.powf(1.0 / 2.4) - 0.055
    }
}

fn analysis_image(image: &DynamicImage, is_raw: bool) -> (Rgb32FImage, Rgb32FImage, Vec<f32>) {
    analysis_image_with_edge(image, is_raw, ANALYSIS_EDGE)
}

fn analysis_image_with_edge(
    image: &DynamicImage,
    is_raw: bool,
    edge: u32,
) -> (Rgb32FImage, Rgb32FImage, Vec<f32>) {
    let factor = (edge as f32 / image.width().max(image.height()) as f32).min(1.0);
    let width = (image.width() as f32 * factor).round().max(1.0) as u32;
    let height = (image.height() as f32 * factor).round().max(1.0) as u32;
    let mut linear_image = image.to_rgb32f();
    linear_image.as_mut().par_iter_mut().for_each(|v| {
        *v = if v.is_finite() { v.max(0.0) } else { 0.0 };
        if !is_raw {
            *v = srgb_to_linear(*v);
        }
    });
    let reduced = image::imageops::resize(
        &linear_image,
        width,
        height,
        image::imageops::FilterType::Triangle,
    );
    let mut encoded = reduced.clone();
    let mut luminance = vec![0.0; (width * height) as usize];
    luminance
        .par_iter_mut()
        .zip(encoded.as_mut().par_chunks_mut(3))
        .zip(reduced.as_raw().par_chunks(3))
        .for_each(|((value, target), source)| {
            *value = source[0] * 0.2126 + source[1] * 0.7152 + source[2] * 0.0722;
            target[0] = linear_to_srgb(source[0]).clamp(0.0, 1.0);
            target[1] = linear_to_srgb(source[1]).clamp(0.0, 1.0);
            target[2] = linear_to_srgb(source[2]).clamp(0.0, 1.0);
        });
    (encoded, reduced, luminance)
}

fn smoothstep(low: f32, high: f32, value: f32) -> f32 {
    let t = ((value - low) / (high - low)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

pub fn sample_field(field: &[f32], width: usize, height: usize, x: f32, y: f32) -> f32 {
    if x < 0.0 || y < 0.0 || x >= width as f32 || y >= height as f32 {
        return 0.0;
    }
    let x = (x - 0.5).max(0.0);
    let y = (y - 0.5).max(0.0);
    let x0 = x.floor() as usize;
    let y0 = y.floor() as usize;
    let x1 = (x0 + 1).min(width - 1);
    let y1 = (y0 + 1).min(height - 1);
    let top = field[y0 * width + x0] * (1.0 - x.fract()) + field[y0 * width + x1] * x.fract();
    let bottom = field[y1 * width + x0] * (1.0 - x.fract()) + field[y1 * width + x1] * x.fract();
    top * (1.0 - y.fract()) + bottom * y.fract()
}

/// f64 积分图避免大图累积误差；输出始终保持有符号浮点。
fn box_mean(values: &[f32], width: usize, height: usize, radius: usize) -> Vec<f32> {
    let stride = width + 1;
    let mut integral = vec![0.0f64; stride * (height + 1)];
    for y in 0..height {
        let mut sum = 0.0;
        for x in 0..width {
            sum += values[y * width + x] as f64;
            integral[(y + 1) * stride + x + 1] = integral[y * stride + x + 1] + sum;
        }
    }
    (0..width * height)
        .into_par_iter()
        .map(|i| {
            let x = i % width;
            let y = i / width;
            let x0 = x.saturating_sub(radius);
            let y0 = y.saturating_sub(radius);
            let x1 = (x + radius + 1).min(width);
            let y1 = (y + radius + 1).min(height);
            ((integral[y1 * stride + x1] - integral[y1 * stride + x0] - integral[y0 * stride + x1]
                + integral[y0 * stride + x0])
                / ((x1 - x0) * (y1 - y0)) as f64) as f32
        })
        .collect()
}

fn guided_smooth(
    log: &[f32],
    mask: &[f32],
    width: usize,
    height: usize,
    radius: usize,
) -> Vec<f32> {
    let weights = box_mean(mask, width, height, radius);
    let first: Vec<f32> = log.iter().zip(mask).map(|(l, m)| l * m).collect();
    let second: Vec<f32> = log.iter().zip(mask).map(|(l, m)| l * l * m).collect();
    let mean = box_mean(&first, width, height, radius);
    let square = box_mean(&second, width, height, radius);
    let mut a = vec![0.0; log.len()];
    let mut b = vec![0.0; log.len()];
    for i in 0..log.len() {
        let mean = mean[i] / weights[i].max(1e-6);
        let variance = (square[i] / weights[i].max(1e-6) - mean * mean).max(0.0);
        let coefficient = variance / (variance + 0.16);
        a[i] = coefficient * mask[i];
        b[i] = mean * (1.0 - coefficient) * mask[i];
    }
    let a = box_mean(&a, width, height, radius);
    let b = box_mean(&b, width, height, radius);
    (0..log.len())
        .map(|i| {
            if weights[i] > 1e-6 {
                (a[i] * log[i] + b[i]) / weights[i]
            } else {
                log[i]
            }
        })
        .collect()
}

/// 同一尺度的引导统计量在 RGB 三通道间复用，避免重复构建积分图。
struct ColorGuide<'a> {
    guide: &'a [f32],
    mask: &'a [f32],
    weights: Vec<f32>,
    mean: Vec<f32>,
    variance: Vec<f32>,
    width: usize,
    height: usize,
    radius: usize,
}

impl<'a> ColorGuide<'a> {
    fn new(guide: &'a [f32], mask: &'a [f32], width: usize, height: usize, radius: usize) -> Self {
        let weights = box_mean(mask, width, height, radius);
        let weighted: Vec<f32> = guide.iter().zip(mask).map(|(g, m)| g * m).collect();
        let mut mean = box_mean(&weighted, width, height, radius);
        let weighted: Vec<f32> = guide.iter().zip(mask).map(|(g, m)| g * g * m).collect();
        let mut variance = box_mean(&weighted, width, height, radius);
        for i in 0..mean.len() {
            mean[i] /= weights[i].max(1e-6);
            variance[i] = (variance[i] / weights[i].max(1e-6) - mean[i] * mean[i]).max(0.0);
        }
        Self {
            guide,
            mask,
            weights,
            mean,
            variance,
            width,
            height,
            radius,
        }
    }

    fn filter(&self, values: &[f32], epsilon: f32) -> Vec<f32> {
        let weighted: Vec<f32> = values.iter().zip(self.mask).map(|(v, m)| v * m).collect();
        let mean = box_mean(&weighted, self.width, self.height, self.radius);
        let weighted: Vec<f32> = values
            .iter()
            .zip(self.guide)
            .zip(self.mask)
            .map(|((v, g), m)| v * g * m)
            .collect();
        let covariance = box_mean(&weighted, self.width, self.height, self.radius);
        let mut a = vec![0.0; values.len()];
        let mut b = vec![0.0; values.len()];
        for i in 0..values.len() {
            let mean = mean[i] / self.weights[i].max(1e-6);
            let covariance = covariance[i] / self.weights[i].max(1e-6) - self.mean[i] * mean;
            let coefficient = covariance / (self.variance[i] + epsilon);
            a[i] = coefficient * self.mask[i];
            b[i] = (mean - coefficient * self.mean[i]) * self.mask[i];
        }
        let a = box_mean(&a, self.width, self.height, self.radius);
        let b = box_mean(&b, self.width, self.height, self.radius);
        (0..values.len())
            .into_par_iter()
            .map(|i| {
                if self.weights[i] > 1e-6 {
                    (a[i] * self.guide[i] + b[i]) / self.weights[i]
                } else {
                    values[i]
                }
            })
            .collect()
    }
}

/// 保存低频修正 S(L)−L 与高频残差 source−low；渲染时按子滑块合成，默认 high=1 不替换纹理。
fn process_frequency_region(
    fields: &mut [[f32; 4]],
    detail_fields: &mut [[f32; 4]],
    linear: &Rgb32FImage,
    rect: (usize, usize, usize, usize),
    mask: &[f32],
    scale: f32,
    is_face: bool,
) {
    let (x0, y0, width, height) = rect;
    if mask.iter().filter(|&&m| m > 0.05).count() < 64 {
        return;
    }
    let index = |i: usize| (y0 + i / width) * linear.width() as usize + x0 + i % width;
    let source: Vec<[f32; 3]> = (0..mask.len())
        .into_par_iter()
        .map(|i| {
            // 蒙版内的二项式预滤波抑制奈奎斯特纹理泄漏，尤其是分离半径仅一像素时。
            // 这只影响低频分析输入，原图高频不经过该滤波。
            let x = i % width;
            let y = i / width;
            let mut sum = [0.0; 3];
            let mut weights = 0.0;
            for dy in 0..3 {
                for dx in 0..3 {
                    let nx = (x + dx).saturating_sub(1).min(width - 1);
                    let ny = (y + dy).saturating_sub(1).min(height - 1);
                    let weight = [1.0, 2.0, 1.0][dx] * [1.0, 2.0, 1.0][dy] * mask[ny * width + nx];
                    let rgb = linear.get_pixel((x0 + nx) as u32, (y0 + ny) as u32);
                    for channel in 0..3 {
                        sum[channel] += rgb[channel] * weight;
                    }
                    weights += weight;
                }
            }
            if weights > 1e-6 {
                sum.map(|v| v / weights)
            } else {
                linear.get_pixel((x0 + x) as u32, (y0 + y) as u32).0
            }
        })
        .collect();
    let luma = |p: [f32; 3]| p[0] * 0.2126 + p[1] * 0.7152 + p[2] * 0.0722;
    let guide: Vec<f32> = source.iter().map(|&p| luma(p)).collect();
    let (separation, smoothing) = frequency_radii(scale, is_face);
    let low: [Vec<f32>; 3] = {
        let filter = ColorGuide::new(&guide, mask, width, height, separation);
        std::array::from_fn(|channel| {
            let values: Vec<f32> = source.iter().map(|p| p[channel]).collect();
            filter.filter(&values, 0.001)
        })
    };
    let guide: Vec<f32> = (0..mask.len())
        .map(|i| luma([low[0][i], low[1][i], low[2][i]]))
        .collect();
    let filter = ColorGuide::new(&guide, mask, width, height, smoothing);
    let smoothed: [Vec<f32>; 3] =
        std::array::from_fn(|channel| filter.filter(&low[channel], 0.012));
    // 商业奶油皮：提高亮度平整比例；仍保留部分 luma 约束以免黄褐斑与五官结构被抹平。
    const LUMA_MIX: f32 = 0.70;
    let mut delta = vec![[0.0; 3]; mask.len()];
    let mut detail = vec![[0.0; 3]; mask.len()];
    for i in 0..mask.len() {
        let low_rgb = [low[0][i], low[1][i], low[2][i]];
        let smoothed_rgb = [smoothed[0][i], smoothed[1][i], smoothed[2][i]];
        let low_y = luma(low_rgb);
        let smoothed_y = luma(smoothed_rgb);
        let scale_y = if smoothed_y > 1e-8 {
            (1.0 - LUMA_MIX) * low_y / smoothed_y + LUMA_MIX
        } else {
            1.0
        };
        for channel in 0..3 {
            let mixed = smoothed_rgb[channel] * scale_y;
            let value = mixed - low_rgb[channel];
            delta[i][channel] = if value.is_finite() {
                value.clamp(-0.32, 0.32) * mask[i]
            } else {
                0.0
            };
            let residual = source[i][channel] - low_rgb[channel];
            detail[i][channel] = if residual.is_finite() {
                residual.clamp(-0.35, 0.35) * mask[i]
            } else {
                0.0
            };
        }
    }
    for i in 0..mask.len() {
        let idx = index(i);
        let target = &mut fields[idx];
        // 脸 pass：蒙版有效即覆盖身体，避免下颌/颊缘被身场占住后拖身体滑块像「脸在变」。
        // 身 pass：仍与已有场比大小，避免压过更强脸权。
        let should_write = if is_face {
            mask[i] > 0.05
        } else {
            mask[i] > target[3].abs()
        };
        if should_write {
            let region = if is_face { mask[i] } else { -mask[i] };
            *target = [delta[i][0], delta[i][1], delta[i][2], region];
            detail_fields[idx] = [detail[i][0], detail[i][1], detail[i][2], region];
        }
    }
}

/// 在保护蒙版内生成柔润场：保留边缘和高频来源，只平滑低频色彩起伏并微提中间调。
fn process_hydrated_region(
    fields: &mut [[f32; 4]],
    linear: &Rgb32FImage,
    rect: (usize, usize, usize, usize),
    mask: &[f32],
    scale: f32,
    is_face: bool,
) {
    let (x0, y0, width, height) = rect;
    if mask.iter().filter(|&&m| m > 0.05).count() < 64 {
        return;
    }
    let index = |i: usize| (y0 + i / width) * linear.width() as usize + x0 + i % width;
    let source: Vec<[f32; 3]> = (0..mask.len())
        .into_par_iter()
        .map(|i| {
            linear
                .get_pixel((x0 + i % width) as u32, (y0 + i / width) as u32)
                .0
        })
        .collect();
    let guide: Vec<f32> = source.iter().map(|&rgb| luma_rgb(rgb)).collect();
    let radius = hydrated_radius(scale, is_face);
    let filter = ColorGuide::new(&guide, mask, width, height, radius);
    let smoothed: [Vec<f32>; 3] = std::array::from_fn(|channel: usize| {
        let values: Vec<f32> = source.iter().map(|rgb| rgb[channel]).collect();
        filter.filter(&values, 0.008)
    });
    let dew_lift = if is_face { 0.014 } else { 0.011 };
    for i in 0..mask.len() {
        let luminance = guide[i];
        let midtone = smoothstep(0.025, 0.22, luminance) * (1.0 - smoothstep(0.62, 0.9, luminance));
        let target = &mut fields[index(i)];
        if mask[i] > target[3].abs() {
            let delta: [f32; 3] = std::array::from_fn(|channel: usize| {
                let smooth_delta = (smoothed[channel][i] - source[i][channel]) * 0.75;
                let value = (smooth_delta + dew_lift * midtone).clamp(-0.09, 0.09) * mask[i];
                if value.is_finite() { value } else { 0.0 }
            });
            *target = [
                delta[0],
                delta[1],
                delta[2],
                if is_face { mask[i] } else { -mask[i] },
            ];
        }
    }
}

fn hydrated_radius(scale: f32, is_face: bool) -> usize {
    let relative = if is_face { 0.065 } else { 0.085 };
    let cap = (ANALYSIS_EDGE as f32 * 0.07).round().max(3.0) as usize;
    ((scale * relative).round().max(3.0) as usize).min(cap)
}

fn luma_rgb(rgb: [f32; 3]) -> f32 {
    rgb[0] * 0.2126 + rgb[1] * 0.7152 + rgb[2] * 0.0722
}

fn frequency_radii(scale: f32, is_face: bool) -> (usize, usize) {
    let (separation, smoothing) = if is_face {
        (0.02, 0.09)
    } else {
        (0.024, 0.11)
    };
    let cap = (ANALYSIS_EDGE as f32 * 0.04).round().max(2.0) as usize;
    let sep = ((scale * separation).round().max(1.0) as usize).min(cap);
    let smooth = ((scale * smoothing).round().max(2.0) as usize)
        .min(cap)
        .max(sep.max(2));
    (sep, smooth)
}

fn face_frequency_scale(faces: &[FaceAnalysis]) -> Option<f32> {
    if faces.is_empty() {
        return None;
    }
    let mut scales: Vec<f32> = faces.iter().map(|face| face.roi.side / 1.5).collect();
    scales.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    Some(scales[scales.len() / 2])
}

fn frequency_region_scale(region_short_side: f32, face_scale: Option<f32>) -> f32 {
    let region = region_short_side.max(16.0);
    match face_scale {
        Some(face) if face.is_finite() && face > 0.0 => region.min(face),
        _ => region,
    }
}

fn in_polygon(point: [f32; 2], points: &[[f32; 3]], indices: &[usize]) -> bool {
    let mut inside = false;
    let mut j = indices.len() - 1;
    for i in 0..indices.len() {
        let a = points[indices[i]];
        let b = points[indices[j]];
        if (a[1] > point[1]) != (b[1] > point[1])
            && point[0] < (b[0] - a[0]) * (point[1] - a[1]) / (b[1] - a[1]) + a[0]
        {
            inside = !inside;
        }
        j = i;
    }
    inside
}

fn point_to_segment_distance(point: [f32; 2], a: [f32; 3], b: [f32; 3]) -> f32 {
    let ab = [b[0] - a[0], b[1] - a[1]];
    let ap = [point[0] - a[0], point[1] - a[1]];
    let denom = ab[0] * ab[0] + ab[1] * ab[1];
    let t = if denom > 1e-12 {
        ((ap[0] * ab[0] + ap[1] * ab[1]) / denom).clamp(0.0, 1.0)
    } else {
        0.0
    };
    let dx = point[0] - (a[0] + ab[0] * t);
    let dy = point[1] - (a[1] + ab[1] * t);
    (dx * dx + dy * dy).sqrt()
}

fn polygon_signed_distance(point: [f32; 2], points: &[[f32; 3]], indices: &[usize]) -> f32 {
    if indices.len() < 2 {
        return if in_polygon(point, points, indices) {
            0.0
        } else {
            -1.0
        };
    }
    let mut distance = f32::MAX;
    let mut j = indices.len() - 1;
    for i in 0..indices.len() {
        distance = distance.min(point_to_segment_distance(
            point,
            points[indices[i]],
            points[indices[j]],
        ));
        j = i;
    }
    if in_polygon(point, points, indices) {
        distance
    } else {
        -distance
    }
}

const OVAL: &[usize] = &[
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152,
    148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109,
];
const LEFT_EYE: &[usize] = &[33, 160, 158, 133, 153, 144];
const RIGHT_EYE: &[usize] = &[362, 385, 387, 263, 373, 380];
const LIPS: &[usize] = &[61, 40, 37, 0, 267, 270, 291, 321, 314, 17, 84, 91];
const LEFT_BROW: &[usize] = &[46, 53, 52, 65, 55, 107, 66, 105, 63, 70];
const RIGHT_BROW: &[usize] = &[276, 283, 282, 295, 285, 336, 296, 334, 293, 300];
/// MediaPipe 478 左／右虹膜中心（与 GPUPixel 大眼原点一致）。
const LEFT_IRIS_CENTER: usize = 468;
const RIGHT_IRIS_CENTER: usize = 473;
/// MediaPipe 虹膜轮廓（左 469–472，右 474–477）。
const LEFT_IRIS_RING: &[usize] = &[469, 470, 471, 472];
const RIGHT_IRIS_RING: &[usize] = &[474, 475, 476, 477];
/// 内眼角（肉阜保护）：左 133，右 362。
const LEFT_INNER_CANTHUS: usize = 133;
const RIGHT_INNER_CANTHUS: usize = 362;
/// 圆挖空 fallback：半径相对「虹膜中心→眼睑」最近距。
const IRIS_RADIUS_RATIO: f32 = 0.55;
/// 提亮：mask * strength * 本常数 * gap_curve * 保护项。
const EYE_WHITE_BRIGHTEN_GAIN: f32 = 0.55;
/// 提亮附带色度增益；默认 0，与亮度解耦。
const EYE_WHITE_CHROMA_GAIN: f32 = 0.0;
/// 提亮软上限（线性 luma）。
const EYE_WHITE_SOFT_CEILING: f32 = 0.90;
/// 羽化相对眼宽（分析像素）的比例。
const SCLERA_FEATHER_EYE_FRAC: f32 = 0.045;
/// 核心内收相对眼宽。
const SCLERA_CORE_INSET_FRAC: f32 = 0.035;
/// 肉阜保护半径相对眼宽。
const CANTHUS_PROTECT_FRAC: f32 = 0.12;
/// Frangi 尺度相对眼宽。
const EYE_VESSEL_SCALE_FRACS: [f32; 3] = [0.004, 0.008, 0.016];

const OVAL_FEATHER: f32 = 0.04;
const FEATURE_FEATHER: f32 = 0.03;
/// 眼睑多边形外扩（脸 ROI 归一化），堵住双线性从眼周皮肤场渗进眼球。
const EYE_EXCLUDE_MARGIN: f32 = 0.02;

/// 眼球排除权重：1=完全排除。多边形内 signed_distance>0；margin 内眼外也抬高排除。
fn eye_exclude_weight(face: &FaceAnalysis, uv: [f32; 2]) -> f32 {
    let mut exclude = 0.0f32;
    for polygon in [LEFT_EYE, RIGHT_EYE] {
        let sd = polygon_signed_distance(uv, &face.points, polygon);
        exclude = exclude.max(smoothstep(-EYE_EXCLUDE_MARGIN, 0.0, sd));
    }
    exclude
}

fn eye_white_debug_enabled() -> bool {
    std::env::var_os("RAPIDRAW_PORTRAIT_EYE_WHITE_DEBUG").is_some_and(|v| {
        let v = v.to_string_lossy();
        v == "1" || v.eq_ignore_ascii_case("true") || v.eq_ignore_ascii_case("yes")
    })
}

fn eye_landmark_span(face: &FaceAnalysis, eyelid: &[usize]) -> f32 {
    let mut min_x = f32::INFINITY;
    let mut max_x = f32::NEG_INFINITY;
    for &idx in eyelid {
        if idx >= face.points.len() {
            continue;
        }
        let x = face.points[idx][0];
        min_x = min_x.min(x);
        max_x = max_x.max(x);
    }
    if !min_x.is_finite() || !max_x.is_finite() {
        return 0.08;
    }
    (max_x - min_x).max(0.04)
}

fn iris_ring_reliable(face: &FaceAnalysis, center_idx: usize, ring: &[usize]) -> bool {
    if face.points.len() <= center_idx || ring.iter().any(|&i| i >= face.points.len()) {
        return false;
    }
    let c = face.points[center_idx];
    let mut radii = [0.0f32; 4];
    for (i, &idx) in ring.iter().enumerate() {
        let p = face.points[idx];
        let d = ((p[0] - c[0]).powi(2) + (p[1] - c[1]).powi(2)).sqrt();
        if !d.is_finite() || d < 1e-4 {
            return false;
        }
        radii[i] = d;
    }
    let mean = radii.iter().sum::<f32>() / ring.len() as f32;
    let max_dev = radii
        .iter()
        .map(|r| (r - mean).abs() / mean.max(1e-4))
        .fold(0.0f32, f32::max);
    mean > 1e-3 && mean < 0.12 && max_dev < 0.85
}

/// 单眼巩膜权重：brighten 写入 A；gate 用于血丝检测门控；reliability 降低不可靠关键点区域。
fn sclera_eye_weights(
    face: &FaceAnalysis,
    uv: [f32; 2],
    eyelid: &[usize],
    iris_center: usize,
    iris_ring: &[usize],
    canthus: usize,
) -> (f32, f32, f32) {
    if face.points.len() <= iris_center || eyelid.len() < 3 {
        return (0.0, 0.0, 0.0);
    }
    let eye_span = eye_landmark_span(face, eyelid);
    let feather = (eye_span * SCLERA_FEATHER_EYE_FRAC).clamp(0.006, 0.03);
    let core_inset = (eye_span * SCLERA_CORE_INSET_FRAC).clamp(0.004, 0.025);
    let lid_sd = polygon_signed_distance(uv, &face.points, eyelid);
    let lid = smoothstep(0.0, feather, lid_sd);
    if lid <= 0.0 {
        return (0.0, 0.0, 0.0);
    }
    let core = smoothstep(core_inset, core_inset + feather, lid_sd);
    let edge = (lid - core).clamp(0.0, 1.0);
    let lid_protect = 1.0 - edge * 0.55;

    let ring_ok = iris_ring_reliable(face, iris_center, iris_ring);
    let iris_keep = if ring_ok {
        let iris_sd = polygon_signed_distance(uv, &face.points, iris_ring);
        // 轮廓外（负距离）为巩膜；轮廓内保护虹膜。
        1.0 - smoothstep(-feather * 0.85, feather * 0.35, iris_sd)
    } else {
        let iris = face.points[iris_center];
        let iris_uv = [iris[0], iris[1]];
        let mut min_edge = f32::INFINITY;
        for &index in eyelid {
            let p = face.points[index];
            let d = ((p[0] - iris_uv[0]).powi(2) + (p[1] - iris_uv[1]).powi(2)).sqrt();
            min_edge = min_edge.min(d);
        }
        if !min_edge.is_finite() || min_edge < 1e-4 {
            return (0.0, 0.0, 0.0);
        }
        let radius = min_edge * IRIS_RADIUS_RATIO;
        let dist_iris = ((uv[0] - iris_uv[0]).powi(2) + (uv[1] - iris_uv[1]).powi(2)).sqrt();
        smoothstep(radius * 0.85, radius, dist_iris)
    };

    let mut canthus_protect = 1.0f32;
    if canthus < face.points.len() {
        let c = face.points[canthus];
        let d = ((uv[0] - c[0]).powi(2) + (uv[1] - c[1]).powi(2)).sqrt();
        let r = (eye_span * CANTHUS_PROTECT_FRAC).clamp(0.015, 0.06);
        canthus_protect = 1.0 - (1.0 - smoothstep(r * 0.35, r, d)) * 0.85;
    }

    let reliability = if ring_ok { 1.0 } else { 0.72 };
    let brighten = (lid * iris_keep * lid_protect * canthus_protect * reliability).clamp(0.0, 1.0);
    let gate = (core.max(lid * 0.65) * iris_keep * canthus_protect * reliability).clamp(0.0, 1.0);
    (brighten, gate, reliability)
}

fn sclera_face_mask(face: &FaceAnalysis, uv: [f32; 2]) -> f32 {
    let (l, _, _) = sclera_eye_weights(
        face,
        uv,
        LEFT_EYE,
        LEFT_IRIS_CENTER,
        LEFT_IRIS_RING,
        LEFT_INNER_CANTHUS,
    );
    let (r, _, _) = sclera_eye_weights(
        face,
        uv,
        RIGHT_EYE,
        RIGHT_IRIS_CENTER,
        RIGHT_IRIS_RING,
        RIGHT_INNER_CANTHUS,
    );
    l.max(r)
}

fn red_opponent(rgb: [f32; 3]) -> f32 {
    (rgb[0] - (rgb[1] + rgb[2]) * 0.5).max(0.0)
}

fn eye_box_filter(src: &[f32], w: usize, h: usize, radius: usize) -> Vec<f32> {
    box_mean(src, w, h, radius.max(1))
}

/// 简易可分高斯模糊（半径 ≈ 2σ）。
fn eye_gaussian_blur(src: &[f32], w: usize, h: usize, sigma: f32) -> Vec<f32> {
    if src.len() != w * h || w < 2 || h < 2 {
        return src.to_vec();
    }
    let sigma = sigma.max(0.35);
    let radius = (sigma * 2.0).ceil().max(1.0) as usize;
    let mut kernel = Vec::with_capacity(radius * 2 + 1);
    let mut sum = 0.0f32;
    for i in 0..=radius * 2 {
        let x = i as f32 - radius as f32;
        let v = (-0.5 * (x / sigma).powi(2)).exp();
        kernel.push(v);
        sum += v;
    }
    for v in &mut kernel {
        *v /= sum.max(1e-6);
    }
    let mut tmp = vec![0.0f32; w * h];
    for y in 0..h {
        for x in 0..w {
            let mut acc = 0.0f32;
            for (k, &kw) in kernel.iter().enumerate() {
                let xx = (x as i32 + k as i32 - radius as i32).clamp(0, w as i32 - 1) as usize;
                acc += src[y * w + xx] * kw;
            }
            tmp[y * w + x] = acc;
        }
    }
    let mut out = vec![0.0f32; w * h];
    for y in 0..h {
        for x in 0..w {
            let mut acc = 0.0f32;
            for (k, &kw) in kernel.iter().enumerate() {
                let yy = (y as i32 + k as i32 - radius as i32).clamp(0, h as i32 - 1) as usize;
                acc += tmp[yy * w + x] * kw;
            }
            out[y * w + x] = acc;
        }
    }
    out
}

/// 多尺度 Frangi-style vesselness（亮结构，作用于红异常图）。
fn eye_vesselness_multiscale(excess: &[f32], w: usize, h: usize, eye_width_px: f32) -> Vec<f32> {
    let n = w * h;
    let mut best = vec![0.0f32; n];
    if n != excess.len() || w < 5 || h < 5 {
        return best;
    }
    let mut scales: Vec<f32> = EYE_VESSEL_SCALE_FRACS
        .iter()
        .map(|f| (eye_width_px * f).clamp(0.6, 4.5))
        .collect();
    scales.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    scales.dedup_by(|a, b| (*a - *b).abs() < 0.15);
    if scales.is_empty() {
        scales.push(1.0);
    }
    let reliability = if eye_width_px < 12.0 {
        0.55
    } else if eye_width_px < 20.0 {
        0.78
    } else {
        1.0
    };
    for sigma in scales {
        let blurred = eye_gaussian_blur(excess, w, h, sigma);
        let at = |x: i32, y: i32| {
            let xx = x.clamp(0, w as i32 - 1) as usize;
            let yy = y.clamp(0, h as i32 - 1) as usize;
            blurred[yy * w + xx]
        };
        for y in 0..h as i32 {
            for x in 0..w as i32 {
                let lxx = at(x - 1, y) - 2.0 * at(x, y) + at(x + 1, y);
                let lyy = at(x, y - 1) - 2.0 * at(x, y) + at(x, y + 1);
                let lxy = 0.25
                    * (at(x + 1, y + 1) + at(x - 1, y - 1)
                        - at(x + 1, y - 1)
                        - at(x - 1, y + 1));
                let tmp = ((lxx - lyy).powi(2) + 4.0 * lxy * lxy).sqrt();
                let mut l1 = 0.5 * (lxx + lyy + tmp);
                let mut l2 = 0.5 * (lxx + lyy - tmp);
                if l1.abs() > l2.abs() {
                    std::mem::swap(&mut l1, &mut l2);
                }
                // 亮脊：λ2 < 0。
                if l2 >= 0.0 {
                    continue;
                }
                let rb = l1.abs() / l2.abs().max(1e-6);
                let s2 = l1 * l1 + l2 * l2;
                let beta2 = 0.5 * 0.5;
                let c2 = (0.5 * s2.max(1e-8)).max(1e-4);
                let v = ((-rb * rb) / (2.0 * beta2)).exp() * (1.0 - (-s2 / (2.0 * c2)).exp());
                let i = y as usize * w + x as usize;
                best[i] = best[i].max(v * reliability);
            }
        }
    }
    // 归一化到 [0,1]
    let max_v = best.iter().copied().fold(0.0f32, f32::max).max(1e-6);
    for v in &mut best {
        *v = (*v / max_v).clamp(0.0, 1.0);
    }
    best
}

fn trimmed_mean_rgb(samples: &[[f32; 3]], low_q: f32, high_q: f32) -> Option<[f32; 3]> {
    if samples.is_empty() {
        return None;
    }
    let mut out = [0.0f32; 3];
    for c in 0..3 {
        let mut vals: Vec<f32> = samples.iter().map(|s| s[c]).collect();
        vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let n = vals.len();
        let lo = ((n as f32) * low_q).floor() as usize;
        let hi = (((n as f32) * high_q).ceil() as usize).max(lo + 1).min(n);
        let slice = &vals[lo..hi];
        out[c] = slice.iter().sum::<f32>() / slice.len() as f32;
    }
    Some(out)
}

/// 巩膜场：按眼处理。RGB=单位祛红增量；A=提亮权重。
fn process_eye_white_region(
    fields: &mut [[f32; 4]],
    linear: &Rgb32FImage,
    rect: (usize, usize, usize, usize),
    face: &FaceAnalysis,
) {
    let (x0, y0, w, h) = rect;
    let n = w * h;
    if w < 5 || h < 5 {
        return;
    }
    let debug = eye_white_debug_enabled();
    let mut dbg_brighten = if debug {
        vec![0.0f32; n]
    } else {
        Vec::new()
    };
    let mut dbg_gate = if debug {
        vec![0.0f32; n]
    } else {
        Vec::new()
    };
    let mut dbg_vessel = if debug {
        vec![0.0f32; n]
    } else {
        Vec::new()
    };
    let mut dbg_donor = if debug {
        vec![0.0f32; n]
    } else {
        Vec::new()
    };
    let mut dbg_conf = if debug {
        vec![0.0f32; n]
    } else {
        Vec::new()
    };

    for (eyelid, iris_center, iris_ring, canthus) in [
        (
            LEFT_EYE,
            LEFT_IRIS_CENTER,
            LEFT_IRIS_RING,
            LEFT_INNER_CANTHUS,
        ),
        (
            RIGHT_EYE,
            RIGHT_IRIS_CENTER,
            RIGHT_IRIS_RING,
            RIGHT_INNER_CANTHUS,
        ),
    ] {
        let mut brighten_w = vec![0.0f32; n];
        let mut gate_w = vec![0.0f32; n];
        // 整眼关键点可靠性（勿在逐像素循环里被眼外返回值覆盖成 0）。
        let reliability = if iris_ring_reliable(face, iris_center, iris_ring) {
            1.0
        } else {
            0.72
        };
        let span_uv = eye_landmark_span(face, eyelid);
        // 脸 ROI 归一化跨度 × 脸边长 ≈ 分析像素眼宽。
        let eye_width_px = (span_uv * face.roi.side).clamp(6.0, 120.0);
        for i in 0..n {
            let px = x0 + i % w;
            let py = y0 + i / w;
            let uv = face
                .roi
                .unproject(px as f32 + 0.5, py as f32 + 0.5);
            let (b, g, _) = sclera_eye_weights(
                face,
                uv,
                eyelid,
                iris_center,
                iris_ring,
                canthus,
            );
            brighten_w[i] = b;
            gate_w[i] = g;
        }
        let gate_coverage: f32 = gate_w.iter().filter(|v| **v > 0.05).count() as f32;
        if gate_coverage < 4.0 {
            continue;
        }

        let mut excess = vec![0.0f32; n];
        let mut rgb_buf = vec![[0.0f32; 3]; n];
        let mut luma_buf = vec![0.0f32; n];
        for i in 0..n {
            if gate_w[i] <= 0.001 && brighten_w[i] <= 0.001 {
                continue;
            }
            let rgb = linear
                .get_pixel((x0 + i % w) as u32, (y0 + i / w) as u32)
                .0;
            if !rgb.iter().all(|v| v.is_finite() && *v >= 0.0) {
                continue;
            }
            rgb_buf[i] = rgb;
            luma_buf[i] = luma_rgb(rgb);
            excess[i] = red_opponent(rgb);
        }

        // 初步供体：门控内、非高光、非过暗，用于估计红基线。
        let mut donor_excess = Vec::new();
        for i in 0..n {
            if gate_w[i] < 0.35 {
                continue;
            }
            let y = luma_buf[i];
            if y < 0.10 || y > 0.92 {
                continue;
            }
            let sat = {
                let mx = rgb_buf[i][0].max(rgb_buf[i][1]).max(rgb_buf[i][2]);
                let mn = rgb_buf[i][0].min(rgb_buf[i][1]).min(rgb_buf[i][2]);
                if mx > 1e-4 {
                    (mx - mn) / mx
                } else {
                    0.0
                }
            };
            // 排除镜面：很亮且低饱和。
            if y > 0.78 && sat < 0.08 {
                continue;
            }
            donor_excess.push(excess[i]);
        }
        donor_excess.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        // 用偏低分位作「干净巩膜」红基线，避免整片偏红时基线被抬高导致漏检。
        let baseline_excess = if donor_excess.is_empty() {
            0.015
        } else {
            let idx = ((donor_excess.len() as f32) * 0.15).floor() as usize;
            donor_excess[idx.min(donor_excess.len() - 1)]
        };

        let vesselness = eye_vesselness_multiscale(&excess, w, h, eye_width_px);
        let blur_r = ((eye_width_px * 0.06).round() as usize)
            .clamp(2, 6)
            .min(w.max(1) / 6)
            .min(h.max(1) / 6);
        let local_excess = eye_box_filter(&excess, w, h, blur_r);

        let mut vessel_p = vec![0.0f32; n];
        for i in 0..n {
            // 检测权重取 gate 与 brighten 的较大者，避免边缘区只有 A、没有祛红。
            let detect = gate_w[i].max(brighten_w[i] * 0.85);
            if detect <= 0.02 {
                continue;
            }
            let delta_e = (excess[i] - baseline_excess).max(0.0);
            // 相对基线为主；整片偏红时走绝对 excess 兜底（仍由 detect 限制范围）。
            let color_anom = smoothstep(0.0015, 0.025, delta_e)
                .max(smoothstep(0.035, 0.09, excess[i]) * 0.75);
            let local_boost =
                smoothstep(0.0, 0.02, (excess[i] - local_excess[i]).max(0.0));
            let v = vesselness[i];
            // 颜色异常为主，细线/局部对比增强；无结构时仍保留部分异常红抑制。
            let structure = v.max(local_boost);
            let p = (color_anom * (0.55 + 0.45 * structure) * detect * reliability)
                .clamp(0.0, 1.0);
            vessel_p[i] = p;
        }

        // 参考白：排除血管/高光/阴影/边缘。
        let mut ref_samples: Vec<[f32; 3]> = Vec::new();
        for i in 0..n {
            if brighten_w[i] < 0.40 || vessel_p[i] > 0.22 {
                continue;
            }
            let y = luma_buf[i];
            if y < 0.12 || y > 0.85 {
                continue;
            }
            let sat = {
                let mx = rgb_buf[i][0].max(rgb_buf[i][1]).max(rgb_buf[i][2]);
                let mn = rgb_buf[i][0].min(rgb_buf[i][1]).min(rgb_buf[i][2]);
                if mx > 1e-4 {
                    (mx - mn) / mx
                } else {
                    0.0
                }
            };
            if y > 0.75 && sat < 0.07 {
                continue;
            }
            if (excess[i] - baseline_excess) > 0.018 {
                continue;
            }
            ref_samples.push(rgb_buf[i]);
            if debug {
                dbg_donor[i] = dbg_donor[i].max(1.0);
            }
        }
        let (reference, ref_conf) = if let Some(r) = trimmed_mean_rgb(&ref_samples, 0.2, 0.8) {
            let conf = (ref_samples.len() as f32 / 24.0).clamp(0.35, 1.0) * reliability;
            (r, conf)
        } else {
            // 保守回退：门控/提亮像素中位数向中性微收，保留环境光。
            let mut fallback = Vec::new();
            for i in 0..n {
                if (gate_w[i] > 0.15 || brighten_w[i] > 0.2) && luma_buf[i] > 0.08 {
                    fallback.push(rgb_buf[i]);
                }
            }
            if let Some(mut r) = trimmed_mean_rgb(&fallback, 0.3, 0.7) {
                let y = luma_rgb(r).max(1e-4);
                // 向同亮度中性微收 15%。
                for c in 0..3 {
                    r[c] = r[c] * 0.85 + y * 0.15;
                }
                (r, 0.28 * reliability)
            } else {
                // 无参考色时仍写入提亮 A，祛红跳过。
                for i in 0..n {
                    let b = brighten_w[i];
                    if b <= 0.001 {
                        continue;
                    }
                    let px = x0 + i % w;
                    let py = y0 + i / w;
                    let idx = py * linear.width() as usize + px;
                    if idx < fields.len() {
                        fields[idx][3] = fields[idx][3].max(b * 0.55);
                    }
                }
                continue;
            }
        };

        for i in 0..n {
            let b = brighten_w[i];
            let p = vessel_p[i];
            if b <= 0.001 && p <= 0.001 {
                continue;
            }
            let rgb = rgb_buf[i];
            let mut delta = [0.0f32; 3];
            if p > 0.005 {
                let excess_now = red_opponent(rgb);
                let conf = ref_conf.max(0.4);
                let excess_delta = (excess_now - baseline_excess)
                    .max(0.0)
                    .max((excess_now - 0.03).max(0.0) * 0.5)
                    * p
                    * conf;
                let suppress = (excess_delta * 0.95).min(0.18);
                // 异常红抑制：主要压 R，少量补 G/B 防偏青。
                delta[0] -= suppress;
                delta[1] += (suppress * 0.22).min(0.05);
                delta[2] += (suppress * 0.30).min(0.06);
                // 有限中性校正（仅血管区）。
                let neutral_pull = p * conf * 0.18;
                for c in 0..3 {
                    delta[c] += (reference[c] - rgb[c]) * neutral_pull;
                }
                delta[0] = delta[0].min(0.0).max(-0.18);
                delta[1] = delta[1].clamp(-0.03, 0.08);
                delta[2] = delta[2].clamp(-0.03, 0.09);
            }
            let px = x0 + i % w;
            let py = y0 + i / w;
            let idx = py * linear.width() as usize + px;
            if idx >= fields.len() {
                continue;
            }
            let prev = fields[idx];
            let out_a = prev[3].max(b * ref_conf.max(0.55));
            let out_rgb = if delta[0] < prev[0] {
                delta
            } else {
                [prev[0], prev[1], prev[2]]
            };
            fields[idx] = [out_rgb[0], out_rgb[1], out_rgb[2], out_a];
            if debug {
                dbg_brighten[i] = dbg_brighten[i].max(b);
                dbg_gate[i] = dbg_gate[i].max(gate_w[i]);
                dbg_vessel[i] = dbg_vessel[i].max(p);
                dbg_conf[i] = dbg_conf[i].max(ref_conf);
            }
        }

    }

    if debug {
        let dir = std::env::temp_dir().join("rapidraw_eye_white_debug");
        let _ = std::fs::create_dir_all(&dir);
        let tag = format!("face_{}_{}x{}", x0 + y0, w, h);
        shine_debug_dump_gray(&dir.join(format!("{tag}_brighten.png")), &dbg_brighten, w, h);
        shine_debug_dump_gray(&dir.join(format!("{tag}_gate.png")), &dbg_gate, w, h);
        shine_debug_dump_gray(&dir.join(format!("{tag}_vessel.png")), &dbg_vessel, w, h);
        shine_debug_dump_gray(&dir.join(format!("{tag}_donor.png")), &dbg_donor, w, h);
        shine_debug_dump_gray(&dir.join(format!("{tag}_ref_conf.png")), &dbg_conf, w, h);
        let mut delta_vis = vec![[0.0f32; 3]; n];
        for i in 0..n {
            let px = x0 + i % w;
            let py = y0 + i / w;
            let idx = py * linear.width() as usize + px;
            if idx < fields.len() {
                let f = fields[idx];
                delta_vis[i] = [f[0], f[1], f[2]];
            }
        }
        shine_debug_dump_delta(&dir.join(format!("{tag}_delta.png")), &delta_vis, w, h);
    }
}

fn face_mask(face: &FaceAnalysis, uv: [f32; 2]) -> f32 {
    if !in_polygon(uv, &face.points, OVAL) {
        return 0.0;
    }
    for polygon in [LIPS, LEFT_BROW, RIGHT_BROW] {
        if in_polygon(uv, &face.points, polygon) {
            return 0.0;
        }
    }
    let eye_keep = 1.0 - eye_exclude_weight(face, uv);
    if eye_keep <= 0.0 {
        return 0.0;
    }
    eye_keep
        * smoothstep(
            0.65,
            0.95,
            sample_field(&face.skin, 512, 512, uv[0] * 512.0, uv[1] * 512.0),
        )
}

fn frequency_face_mask(face: &FaceAnalysis, uv: [f32; 2]) -> f32 {
    let oval = smoothstep(
        0.0,
        OVAL_FEATHER,
        polygon_signed_distance(uv, &face.points, OVAL),
    );
    if oval <= 0.0 {
        return 0.0;
    }
    let mut feature_keep: f32 = 1.0 - eye_exclude_weight(face, uv);
    for polygon in [LIPS, LEFT_BROW, RIGHT_BROW] {
        let distance = polygon_signed_distance(uv, &face.points, polygon);
        feature_keep = feature_keep.min(1.0 - smoothstep(-FEATURE_FEATHER, 0.0, distance));
    }
    oval * feature_keep
        * smoothstep(
            0.65,
            0.95,
            sample_field(&face.skin, 512, 512, uv[0] * 512.0, uv[1] * 512.0),
        )
}

/// 人脸 ROI 内对眼球硬清零中性灰／高低频／水润场，避免引导滤波边界残留进 atlas。
fn clear_eyeball_retouch_fields(
    face: &FaceAnalysis,
    image_width: usize,
    rect: (usize, usize, usize, usize),
    pixels: &mut [[f32; 4]],
    evenness_chroma: &mut [[f32; 4]],
    frequency: &mut [[f32; 4]],
    frequency_detail: &mut [[f32; 4]],
    hydrated: &mut [[f32; 4]],
) {
    let (x0, y0, w, h) = rect;
    for i in 0..w * h {
        let uv = face
            .roi
            .unproject((x0 + i % w) as f32 + 0.5, (y0 + i / w) as f32 + 0.5);
        if eye_exclude_weight(face, uv) <= 0.5 {
            continue;
        }
        let gi = (y0 + i / w) * image_width + x0 + i % w;
        let luma = pixels[gi][3];
        pixels[gi] = [0.0, 0.0, 0.0, luma];
        evenness_chroma[gi] = [0.0; 4];
        frequency[gi] = [0.0; 4];
        frequency_detail[gi] = [0.0; 4];
        hydrated[gi] = [0.0; 4];
    }
}

fn oval_body_keep(face: &FaceAnalysis, uv: [f32; 2]) -> f32 {
    1.0 - smoothstep(
        0.0,
        OVAL_FEATHER,
        polygon_signed_distance(uv, &face.points, OVAL),
    )
}

fn body_skin_likelihood(rgb: [f32; 3]) -> f32 {
    if !rgb.iter().all(|v| v.is_finite() && *v >= 0.0) {
        return 0.0;
    }
    let luma = luma_rgb(rgb);
    let maxc = rgb[0].max(rgb[1]).max(rgb[2]);
    let minc = rgb[0].min(rgb[1]).min(rgb[2]);
    let sat = if maxc > 1e-6 {
        (maxc - minc) / maxc
    } else {
        0.0
    };
    smoothstep(-0.02, 0.02, rgb[0] - rgb[1])
        * smoothstep(-0.05, 0.02, rgb[1] - rgb[2])
        * (1.0 - smoothstep(0.62, 0.92, sat))
        * smoothstep(0.015, 0.06, sat)
        * smoothstep(0.005, 0.03, luma)
        * (1.0 - smoothstep(0.85, 1.4, luma))
}

fn feather_retouch_mask(
    mask: &[f32],
    luminance: &[f32],
    image_width: usize,
    rect: (usize, usize, usize, usize),
    scale: f32,
    gradient_lo: f32,
    gradient_hi: f32,
) -> Vec<f32> {
    let (x0, y0, width, height) = rect;
    let index = |i: usize| (y0 + i / width) * image_width + x0 + i % width;
    let log: Vec<f32> = (0..width * height)
        .map(|i| luminance[index(i)].max(1e-6).log2())
        .collect();
    let feather = box_mean(
        mask,
        width,
        height,
        (scale * 0.008).round().max(1.0) as usize,
    );
    let mut out = mask.to_vec();
    for i in 0..out.len() {
        let x = i % width;
        let y = i / width;
        let gradient = (log[y * width + (x + 1).min(width - 1)]
            - log[y * width + x.saturating_sub(1)])
        .abs()
        .max(
            (log[(y + 1).min(height - 1) * width + x] - log[y.saturating_sub(1) * width + x]).abs(),
        );
        let lum = luminance[index(i)];
        out[i] *= feather[i]
            * (1.0 - smoothstep(gradient_lo, gradient_hi, gradient))
            * smoothstep(0.005, 0.035, lum)
            * (1.0 - smoothstep(0.75, 1.2, lum));
    }
    out
}

/// 鼻孔和五官边缘不参与瑕疵修复；不以亮度梯度排除痘痘本身。
fn blemish_face_mask(face: &FaceAnalysis, uv: [f32; 2]) -> f32 {
    for index in [98, 327] {
        let point = face.points[index];
        if ((uv[0] - point[0]) / 0.025).powi(2) + ((uv[1] - point[1]) / 0.018).powi(2) < 1.0 {
            return 0.0;
        }
    }
    face_mask(face, uv)
}

/// 祛油光脸皮蒙版：椭圆／五官 signed-distance 羽化 + 鼻孔软排除（不硬切眼下／唇周）。
fn shine_face_mask(face: &FaceAnalysis, uv: [f32; 2]) -> f32 {
    let oval = smoothstep(
        0.0,
        OVAL_FEATHER,
        polygon_signed_distance(uv, &face.points, OVAL),
    );
    if oval <= 0.0 {
        return 0.0;
    }
    let mut feature_keep: f32 = 1.0;
    for polygon in [LEFT_EYE, RIGHT_EYE, LIPS, LEFT_BROW, RIGHT_BROW] {
        let distance = polygon_signed_distance(uv, &face.points, polygon);
        feature_keep =
            feature_keep.min(1.0 - smoothstep(-SHINE_FEATURE_FEATHER, 0.0, distance));
    }
    let mut nostril_keep = 1.0f32;
    if face.points.len() > 327 {
        for index in [98usize, 327] {
            let point = face.points[index];
            if !point[0].is_finite() || !point[1].is_finite() {
                continue;
            }
            let r = (((uv[0] - point[0]) / 0.025).powi(2) + ((uv[1] - point[1]) / 0.018).powi(2))
                .sqrt();
            nostril_keep = nostril_keep.min(smoothstep(SHINE_NOSTRIL_INNER, SHINE_NOSTRIL_OUTER, r));
        }
    }
    oval
        * feature_keep
        * nostril_keep
        * smoothstep(
            0.65,
            0.95,
            sample_field(&face.skin, 512, 512, uv[0] * 512.0, uv[1] * 512.0),
        )
}

fn shine_elliptical_prior(uv: [f32; 2], center: [f32; 3], radii: [f32; 2]) -> f32 {
    if !center[0].is_finite() || !center[1].is_finite() || radii[0] <= 1e-6 || radii[1] <= 1e-6 {
        return 0.0;
    }
    let r = (((uv[0] - center[0]) / radii[0]).powi(2) + ((uv[1] - center[1]) / radii[1]).powi(2))
        .sqrt();
    1.0 - smoothstep(0.35, 1.15, r)
}

/// T 区先验：额／鼻梁／颧骨接近 1，其余脸皮保底，减轻强侧光误伤。
fn shine_t_zone_prior(face: &FaceAnalysis, uv: [f32; 2]) -> f32 {
    if face.points.len() < 331 {
        return 1.0;
    }
    let nose = [
        0.5 * (face.points[168][0] + face.points[1][0]),
        0.5 * (face.points[168][1] + face.points[1][1]),
        0.0,
    ];
    let score = shine_elliptical_prior(uv, face.points[151], [0.18, 0.11])
        .max(shine_elliptical_prior(uv, nose, [0.055, 0.15]))
        .max(shine_elliptical_prior(uv, face.points[50], [0.10, 0.085]))
        .max(shine_elliptical_prior(uv, face.points[280], [0.10, 0.085]))
        .max(shine_elliptical_prior(uv, face.points[101], [0.08, 0.07]))
        .max(shine_elliptical_prior(uv, face.points[330], [0.08, 0.07]))
        .clamp(0.0, 1.0);
    SHINE_T_ZONE_FLOOR + (1.0 - SHINE_T_ZONE_FLOOR) * score
}

fn shine_saturation(rgb: [f32; 3]) -> f32 {
    let hi = rgb.into_iter().fold(0.0f32, f32::max);
    let lo = rgb.into_iter().fold(f32::INFINITY, f32::min);
    (hi - lo).max(0.0) / hi.max(1e-6)
}

fn shine_chroma(rgb: [f32; 3]) -> [f32; 2] {
    let y = luma_rgb(rgb).max(1e-6);
    [rgb[0] - y, rgb[2] - y]
}

/// 渲染期安全限：保色分通道增量，限制相对变化与亮度损失（CPU／WGSL 同式）。
fn apply_shine_safety(current: [f32; 3], delta: [f32; 3]) -> [f32; 3] {
    if !current.iter().chain(delta.iter()).all(|v| v.is_finite()) {
        return [0.0; 3];
    }
    let current = current.map(|v| v.max(0.0));
    let mut delta = delta;
    for c in 0..3 {
        let limit = current[c].max(1e-4) * SHINE_APPLY_REL_CAP;
        delta[c] = delta[c].clamp(-limit, limit);
    }
    let cur_luma = luma_rgb(current).max(1e-6);
    let next = [
        (current[0] + delta[0]).max(0.0),
        (current[1] + delta[1]).max(0.0),
        (current[2] + delta[2]).max(0.0),
    ];
    let loss = (cur_luma - luma_rgb(next)).max(0.0);
    let max_loss = cur_luma * SHINE_MAX_REDUCTION_HIGH;
    if loss > max_loss && loss > 1e-8 {
        let scale = max_loss / loss;
        delta = delta.map(|v| v * scale);
    }
    for c in 0..3 {
        if current[c] + delta[c] < 0.0 {
            delta[c] = -current[c];
        }
    }
    delta
}

fn shine_debug_enabled() -> bool {
    std::env::var_os("RAPIDRAW_PORTRAIT_SHINE_DEBUG").is_some_and(|v| {
        let v = v.to_string_lossy();
        v == "1" || v.eq_ignore_ascii_case("true") || v.eq_ignore_ascii_case("yes")
    })
}

fn shine_debug_dump_gray(path: &std::path::Path, values: &[f32], width: usize, height: usize) {
    if values.len() != width * height {
        return;
    }
    let max = values.iter().copied().fold(0.0f32, f32::max).max(1e-6);
    let bytes: Vec<u8> = values
        .iter()
        .map(|v| ((v / max).clamp(0.0, 1.0) * 255.0).round() as u8)
        .collect();
    let _ = image::GrayImage::from_raw(width as u32, height as u32, bytes)
        .map(|img| img.save(path));
}

fn shine_debug_dump_delta(path: &std::path::Path, delta: &[[f32; 3]], width: usize, height: usize) {
    if delta.len() != width * height {
        return;
    }
    let bytes: Vec<u8> = delta
        .iter()
        .flat_map(|rgb| {
            let y = (-luma_rgb(*rgb)).max(0.0);
            let n = (y * 4.0).clamp(0.0, 1.0);
            let b = (n * 255.0).round() as u8;
            [b, b, b]
        })
        .collect();
    let _ = image::RgbImage::from_raw(width as u32, height as u32, bytes).map(|img| img.save(path));
}

/// 检测与修复分离：烘焙保色线性 RGB 增量，原分辨率纹理不进入修正场。
/// `prior` 为可选空间权重（如 T 区）；与 mask 同尺寸，缺省视为 1。
fn process_shine_region(
    fields: &mut [[f32; 4]],
    linear: &Rgb32FImage,
    rect: (usize, usize, usize, usize),
    mask: &[f32],
    scale: f32,
    prior: Option<&[f32]>,
) {
    let (x0, y0, w, h) = rect;
    let n = w * h;
    if w < 5 || h < 5 || mask.len() != n || !scale.is_finite() {
        return;
    }
    if prior.is_some_and(|p| p.len() != n) {
        return;
    }
    let mut skin = mask.to_vec();
    let source: Vec<[f32; 3]> = (0..n)
        .map(|i| {
            let rgb = linear.get_pixel((x0 + i % w) as u32, (y0 + i / w) as u32).0;
            if !rgb.iter().all(|v| v.is_finite() && *v >= 0.0) || !skin[i].is_finite() {
                skin[i] = 0.0;
                [0.0; 3]
            } else {
                skin[i] = skin[i].clamp(0.0, 1.0);
                rgb
            }
        })
        .collect();
    if skin.iter().filter(|&&m| m > 0.5).count() < 64 {
        return;
    }
    let fine = (scale * 0.012).round().max(1.0) as usize;
    let local = (scale * 0.035).round().max(2.0) as usize;
    let broad = (scale * 0.08).round().max(4.0) as usize;
    let feather = box_mean(&skin, w, h, (scale * 0.008).round().max(1.0) as usize);
    // 二项式预滤波阻止毛孔等高频信号进入修正场，且不采入五官和背景。
    let filtered: Vec<[f32; 3]> = (0..n)
        .into_par_iter()
        .map(|i| {
            let mut sum = [0.0; 3];
            let mut weight = 0.0;
            for dy in 0..3 {
                for dx in 0..3 {
                    let x = (i % w + dx).saturating_sub(1).min(w - 1);
                    let y = (i / w + dy).saturating_sub(1).min(h - 1);
                    let j = y * w + x;
                    let m = [1.0, 2.0, 1.0][dx] * [1.0, 2.0, 1.0][dy] * skin[j];
                    for c in 0..3 {
                        sum[c] += source[j][c] * m;
                    }
                    weight += m;
                }
            }
            if weight > 1e-6 {
                sum.map(|v| v / weight)
            } else {
                source[i]
            }
        })
        .collect();
    let guide: Vec<f32> = filtered.iter().map(|&rgb| luma_rgb(rgb)).collect();
    let mean_luma = guide.iter().zip(&skin).map(|(v, m)| v * m).sum::<f32>()
        / skin.iter().sum::<f32>().max(1e-6);
    let filter = ColorGuide::new(&guide, &skin, w, h, fine);
    let low: [Vec<f32>; 3] = std::array::from_fn(|c| {
        filter.filter(
            &filtered.iter().map(|p| p[c]).collect::<Vec<_>>(),
            (mean_luma * 0.15).powi(2).max(1e-6),
        )
    });
    let low_rgb = |i: usize| [low[0][i], low[1][i], low[2][i]];

    // —— 1) 粗参考：皮肤宽核均值，仅用于初检 ——
    let skin_support = box_mean(&skin, w, h, broad);
    let initial: [Vec<f32>; 3] = std::array::from_fn(|c| {
        let weighted: Vec<f32> = low[c].iter().zip(&skin).map(|(v, m)| v * m).collect();
        box_mean(&weighted, w, h, broad)
            .iter()
            .zip(&skin_support)
            .map(|(v, m)| v / m.max(1e-6))
            .collect()
    });

    // 局部亮度梯度：强平滑形体光降权，减轻鼻梁／颧骨自然高光误伤。
    let mut gradient = vec![0.0f32; n];
    for i in 0..n {
        let x = i % w;
        let y = i / w;
        let xm = if x == 0 { 0 } else { x - 1 };
        let xp = (x + 1).min(w - 1);
        let ym = if y == 0 { 0 } else { y - 1 };
        let yp = (y + 1).min(h - 1);
        let gx = guide[y * w + xp] - guide[y * w + xm];
        let gy = guide[yp * w + x] - guide[ym * w + x];
        gradient[i] = (gx * gx + gy * gy).sqrt() / mean_luma.max(1e-6);
    }
    let gradient = box_mean(&gradient, w, h, fine.max(1));

    let mut highlight = vec![0.0f32; n];
    for i in 0..n {
        if skin[i] <= 0.0 {
            continue;
        }
        let rgb = low_rgb(i);
        let diffuse = [initial[0][i], initial[1][i], initial[2][i]];
        let y = luma_rgb(rgb);
        let yd = luma_rgb(diffuse).max(1e-6);
        let ev = (y.max(1e-6) / yd).log2();
        let sat_drop = shine_saturation(diffuse) - shine_saturation(rgb);
        let oil = smoothstep(0.06, 0.45, ev)
            * smoothstep(0.012, 0.11, sat_drop)
            * smoothstep(0.005, 0.025, y);
        // 形体光：有梯度但缺褪饱和 → 降权；油光常伴褪色，保留。
        let form = 1.0
            - smoothstep(0.025, 0.10, gradient[i]) * (1.0 - smoothstep(0.02, 0.08, sat_drop));
        let blown = 1.0 - 0.55 * smoothstep(0.85, 1.35, y) * (1.0 - oil);
        highlight[i] = (oil * form * blown * skin[i]).clamp(0.0, 1.0);
    }

    // —— 2) 供体：附近、色近、未明显反光 ——
    let clean: Vec<f32> = (0..n)
        .map(|i| skin[i] * (1.0 - highlight[i]).powi(2))
        .collect();
    let clean_support = box_mean(&clean, w, h, local);
    let local_chroma: Vec<[f32; 2]> = {
        let cr: Vec<f32> = (0..n)
            .map(|i| shine_chroma(low_rgb(i))[0] * clean[i])
            .collect();
        let cb: Vec<f32> = (0..n)
            .map(|i| shine_chroma(low_rgb(i))[1] * clean[i])
            .collect();
        let cr = box_mean(&cr, w, h, local);
        let cb = box_mean(&cb, w, h, local);
        (0..n)
            .map(|i| {
                let s = clean_support[i].max(1e-6);
                [cr[i] / s, cb[i] / s]
            })
            .collect()
    };
    let local_luma: Vec<f32> = {
        let weighted: Vec<f32> = guide.iter().zip(&clean).map(|(v, m)| v * m).collect();
        box_mean(&weighted, w, h, local)
            .iter()
            .zip(&clean_support)
            .map(|(v, m)| v / m.max(1e-6))
            .collect()
    };

    let mut donors = vec![0.0f32; n];
    for i in 0..n {
        if skin[i] <= 0.0 {
            continue;
        }
        let rgb = low_rgb(i);
        let ch = shine_chroma(rgb);
        let dcr = ch[0] - local_chroma[i][0];
        let dcb = ch[1] - local_chroma[i][1];
        let chroma_dist = (dcr * dcr + dcb * dcb).sqrt();
        let color_sim = (-chroma_dist / SHINE_CHROMA_SIGMA).exp();
        let y = luma_rgb(rgb);
        let luma_ok = 1.0
            - smoothstep(0.35, 0.85, (y - local_luma[i]).abs() / local_luma[i].max(1e-6));
        // 明暗边界：梯度过大处不作供体，避免把硬阴影平均进目标。
        let edge = 1.0 - smoothstep(0.08, 0.22, gradient[i]);
        donors[i] = skin[i]
            * (1.0 - highlight[i]).powi(4)
            * color_sim
            * luma_ok
            * edge
            * smoothstep(0.005, 0.02, y);
    }

    let donor_support = box_mean(&donors, w, h, local);
    let diffuse: [Vec<f32>; 3] = std::array::from_fn(|c| {
        let weighted: Vec<f32> = low[c].iter().zip(&donors).map(|(v, m)| v * m).collect();
        box_mean(&weighted, w, h, local)
            .iter()
            .zip(&donor_support)
            .map(|(v, m)| v / m.max(1e-6))
            .collect()
    });

    // —— 3) 修复：轻度保色亮度；强反光且供体可靠时混入供体色 ——
    let mut delta_r = vec![0.0f32; n];
    let mut delta_g = vec![0.0f32; n];
    let mut delta_b = vec![0.0f32; n];
    let mut repair_conf = vec![0.0f32; n];
    for i in 0..n {
        if skin[i] <= 0.0 || feather[i] <= 1e-6 {
            continue;
        }
        let zone = prior.map(|p| p[i].clamp(0.0, 1.0)).unwrap_or(1.0);
        if zone <= 1e-7 {
            continue;
        }
        let h = highlight[i];
        if h <= 1e-5 {
            continue;
        }
        let rgb = low_rgb(i);
        let y = luma_rgb(rgb).max(1e-6);
        let donor_rel = smoothstep(0.08, 0.35, donor_support[i])
            * smoothstep(0.12, 0.45, donor_support[i] / skin_support[i].max(1e-6));
        let blown_soft = 1.0 - 0.65 * smoothstep(0.95, 1.4, y) * (1.0 - donor_rel);
        let conf = (h * (0.35 + 0.65 * donor_rel) * blown_soft).clamp(0.0, 1.0);
        if conf <= 1e-5 {
            continue;
        }

        let donor = [diffuse[0][i], diffuse[1][i], diffuse[2][i]];
        let y_donor = luma_rgb(donor).max(1e-6);
        // 供体不足：仅温和亮度压制，不用不可靠参考色。
        let y_target = if donor_rel < 0.2 {
            y * (1.0 - SHINE_MAX_REDUCTION_LOW * 0.45 * h)
        } else {
            y_donor.clamp(y * (1.0 - SHINE_MAX_REDUCTION_HIGH), y)
        };
        if y_target >= y - 1e-7 {
            continue;
        }
        let luma_scaled = rgb.map(|c| c * (y_target / y));
        // 仅当油光相对供体明显褪色时才混入供体色度；色近时保持纯亮度比例。
        let wash = {
            let a = shine_chroma(rgb);
            let b = shine_chroma(donor);
            ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2)).sqrt()
        };
        let chroma_blend =
            smoothstep(0.35, 0.75, h) * donor_rel * smoothstep(0.015, 0.055, wash);
        let target = if chroma_blend <= 1e-5 {
            luma_scaled
        } else {
            // 供体 Cr/Cb 落到目标亮度；G 由 Rec.709 反推，避免阴影亮度灌入。
            let dch = shine_chroma(donor);
            let r = (y_target + dch[0]).max(0.0);
            let b = (y_target + dch[1]).max(0.0);
            let g = ((y_target - 0.2126 * r - 0.0722 * b) / 0.7152).max(0.0);
            let donor_chroma = [r, g, b];
            [
                luma_scaled[0] * (1.0 - chroma_blend) + donor_chroma[0] * chroma_blend,
                luma_scaled[1] * (1.0 - chroma_blend) + donor_chroma[1] * chroma_blend,
                luma_scaled[2] * (1.0 - chroma_blend) + donor_chroma[2] * chroma_blend,
            ]
        };
        let max_reduce = y
            * (SHINE_MAX_REDUCTION_LOW
                + (SHINE_MAX_REDUCTION_HIGH - SHINE_MAX_REDUCTION_LOW) * conf * donor_rel);
        let mut d = [
            (target[0] - rgb[0]) * conf,
            (target[1] - rgb[1]) * conf,
            (target[2] - rgb[2]) * conf,
        ];
        let dy = -luma_rgb(d);
        if dy > max_reduce && dy > 1e-8 {
            let s = max_reduce / dy;
            d = d.map(|v| v * s);
        }
        let gate = skin[i] * feather[i] * zone;
        delta_r[i] = d[0] * gate;
        delta_g[i] = d[1] * gate;
        delta_b[i] = d[2] * gate;
        repair_conf[i] = conf * gate;
    }

    // 羽化只缩小幅度：平滑后与原始逐通道取更接近 0 的值，避免外扩到正常皮肤。
    let smooth_r = box_mean(&delta_r, w, h, fine);
    let smooth_g = box_mean(&delta_g, w, h, fine);
    let smooth_b = box_mean(&delta_b, w, h, fine);
    let smooth_a = box_mean(&repair_conf, w, h, fine);

    if shine_debug_enabled() {
        let dir = std::env::temp_dir().join("rapidraw_shine_debug");
        let _ = std::fs::create_dir_all(&dir);
        let tag = format!("{}x{}_{}_{}", w, h, x0, y0);
        shine_debug_dump_gray(&dir.join(format!("{tag}_skin.png")), &skin, w, h);
        shine_debug_dump_gray(&dir.join(format!("{tag}_highlight.png")), &highlight, w, h);
        shine_debug_dump_gray(&dir.join(format!("{tag}_donors.png")), &donors, w, h);
        shine_debug_dump_gray(&dir.join(format!("{tag}_repair_conf.png")), &repair_conf, w, h);
        let deltas: Vec<[f32; 3]> = (0..n)
            .map(|i| [delta_r[i], delta_g[i], delta_b[i]])
            .collect();
        shine_debug_dump_delta(&dir.join(format!("{tag}_delta.png")), &deltas, w, h);
        log::debug!(
            "[portrait-shine] dumped debug masks to {}",
            dir.display()
        );
    }

    for i in 0..n {
        let zone = prior.map(|p| p[i].clamp(0.0, 1.0)).unwrap_or(1.0);
        if zone <= 1e-7 || skin[i] <= 0.0 {
            continue;
        }
        // 与平滑场取“更弱”的修正：同号取绝对值更小者。
        let shrink = |a: f32, b: f32| {
            if a == 0.0 || b == 0.0 {
                0.0
            } else if a.signum() != b.signum() {
                0.0
            } else if a.abs() < b.abs() {
                a
            } else {
                b
            }
        };
        let d = [
            shrink(delta_r[i], smooth_r[i]),
            shrink(delta_g[i], smooth_g[i]),
            shrink(delta_b[i], smooth_b[i]),
        ];
        let weight = repair_conf[i].min(smooth_a[i]);
        if weight <= 1e-7 || !d.iter().all(|v| v.is_finite()) {
            continue;
        }
        if d.iter().all(|v| v.abs() <= 1e-7) {
            continue;
        }
        let target = &mut fields[(y0 + i / w) * linear.width() as usize + x0 + i % w];
        if weight > target[3] {
            *target = [d[0], d[1], d[2], weight];
        }
    }
}

/// 检测与供体选择只依赖基底。强度变化不重新执行本算法。
fn process_blemish_region(
    repair: &mut [[f32; 4]],
    texture: &mut [[f32; 4]],
    linear: &Rgb32FImage,
    rect: (usize, usize, usize, usize),
    mut mask: Vec<f32>,
    scale: f32,
) {
    let (x0, y0, w, h) = rect;
    let n = w * h;
    if w < 5 || h < 5 || mask.iter().filter(|&&m| m > 0.85).count() < 64 {
        return;
    }
    let source: Vec<[f32; 3]> = (0..n)
        .map(|i| linear.get_pixel((x0 + i % w) as u32, (y0 + i / w) as u32).0)
        .collect();
    let luma: Vec<f32> = source
        .iter()
        .map(|p| p[0] * 0.2126 + p[1] * 0.7152 + p[2] * 0.0722)
        .collect();
    let log: Vec<f32> = luma.iter().map(|v| v.max(1e-6).ln()).collect();
    let red: Vec<f32> = source
        .iter()
        .map(|p| (p[0] - p[1]) / (p.iter().sum::<f32>()).max(1e-6))
        .collect();
    let border = box_mean(&mask, w, h, (scale * 0.008).round().max(1.0) as usize);
    for i in 0..n {
        mask[i] *= smoothstep(0.9, 1.0, border[i])
            * smoothstep(0.005, 0.025, luma[i])
            * (1.0 - smoothstep(0.85, 1.2, luma[i]));
        if source[i].iter().any(|v| !v.is_finite()) {
            return;
        }
    }
    let weighted_mean = |values: &[f32], weights: &[f32], radius: usize| {
        let weighted: Vec<f32> = values.iter().zip(weights).map(|(v, m)| v * m).collect();
        let sums = box_mean(&weighted, w, h, radius);
        let counts = box_mean(weights, w, h, radius);
        sums.iter()
            .zip(counts)
            .zip(values)
            .map(
                |((v, count), original)| {
                    if count > 1e-6 { v / count } else { *original }
                },
            )
            .collect::<Vec<f32>>()
    };
    let fine_log = box_mean(&log, w, h, 1);
    let fine_red = box_mean(&red, w, h, 1);
    let noise: Vec<f32> = log
        .iter()
        .zip(&fine_log)
        .map(|(a, b)| (a - b).abs())
        .collect();
    let color_noise: Vec<f32> = red
        .iter()
        .zip(&fine_red)
        .map(|(a, b)| (a - b).abs())
        .collect();
    let mut score = vec![0.0f32; n];
    for factor in BLEMISH_SCALES {
        let radius = (scale * factor).round().max(2.0) as usize;
        let mean = weighted_mean(&log, &mask, radius);
        let color = weighted_mean(&red, &mask, radius);
        let deviation = box_mean(&noise, w, h, radius * 2);
        let color_deviation = box_mean(&color_noise, w, h, radius * 2);
        for i in 0..n {
            if mask[i] < 0.85 {
                continue;
            }
            let dark = smoothstep(
                BLEMISH_DARK_RANGE[0] + deviation[i],
                BLEMISH_DARK_RANGE[1] + deviation[i],
                mean[i] - log[i],
            );
            let redness = smoothstep(
                BLEMISH_RED_RANGE[0] + color_deviation[i],
                BLEMISH_RED_RANGE[1] + color_deviation[i],
                red[i] - color[i],
            );
            score[i] = score[i].max(dark.max(redness));
        }
    }
    let mut visited = vec![false; n];
    let mut regions = Vec::new();
    for seed in 0..n {
        if visited[seed] || score[seed] < BLEMISH_CANDIDATE_SCORE {
            continue;
        }
        let mut points = vec![seed];
        visited[seed] = true;
        let (mut left, mut top, mut right, mut bottom) = (w, h, 0, 0);
        let mut peak = 0.0f32;
        let mut cursor = 0;
        while cursor < points.len() {
            let i = points[cursor];
            cursor += 1;
            let (x, y) = (i % w, i / w);
            left = left.min(x);
            top = top.min(y);
            right = right.max(x + 1);
            bottom = bottom.max(y + 1);
            peak = peak.max(score[i]);
            for (nx, ny) in [
                (x.wrapping_sub(1), y),
                (x + 1, y),
                (x, y.wrapping_sub(1)),
                (x, y + 1),
            ] {
                if nx < w && ny < h {
                    let j = ny * w + nx;
                    if !visited[j] && score[j] >= BLEMISH_CANDIDATE_SCORE {
                        visited[j] = true;
                        points.push(j);
                    }
                }
            }
        }
        let (rw, rh) = (right - left, bottom - top);
        let diameter = rw.max(rh);
        if points.len() < BLEMISH_MIN_PIXELS
            || peak < BLEMISH_PEAK_SCORE
            || diameter as f32 > (scale * BLEMISH_MAX_DIAMETER).max(3.0)
            || diameter as f32 / rw.min(rh) as f32 > BLEMISH_MAX_ASPECT
            || points.len() as f32 / ((rw * rh) as f32) < 0.3
        {
            continue;
        }
        // 平坦的痘痘中心可能没有局部色差，填充闭合候选内部，防止仅修边缘。
        let mut inside = vec![false; rw * rh];
        for &i in &points {
            inside[(i / w - top) * rw + i % w - left] = true;
        }
        let mut outside = vec![false; rw * rh];
        let mut queue = VecDeque::new();
        for i in 0..rw * rh {
            if !inside[i] && (i % rw == 0 || i % rw + 1 == rw || i / rw == 0 || i / rw + 1 == rh) {
                outside[i] = true;
                queue.push_back(i);
            }
        }
        while let Some(i) = queue.pop_front() {
            let (x, y) = (i % rw, i / rw);
            for (nx, ny) in [
                (x.wrapping_sub(1), y),
                (x + 1, y),
                (x, y.wrapping_sub(1)),
                (x, y + 1),
            ] {
                if nx < rw && ny < rh {
                    let j = ny * rw + nx;
                    if !inside[j] && !outside[j] {
                        outside[j] = true;
                        queue.push_back(j);
                    }
                }
            }
        }
        for i in 0..rw * rh {
            if !outside[i] && !inside[i] {
                let j = (top + i / rw) * w + left + i % rw;
                points.push(j);
                score[j] = BLEMISH_CANDIDATE_SCORE;
                visited[j] = true;
            }
        }
        regions.push((points, left, top, right, bottom));
    }
    if regions.is_empty() {
        return;
    }
    // 所有疑似瑕疵均排除为供体，包括因形状不可靠而未修复的区域。
    let clean: Vec<f32> = (0..n)
        .map(|i| if score[i] < 0.05 { mask[i] } else { 0.0 })
        .collect();
    let clean_border = box_mean(&clean, w, h, 1);
    let radius = (scale * 0.04).round().max(4.0) as usize;
    let support = box_mean(&clean, w, h, radius);
    let channels: [Vec<f32>; 3] = std::array::from_fn(|c| source.iter().map(|p| p[c]).collect());
    let base: [Vec<f32>; 3] = std::array::from_fn(|c| weighted_mean(&channels[c], &clean, radius));
    let low: [Vec<f32>; 3] = std::array::from_fn(|c| box_mean(&channels[c], w, h, 1));
    let feather = (scale * 0.003).round().clamp(1.0, 3.0) as usize;
    for (points, left, top, right, bottom) in regions {
        let (left, top) = (left.saturating_sub(feather), top.saturating_sub(feather));
        let (right, bottom) = ((right + feather).min(w), (bottom + feather).min(h));
        let (rw, rh) = (right - left, bottom - top);
        if left == 0 || top == 0 || right == w || bottom == h {
            continue;
        }
        // 有限、确定性的邻域搜索；整块供体必须是无瑕疵皮肤。
        let step = rw.max(rh) as isize + 2;
        let mut donor = None;
        let mut best = f32::INFINITY;
        for distance in [1, 2, 3] {
            for (dx, dy) in [
                (-1, -1),
                (0, -1),
                (1, -1),
                (-1, 0),
                (1, 0),
                (-1, 1),
                (0, 1),
                (1, 1),
            ] {
                let ox = dx * step * distance;
                let oy = dy * step * distance;
                if left as isize + ox < 1
                    || top as isize + oy < 1
                    || right as isize + ox >= w as isize
                    || bottom as isize + oy >= h as isize
                {
                    continue;
                }
                let mut cost = 0.0;
                let mut valid = true;
                for y in top..bottom {
                    for x in left..right {
                        let i = y * w + x;
                        let j = (y as isize + oy) as usize * w + (x as isize + ox) as usize;
                        if clean_border[j] < 0.98 || mask[i] < 0.85 || support[i] < 0.25 {
                            valid = false;
                            break;
                        }
                        if x == left || x + 1 == right || y == top || y + 1 == bottom {
                            cost += (log[i] - log[j]).abs() + 2.0 * (red[i] - red[j]).abs();
                        }
                    }
                    if !valid {
                        break;
                    }
                }
                cost = cost / (2 * rw + 2 * rh - 4) as f32 + distance as f32 * 0.002;
                if valid && cost < best {
                    best = cost;
                    donor = Some((ox, oy));
                }
            }
        }
        let Some((ox, oy)) = donor else {
            continue;
        };
        if best > 0.25 {
            continue;
        }
        // 局部距离羽化不扩散到五官；重叠候选只保留更强修正。
        let mut distance = vec![usize::MAX / 2; rw * rh];
        let mut queue = VecDeque::new();
        for i in points {
            let local = (i / w - top) * rw + i % w - left;
            distance[local] = 0;
            queue.push_back(local);
        }
        while let Some(i) = queue.pop_front() {
            if distance[i] >= feather {
                continue;
            }
            let (x, y) = (i % rw, i / rw);
            for (nx, ny) in [
                (x.wrapping_sub(1), y),
                (x + 1, y),
                (x, y.wrapping_sub(1)),
                (x, y + 1),
            ] {
                if nx < rw && ny < rh {
                    let j = ny * rw + nx;
                    if distance[j] > distance[i] + 1 {
                        distance[j] = distance[i] + 1;
                        queue.push_back(j);
                    }
                }
            }
        }
        for y in top..bottom {
            for x in left..right {
                let i = y * w + x;
                let d = distance[(y - top) * rw + x - left];
                if d > feather {
                    continue;
                }
                let weight = mask[i] * (1.0 - smoothstep(0.0, (feather + 1) as f32, d as f32));
                let target = (y0 + y) * linear.width() as usize + x0 + x;
                if weight <= repair[target][3] {
                    continue;
                }
                let j = (y as isize + oy) as usize * w + (x as isize + ox) as usize;
                for c in 0..3 {
                    repair[target][c] = (base[c][i] - source[i][c]).clamp(-0.25, 0.25) * weight;
                    let limit = (luma[i] * 0.15).min(0.025);
                    texture[target][c] = (source[j][c] - low[c][j]).clamp(-limit, limit) * weight;
                }
                repair[target][3] = weight;
                texture[target][3] = weight;
            }
        }
    }
}

fn gaussian(uv: [f32; 2], center: [f32; 2], radius: [f32; 2]) -> f32 {
    let d = ((uv[0] - center[0]) / radius[0]).powi(2) + ((uv[1] - center[1]) / radius[1]).powi(2);
    (-0.5 * d).exp()
}

const CONTOUR_BLOB_COUNT: usize = 17;

fn contour_blobs(face: &FaceAnalysis) -> [([f32; 2], [f32; 2], f32); CONTOUR_BLOB_COUNT] {
    let p = &face.points;
    let fw = (p[454][0] - p[234][0]).abs().max(0.1);
    let fh = (p[152][1] - p[10][1]).abs().max(0.1);
    let center = |a: usize, b: usize, t: f32| {
        [
            p[a][0] * (1.0 - t) + p[b][0] * t,
            p[a][1] * (1.0 - t) + p[b][1] * t,
        ]
    };
    [
        // 五官 Dodge：额、鼻梁窄带、颧峰、颏垫
        (center(10, 151, 0.55), [fw * 0.17, fh * 0.10], 0.42),
        (center(168, 1, 0.45), [fw * 0.038, fh * 0.13], 0.48),
        (center(50, 101, 0.5), [fw * 0.12, fh * 0.075], 0.38),
        (center(280, 330, 0.5), [fw * 0.12, fh * 0.075], 0.38),
        (center(17, 152, 0.6), [fw * 0.13, fh * 0.055], 0.34),
        // 轮廓 Burn：颞、下颌、颌缘
        (center(127, 162, 0.5), [fw * 0.09, fh * 0.13], -0.40),
        (center(356, 389, 0.5), [fw * 0.09, fh * 0.13], -0.40),
        (center(234, 172, 0.5), [fw * 0.11, fh * 0.12], -0.48),
        (center(454, 397, 0.5), [fw * 0.11, fh * 0.12], -0.48),
        (center(172, 150, 0.5), [fw * 0.10, fh * 0.055], -0.32),
        (center(397, 379, 0.5), [fw * 0.10, fh * 0.055], -0.32),
        // 轮廓 Burn：鼻翼侧、鼻唇沟、眶下缘（仅用合成脸与实拍都可靠的 landmark）
        (center(98, 1, 0.25), [fw * 0.045, fh * 0.055], -0.22),
        (center(327, 1, 0.25), [fw * 0.045, fh * 0.055], -0.22),
        (center(50, 152, 0.4), [fw * 0.07, fh * 0.05], -0.20),
        (center(280, 152, 0.4), [fw * 0.07, fh * 0.05], -0.20),
        (center(50, 168, 0.35), [fw * 0.065, fh * 0.04], -0.14),
        (center(280, 168, 0.35), [fw * 0.065, fh * 0.04], -0.14),
    ]
}

/// 按局部光影残差缩放化妆打光：同侧保留，逆光衰减。
fn contour_template_gains(
    face: &FaceAnalysis,
    residual_at: impl Fn([f32; 2]) -> f32,
) -> [f32; CONTOUR_BLOB_COUNT] {
    contour_blobs(face).map(|(center, _, amount)| {
        let residual = residual_at(center);
        let agree = smoothstep(-0.03, 0.05, residual * amount.signum());
        0.9 * agree
    })
}

fn face_contour_features(face: &FaceAnalysis, uv: [f32; 2], gains: &[f32; CONTOUR_BLOB_COUNT]) -> f32 {
    contour_blobs(face)
        .iter()
        .zip(gains)
        .filter(|((_, _, amount), _)| *amount > 0.0)
        .map(|((center, radius, amount), gain)| gaussian(uv, *center, *radius) * amount * gain)
        .sum()
}

fn face_contour_outline(face: &FaceAnalysis, uv: [f32; 2], gains: &[f32; CONTOUR_BLOB_COUNT]) -> f32 {
    contour_blobs(face)
        .iter()
        .zip(gains)
        .filter(|((_, _, amount), _)| *amount < 0.0)
        .map(|((center, radius, amount), gain)| gaussian(uv, *center, *radius) * amount * gain)
        .sum()
}

fn balance(values: &mut [f32], mask: &[f32], limit: f32) {
    if values.iter().all(|v| *v == 0.0) {
        return;
    }
    let mut lo = -2.0;
    let mut hi = 2.0;
    for _ in 0..24 {
        let shift = (lo + hi) * 0.5;
        let sum: f64 = values
            .iter()
            .zip(mask)
            .map(|(&v, &m)| ((v - shift).clamp(-limit, limit) * m) as f64)
            .sum();
        if sum > 0.0 {
            lo = shift;
        } else {
            hi = shift;
        }
    }
    let shift = (lo + hi) * 0.5;
    for (value, &weight) in values.iter_mut().zip(mask) {
        *value = (*value - shift).clamp(-limit, limit) * weight;
    }
}

fn build_fields(
    linear: &Rgb32FImage,
    luminance: &[f32],
    segmentation: Segmentation,
) -> (PortraitAnalysis, Vec<f32>) {
    let width = linear.width() as usize;
    let height = linear.height() as usize;
    let n = width * height;
    let mut pixels: Vec<[f32; 4]> = luminance.iter().map(|&y| [0.0, 0.0, 0.0, y]).collect();
    let mut frequency_pixels = vec![[0.0; 4]; n];
    let mut frequency_detail_pixels = vec![[0.0; 4]; n];
    let mut hydrated_pixels = vec![[0.0; 4]; n];
    let mut evenness_chroma_pixels = vec![[0.0; 4]; n];
    let mut blemish_pixels = vec![[0.0; 4]; n];
    let mut texture_pixels = vec![[0.0; 4]; n];
    let mut shine_pixels = vec![[0.0; 4]; n];
    let eye_white_pixels = vec![[0.0; 4]; n];
    let has_face = !segmentation.faces.is_empty();
    let face_scale = face_frequency_scale(&segmentation.faces);
    let mut body: Vec<f32> = (0..n)
        .into_par_iter()
        .map(|i| {
            smoothstep(
                0.7,
                0.95,
                sample_field(
                    &segmentation.body,
                    256,
                    256,
                    (i % width) as f32 * 256.0 / width as f32 + 128.0 / width as f32,
                    (i / width) as f32 * 256.0 / height as f32 + 128.0 / height as f32,
                ),
            )
        })
        .collect();
    body.par_iter_mut().enumerate().for_each(|(i, weight)| {
        let mut keep = 1.0f32;
        for face in &segmentation.faces {
            let uv = face
                .roi
                .unproject((i % width) as f32 + 0.5, (i / width) as f32 + 0.5);
            keep = keep.min(oval_body_keep(face, uv));
        }
        *weight *= keep;
    });
    // 关键点失败的检测区域不能退回身体修容，宁可跳过不可靠区域。
    for roi in &segmentation.rejected_faces {
        body.par_iter_mut().enumerate().for_each(|(i, weight)| {
            let uv = roi.unproject((i % width) as f32 + 0.5, (i / width) as f32 + 0.5);
            if (uv[0] - 0.5).abs() < 0.38 && (uv[1] - 0.5).abs() < 0.4 {
                *weight = 0.0;
            }
        });
    }
    // 连通皮肤区域独立确定尺度和平衡曝光，避免多人之间相互影响。
    let mut visited = vec![false; n];
    for seed in 0..n {
        if visited[seed] || body[seed] <= 0.05 {
            continue;
        }
        let mut indices = vec![seed];
        visited[seed] = true;
        let mut cursor = 0;
        let (mut x0, mut y0, mut x1, mut y1) = (width, height, 0, 0);
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
                    if !visited[j] && body[j] > 0.05 {
                        visited[j] = true;
                        indices.push(j);
                    }
                }
            }
        }
        if indices.len() < 64 {
            continue;
        }
        let w = x1 - x0;
        let h = y1 - y0;
        let mut mask = vec![0.0; w * h];
        for &i in &indices {
            mask[(i / width - y0) * w + i % width - x0] = body[i];
        }
        let frequency_mask = if has_face {
            let mut frequency_mask = mask.clone();
            for i in 0..frequency_mask.len() {
                if frequency_mask[i] <= 0.0 {
                    continue;
                }
                let rgb = linear.get_pixel((x0 + i % w) as u32, (y0 + i / w) as u32).0;
                frequency_mask[i] *= body_skin_likelihood(rgb);
            }
            Some(feather_retouch_mask(
                &frequency_mask,
                luminance,
                width,
                (x0, y0, w, h),
                frequency_region_scale(w.min(h) as f32, face_scale),
                0.8,
                2.0,
            ))
        } else {
            None
        };
        let hydrated_mask = feather_retouch_mask(
            &mask,
            luminance,
            width,
            (x0, y0, w, h),
            frequency_region_scale(w.min(h) as f32, face_scale),
            0.8,
            2.0,
        );
        process_region(
            &mut pixels,
            &mut evenness_chroma_pixels,
            linear,
            luminance,
            width,
            (x0, y0, w, h),
            mask,
            None,
        );
        if let Some(frequency_mask) = frequency_mask {
            process_frequency_region(
                &mut frequency_pixels,
                &mut frequency_detail_pixels,
                linear,
                (x0, y0, w, h),
                &frequency_mask,
                frequency_region_scale(w.min(h) as f32, face_scale),
                false,
            );
        }
        process_hydrated_region(
            &mut hydrated_pixels,
            linear,
            (x0, y0, w, h),
            &hydrated_mask,
            frequency_region_scale(w.min(h) as f32, face_scale),
            false,
        );
    }
    for face in &segmentation.faces {
        let corners =
            [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]].map(|p| face.roi.project(p[0], p[1]));
        let x0 = corners
            .iter()
            .map(|p| p[0])
            .fold(f32::INFINITY, f32::min)
            .floor()
            .max(0.0)
            .min(width as f32) as usize;
        let y0 = corners
            .iter()
            .map(|p| p[1])
            .fold(f32::INFINITY, f32::min)
            .floor()
            .max(0.0)
            .min(height as f32) as usize;
        let x1 = corners
            .iter()
            .map(|p| p[0])
            .fold(f32::NEG_INFINITY, f32::max)
            .ceil()
            .max(0.0)
            .min(width as f32) as usize;
        let y1 = corners
            .iter()
            .map(|p| p[1])
            .fold(f32::NEG_INFINITY, f32::max)
            .ceil()
            .max(0.0)
            .min(height as f32) as usize;
        if x1 <= x0 || y1 <= y0 {
            continue;
        }
        let w = x1 - x0;
        let h = y1 - y0;
        let face_uv = |i: usize| {
            face.roi
                .unproject((x0 + i % w) as f32 + 0.5, (y0 + i / w) as f32 + 0.5)
        };
        let shine_mask: Vec<f32> = (0..w * h)
            .into_par_iter()
            .map(|i| shine_face_mask(face, face_uv(i)))
            .collect();
        let shine_prior: Vec<f32> = (0..w * h)
            .into_par_iter()
            .map(|i| shine_t_zone_prior(face, face_uv(i)))
            .collect();
        process_shine_region(
            &mut shine_pixels,
            linear,
            (x0, y0, w, h),
            &shine_mask,
            face.roi.side / 1.5,
            Some(&shine_prior),
        );
        // 眼白已迁 GPUPixel；不再写入 atlas 眼白场。
        let blemish_mask: Vec<f32> = (0..w * h)
            .into_par_iter()
            .map(|i| blemish_face_mask(face, face_uv(i)))
            .collect();
        process_blemish_region(
            &mut blemish_pixels,
            &mut texture_pixels,
            linear,
            (x0, y0, w, h),
            blemish_mask,
            face.roi.side / 1.5,
        );
        let mask = (0..w * h)
            .into_par_iter()
            .map(|i| {
                face_mask(
                    face,
                    face.roi
                        .unproject((x0 + i % w) as f32 + 0.5, (y0 + i / w) as f32 + 0.5),
                )
            })
            .collect();
        let frequency_mask: Vec<f32> = (0..w * h)
            .into_par_iter()
            .map(|i| {
                frequency_face_mask(
                    face,
                    face.roi
                        .unproject((x0 + i % w) as f32 + 0.5, (y0 + i / w) as f32 + 0.5),
                )
            })
            .collect();
        let frequency_mask = feather_retouch_mask(
            &frequency_mask,
            luminance,
            width,
            (x0, y0, w, h),
            face.roi.side / 1.5,
            0.8,
            2.0,
        );
        process_region(
            &mut pixels,
            &mut evenness_chroma_pixels,
            linear,
            luminance,
            width,
            (x0, y0, w, h),
            mask,
            Some(face),
        );
        process_frequency_region(
            &mut frequency_pixels,
            &mut frequency_detail_pixels,
            linear,
            (x0, y0, w, h),
            &frequency_mask,
            face.roi.side / 1.5,
            true,
        );
        process_hydrated_region(
            &mut hydrated_pixels,
            linear,
            (x0, y0, w, h),
            &frequency_mask,
            face.roi.side / 1.5,
            true,
        );
        clear_eyeball_retouch_fields(
            face,
            width,
            (x0, y0, w, h),
            &mut pixels,
            &mut evenness_chroma_pixels,
            &mut frequency_pixels,
            &mut frequency_detail_pixels,
            &mut hydrated_pixels,
        );
    }
    let has_skin = evenness_chroma_pixels.iter().any(|p| p[3] > 0.05)
        || pixels
            .iter()
            .any(|p| p[0].abs() > 0.001 || p[1].abs() > 0.001 || p[2].abs() > 0.001);
    let even_pixels = vec![[0.0; 4]; n];
    let whitening_pixels = vec![[0.0; 4]; n];
    (
        PortraitAnalysis {
            summary: PortraitSummary {
                key: String::new(),
                face_count: segmentation.faces.len(),
                skipped_faces: segmentation.skipped_faces,
                has_skin,
            },
            width: width as u32,
            height: height as u32,
            full_dimensions: (width as u32, height as u32),
            pixels: Arc::from(pixels),
            frequency_pixels: Arc::from(frequency_pixels),
            frequency_detail_pixels: Arc::from(frequency_detail_pixels),
            hydrated_pixels: Arc::from(hydrated_pixels),
            even_pixels: Arc::from(even_pixels),
            whitening_pixels: Arc::from(whitening_pixels),
            evenness_chroma_pixels: Arc::from(evenness_chroma_pixels),
            blemish_pixels: Arc::from(blemish_pixels),
            texture_pixels: Arc::from(texture_pixels),
            shine_pixels: Arc::from(shine_pixels),
            eye_white_pixels: Arc::from(eye_white_pixels),
        },
        body,
    )
}

/// AI 完整美肤替换普通祛痘场；先完成 sRGB 合成再求线性差值，不能再次乘瑕疵概率。
fn apply_ai_blemishes(
    models: &mut PortraitModels,
    encoded: &Rgb32FImage,
    linear: &Rgb32FImage,
    faces: &[[f32; 4]],
    should_stop: &dyn Fn() -> bool,
) -> Result<Vec<[f32; 4]>, String> {
    let repaired = models.beautify_interruptible(encoded, faces, should_stop)?;
    Ok(beauty_delta_fields(&repaired, encoded, linear))
}

/// 普通滑块：仅局部检测+修复，不含匀肤/美白。
fn apply_retouch_blemishes(
    models: &mut PortraitModels,
    encoded: &Rgb32FImage,
    linear: &Rgb32FImage,
    faces: &[[f32; 4]],
    should_stop: &dyn Fn() -> bool,
) -> Result<Vec<[f32; 4]>, String> {
    let repaired = models.retouch_blemishes_only_interruptible(encoded, faces, should_stop)?;
    Ok(beauty_delta_fields(&repaired, encoded, linear))
}

/// 匀肤结果只保留低频色调，高频（痘/斑/毛孔）从原图加回，避免干涉祛斑祛痘。
fn restore_even_high_frequency(
    evened: &Rgb32FImage,
    original: &Rgb32FImage,
    region: &[f32],
) -> Rgb32FImage {
    let (w, h) = (evened.width() as usize, evened.height() as usize);
    debug_assert_eq!(evened.dimensions(), original.dimensions());
    debug_assert_eq!(region.len(), w * h);
    if !region.iter().any(|&r| r != 0.0) {
        return evened.clone();
    }
    let scale = (w.min(h) as f32).max(64.0);
    // 与瑕疵最大尺度同级：痘斑进高频，大面积肤色不均留在 generator 低频结果。
    let radius = (scale * 0.022).round().clamp(3.0, 48.0) as usize;
    let even_ch: [Vec<f32>; 3] = std::array::from_fn(|c| evened.pixels().map(|p| p.0[c]).collect());
    let orig_ch: [Vec<f32>; 3] =
        std::array::from_fn(|c| original.pixels().map(|p| p.0[c]).collect());
    let even_lf: [Vec<f32>; 3] = std::array::from_fn(|c| box_mean(&even_ch[c], w, h, radius));
    let orig_lf: [Vec<f32>; 3] = std::array::from_fn(|c| box_mean(&orig_ch[c], w, h, radius));
    Rgb32FImage::from_fn(w as u32, h as u32, |x, y| {
        let i = y as usize * w + x as usize;
        if region[i] == 0.0 {
            return *original.get_pixel(x, y);
        }
        Rgb(std::array::from_fn(|c| {
            (even_lf[c][i] + orig_ch[c][i] - orig_lf[c][i]).clamp(0.0, 1.0)
        }))
    })
}

/// 均匀肤色：generator 匀肤后写脸(+)/身(-) 分号增量场。
fn apply_even_skin(
    models: &mut PortraitModels,
    encoded: &Rgb32FImage,
    linear: &Rgb32FImage,
    faces: &[[f32; 4]],
    body: &[f32],
) -> Result<Vec<[f32; 4]>, String> {
    let (repaired, region) = models.even_skin_compose(encoded, faces, body)?;
    let repaired = restore_even_high_frequency(&repaired, encoded, &region);
    let mut fields = beauty_delta_fields(&repaired, encoded, linear);
    for (pixel, &sign) in fields.iter_mut().zip(region.iter()) {
        if sign == 0.0 {
            *pixel = [0.0; 4];
            continue;
        }
        // 未改动像素 beauty_delta 已为零；有改动时用区域符号写入 A。
        if pixel[3] > 0.0 || pixel[0] != 0.0 || pixel[1] != 0.0 || pixel[2] != 0.0 {
            pixel[3] = sign.signum() * sign.abs().clamp(0.0, 1.0).max(0.05);
        } else if sign.abs() > 0.05 {
            // ROI 内无数值差但仍属区域：保留弱覆盖供强度门控（通常仍为零色）。
            pixel[3] = sign.signum() * 0.05;
        }
    }
    Ok(fields)
}

fn attach_even_pixels(
    base: &Arc<PortraitAnalysis>,
    layer: &EvenSkinLayer,
) -> Arc<PortraitAnalysis> {
    Arc::new(PortraitAnalysis {
        summary: PortraitSummary {
            key: even_variant_key(&base.summary.key),
            face_count: base.summary.face_count,
            skipped_faces: base.summary.skipped_faces,
            has_skin: base.summary.has_skin,
        },
        width: base.width,
        height: base.height,
        full_dimensions: base.full_dimensions,
        pixels: Arc::clone(&base.pixels),
        frequency_pixels: Arc::clone(&base.frequency_pixels),
        frequency_detail_pixels: Arc::clone(&base.frequency_detail_pixels),
        hydrated_pixels: Arc::clone(&base.hydrated_pixels),
        even_pixels: Arc::clone(&layer.even),
        whitening_pixels: Arc::clone(&base.whitening_pixels),
        evenness_chroma_pixels: Arc::clone(&base.evenness_chroma_pixels),
        blemish_pixels: Arc::clone(&base.blemish_pixels),
        texture_pixels: Arc::clone(&base.texture_pixels),
        shine_pixels: Arc::clone(&base.shine_pixels),
        eye_white_pixels: Arc::clone(&base.eye_white_pixels),
    })
}

/// 独立美白：整图美白后写脸(+)/身(-) 分号增量场。
fn apply_skin_whitening(
    models: &mut PortraitModels,
    encoded: &Rgb32FImage,
    linear: &Rgb32FImage,
    faces: &[[f32; 4]],
    body: &[f32],
) -> Result<Vec<[f32; 4]>, String> {
    let (repaired, region) = models.whitening_compose(encoded, faces, body)?;
    let mut fields = beauty_delta_fields(&repaired, encoded, linear);
    for (pixel, &sign) in fields.iter_mut().zip(region.iter()) {
        if sign == 0.0 {
            *pixel = [0.0; 4];
            continue;
        }
        if pixel[3] > 0.0 || pixel[0] != 0.0 || pixel[1] != 0.0 || pixel[2] != 0.0 {
            pixel[3] = sign.signum() * sign.abs().clamp(0.0, 1.0).max(0.05);
        } else if sign.abs() > 0.05 {
            pixel[3] = sign.signum() * 0.05;
        }
    }
    Ok(fields)
}

fn attach_whitening_pixels(
    base: &Arc<PortraitAnalysis>,
    layer: &WhiteningLayer,
) -> Arc<PortraitAnalysis> {
    Arc::new(PortraitAnalysis {
        summary: PortraitSummary {
            key: whitening_variant_key(&base.summary.key),
            face_count: base.summary.face_count,
            skipped_faces: base.summary.skipped_faces,
            has_skin: base.summary.has_skin,
        },
        width: base.width,
        height: base.height,
        full_dimensions: base.full_dimensions,
        pixels: Arc::clone(&base.pixels),
        frequency_pixels: Arc::clone(&base.frequency_pixels),
        frequency_detail_pixels: Arc::clone(&base.frequency_detail_pixels),
        hydrated_pixels: Arc::clone(&base.hydrated_pixels),
        even_pixels: Arc::clone(&base.even_pixels),
        whitening_pixels: Arc::clone(&layer.whitening),
        evenness_chroma_pixels: Arc::clone(&base.evenness_chroma_pixels),
        blemish_pixels: Arc::clone(&base.blemish_pixels),
        texture_pixels: Arc::clone(&base.texture_pixels),
        shine_pixels: Arc::clone(&base.shine_pixels),
        eye_white_pixels: Arc::clone(&base.eye_white_pixels),
    })
}

pub(crate) fn beauty_delta_fields(
    repaired: &Rgb32FImage,
    encoded: &Rgb32FImage,
    linear: &Rgb32FImage,
) -> Vec<[f32; 4]> {
    assert_eq!(repaired.dimensions(), encoded.dimensions());
    assert_eq!(linear.dimensions(), encoded.dimensions());
    repaired
        .pixels()
        .zip(encoded.pixels())
        .zip(linear.pixels())
        .map(|((after, before), original)| {
            // 编码图相同的像素保持严格零增量，包括 RAW 超范围高光。
            if after
                .0
                .iter()
                .zip(before.0)
                .all(|(a, b)| (*a - b).abs() <= 1e-7)
            {
                return [0.0; 4];
            }
            let rgb = after.0.map(srgb_to_linear);
            [
                rgb[0] - original[0],
                rgb[1] - original[1],
                rgb[2] - original[2],
                1.0,
            ]
        })
        .collect()
}

fn process_region(
    pixels: &mut [[f32; 4]],
    chroma: &mut [[f32; 4]],
    linear: &Rgb32FImage,
    luminance: &[f32],
    image_width: usize,
    rect: (usize, usize, usize, usize),
    mask: Vec<f32>,
    face: Option<&FaceAnalysis>,
) -> Vec<f32> {
    let (x0, y0, width, height) = rect;
    let index = |i: usize| (y0 + i / width) * image_width + x0 + i % width;
    let scale = face.map_or((width.min(height) as f32).max(16.0), |f| f.roi.side / 1.5);
    // 羽化只能减小语义区域，不能把修正扩散到原先为零的保护区。
    let mask = feather_retouch_mask(&mask, luminance, image_width, rect, scale, 0.3, 1.2);
    let log: Vec<f32> = (0..width * height)
        .map(|i| luminance[index(i)].max(1e-6).log2())
        .collect();
    // 平整独立频带：约 1.8%–5.5% 脸宽，避开毛孔级并少吃半脸阴影。
    let even_medium_r = (scale * 0.018).round().max(2.0) as usize;
    let even_blotch_r = (scale * 0.055).round().max(3.0) as usize;
    let even_medium = guided_smooth(&log, &mask, width, height, even_medium_r);
    let even_blotch = guided_smooth(&log, &mask, width, height, even_blotch_r);
    // 立体仍用原斑驳带（约 3.5%），避免扩平整时削掉轮廓体量。
    let blotch = guided_smooth(
        &log,
        &mask,
        width,
        height,
        (scale * 0.035).round().max(3.0) as usize,
    );
    let illumination = guided_smooth(
        &log,
        &mask,
        width,
        height,
        (scale * 0.18).round().max(8.0) as usize,
    );
    let mut even: Vec<f32> = even_blotch
        .iter()
        .zip(&even_medium)
        .map(|(b, m)| (b - m) * 0.92)
        .collect();
    // 五官通道耦合部分中频平整，使「仅拉立体」也能压脸颊斑驳起伏。
    let mut features: Vec<f32> = even.iter().map(|e| e * 0.4).collect();
    let mut outline: Vec<f32> = blotch
        .iter()
        .zip(&illumination)
        .map(|(b, l)| (b - l) * 0.85)
        .collect();
    if let Some(face) = face {
        if face.contour_reliable {
            let residual = outline.clone();
            let gains = contour_template_gains(face, |center| {
                let xy = face.roi.project(center[0], center[1]);
                sample_field(
                    &residual,
                    width,
                    height,
                    xy[0] - x0 as f32,
                    xy[1] - y0 as f32,
                )
            });
            for i in 0..mask.len() {
                let uv = face
                    .roi
                    .unproject((x0 + i % width) as f32 + 0.5, (y0 + i / width) as f32 + 0.5);
                features[i] += face_contour_features(face, uv, &gains);
                outline[i] += face_contour_outline(face, uv, &gains);
            }
        }
    }
    balance(&mut even, &mask, 0.45);
    let contour_limit = if face.is_some() { 0.55 } else { 0.35 };
    balance(&mut features, &mask, contour_limit);
    balance(&mut outline, &mask, contour_limit);

    // 色度中频：弱增益、不做 balance，避免脸内橙灰块重分配。
    let mut cr = vec![0.0f32; width * height];
    let mut cb = vec![0.0f32; width * height];
    for i in 0..width * height {
        let rgb = linear.get_pixel((x0 + i % width) as u32, (y0 + i / width) as u32).0;
        let y = luminance[index(i)];
        cr[i] = rgb[0] - y;
        cb[i] = rgb[2] - y;
    }
    let cr_medium = guided_smooth(&cr, &mask, width, height, even_medium_r);
    let cr_blotch = guided_smooth(&cr, &mask, width, height, even_blotch_r);
    let cb_medium = guided_smooth(&cb, &mask, width, height, even_medium_r);
    let cb_blotch = guided_smooth(&cb, &mask, width, height, even_blotch_r);
    let d_cr: Vec<f32> = cr_blotch
        .iter()
        .zip(&cr_medium)
        .zip(&mask)
        .map(|((b, m), w)| ((b - m) * 0.35).clamp(-0.05, 0.05) * w)
        .collect();
    let d_cb: Vec<f32> = cb_blotch
        .iter()
        .zip(&cb_medium)
        .zip(&mask)
        .map(|((b, m), w)| ((b - m) * 0.35).clamp(-0.05, 0.05) * w)
        .collect();

    for i in 0..mask.len() {
        let gi = index(i);
        // 覆盖权重改存 chroma.a；EV 打包为 [even, features, outline, luma]。
        if mask[i] > chroma[gi][3] {
            let target = &mut pixels[gi];
            *target = [even[i], features[i], outline[i], target[3]];
            let dr = d_cr[i];
            let db = d_cb[i];
            // 保持局部亮度：ΔG 抵消 R/B 对 luma 的贡献。
            let dg = -(0.2126 * dr + 0.0722 * db) / 0.7152;
            chroma[gi] = [dr, dg, db, mask[i]];
        }
    }
    mask
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn background_prewarm_yields_to_foreground_unless_promoted() {
        assert!(!should_yield(0, 0, false));
        assert!(should_yield(1, 0, false));
        assert!(should_yield(0, 1, false));
        assert!(should_yield(2, 3, false));
        assert!(!should_yield(0, 0, true));
        assert!(!should_yield(1, 0, true));
        assert!(!should_yield(0, 1, true));
    }

    #[test]
    fn portrait_analysis_generation_rejects_stale_work_and_late_cancel() {
        let state = PortraitState::default();
        let first = state.begin_analysis(1).unwrap();
        let second = state.begin_analysis(2).unwrap();
        assert!(!state.analysis_current(first));
        assert!(state.analysis_current(second));
        state.cancel_analysis(first);
        assert!(state.analysis_current(second), "迟到取消不能误伤新请求");
        state.cancel_analysis(second);
        assert!(!state.analysis_current(second));
        assert!(state.begin_analysis(1).is_err());
        assert_eq!(state.begin_analysis(3).unwrap(), 3);
    }

    #[test]
    fn foreground_guard_tracks_activity_and_idle() {
        let state = PortraitState::default();
        assert!(state.foreground_idle_for(Duration::ZERO));
        {
            let _outer = state.foreground();
            let _inner = state.foreground();
            assert_eq!(state.foreground_active.load(Ordering::SeqCst), 2);
            assert!(!state.foreground_idle_for(Duration::ZERO));
        }
        assert_eq!(state.foreground_active.load(Ordering::SeqCst), 0);
        assert!(state.foreground_idle_for(Duration::ZERO));
        assert!(!state.foreground_idle_for(Duration::from_secs(3600)));
        drop(state.lock_models_foreground().unwrap());
        assert_eq!(state.foreground_waiters.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn intermediates_survive_cloned_reads() {
        let mut list = LruList::default();
        let value = AnalysisIntermediates {
            linear: Rgb32FImage::from_pixel(2, 2, Rgb([0.25; 3])),
            faces: vec![[0.0, 0.0, 1.0, 1.0]],
            body: vec![0.5; 4],
        };
        list.insert("base".into(), value.byte_size(), value, INTERMEDIATE_BUDGET);
        let first = list.get("base").cloned().unwrap();
        let second = list.get("base").cloned().unwrap();
        assert_eq!(first.faces, second.faces);
        assert_eq!(first.linear.dimensions(), (2, 2));
        assert!(list.get("base").is_some());
    }

    #[test]
    fn beauty_delta_reconstructs_full_result_and_preserves_hdr_identity() {
        let encoded = Rgb32FImage::from_pixel(3, 1, Rgb([0.5; 3]));
        let mut linear = Rgb32FImage::from_pixel(3, 1, Rgb([srgb_to_linear(0.5); 3]));
        linear.put_pixel(2, 0, Rgb([3.0, 2.0, 4.0]));
        let mut result = encoded.clone();
        result.put_pixel(0, 0, Rgb([0.7, 0.6, 0.55]));
        result.put_pixel(1, 0, Rgb([0.3, 0.4, 0.45]));
        let delta = beauty_delta_fields(&result, &encoded, &linear);
        assert_eq!(delta[2], [0.0; 4], "编码不变时不能截断 HDR 高光");
        for i in 0..2 {
            assert_eq!(delta[i][3], 1.0, "新增匀肤和美白也必须写入覆盖标记");
            for c in 0..3 {
                assert!(
                    (linear.get_pixel(i as u32, 0)[c] + delta[i][c]
                        - srgb_to_linear(result.get_pixel(i as u32, 0)[c]))
                    .abs()
                        < 1e-7
                );
            }
        }
        assert!(delta[0][0] > 0.0 && delta[1][0] < 0.0);
        assert!(
            beauty_delta_fields(&encoded, &encoded, &linear)
                .iter()
                .all(|v| *v == [0.0; 4])
        );
    }

    #[test]
    fn shine_strength_is_independent_and_visibility_gated() {
        for (value, expected) in [
            (json!(50.5), 0.51),
            (json!(101), 1.0),
            (json!(-2), 0.0),
            (json!("50"), 0.0),
            (Value::Null, 0.0),
        ] {
            let mut a = json!({"portrait": {"faceShineRemoval": value, "textureRetention": 100}, "sectionVisibility": {"portraitSkin": false}});
            assert_eq!(blemish_strengths(&a), [0.0, 0.0, expected, 0.0]);
            assert_eq!(enabled(&a), expected > 0.0);
            assert_eq!(base_adjustments(&a), base_adjustments(&json!({})));
            a["portrait"]["aiBlemishRemoval"] = json!(40);
            assert_eq!(blemish_strengths(&a), [0.4, 0.0, expected, 0.0]);
            a["sectionVisibility"]["portraitBlemishes"] = json!(false);
            assert!(!enabled(&a));
        }
    }

    fn shine_fixture(base: [f32; 3], amount: f32, texture: bool) -> Rgb32FImage {
        Rgb32FImage::from_fn(128, 128, |x, y| {
            let distance = ((x as f32 - 64.0).powi(2) + (y as f32 - 64.0).powi(2)).sqrt();
            let shine = amount * (1.0 - smoothstep(3.0, 9.0, distance));
            let detail = if texture {
                if (x + y) % 2 == 0 { 0.002 } else { -0.002 }
            } else {
                0.0
            };
            Rgb(base.map(|v| v + shine + detail))
        })
    }

    #[test]
    fn shine_reduces_neutral_reflections_and_preserves_texture_and_hdr() {
        for (base, amount) in [
            ([0.32, 0.22, 0.16], 0.35),
            ([0.09, 0.05, 0.03], 0.15),
            ([0.7, 0.55, 0.45], 0.4),
            ([1.4, 0.9, 0.6], 1.2),
        ] {
            let image = shine_fixture(base, amount, true);
            let mut fields = vec![[0.0; 4]; 128 * 128];
            process_shine_region(
                &mut fields,
                &image,
                (0, 0, 128, 128),
                &vec![1.0; 128 * 128],
                128.0,
                None,
            );
            let center = 64 * 128 + 64;
            let center_y = luma_rgb([fields[center][0], fields[center][1], fields[center][2]]);
            assert!(
                center_y < -0.005,
                "未抑制油光：{base:?}, {:?}",
                fields[center]
            );
            assert_eq!(fields[0], [0.0; 4], "正常皮肤不变");
            let lum = skin_luminance(&image);
            for (i, field) in fields.iter().enumerate() {
                assert!(field.iter().all(|v| v.is_finite()));
                let dy = -luma_rgb([field[0], field[1], field[2]]);
                assert!(dy >= -0.001 && dy <= lum[i] * SHINE_MAX_REDUCTION_HIGH + 0.002);
            }
            // 色度肤色上中心修正不应退化为等通道灰减。
            if (base[0] - base[2]).abs() > 0.05 {
                let f = fields[center];
                let dy = -luma_rgb([f[0], f[1], f[2]]);
                if dy > 0.005 {
                    assert!(
                        (f[0] - f[2]).abs() > 1e-5,
                        "色度肤色上不应退化为等通道减光：{f:?}"
                    );
                }
            }
            let plain = shine_fixture(base, amount, false);
            let mut plain_fields = vec![[0.0; 4]; 128 * 128];
            process_shine_region(
                &mut plain_fields,
                &plain,
                (0, 0, 128, 128),
                &vec![1.0; 128 * 128],
                128.0,
                None,
            );
            for y in 61..=67 {
                for x in 61..=67 {
                    let i = y * 128 + x;
                    for c in 0..3 {
                        assert!(
                            (fields[i][c] - plain_fields[i][c]).abs() < 0.0005,
                            "不能把高频纹理写进修正场"
                        );
                    }
                }
            }
            if base[0] > 1.0 {
                assert!(image.get_pixel(64, 64)[0] + fields[center][0] > 1.0);
            }
        }
    }

    #[test]
    fn shine_preserves_chroma_better_than_equal_rgb_subtraction() {
        let base = [0.32, 0.22, 0.16];
        let image = shine_fixture(base, 0.35, false);
        let mut fields = vec![[0.0; 4]; 128 * 128];
        process_shine_region(
            &mut fields,
            &image,
            (0, 0, 128, 128),
            &vec![1.0; 128 * 128],
            128.0,
            None,
        );
        let c = 64 * 128 + 64;
        let rgb = image.get_pixel(64, 64).0;
        let delta = [fields[c][0], fields[c][1], fields[c][2]];
        let repaired = apply_shine_safety(rgb, delta);
        let out = [
            (rgb[0] + repaired[0]).max(0.0),
            (rgb[1] + repaired[1]).max(0.0),
            (rgb[2] + repaired[2]).max(0.0),
        ];
        let equal_amt = (-luma_rgb(delta)).max(0.0);
        let gray = [
            (rgb[0] - equal_amt).max(0.0),
            (rgb[1] - equal_amt).max(0.0),
            (rgb[2] - equal_amt).max(0.0),
        ];
        let chroma = |p: [f32; 3]| {
            let y = luma_rgb(p).max(1e-6);
            ((p[0] / y - base[0] / luma_rgb(base).max(1e-6)).powi(2)
                + (p[2] / y - base[2] / luma_rgb(base).max(1e-6)).powi(2))
            .sqrt()
        };
        assert!(
            chroma(out) < chroma(gray) + 0.02,
            "保色路径色度偏移应不差于等通道减光：out={} gray={}",
            chroma(out),
            chroma(gray)
        );
        assert!(luma_rgb(out) < luma_rgb(rgb) - 0.01);
    }

    #[test]
    fn shine_donor_fallback_avoids_shadow_color_bleed() {
        // 左半阴影供体、右半油光：不可把阴影色强行灌入高光。
        let shadow = [0.08f32, 0.05, 0.04];
        let cheek = [0.32f32, 0.22, 0.16];
        let image = Rgb32FImage::from_fn(128, 128, |x, y| {
            let d = ((x as f32 - 96.0).powi(2) + (y as f32 - 64.0).powi(2)).sqrt();
            let shine = 0.4 * (1.0 - smoothstep(3.0, 10.0, d));
            if x < 48 {
                Rgb(shadow)
            } else {
                Rgb([cheek[0] + shine, cheek[1] + shine, cheek[2] + shine])
            }
        });
        let mut fields = vec![[0.0; 4]; 128 * 128];
        process_shine_region(
            &mut fields,
            &image,
            (0, 0, 128, 128),
            &vec![1.0; 128 * 128],
            128.0,
            None,
        );
        let c = 64 * 128 + 96;
        let rgb = image.get_pixel(96, 64).0;
        let out = [
            (rgb[0] + fields[c][0]).max(0.0),
            (rgb[1] + fields[c][1]).max(0.0),
            (rgb[2] + fields[c][2]).max(0.0),
        ];
        let chroma_dist = |a: [f32; 3], b: [f32; 3]| {
            let ca = shine_chroma(a);
            let cb = shine_chroma(b);
            ((ca[0] - cb[0]).powi(2) + (ca[1] - cb[1]).powi(2)).sqrt()
        };
        assert!(
            chroma_dist(out, shadow) >= chroma_dist(out, cheek) - 0.01,
            "修复色度不应更接近阴影供体：out={out:?} d_shadow={} d_cheek={}",
            chroma_dist(out, shadow),
            chroma_dist(out, cheek)
        );
        // 仅阴影可靠时，修复置信应明显低于满权均匀肤色场景。
        let uniform = shine_fixture(cheek, 0.4, false);
        let mut full = vec![[0.0; 4]; 128 * 128];
        process_shine_region(
            &mut full,
            &uniform,
            (0, 0, 128, 128),
            &vec![1.0; 128 * 128],
            128.0,
            None,
        );
        assert!(
            fields[c][3] < full[64 * 128 + 64][3] * 0.9 + 0.05,
            "阴影侧供体不足时应降低修复置信：{} vs {}",
            fields[c][3],
            full[64 * 128 + 64][3]
        );
    }

    #[test]
    fn apply_shine_safety_caps_relative_and_luma_loss() {
        let current = [0.4, 0.28, 0.2];
        let huge = [-0.5, -0.5, -0.5];
        let capped = apply_shine_safety(current, huge);
        for c in 0..3 {
            assert!(-capped[c] <= current[c] * SHINE_APPLY_REL_CAP + 1e-5);
            assert!(current[c] + capped[c] >= -1e-5);
        }
        let loss = luma_rgb(current) - luma_rgb([
            current[0] + capped[0],
            current[1] + capped[1],
            current[2] + capped[2],
        ]);
        assert!(loss <= luma_rgb(current) * SHINE_MAX_REDUCTION_HIGH + 1e-4);
        assert_eq!(apply_shine_safety(current, [0.0; 3]), [0.0; 3]);
    }

    #[test]
    fn shine_skips_normal_lighting_protection_and_insufficient_skin() {
        let oil = shine_fixture([0.32, 0.22, 0.16], 0.35, false);
        for (image, mask) in [
            (
                Rgb32FImage::from_pixel(128, 128, Rgb([0.7, 0.6, 0.5])),
                vec![1.0; 128 * 128],
            ),
            (
                Rgb32FImage::from_pixel(128, 128, Rgb([0.001; 3])),
                vec![1.0; 128 * 128],
            ),
            (
                Rgb32FImage::from_fn(128, 128, |x, _| {
                    Rgb([0.32, 0.22, 0.16].map(|v| v * (0.5 + x as f32 / 128.0)))
                }),
                vec![1.0; 128 * 128],
            ),
            (oil.clone(), vec![0.0; 128 * 128]),
            (
                oil.clone(),
                (0..128 * 128)
                    .map(|i| if i % 128 < 4 && i / 128 < 4 { 1.0 } else { 0.0 })
                    .collect(),
            ),
            (
                Rgb32FImage::from_pixel(128, 128, Rgb([f32::NAN, 0.3, 0.2])),
                vec![1.0; 128 * 128],
            ),
        ] {
            let mut fields = vec![[0.0; 4]; 128 * 128];
            process_shine_region(&mut fields, &image, (0, 0, 128, 128), &mask, 128.0, None);
            assert!(fields.iter().all(|p| *p == [0.0; 4]));
        }
        let mask: Vec<f32> = (0..128 * 128)
            .map(|i| if i % 128 < 64 { 0.0 } else { 1.0 })
            .collect();
        let mut fields = vec![[0.0; 4]; 128 * 128];
        process_shine_region(&mut fields, &oil, (0, 0, 128, 128), &mask, 128.0, None);
        assert!(
            fields
                .iter()
                .enumerate()
                .filter(|(i, _)| i % 128 < 64)
                .all(|(_, p)| *p == [0.0; 4])
        );
    }

    #[test]
    fn shine_face_mask_feathers_features_and_t_zone_prioritizes_forehead() {
        let face = synthetic_face();
        // 五官中心仍清零。
        for uv in [[0.33, 0.4], [0.67, 0.4], [0.5, 0.68]] {
            assert_eq!(shine_face_mask(&face, uv), 0.0);
        }
        // shine 在眼缘外过渡带应羽化减弱；face_mask 在排除带外应接近满权。
        let soft_edge = [0.33 + 0.07 + 0.012, 0.4];
        let hard_edge = [0.33 + 0.07 + EYE_EXCLUDE_MARGIN + 0.025, 0.4];
        let soft = shine_face_mask(&face, soft_edge);
        let hard = face_mask(&face, hard_edge);
        assert!(hard > 0.9, "硬切蒙版在五官外应接近满权：{hard}");
        assert!(
            soft > 0.0 && soft < 0.95,
            "五官外侧 shine 应羽化减弱：soft={soft}"
        );
        let forehead = shine_t_zone_prior(&face, [0.50, 0.28]);
        let cheek = shine_t_zone_prior(&face, [0.34, 0.52]);
        let jaw = shine_t_zone_prior(&face, [0.50, 0.88]);
        assert!(forehead > 0.85, "额心应接近峰值：{forehead}");
        assert!(cheek > jaw, "颧骨应高于下颌：{cheek} vs {jaw}");
        assert!(
            (jaw - SHINE_T_ZONE_FLOOR).abs() < 0.08,
            "下颌应接近保底：{jaw}"
        );
        // prior 近似线性调制修正幅度。
        let oil = shine_fixture([0.32, 0.22, 0.16], 0.35, false);
        let mask = vec![1.0; 128 * 128];
        let prior = vec![0.5; 128 * 128];
        let mut full = vec![[0.0; 4]; 128 * 128];
        let mut half = vec![[0.0; 4]; 128 * 128];
        process_shine_region(&mut full, &oil, (0, 0, 128, 128), &mask, 128.0, None);
        process_shine_region(
            &mut half,
            &oil,
            (0, 0, 128, 128),
            &mask,
            128.0,
            Some(&prior),
        );
        let c = 64 * 128 + 64;
        let full_y = luma_rgb([full[c][0], full[c][1], full[c][2]]);
        let half_y = luma_rgb([half[c][0], half[c][1], half[c][2]]);
        assert!(full_y < -0.005);
        assert!(
            (half_y - 0.5 * full_y).abs() < 1e-3,
            "prior=0.5 应近似减半：full={:?} half={:?}",
            full[c],
            half[c]
        );
    }

    #[test]
    fn shine_fields_are_face_only_and_overlaps_do_not_stack() {
        let image = Rgb32FImage::from_fn(256, 128, |x, y| {
            let cx = if x < 128 { 48.0 } else { 200.0 };
            let d = ((x as f32 - cx).powi(2) + (y as f32 - 75.0).powi(2)).sqrt();
            Rgb([0.32, 0.22, 0.16].map(|v| v + 0.35 * (1.0 - smoothstep(1.0, 4.0, d))))
        });
        let luma = skin_luminance(&image);
        let (empty, _) = build_fields(&image, &luma, synthetic_skin(vec![]));
        assert!(empty.shine_pixels.iter().all(|p| *p == [0.0; 4]));
        let (one, _) = build_fields(&image, &luma, synthetic_skin(vec![synthetic_face()]));
        let (duplicate, _) = build_fields(
            &image,
            &luma,
            synthetic_skin(vec![synthetic_face(), synthetic_face()]),
        );
        assert_eq!(one.shine_pixels, duplicate.shine_pixels);
        assert!(one.shine_pixels.iter().any(|p| p[0] < 0.0));
        assert!(
            one.shine_pixels
                .iter()
                .enumerate()
                .filter(|(i, _)| i % 256 >= 128)
                .all(|(_, p)| *p == [0.0; 4])
        );
        let face = synthetic_face();
        for y in 16..112 {
            for x in 16..112 {
                if shine_face_mask(&face, face.roi.unproject(x as f32 + 0.5, y as f32 + 0.5))
                    <= 1e-6
                {
                    assert_eq!(one.shine_pixels[y * 256 + x], [0.0; 4]);
                }
            }
        }
        let mut other = synthetic_face();
        other.roi.center[0] += 152.0;
        let (two, _) = build_fields(&image, &luma, synthetic_skin(vec![synthetic_face(), other]));
        assert!(
            luma_rgb([
                two.shine_pixels[75 * 256 + 200][0],
                two.shine_pixels[75 * 256 + 200][1],
                two.shine_pixels[75 * 256 + 200][2],
            ]) < -0.001
        );
        assert_eq!(
            two.shine_pixels[75 * 256 + 48],
            one.shine_pixels[75 * 256 + 48]
        );
    }

    #[test]
    fn normalization_and_independence() {
        assert_eq!(strengths(&json!({})), [0.0; 4]);
        assert_eq!(
            strengths(&json!({"portrait":{"neutralGrayEvenness":101, "neutralGrayContour":50.5}})),
            [1.0, 0.51, 0.0, 0.0]
        );
        assert_eq!(
            strengths(&json!({"portrait":{"neutralGrayEvenness":"50", "neutralGrayContour":-2}})),
            [0.0; 4]
        );
        assert_eq!(
            strengths(
                &json!({"portrait":{"neutralGrayEvenness":50},"sectionVisibility":{"portraitSkin":false}})
            ),
            [0.0; 4]
        );
        let a =
            json!({"portrait":{"neutralGrayEvenness":5},"crop":{"x":2},"exposure":1,"rotation":5});
        let b = json!({"portrait":{"neutralGrayContour":100},"exposure":3,"rotation":5});
        assert_eq!(base_adjustments(&a), base_adjustments(&b));
        assert_eq!(
            contour_band_strengths(&json!({"portrait":{}})),
            [0.8, 0.8, 0.0, 0.0],
            "缺省五官/轮廓子带为 0.8"
        );
        assert_eq!(
            contour_band_strengths(&json!({"portrait":{
                "neutralGrayContourFeatures":40.4,
                "neutralGrayContourOutline":0
            }})),
            [0.4, 0.0, 0.0, 0.0]
        );
        assert_eq!(
            contour_band_strengths(&json!({"portrait":{
                "neutralGrayContourFeatures":200,
                "neutralGrayContourOutline":-5
            }})),
            [1.0, 0.0, 0.0, 0.0]
        );
    }

    #[test]
    fn frequency_strengths_are_independent_and_visibility_gated() {
        for (slot, key) in ["frequencySeparationFace", "frequencySeparationBody"]
            .iter()
            .enumerate()
        {
            for (value, expected) in [
                (json!(50.5), 0.51),
                (json!(200), 1.0),
                (json!(-4), 0.0),
                (json!("50"), 0.0),
                (Value::Null, 0.0),
            ] {
                let mut adjustments = json!({"portrait": {}});
                adjustments["portrait"][*key] = value;
                let mut amounts = [0.0; 4];
                amounts[slot + 2] = expected;
                assert_eq!(strengths(&adjustments), amounts);
                assert_eq!(enabled(&adjustments), expected > 0.0);
                adjustments["sectionVisibility"] = json!({"portraitSkin": false});
                assert!(!enabled(&adjustments));
            }
        }
        assert!(enabled(&json!({"portrait":{"hydratedSmoothingFace":100}})));
        assert!(enabled(&json!({"portrait":{"hydratedSmoothingBody":100}})));
        assert_eq!(
            hydrated_strengths(
                &json!({"portrait":{"hydratedSmoothingFace":50.5,"hydratedSmoothingBody":200}})
            ),
            [0.51, 1.0, 0.0, 0.0],
        );
        assert_eq!(
            hydrated_strengths(
                &json!({"portrait":{"hydratedSmoothingFace":100},"sectionVisibility":{"portraitSkin":false}})
            ),
            [0.0; 4],
        );
        assert_eq!(
            frequency_band_strengths(&json!({"portrait":{}})),
            [1.0; 4],
            "缺省：低频满量、高频中性 high_amt=1"
        );
        assert_eq!(
            frequency_band_strengths(&json!({"portrait":{
                "frequencySeparationFaceLow":40.4,
                "frequencySeparationFaceHigh":-100,
                "frequencySeparationBodyLow":80,
                "frequencySeparationBodyHigh":10
            }})),
            [0.4, 2.0, 0.8, 0.9],
        );
        assert_eq!(
            frequency_band_strengths(&json!({"portrait":{
                "frequencySeparationLow":40.4,
                "frequencySeparationHigh":0
            }})),
            [0.4, 1.0, 0.4, 1.0],
            "共享旧键：High=0 → high_amt=1 中性"
        );
        assert_eq!(
            frequency_band_strengths(&json!({"portrait":{
                "frequencySeparationFaceHigh":100,
                "frequencySeparationBodyHigh":-50
            }})),
            [1.0, 0.0, 1.0, 1.5],
            "双极：+100→0.0 满柔化，-50→1.5 增强"
        );
        assert!(!enabled(&json!({"portrait":{
            "frequencySeparationFaceLow":0,
            "frequencySeparationFaceHigh":0,
            "frequencySeparationBodyLow":0,
            "frequencySeparationBodyHigh":0
        }})));
    }

    #[test]
    fn restore_even_high_frequency_keeps_blemishes_and_applies_low_freq_tone() {
        let (w, h) = (256usize, 256usize);
        let mut original = Rgb32FImage::from_pixel(w as u32, h as u32, Rgb([0.55, 0.45, 0.40]));
        for y in 0..h {
            for x in 0..w {
                let wash = (x as f32 / w as f32).powi(2) * 0.18;
                let p = original.get_pixel_mut(x as u32, y as u32);
                p.0[0] = (0.55 + wash).min(1.0);
            }
        }
        // 模拟痘点：远小于匀肤模糊核的局部凹陷。
        for y in 126..130 {
            for x in 126..130 {
                original.put_pixel(x as u32, y as u32, Rgb([0.22, 0.16, 0.14]));
            }
        }
        let evened = Rgb32FImage::from_pixel(w as u32, h as u32, Rgb([0.55, 0.45, 0.40]));
        let region = vec![1.0f32; w * h];
        let restored = restore_even_high_frequency(&evened, &original, &region);
        let spot = restored.get_pixel(127, 127).0[0];
        let flat = restored.get_pixel(140, 127).0[0];
        assert!(
            spot < flat - 0.12,
            "痘斑应作为高频保留：spot={spot} neighbor={flat}"
        );
        let right = restored.get_pixel(230, 128).0[0];
        let orig_right = original.get_pixel(230, 128).0[0];
        assert!(
            (right - 0.55).abs() < (orig_right - 0.55).abs() * 0.45,
            "低频肤色不均应被匀开：restored={right} orig={orig_right}"
        );
        let outside = restore_even_high_frequency(&evened, &original, &vec![0.0; w * h]);
        assert_eq!(outside.get_pixel(127, 127), evened.get_pixel(127, 127));
    }

    #[test]
    fn even_strengths_scale_and_gate_by_skin_visibility() {
        assert_eq!(
            even_strengths(&json!({"portrait": {"evenSkinFace": 50, "evenSkinBody": 80}})),
            [0.5, 0.8, 0.0, 0.0]
        );
        assert!(even_skin_enabled(&json!({"portrait": {"evenSkinFace": 1}})));
        assert!(!even_skin_enabled(
            &json!({"portrait": {"evenSkinFace": 0, "evenSkinBody": 0}})
        ));
        let mut hidden = json!({"portrait": {"evenSkinFace": 100, "evenSkinBody": 100}});
        hidden["sectionVisibility"] = json!({"portraitSkin": false});
        assert_eq!(even_strengths(&hidden), [0.0; 4]);
        assert!(!even_skin_enabled(&hidden));
        assert_eq!(even_variant_key("base"), "base:even");
        assert!(enabled(&json!({"portrait": {"evenSkinBody": 40}})));
    }

    #[test]
    fn whitening_strengths_scale_and_gate_by_skin_visibility() {
        assert_eq!(
            whitening_strengths(&json!({"portrait": {"whiteningFace": 40, "whiteningBody": 70}})),
            [0.4, 0.7, 0.0, 0.0]
        );
        assert!(whitening_enabled(
            &json!({"portrait": {"whiteningFace": 1}})
        ));
        assert!(!whitening_enabled(
            &json!({"portrait": {"whiteningFace": 0, "whiteningBody": 0}})
        ));
        let mut hidden = json!({"portrait": {"whiteningFace": 100, "whiteningBody": 100}});
        hidden["sectionVisibility"] = json!({"portraitSkin": false});
        assert_eq!(whitening_strengths(&hidden), [0.0; 4]);
        assert!(!whitening_enabled(&hidden));
        assert_eq!(whitening_variant_key("base"), "base:whiten");
        assert!(enabled(&json!({"portrait": {"whiteningBody": 25}})));
    }

    #[test]
    fn whitening_strength_for_region_blends_face_into_body() {
        assert!((whitening_strength_for_region(0.5, 1.0, 0.0) - 0.5).abs() < 1e-6);
        assert!((whitening_strength_for_region(1.0, 1.0, 0.0) - 1.0).abs() < 1e-6);
        assert!((whitening_strength_for_region(-0.8, 1.0, 0.0) - 0.0).abs() < 1e-6);
        assert!((whitening_strength_for_region(-0.8, 1.0, 0.5) - 0.4).abs() < 1e-6);
        // 0.8*0.25 + 0.2*0.75
        assert!((whitening_strength_for_region(0.25, 0.8, 0.2) - 0.35).abs() < 1e-6);
        // 脸身都满时，羽化带不能掉强度，否则交界发黑。
        assert!((whitening_strength_for_region(0.15, 1.0, 1.0) - 1.0).abs() < 1e-6);
        assert!((whitening_strength_for_region(0.15, 1.0, 0.0) - 0.15).abs() < 1e-6);
    }

    #[test]
    fn whitening_sample_keeps_full_delta_across_equal_strength_boundary() {
        let width = 2u32;
        let height = 2u32;
        let mut field = vec![[0.0; 4]; 4];
        let reference = vec![[0.0, 0.0, 0.0, 0.5]; 4];
        for x in 0..2 {
            field[x] = [0.4, 0.4, 0.4, 1.0];
            field[2 + x] = [0.4, 0.4, 0.4, -1.0];
        }
        // 落在两行正中，四邻点权重接近。
        let uv = [0.5, 0.5];
        let both = sample_whitening_delta(&field, &reference, width, height, uv, 0.5, 1.0, 1.0);
        assert!(
            (both[0] - 0.4).abs() < 0.02,
            "脸身同强度时交界不应发暗，got {}",
            both[0]
        );
        let face_only =
            sample_whitening_delta(&field, &reference, width, height, uv, 0.5, 1.0, 0.0);
        assert!(
            face_only[0] > 0.3,
            "仅脸部时交界仍应保留脸侧增量，got {}",
            face_only[0]
        );
    }

    #[test]
    fn eye_white_direct_process_writes_redness_delta() {
        let width = 128u32;
        let height = 128u32;
        let mut image = Rgb32FImage::from_pixel(width, height, Rgb([0.50, 0.48, 0.46]));
        let face = synthetic_face();
        let mut painted = 0usize;
        for y in 0..height {
            for x in 0..width {
                let uv = face.roi.unproject(x as f32 + 0.5, y as f32 + 0.5);
                let (b, g, _) = sclera_eye_weights(
                    &face,
                    uv,
                    LEFT_EYE,
                    LEFT_IRIS_CENTER,
                    LEFT_IRIS_RING,
                    LEFT_INNER_CANTHUS,
                );
                if b > 0.2 {
                    image.put_pixel(x, y, Rgb([0.80, 0.40, 0.38]));
                    painted += 1;
                }
                let _ = g;
            }
        }
        assert!(painted > 10, "synthetic sclera should paint, got {painted}");
        let n = (width * height) as usize;
        let mut fields = vec![[0.0f32; 4]; n];
        process_eye_white_region(&mut fields, &image, (0, 0, width as usize, height as usize), &face);
        let neg_r = fields.iter().filter(|p| p[0] < -0.01).count();
        let mask_n = fields.iter().filter(|p| p[3] > 0.2).count();
        assert!(mask_n > 0, "expected brighten mask");
        assert!(
            neg_r > 0,
            "expected redness deltas; mask={mask_n} neg_r={neg_r} sample={:?}",
            fields.iter().find(|p| p[3] > 0.2)
        );
    }

    #[test]
    fn eye_white_strengths_gate_on_portrait_eyes() {
        // Atlas 强度恒 0；启用态由 GPUPixel 参数表达。
        assert_eq!(
            eye_white_strengths(&json!({"portrait": {"eyeWhiteBrighten": 40, "eyeRedness": 70}})),
            [0.0; 4]
        );
        assert!(eye_white_enabled(
            &json!({"portrait": {"eyeWhiteBrighten": 1}})
        ));
        assert!(!eye_white_enabled(
            &json!({"portrait": {"eyeWhiteBrighten": 0, "eyeRedness": 0}})
        ));
        let mut hidden = json!({"portrait": {"eyeWhiteBrighten": 100, "eyeRedness": 100}});
        hidden["sectionVisibility"] = json!({"portraitEyes": false});
        assert_eq!(eye_white_strengths(&hidden), [0.0; 4]);
        assert!(!eye_white_enabled(&hidden));
        // 仅眼白不再打开 Rust enabled()（不跑 atlas）；GPUPixel requested 另测。
        assert!(!enabled(&json!({"portrait": {"eyeRedness": 25}})));
        assert!(crate::portrait_gpupixel::requested(&json!({
            "portrait": {"eyeRedness": 25}
        })));
    }

    #[test]
    fn eye_white_field_reduces_sclera_redness_and_spares_iris() {
        let width = 128u32;
        let height = 128u32;
        let mut image = Rgb32FImage::from_pixel(width, height, Rgb([0.45, 0.35, 0.30]));
        let face = synthetic_face();
        let iris_xy = face.roi.project(0.33, 0.4);
        for y in 40..60 {
            for x in 30..55 {
                let uv = face.roi.unproject(x as f32 + 0.5, y as f32 + 0.5);
                let sclera = sclera_face_mask(&face, uv);
                if sclera > 0.25 {
                    image.put_pixel(x, y, Rgb([0.72, 0.42, 0.40]));
                } else if ((uv[0] - 0.33).powi(2) + (uv[1] - 0.4).powi(2)).sqrt() < 0.02 {
                    image.put_pixel(x, y, Rgb([0.12, 0.08, 0.05]));
                }
            }
        }
        let n = (width * height) as usize;
        let mut fields = vec![[0.0f32; 4]; n];
        process_eye_white_region(&mut fields, &image, (0, 0, width as usize, height as usize), &face);
        assert!(fields.iter().any(|p| p[3] > 0.25), "应生成巩膜蒙版");
        assert!(
            fields.iter().any(|p| p[3] > 0.25 && p[0] < -0.01),
            "偏红巩膜应有负 R 增量"
        );
        let iris_i = (iris_xy[1].round() as usize).min(height as usize - 1) * width as usize
            + (iris_xy[0].round() as usize).min(width as usize - 1);
        assert!(
            fields[iris_i][3] < 0.15,
            "虹膜中心蒙版应弱：{}",
            fields[iris_i][3]
        );
        // 合成侧：直接采样单位场（atlas 强度已关，产品走 GPUPixel）。
        let mut found_red = false;
        let mut found_lift = false;
        for y in 40..60 {
            for x in 30..55 {
                let i = (y * width + x) as usize;
                if fields[i][3] < 0.25 {
                    continue;
                }
                let uv = [
                    (x as f32 + 0.5) / width as f32,
                    (y as f32 + 0.5) / height as f32,
                ];
                let p = image.get_pixel(x, y).0;
                let luma = p[0] * 0.2126 + p[1] * 0.7152 + p[2] * 0.0722;
                let d_r = sample_eye_white_delta(&fields, width, height, uv, luma, 0.0, 1.0, p);
                let d_b = sample_eye_white_delta(&fields, width, height, uv, luma, 1.0, 0.0, p);
                if d_r[0] < -1e-4 {
                    found_red = true;
                }
                if d_b[0] > 1e-4 {
                    found_lift = true;
                }
            }
        }
        assert!(found_red, "祛红应降低巩膜 R");
        assert!(found_lift, "提亮应抬升巩膜亮度");
    }

    #[test]
    fn eye_white_field_boosts_thin_vessel_over_clean_sclera() {
        let width = 128u32;
        let height = 128u32;
        let mut image = Rgb32FImage::from_pixel(width, height, Rgb([0.45, 0.35, 0.30]));
        let face = synthetic_face();
        // 干净巩膜底 + 细红丝，验证局部 excess 响应。
        let mut vessel_xy = None;
        let mut clean_xy = None;
        for y in 40..60 {
            for x in 30..55 {
                let uv = face.roi.unproject(x as f32 + 0.5, y as f32 + 0.5);
                let sclera = sclera_face_mask(&face, uv);
                if sclera <= 0.25 {
                    continue;
                }
                image.put_pixel(x, y, Rgb([0.62, 0.60, 0.58]));
                if clean_xy.is_none() && sclera > 0.4 {
                    clean_xy = Some((x, y));
                }
                // 竖直细丝（约 1px），偏红但不过饱和。
                if x == 42 && sclera > 0.35 {
                    image.put_pixel(x, y, Rgb([0.78, 0.42, 0.40]));
                    vessel_xy = Some((x, y));
                }
            }
        }
        let (vessel_x, vessel_y) = vessel_xy.expect("应找到巩膜细丝像素");
        let (clean_x, clean_y) = clean_xy.expect("应找到干净巩膜像素");
        let n = (width * height) as usize;
        let mut fields = vec![[0.0f32; 4]; n];
        process_eye_white_region(&mut fields, &image, (0, 0, width as usize, height as usize), &face);
        let vessel_i = vessel_y as usize * width as usize + vessel_x as usize;
        let clean_i = clean_y as usize * width as usize + clean_x as usize;
        assert!(fields[vessel_i][3] > 0.25, "细丝处应有巩膜蒙版");
        assert!(
            fields[vessel_i][0] < -0.01,
            "细丝处应有负 R 增量：{}",
            fields[vessel_i][0]
        );
        assert!(
            fields[vessel_i][0] < fields[clean_i][0] - 0.005,
            "细丝负 R 应强于干净眼白：vessel={} clean={}",
            fields[vessel_i][0],
            fields[clean_i][0]
        );
    }

    #[test]
    fn eye_white_brighten_preserves_chroma() {
        let field = [[0.0, 0.0, 0.0, 1.0]; 1];
        let current = [0.55f32, 0.52, 0.38];
        let luma = current[0] * 0.2126 + current[1] * 0.7152 + current[2] * 0.0722;
        let chroma = |c: [f32; 3]| {
            let y = c[0] * 0.2126 + c[1] * 0.7152 + c[2] * 0.0722;
            (c[0] - y).abs() + (c[1] - y).abs() + (c[2] - y).abs()
        };
        let d = sample_eye_white_delta(&field, 1, 1, [0.5, 0.5], luma, 1.0, 0.0, current);
        let after = [current[0] + d[0], current[1] + d[1], current[2] + d[2]];
        let luma_after = after[0] * 0.2126 + after[1] * 0.7152 + after[2] * 0.0722;
        assert!(luma_after > luma + 1e-4, "提亮应抬升亮度");
        assert!(
            (chroma(after) - chroma(current)).abs() < 0.02,
            "提亮不应明显改变色度"
        );
    }

    #[test]
    fn eye_white_compose_zero_strength_is_identity() {
        let delta = sample_eye_white_delta(
            &[[0.0, 0.0, 0.0, 0.8]; 4],
            2,
            2,
            [0.5, 0.5],
            0.5,
            0.0,
            0.0,
            [0.5, 0.5, 0.5],
        );
        assert_eq!(delta, [0.0; 3]);
    }

    /// 视觉回归：设置 `RAPIDRAW_PORTRAIT_EYE_WHITE_DEBUG=1` 后在应用中打开
    /// `paileme-miniapp/tempPic` 下四张测试图，调试蒙版输出到 `%TEMP%/rapidraw_eye_white_debug/`。
    #[test]
    fn eye_white_temp_pic_paths_documented() {
        let candidates = [
            std::path::Path::new(r"E:\paileme_ai\paileme-miniapp\tempPic"),
            std::path::Path::new("../paileme-miniapp/tempPic"),
        ];
        let found = candidates.iter().find(|p| p.exists());
        if let Some(dir) = found {
            let names = [
                "轻度红血丝.jpg",
                "明显红血丝.jpg",
                "偏黄眼白.jpg",
                "深色虹膜.jpg",
            ];
            for name in names {
                assert!(
                    dir.join(name).exists(),
                    "missing regression fixture {} under {}",
                    name,
                    dir.display()
                );
            }
        }
    }

    /// 与 WGSL `portrait_eye_white_delta` 同公式的标量对照（允许微小浮点差）。
    #[test]
    fn eye_white_cpu_matches_wgsl_compose_formula() {
        let field = [[-0.06, 0.012, 0.018, 0.85]; 1];
        let luma = 0.42f32;
        let current = [0.48f32, 0.40, 0.36];
        let brighten = 0.8f32;
        let redness = 0.6f32;
        let cpu = sample_eye_white_delta(&field, 1, 1, [0.5, 0.5], luma, brighten, redness, current);
        // 手算 WGSL 对齐路径。
        let soft_ceiling = 0.90f32;
        let gap = (soft_ceiling - luma).max(0.0);
        let gap_curve = gap * (0.40 + 0.60 * (gap / soft_ceiling).clamp(0.0, 1.0));
        let shadow_protect = smoothstep(0.04, 0.16, luma);
        let highlight_protect = 1.0 - smoothstep(soft_ceiling - 0.06, soft_ceiling, luma);
        let lift = 0.85 * brighten * 0.55 * gap_curve * shadow_protect * highlight_protect;
        let expected = [
            -0.06 * redness + lift,
            0.012 * redness + lift,
            0.018 * redness + lift,
        ];
        for c in 0..3 {
            assert!(
                (cpu[c] - expected[c]).abs() < 1e-5,
                "channel {c}: cpu={} expected={}",
                cpu[c],
                expected[c]
            );
        }
    }

    #[test]
    fn eye_white_brighten_and_redness_are_independent() {
        // A=1 提亮蒙版；RGB 为单位祛红。
        let field = [[-0.08, 0.01, 0.02, 1.0]; 1];
        let only_brighten = sample_eye_white_delta(
            &field,
            1,
            1,
            [0.5, 0.5],
            0.45,
            1.0,
            0.0,
            [0.5, 0.4, 0.35],
        );
        let only_red = sample_eye_white_delta(
            &field,
            1,
            1,
            [0.5, 0.5],
            0.45,
            0.0,
            1.0,
            [0.5, 0.4, 0.35],
        );
        assert!(only_brighten[0] > 0.0 && only_brighten[1] > 0.0 && only_brighten[2] > 0.0);
        assert!((only_brighten[0] - only_brighten[1]).abs() < 1e-5);
        assert!(only_red[0] < -0.05);
        assert!(only_red[0] < only_red[1]);
    }

    #[test]
    fn attach_whitening_preserves_blemish_and_even_fields() {
        let image = textured_skin();
        let luminance = skin_luminance(&image);
        let (mut base_raw, _) =
            build_fields(&image, &luminance, synthetic_skin(vec![synthetic_face()]));
        base_raw.summary.key = "base-key".into();
        let base = Arc::new(base_raw);
        let pixel_count = (base.width as usize) * (base.height as usize);
        let even_layer = EvenSkinLayer {
            even: Arc::from(vec![[0.01, 0.01, 0.01, 1.0]; pixel_count]),
        };
        let with_even = attach_even_pixels(&base, &even_layer);
        let whitening_layer = WhiteningLayer {
            whitening: Arc::from(vec![[0.02, 0.015, 0.01, -1.0]; pixel_count]),
        };
        let with_whiten = attach_whitening_pixels(&with_even, &whitening_layer);
        assert!(Arc::ptr_eq(&with_whiten.even_pixels, &even_layer.even));
        assert!(Arc::ptr_eq(
            &with_whiten.whitening_pixels,
            &whitening_layer.whitening
        ));
        assert!(Arc::ptr_eq(
            &with_whiten.blemish_pixels,
            &base.blemish_pixels
        ));
        assert_eq!(with_whiten.summary.key, "base-key:even:whiten");
    }

    #[test]
    fn neural_preview_bake_key_ignores_grade_and_tracks_ai_strength() {
        let base = json!({
            "portrait": {"aiBlemishRemoval": 80, "blemishRemoval": 0},
            "crop": {"x": 0, "y": 0, "width": 100, "height": 100},
            "exposure": 0.0,
            "sectionVisibility": {"portraitBlemishes": true, "portraitSkin": true}
        });
        let graded = {
            let mut v = base.clone();
            v["exposure"] = json!(1.5);
            v["contrast"] = json!(20);
            v["temperature"] = json!(10);
            v
        };
        let ai_changed = {
            let mut v = base.clone();
            v["portrait"]["aiBlemishRemoval"] = json!(40);
            v
        };
        let transform = crate::cache_utils::calculate_transform_hash(&base);
        let k_base = neural_preview_bake_key("img.raw", &base, "analysis:ai", 1280, 720, transform);
        let k_grade = neural_preview_bake_key(
            "img.raw",
            &graded,
            "analysis:ai",
            1280,
            720,
            crate::cache_utils::calculate_transform_hash(&graded),
        );
        let k_ai = neural_preview_bake_key(
            "img.raw",
            &ai_changed,
            "analysis:ai",
            1280,
            720,
            crate::cache_utils::calculate_transform_hash(&ai_changed),
        );
        assert_eq!(k_base, k_grade, "曝光/对比度/色温不得改变 bake 键");
        assert_ne!(k_base, k_ai, "AI 美肤强度变化必须使 bake 键失效");
        let k_size = neural_preview_bake_key("img.raw", &base, "analysis:ai", 960, 540, transform);
        assert_ne!(k_base, k_size, "预览尺寸变化必须使 bake 键失效");
    }

    #[test]
    fn bake_neural_preview_applies_blemish_delta_on_linear_base() {
        let width = 4u32;
        let height = 4u32;
        let pixel_count = (width * height) as usize;
        let mut blemish = vec![[0.0; 4]; pixel_count];
        // 中心像素写入可观测增量。
        blemish[5] = [0.2, 0.1, 0.05, 1.0];
        let analysis = Arc::new(PortraitAnalysis {
            summary: PortraitSummary {
                key: "bake-test".into(),
                face_count: 1,
                skipped_faces: 0,
                has_skin: true,
            },
            width,
            height,
            full_dimensions: (width, height),
            pixels: Arc::from(vec![[0.0, 0.0, 1.0, 0.5]; pixel_count]),
            frequency_pixels: Arc::from(vec![[0.0; 4]; pixel_count]),
            frequency_detail_pixels: Arc::from(vec![[0.0; 4]; pixel_count]),
            hydrated_pixels: Arc::from(vec![[0.0; 4]; pixel_count]),
            even_pixels: Arc::from(vec![[0.0; 4]; pixel_count]),
            whitening_pixels: Arc::from(vec![[0.0; 4]; pixel_count]),
            evenness_chroma_pixels: Arc::from(vec![[0.0; 4]; pixel_count]),
            blemish_pixels: Arc::from(blemish),
            texture_pixels: Arc::from(vec![[0.0; 4]; pixel_count]),
            shine_pixels: Arc::from(vec![[0.0; 4]; pixel_count]),
            eye_white_pixels: Arc::from(vec![[0.0; 4]; pixel_count]),
        });
        let maps = PortraitRenderMaps {
            analysis,
            uv_rect: [0.0, 0.0, 1.0, 1.0],
        };
        // 模拟编辑器预览基底：已是场景线性（非 RAW 也会先转线性）。
        let source = DynamicImage::ImageRgba32F(image::Rgba32FImage::from_pixel(
            width,
            height,
            image::Rgba([0.1, 0.1, 0.1, 1.0]),
        ));
        let adj = json!({"portrait": {"aiBlemishRemoval": 100}, "sectionVisibility": {"portraitBlemishes": true}});
        let baked = bake_neural_preview_onto(&source, &maps, &adj).expect("bake");
        let rgba = baked.as_rgba32f().expect("rgba32f");
        let px = rgba.get_pixel(1, 1).0;
        assert!(
            (px[0] - 0.3).abs() < 0.02,
            "线性域应近似 0.1+0.2，不能被错误 sRGB 往返抬成近白: {:?}",
            px
        );
        let baked2 = bake_neural_preview_onto(&source, &maps, &adj).expect("bake2");
        let px2 = baked2.as_rgba32f().expect("rgba32f").get_pixel(1, 1).0;
        assert!((px[0] - px2[0]).abs() < 1e-5);
    }

    #[test]
    fn ai_strength_scales_blemishes_and_separates_analysis_keys() {
        let visible =
            json!({"portrait":{"aiBlemishRemoval":100,"blemishRemoval":20,"textureRetention":80}});
        assert_eq!(blemish_strengths(&visible), [1.0, 0.0, 0.0, 0.0]);
        let half =
            json!({"portrait":{"aiBlemishRemoval":50,"blemishRemoval":20,"textureRetention":80}});
        assert_eq!(blemish_strengths(&half), [0.5, 0.0, 0.0, 0.0]);
        let off =
            json!({"portrait":{"aiBlemishRemoval":0,"blemishRemoval":20,"textureRetention":80}});
        // 神经祛痘场无质感回加；textureRetention 恒为 0。
        assert_eq!(blemish_strengths(&off), [0.2, 0.0, 0.0, 0.0]);
        assert!(ordinary_neural_blemish_enabled(&off));
        assert!(!ordinary_neural_blemish_enabled(&visible));
        // 旧照片布尔开关兼容。
        let legacy =
            json!({"portrait":{"aiBlemishRemoval":true,"blemishRemoval":20,"textureRetention":80}});
        assert_eq!(blemish_strengths(&legacy), [1.0, 0.0, 0.0, 0.0]);
        let mut hidden = visible.clone();
        hidden["sectionVisibility"] = json!({"portraitBlemishes":false});
        assert_eq!(blemish_strengths(&hidden), [0.0; 4]);
        assert!(!ordinary_neural_blemish_enabled(&hidden));
        // 分组显隐、开关与强度都不进入基础缓存键，避免触发重新识别。
        assert_eq!(base_adjustments(&visible), base_adjustments(&hidden));
        assert_eq!(base_adjustments(&visible), base_adjustments(&half));
        assert_eq!(base_adjustments(&off), base_adjustments(&half));
        // AI / 普通神经变体键独立于基础键，保证 GPU 纹理缓存区分。
        assert_ne!(ai_variant_key("base"), "base");
        assert_eq!(ai_variant_key("base"), format!("{}:ai", "base"));
        assert_eq!(retouch_variant_key("base"), format!("{}:retouch", "base"));
        assert_ne!(ai_variant_key("base"), retouch_variant_key("base"));
    }

    #[test]
    fn ai_variant_assembles_shared_fields_with_separate_key() {
        let image = textured_skin();
        let luminance = skin_luminance(&image);
        let (mut base_raw, _) =
            build_fields(&image, &luminance, synthetic_skin(vec![synthetic_face()]));
        base_raw.summary.key = "base-key".into();
        let base = Arc::new(base_raw);
        let pixel_count = (base.width as usize) * (base.height as usize);
        let layer = Arc::new(AiBlemishLayer {
            blemish: Arc::from(vec![[0.05, 0.02, 0.01, 1.0]; pixel_count]),
            zeros: Arc::from(vec![[0.0; 4]; pixel_count]),
        });
        let variant = assemble_neural_variant(&base, Some(&layer), "base-key:ai");
        // 基础场共享内存，不复制像素数据。
        assert!(Arc::ptr_eq(&variant.pixels, &base.pixels));
        assert!(Arc::ptr_eq(
            &variant.frequency_pixels,
            &base.frequency_pixels
        ));
        assert!(Arc::ptr_eq(
            &variant.frequency_detail_pixels,
            &base.frequency_detail_pixels
        ));
        assert!(Arc::ptr_eq(&variant.shine_pixels, &base.shine_pixels));
        assert!(Arc::ptr_eq(&variant.blemish_pixels, &layer.blemish));
        assert!(Arc::ptr_eq(&variant.texture_pixels, &layer.zeros));
        assert_eq!(variant.width, base.width);
        assert_eq!(variant.height, base.height);
        assert_eq!(variant.full_dimensions, base.full_dimensions);
        assert_eq!(variant.summary.face_count, base.summary.face_count);
        assert_eq!(variant.summary.has_skin, base.summary.has_skin);
        assert_eq!(variant.summary.key, "base-key:ai");
        let retouch = assemble_neural_variant(&base, Some(&layer), "base-key:retouch");
        assert_eq!(retouch.summary.key, "base-key:retouch");
        assert!(Arc::ptr_eq(&retouch.blemish_pixels, &layer.blemish));
        // 无人脸时直接复用基础零场，不分配新层。
        let no_face = assemble_neural_variant(&base, None, "base-key:ai");
        assert!(Arc::ptr_eq(&no_face.blemish_pixels, &base.blemish_pixels));
        assert!(Arc::ptr_eq(&no_face.texture_pixels, &base.texture_pixels));
        assert_eq!(no_face.summary.key, "base-key:ai");
    }

    #[test]
    fn lru_list_evicts_by_budget_and_refreshes_on_get() {
        let mut list: LruList<u32> = LruList::default();
        list.insert("a".into(), 40, 1, 100);
        list.insert("b".into(), 50, 2, 100);
        assert_eq!(list.get("a"), Some(&1));
        assert_eq!(list.get("b"), Some(&2));
        assert_eq!(list.bytes, 90);
        // 超预算时插入立即淘汰最久未使用的条目（get 刷新后 a 最旧）。
        list.insert("c".into(), 50, 3, 100);
        assert_eq!(list.get("b"), Some(&2));
        assert_eq!(list.get("c"), Some(&3));
        assert_eq!(list.get("a"), None);
        assert_eq!(list.bytes, 100);
        // 取出即移除并释放字节。
        assert_eq!(list.remove("c"), Some(3));
        assert_eq!(list.bytes, 50);
        assert_eq!(list.remove("c"), None);
        // 同键覆盖插入时释放旧字节。
        list.insert("b".into(), 30, 20, 100);
        assert_eq!(list.bytes, 30);
        assert_eq!(list.get("b"), Some(&20));
    }

    #[test]
    fn blemish_parameters_are_independent_and_texture_is_not_an_effect() {
        for (value, expected) in [
            (json!(50.5), 0.51),
            (json!(101), 1.0),
            (json!(-3), 0.0),
            (json!("50"), 0.0),
            (Value::Null, 0.0),
        ] {
            let a = json!({"portrait": {"blemishRemoval": value, "textureRetention": 75}, "sectionVisibility": {"portraitSkin": false}});
            assert_eq!(blemish_strengths(&a), [expected, 0.0, 0.0, 0.0]);
            assert_eq!(enabled(&a), expected > 0.0);
            let mut hidden = a.clone();
            hidden["sectionVisibility"]["portraitBlemishes"] = json!(false);
            assert!(!enabled(&hidden));
            hidden["sectionVisibility"]["portraitSkin"] = json!(true);
            hidden["portrait"]["frequencySeparationFace"] = json!(50);
            assert!(enabled(&hidden));
        }
        assert!(!enabled(&json!({"portrait": {"textureRetention": 100}})));
        assert_eq!(
            blemish_strengths(
                &json!({"portrait": {"blemishRemoval": 50, "textureRetention": "100"}})
            ),
            [0.5, 0.0, 0.0, 0.0]
        );
    }

    fn blemish_fixture(redness: bool) -> Rgb32FImage {
        Rgb32FImage::from_fn(128, 128, |x, y| {
            let detail = if (x + y) % 2 == 0 { 0.003 } else { -0.003 };
            let mut rgb = [0.32 + detail, 0.22 + detail, 0.16 + detail];
            if (62..=66).contains(&x) && (62..=66).contains(&y) {
                if redness {
                    rgb[0] += 0.06;
                    rgb[1] -= 0.045;
                    rgb[2] -= 0.03;
                } else {
                    rgb = rgb.map(|v| v * 0.65);
                }
            }
            Rgb(rgb)
        })
    }

    #[test]
    fn blemishes_reduce_dark_and_red_contrast_and_restore_clean_texture() {
        for redness in [false, true] {
            let image = blemish_fixture(redness);
            let mut repair = vec![[0.0; 4]; 128 * 128];
            let mut texture = repair.clone();
            process_blemish_region(
                &mut repair,
                &mut texture,
                &image,
                (0, 0, 128, 128),
                vec![1.0; 128 * 128],
                128.0,
            );
            let mut original_error = 0.0;
            let mut smooth_error = 0.0;
            let mut retained_error = 0.0;
            let mut smooth_detail = 0.0;
            let mut retained_detail = 0.0;
            for y in 62..=66 {
                for x in 62..=66 {
                    let i = y * 128 + x;
                    let source = image.get_pixel(x as u32, y as u32);
                    assert!(repair[i][3] > 0.9, "未识别瑕疵中心：{redness}, {x}, {y}");
                    for (c, baseline) in [0.32, 0.22, 0.16].into_iter().enumerate() {
                        original_error += (source[c] - baseline).abs();
                        smooth_error += (source[c] + repair[i][c] - baseline).abs();
                        retained_error +=
                            (source[c] + repair[i][c] + texture[i][c] - baseline).abs();
                    }
                    let phase = if (x + y) % 2 == 0 { 1.0 } else { -1.0 };
                    smooth_detail += (source[0] + repair[i][0] - 0.32) * phase;
                    retained_detail += (source[0] + repair[i][0] + texture[i][0] - 0.32) * phase;
                }
            }
            assert!(
                smooth_error < original_error * 0.5,
                "平滑修复不足：{smooth_error}/{original_error}"
            );
            assert!(
                retained_error < original_error * 0.5,
                "质感回加重现瑕疵：{retained_error}/{original_error}"
            );
            assert!(
                retained_detail.abs() > smooth_detail.abs() + 0.01,
                "未恢复清洁纹理"
            );
            assert!(
                repair
                    .iter()
                    .chain(&texture)
                    .flatten()
                    .all(|v| v.is_finite())
            );
            for y in 0..128 {
                for x in 0..128 {
                    if !(60..=68).contains(&x) || !(60..=68).contains(&y) {
                        assert_eq!(repair[y * 128 + x], [0.0; 4], "不能变成整脸磨皮");
                        assert_eq!(texture[y * 128 + x], [0.0; 4]);
                    }
                }
            }
        }
    }

    #[test]
    fn blemishes_fill_flat_centers_without_using_them_as_donors() {
        let mut image = Rgb32FImage::from_pixel(128, 128, Rgb([0.32, 0.22, 0.16]));
        for y in 60..=68 {
            for x in 60..=68 {
                image.put_pixel(x, y, Rgb([0.16, 0.11, 0.08]));
            }
        }
        let mut repair = vec![[0.0; 4]; 128 * 128];
        let mut texture = repair.clone();
        process_blemish_region(
            &mut repair,
            &mut texture,
            &image,
            (0, 0, 128, 128),
            vec![1.0; 128 * 128],
            128.0,
        );
        for y in 60..=68 {
            for x in 60..=68 {
                let i = y * 128 + x;
                assert!(repair[i][3] > 0.9, "平坦中心必须修复");
                assert!(
                    (image.get_pixel(x as u32, y as u32)[0] + repair[i][0] - 0.32).abs() < 0.08
                );
            }
        }
    }

    #[test]
    fn blemishes_skip_uniform_skin_lines_protected_regions_and_missing_donors() {
        let uniform = Rgb32FImage::from_pixel(128, 128, Rgb([0.32, 0.22, 0.16]));
        let mut line = uniform.clone();
        for y in 20..100 {
            line.put_pixel(64, y, Rgb([0.12, 0.08, 0.06]));
        }
        let mut pore = uniform.clone();
        pore.put_pixel(64, 64, Rgb([0.12, 0.08, 0.06]));
        for (image, mask) in [
            (uniform, vec![1.0; 128 * 128]),
            (line, vec![1.0; 128 * 128]),
            (pore, vec![1.0; 128 * 128]),
            (blemish_fixture(false), vec![0.0; 128 * 128]),
            (
                blemish_fixture(false),
                (0..128 * 128)
                    .map(|i| {
                        if (58..=70).contains(&(i % 128)) && (58..=70).contains(&(i / 128)) {
                            1.0
                        } else {
                            0.0
                        }
                    })
                    .collect(),
            ),
        ] {
            let mut repair = vec![[0.0; 4]; 128 * 128];
            let mut texture = repair.clone();
            process_blemish_region(
                &mut repair,
                &mut texture,
                &image,
                (0, 0, 128, 128),
                mask,
                128.0,
            );
            assert!(repair.iter().chain(&texture).all(|p| *p == [0.0; 4]));
        }
    }

    #[test]
    fn blemish_fields_are_face_only_and_duplicate_faces_do_not_stack() {
        let mut image = Rgb32FImage::from_pixel(256, 128, Rgb([0.32, 0.22, 0.16]));
        for cx in [48, 200] {
            for y in 74..77 {
                for x in cx..cx + 3 {
                    image.put_pixel(x, y, Rgb([0.18, 0.13, 0.10]));
                }
            }
        }
        let luminance = skin_luminance(&image);
        let (no_face, _) = build_fields(&image, &luminance, synthetic_skin(vec![]));
        assert!(no_face.blemish_pixels.iter().all(|p| *p == [0.0; 4]));
        let (one, _) = build_fields(&image, &luminance, synthetic_skin(vec![synthetic_face()]));
        let (two, _) = build_fields(
            &image,
            &luminance,
            synthetic_skin(vec![synthetic_face(), synthetic_face()]),
        );
        assert_eq!(one.blemish_pixels, two.blemish_pixels);
        assert_eq!(one.texture_pixels, two.texture_pixels);
        assert!(one.blemish_pixels.iter().any(|p| p[3] > 0.0));
        assert_eq!(one.blemish_pixels[75 * 256 + 201], [0.0; 4]);
        for uv in [
            [0.33, 0.4],
            [0.67, 0.4],
            [0.33, 0.31],
            [0.67, 0.31],
            [0.5, 0.68],
            [0.47, 0.56], // 鼻孔 landmark 98
            [0.53, 0.56], // 鼻孔 landmark 327
        ] {
            assert_eq!(blemish_face_mask(&synthetic_face(), uv), 0.0);
        }
    }

    #[test]
    fn maximum_analysis_fits_the_cache_budget() {
        let pixel_bytes = std::mem::size_of::<[f32; 4]>();
        // 10 层场：正方形最坏情况须能缓存；常见 2:3 人像须容纳两张以支持来回切图。
        let square = ANALYSIS_EDGE as usize * ANALYSIS_EDGE as usize * pixel_bytes * 10;
        assert!(square <= LEGACY_ANALYSIS_CACHE_BUDGET);
        let portrait = ANALYSIS_EDGE as usize * (ANALYSIS_EDGE as usize * 2 / 3) * pixel_bytes * 10;
        assert!(portrait * 2 <= LEGACY_ANALYSIS_CACHE_BUDGET);
        let layer = ANALYSIS_EDGE as usize * (ANALYSIS_EDGE as usize * 2 / 3) * pixel_bytes * 2;
        assert!(layer * 2 <= LEGACY_LAYER_BUDGET);
    }

    #[test]
    fn native_analysis_fits_cache_and_gpu_atlas_budget() {
        let edge = NATIVE_ANALYSIS_EDGE as usize;
        let pixels = edge * edge;
        // 10 层 f32x4 分析场 + 线性中间图（最坏情况为正方形）。
        let cpu = pixels * std::mem::size_of::<[f32; 4]>() * 10 + pixels * 12;
        assert!(cpu <= CACHE_BUDGET);
        // atlas 以 8 字节/像素上传，峰值约为纹理 + 暂存两份。
        let gpu = (pixels * 8 * 10 * 2) as u64;
        assert!(gpu < crate::portrait_tiles::GPU_BUDGET);
    }

    #[test]
    fn balanced_exposure_is_bounded_and_skin_limited() {
        let mut values = vec![0.7, -0.8, 2.0, 0.1];
        let mask = vec![1.0, 0.8, 0.0, 1.0];
        balance(&mut values, &mask, 0.4);
        assert!(values.iter().all(|v| v.is_finite() && v.abs() <= 0.4));
        assert_eq!(values[2], 0.0);
        assert!(values.iter().sum::<f32>().abs() < 1e-5);
    }

    #[test]
    fn empty_skin_preserves_zero_fields() {
        let image = Rgb32FImage::from_pixel(32, 24, Rgb([0.5; 3]));
        let (result, _) = build_fields(
            &image,
            &vec![0.2; 32 * 24],
            Segmentation {
                body: vec![0.0; 256 * 256],
                faces: vec![],
                skipped_faces: 0,
                rejected_faces: vec![],
            },
        );
        assert!(!result.summary.has_skin);
        assert!(result
            .pixels
            .iter()
            .all(|p| p[0] == 0.0 && p[1] == 0.0 && p[2] == 0.0));
        assert!(result.frequency_pixels.iter().all(|p| *p == [0.0; 4]));
    }

    fn synthetic_face() -> FaceAnalysis {
        synthetic_face_pose(false)
    }

    fn synthetic_face_pose(contour_reliable: bool) -> FaceAnalysis {
        let mut points = vec![[0.5, 0.5, 0.0]; 478];
        for (indices, center, radius) in [
            (OVAL, [0.5, 0.5], [0.4, 0.4]),
            (LEFT_EYE, [0.33, 0.4], [0.07, 0.035]),
            (RIGHT_EYE, [0.67, 0.4], [0.07, 0.035]),
            (LEFT_BROW, [0.33, 0.31], [0.08, 0.025]),
            (RIGHT_BROW, [0.67, 0.31], [0.08, 0.025]),
            (LIPS, [0.5, 0.68], [0.1, 0.04]),
        ] {
            for (i, &index) in indices.iter().enumerate() {
                let angle = i as f32 * std::f32::consts::TAU / indices.len() as f32;
                points[index] = [
                    center[0] + radius[0] * angle.cos(),
                    center[1] + radius[1] * angle.sin(),
                    0.0,
                ];
            }
        }
        for (index, point) in [
            (151, [0.50, 0.28]),
            (168, [0.50, 0.40]),
            (1, [0.50, 0.52]),
            (50, [0.34, 0.52]),
            (101, [0.38, 0.50]),
            (280, [0.66, 0.52]),
            (330, [0.62, 0.50]),
            (98, [0.47, 0.56]),
            (327, [0.53, 0.56]),
            (LEFT_IRIS_CENTER, [0.33, 0.4]),
            (RIGHT_IRIS_CENTER, [0.67, 0.4]),
        ] {
            points[index] = [point[0], point[1], 0.0];
        }
        // 合成虹膜轮廓，供巩膜 Mask v2 多边形挖空。
        for (ring, center) in [
            (LEFT_IRIS_RING, [0.33f32, 0.4]),
            (RIGHT_IRIS_RING, [0.67f32, 0.4]),
        ] {
            let radius = 0.028f32;
            for (i, &index) in ring.iter().enumerate() {
                let angle = i as f32 * std::f32::consts::TAU / ring.len() as f32;
                points[index] = [
                    center[0] + radius * angle.cos(),
                    center[1] + radius * angle.sin(),
                    0.0,
                ];
            }
        }
        FaceAnalysis {
            roi: crate::portrait_models::FaceRoi {
                center: [64.0, 64.0],
                side: 96.0,
                angle: 0.0,
            },
            bounds: [16.0, 16.0, 112.0, 112.0],
            points,
            skin: vec![1.0; 512 * 512],
            contour_reliable,
        }
    }

    fn synthetic_skin(faces: Vec<FaceAnalysis>) -> Segmentation {
        Segmentation {
            body: (0..256 * 256)
                .map(|i| {
                    if (i % 256 < 112 || i % 256 > 168) && (24..232).contains(&(i / 256)) {
                        1.0
                    } else {
                        0.0
                    }
                })
                .collect(),
            faces,
            skipped_faces: 0,
            rejected_faces: vec![],
        }
    }

    fn textured_skin() -> Rgb32FImage {
        Rgb32FImage::from_fn(256, 128, |x, _| {
            let wave = (x as f32 * std::f32::consts::TAU / 32.0).sin() * 0.025;
            let texture = if x % 2 == 0 { 0.003 } else { -0.003 };
            Rgb([
                0.3 + wave + texture,
                0.22 - wave * 0.3 + texture,
                0.16 + texture,
            ])
        })
    }

    fn skin_luminance(image: &Rgb32FImage) -> Vec<f32> {
        image
            .pixels()
            .map(|p| p[0] * 0.2126 + p[1] * 0.7152 + p[2] * 0.0722)
            .collect()
    }

    #[test]
    fn frequency_requires_face_and_keeps_regions_and_features_separate() {
        let image = textured_skin();
        let luminance = skin_luminance(&image);
        let (no_face, _) = build_fields(&image, &luminance, synthetic_skin(vec![]));
        assert!(no_face.summary.has_skin);
        assert!(no_face.frequency_pixels.iter().all(|p| *p == [0.0; 4]));
        let (one, _) = build_fields(&image, &luminance, synthetic_skin(vec![synthetic_face()]));
        let (duplicate, _) = build_fields(
            &image,
            &luminance,
            synthetic_skin(vec![synthetic_face(), synthetic_face()]),
        );
        assert_eq!(
            one.frequency_pixels, duplicate.frequency_pixels,
            "重叠面部不叠加修正"
        );
        assert!(one.frequency_pixels[64 * 256 + 64][3] > 0.0);
        assert!(one.frequency_pixels[64 * 256 + 200][3] < 0.0);
        assert!(
            one.frequency_pixels
                .iter()
                .any(|p| p[3] > 0.0 && p[0].abs() > 1e-5)
        );
        assert!(
            one.frequency_pixels
                .iter()
                .any(|p| p[3] < 0.0 && p[0].abs() > 1e-5)
        );
        for uv in [
            [0.33, 0.4],
            [0.67, 0.4],
            [0.33, 0.31],
            [0.67, 0.31],
            [0.5, 0.68],
        ] {
            let xy = synthetic_face().roi.project(uv[0], uv[1]);
            assert_eq!(
                one.frequency_pixels[xy[1] as usize * 256 + xy[0] as usize],
                [0.0; 4],
                "五官保护"
            );
        }
        for y in 0..128 {
            assert_eq!(one.frequency_pixels[y * 256 + 140], [0.0; 4], "背景保护");
        }
        assert!(one.frequency_pixels.iter().flatten().all(|v| v.is_finite()));
    }

    #[test]
    fn frequency_reduces_low_frequency_color_variation_and_retains_texture() {
        let image = textured_skin();
        let (w, h) = (256, 128);
        for is_face in [true, false] {
            let mut fields = vec![[0.0; 4]; w * h];
            let mut detail = vec![[0.0; 4]; w * h];
            process_frequency_region(
                &mut fields,
                &mut detail,
                &image,
                (0, 0, w, h),
                &vec![1.0; w * h],
                128.0,
                is_face,
            );
            let amplitude = |channel: usize, corrected: bool, high: bool| -> f32 {
                (32..224)
                    .map(|x| {
                        let value = image.get_pixel(x as u32, 64)[channel]
                            + if corrected {
                                fields[64 * w + x][channel]
                            } else {
                                0.0
                            };
                        let wave = if high {
                            if x % 2 == 0 { 1.0 } else { -1.0 }
                        } else {
                            (x as f32 * std::f32::consts::TAU / 32.0).sin()
                        };
                        value * wave
                    })
                    .sum::<f32>()
                    .abs()
            };
            for channel in [0, 1] {
                let low_ratio = amplitude(channel, true, false) / amplitude(channel, false, false);
                let high_ratio = amplitude(channel, true, true) / amplitude(channel, false, true);
                assert!(low_ratio < 0.95, "低频衰减不足：{low_ratio}");
                assert!(
                    (high_ratio - 1.0).abs() < 0.1,
                    "高频纹理发生变化：{high_ratio}"
                );
            }
            let luma_energy: f32 = fields
                .iter()
                .map(|p| luma_rgb([p[0], p[1], p[2]]).abs())
                .sum();
            let rgb_energy: f32 = fields
                .iter()
                .map(|p| p[0].abs() + p[1].abs() + p[2].abs())
                .sum();
            assert!(
                luma_energy < 0.85 * rgb_energy,
                "亮度修正不应独占能量：luma={luma_energy} rgb={rgb_energy}"
            );
            assert!(fields.iter().all(|p| (p[3] > 0.0) == is_face));
            assert!(
                detail
                    .iter()
                    .any(|p| p[0].abs() + p[1].abs() + p[2].abs() > 1e-5)
            );
            assert!(detail.iter().all(|p| (p[3] > 0.0) == is_face));
            // 子带合成：high_amt=1 中性；UI +100→0 满柔化；UI -100→2 为 +detail 锐化。
            let sample = detail
                .iter()
                .enumerate()
                .find(|(_, p)| p[0].abs() > 1e-4)
                .map(|(i, _)| i)
                .expect("应存在非零高频残差");
            let delta = |low_amt: f32, high_amt: f32| {
                low_amt * fields[sample][0] + (high_amt - 1.0) * detail[sample][0]
            };
            assert!((delta(1.0, 1.0) - fields[sample][0]).abs() < 1e-6);
            assert!((delta(1.0, 0.0) - (fields[sample][0] - detail[sample][0])).abs() < 1e-6);
            assert!((delta(1.0, 2.0) - (fields[sample][0] + detail[sample][0])).abs() < 1e-6);
            assert!(
                (delta(1.0, 0.0) - delta(1.0, 1.0)).abs() > 1e-5,
                "UI +100 柔化应改变合成增量"
            );
            assert!(
                (delta(1.0, 2.0) - delta(1.0, 1.0)).abs() > 1e-5,
                "UI -100 增强应改变合成增量"
            );
            let low_amp = |low_amt: f32| -> f32 {
                (32..224)
                    .map(|x| {
                        let i = 64 * w + x;
                        let value = image.get_pixel(x as u32, 64)[0]
                            + low_amt * fields[i][0]
                            + 0.0 * detail[i][0];
                        value * (x as f32 * std::f32::consts::TAU / 32.0).sin()
                    })
                    .sum::<f32>()
                    .abs()
            };
            assert!(
                low_amp(0.25) > low_amp(1.0),
                "降低低频应减弱斑驳平滑：{} vs {}",
                low_amp(0.25),
                low_amp(1.0)
            );
        }
    }

    #[test]
    fn frequency_skips_tiny_regions_and_preserves_mask_zeros_and_uniform_skin() {
        let image = Rgb32FImage::from_pixel(32, 32, Rgb([0.3, 0.2, 0.1]));
        let mut fields = vec![[0.0; 4]; 32 * 32];
        let mut detail = vec![[0.0; 4]; 32 * 32];
        let mut mask = vec![0.0; 32 * 32];
        mask[0] = 1.0;
        process_frequency_region(
            &mut fields,
            &mut detail,
            &image,
            (0, 0, 32, 32),
            &mask,
            32.0,
            true,
        );
        assert!(fields.iter().all(|p| *p == [0.0; 4]));
        assert!(detail.iter().all(|p| *p == [0.0; 4]));
        mask[..512].fill(1.0);
        process_frequency_region(
            &mut fields,
            &mut detail,
            &image,
            (0, 0, 32, 32),
            &mask,
            32.0,
            true,
        );
        assert!(
            fields[..512]
                .iter()
                .all(|p| p[..3].iter().all(|v| v.abs() < 1e-6))
        );
        assert!(fields[512..].iter().all(|p| *p == [0.0; 4]));
        assert!(detail[512..].iter().all(|p| *p == [0.0; 4]));
    }

    #[test]
    fn frequency_face_pass_overwrites_weaker_body_overlap() {
        let image = textured_skin();
        let w = image.width() as usize;
        let h = image.height() as usize;
        let mut fields = vec![[0.0; 4]; w * h];
        let mut detail = vec![[0.0; 4]; w * h];
        let body_mask = vec![0.9; w * h];
        process_frequency_region(
            &mut fields,
            &mut detail,
            &image,
            (0, 0, w, h),
            &body_mask,
            64.0,
            false,
        );
        assert!(fields.iter().any(|p| p[3] < -0.05), "身 pass 应写入负 A");
        // 脸蒙版弱于已写入 |body|，旧逻辑会输掉；现要求脸有效即覆盖。
        let face_mask = vec![0.2; w * h];
        process_frequency_region(
            &mut fields,
            &mut detail,
            &image,
            (0, 0, w, h),
            &face_mask,
            64.0,
            true,
        );
        assert!(
            fields.iter().all(|p| p[3] > 0.05),
            "脸 pass 应覆盖重叠身场，A 全为正"
        );
        assert!(detail.iter().all(|p| p[3] > 0.05));
    }

    #[test]
    fn frequency_sample_ignores_cross_region_neighbors_in_weight_sum() {
        let (w, h) = (4u32, 4u32);
        let mut field = vec![[0.0; 4]; (w * h) as usize];
        let mut detail = vec![[0.0; 4]; (w * h) as usize];
        let reference = vec![[0.0, 0.0, 0.0, 0.25]; (w * h) as usize];
        // 左上脸、右上身：nearest 落在脸格时不应被身邻点稀释到 0。
        field[0] = [0.04, 0.0, 0.0, 0.8];
        detail[0] = [0.02, 0.0, 0.0, 0.8];
        field[1] = [0.20, 0.0, 0.0, -0.9];
        detail[1] = [0.10, 0.0, 0.0, -0.9];
        let face_only = sample_region_weighted_rgb(
            &field,
            Some(&detail),
            &reference,
            w,
            h,
            [0.5 / w as f32, 0.5 / h as f32],
            0.25,
            |_| (1.0, 1.0, 0.0),
            true,
        );
        assert!(
            face_only[0].abs() > 1e-4,
            "频分同区采样不应被异区邻点把结果冲掉：{face_only:?}"
        );
    }

    #[test]
    fn frequency_body_scale_follows_face_and_caps_smoothing() {
        let face = synthetic_face();
        assert_eq!(
            face_frequency_scale(&[synthetic_face()]),
            Some(face.roi.side / 1.5)
        );
        assert_eq!(frequency_region_scale(1800.0, Some(64.0)), 64.0);
        assert_eq!(frequency_region_scale(32.0, Some(64.0)), 32.0);
        let cap = (ANALYSIS_EDGE as f32 * 0.04).round() as usize;
        let (sep, smooth) = frequency_radii(10_000.0, false);
        assert_eq!(smooth, cap);
        assert!(sep <= smooth);
        let (_, face_smooth) = frequency_radii(64.0, true);
        let body_scale = frequency_region_scale(1800.0, Some(64.0));
        let (_, body_smooth) = frequency_radii(body_scale, false);
        assert!(body_smooth <= cap);
        assert!(body_smooth >= face_smooth);
    }

    #[test]
    fn frequency_feathers_oval_and_keeps_feature_centers_clear() {
        let image = textured_skin();
        let luminance = skin_luminance(&image);
        let (one, _) = build_fields(&image, &luminance, synthetic_skin(vec![synthetic_face()]));
        let mut previous = one.frequency_pixels[64 * 256 + 64][3];
        let mut crossed = false;
        for x in 64..130 {
            let alpha = one.frequency_pixels[64 * 256 + x][3];
            if previous > 0.0 && alpha < 0.0 {
                assert!(
                    previous < 0.85 && alpha.abs() < 0.85,
                    "OVAL 边界 alpha 硬切：{previous} -> {alpha}"
                );
                crossed = true;
            }
            previous = alpha;
        }
        assert!(crossed, "应经过脸到身的符号切换");
        for uv in [
            [0.33, 0.4],
            [0.67, 0.4],
            [0.33, 0.31],
            [0.67, 0.31],
            [0.5, 0.68],
        ] {
            assert_eq!(frequency_face_mask(&synthetic_face(), uv), 0.0);
        }
        assert!(body_skin_likelihood([0.08, 0.14, 0.42]) < 0.05);
        assert!(body_skin_likelihood([0.30, 0.22, 0.16]) > 0.5);
    }

    #[test]
    fn eyeball_exclude_clears_retouch_fields_at_iris_and_margin() {
        let face = synthetic_face();
        // 虹膜中心与近巩膜采样点：蒙版必须为 0。
        for uv in [[0.33, 0.4], [0.67, 0.4], [0.30, 0.4], [0.36, 0.4]] {
            assert!(
                eye_exclude_weight(&face, uv) > 0.5,
                "眼球内应强排除：{uv:?}"
            );
            assert_eq!(face_mask(&face, uv), 0.0);
            assert_eq!(frequency_face_mask(&face, uv), 0.0);
        }
        // 眼缘外扩：略出多边形但仍在 margin 内。
        let margin_uv = [0.33 + 0.07 + 0.008, 0.4];
        assert!(
            eye_exclude_weight(&face, margin_uv) > 0.0,
            "眼缘 margin 应排除：{}",
            eye_exclude_weight(&face, margin_uv)
        );
        let image = textured_skin();
        let luminance = skin_luminance(&image);
        let (analysis, _) = build_fields(&image, &luminance, synthetic_skin(vec![synthetic_face()]));
        for center in [[0.33, 0.4], [0.67, 0.4]] {
            let xy = face.roi.project(center[0], center[1]);
            let x = xy[0].round().clamp(0.0, 255.0) as usize;
            let y = xy[1].round().clamp(0.0, 127.0) as usize;
            let i = y * 256 + x;
            assert_eq!(
                analysis.pixels[i][0],
                0.0,
                "眼心 EV even 应为 0：{center:?}"
            );
            assert_eq!(
                analysis.pixels[i][1],
                0.0,
                "眼心 EV features 应为 0：{center:?}"
            );
            assert_eq!(
                analysis.pixels[i][2],
                0.0,
                "眼心 EV outline 应为 0：{center:?}"
            );
            assert_eq!(
                analysis.evenness_chroma_pixels[i],
                [0.0; 4],
                "眼心 chroma 应为 0：{center:?}"
            );
            assert_eq!(
                analysis.frequency_pixels[i][3].abs(),
                0.0,
                "眼心 frequency 覆盖应为 0：{center:?}"
            );
            assert_eq!(
                analysis.hydrated_pixels[i],
                [0.0; 4],
                "眼心 hydrated 应为 0：{center:?}"
            );
        }
    }

    #[test]
    fn frequency_body_skips_cool_clothing_colors() {
        let mut image = textured_skin();
        for y in 0..128u32 {
            for x in 180..256u32 {
                image.put_pixel(x, y, Rgb([0.08, 0.14, 0.42]));
            }
        }
        let luminance = skin_luminance(&image);
        let (one, _) = build_fields(&image, &luminance, synthetic_skin(vec![synthetic_face()]));
        assert!(one.frequency_pixels[64 * 256 + 64][3] > 0.0);
        assert_eq!(one.frequency_pixels[64 * 256 + 200], [0.0; 4]);
    }

    #[test]
    fn analysis_retains_hdr_and_sanitizes_non_finite_inputs() {
        let image = Rgb32FImage::from_pixel(8, 8, Rgb([2.0, f32::NAN, -1.0]));
        let (encoded, linear, luminance) = analysis_image(&DynamicImage::ImageRgb32F(image), true);
        assert_eq!(encoded.get_pixel(0, 0)[0], 1.0);
        assert_eq!(linear.get_pixel(0, 0).0, [2.0, 0.0, 0.0]);
        assert!(luminance.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn roi_round_trip() {
        let roi = crate::portrait_models::FaceRoi {
            center: [200.0, 140.0],
            side: 80.0,
            angle: 0.42,
        };
        let xy = roi.project(0.23, 0.64);
        let uv = roi.unproject(xy[0], xy[1]);
        assert!((uv[0] - 0.23).abs() < 1e-6 && (uv[1] - 0.64).abs() < 1e-6);
    }

    fn region_wave_field(
        period: f32,
        texture: bool,
    ) -> (usize, usize, Vec<f32>, Vec<f32>, Vec<[f32; 4]>) {
        let (w, h) = (256, 128);
        let log: Vec<f32> = (0..w * h)
            .map(|i| {
                let wave = ((i % w) as f32 * std::f32::consts::TAU / period).sin();
                -2.0 + 0.2 * wave
                    + if texture && i % 2 == 0 {
                        0.025
                    } else if texture {
                        -0.025
                    } else {
                        0.0
                    }
            })
            .collect();
        let luminance: Vec<f32> = log.iter().map(|l| l.exp2()).collect();
        let pixels: Vec<_> = luminance.iter().map(|y| [0.0, 0.0, 0.0, *y]).collect();
        (w, h, log, luminance, pixels)
    }

    fn wave_amplitude(values: &[f32], w: usize, period: f32, offset: impl Fn(usize) -> f32) -> f32 {
        let wave = |x: usize| (x as f32 * std::f32::consts::TAU / period).sin();
        let mut sum = 0.0;
        for x in 32..224 {
            let i = 64 * w + x;
            sum += (values[i] + offset(i)) * wave(x);
        }
        sum.abs()
    }

    fn gray_linear(w: usize, h: usize, luminance: &[f32]) -> Rgb32FImage {
        Rgb32FImage::from_fn(w as u32, h as u32, |x, y| {
            let yv = luminance[y as usize * w + x as usize];
            Rgb([yv, yv, yv])
        })
    }

    #[test]
    fn evenness_reduces_mid_frequency_and_retains_texture() {
        // 周期约落在 1.8%–5.5% 脸宽的平整带内（scale=128 时半径约 2–7）。
        let (w, h, log, luminance, mut pixels) = region_wave_field(10.0, true);
        let linear = gray_linear(w, h, &luminance);
        let mut chroma = vec![[0.0; 4]; w * h];
        process_region(
            &mut pixels,
            &mut chroma,
            &linear,
            &luminance,
            w,
            (0, 0, w, h),
            vec![1.0; w * h],
            None,
        );
        let mid = |corrected: bool| {
            wave_amplitude(&log, w, 10.0, |i| if corrected { pixels[i][0] } else { 0.0 })
        };
        let high = |corrected: bool| -> f32 {
            let mut sum = 0.0;
            for x in 32..224 {
                let i = 64 * w + x;
                let value = log[i] + if corrected { pixels[i][0] } else { 0.0 };
                sum += value * if x % 2 == 0 { 1.0 } else { -1.0 };
            }
            sum.abs()
        };
        assert!(mid(true) < mid(false) * 0.9);
        assert!((high(true) / high(false) - 1.0).abs() < 0.1);
        assert!(
            pixels.iter().all(|p| {
                p[0].abs() <= 0.45 && p[1].abs() <= 0.35 && p[2].abs() <= 0.35
            })
        );
        let mean: f32 = pixels
            .iter()
            .map(|p| p[0] + p[1] + p[2])
            .sum::<f32>()
            / pixels.len() as f32;
        assert!(mean.abs() < 0.02);
    }

    #[test]
    fn evenness_preserves_illumination_volume_in_contour() {
        let (w, h, log, luminance, mut pixels) = region_wave_field(64.0, false);
        let linear = gray_linear(w, h, &luminance);
        let mut chroma = vec![[0.0; 4]; w * h];
        process_region(
            &mut pixels,
            &mut chroma,
            &linear,
            &luminance,
            w,
            (0, 0, w, h),
            vec![1.0; w * h],
            None,
        );
        let original = wave_amplitude(&log, w, 64.0, |_| 0.0);
        let even_field = wave_amplitude(&log, w, 64.0, |i| pixels[i][0]) - original;
        // 大尺度光影残差写入轮廓通道（outline = pixels[].b）。
        let contour_field = wave_amplitude(
            &pixels.iter().map(|p| p[2]).collect::<Vec<_>>(),
            w,
            64.0,
            |_| 0.0,
        );
        assert!(
            even_field.abs() < original * 0.20,
            "平整不应抹掉大尺度光影：{even_field} vs {original}"
        );
        assert!(
            contour_field > original * 0.05,
            "立体应承接大尺度光影：{contour_field} vs {original}"
        );
    }

    #[test]
    fn evenness_reduces_chroma_blotch_while_preserving_luma() {
        let (w, h) = (256usize, 128usize);
        let y = 0.25f32;
        let period = 10.0f32;
        let luminance = vec![y; w * h];
        let linear = Rgb32FImage::from_fn(w as u32, h as u32, |x, _y| {
            let wave = (x as f32 * std::f32::consts::TAU / period).sin();
            let cr = 0.06 * wave;
            let r = (y + cr).max(0.0);
            let b = y;
            let g = ((y - 0.2126 * r - 0.0722 * b) / 0.7152).max(0.0);
            Rgb([r, g, b])
        });
        let mut pixels: Vec<_> = luminance.iter().map(|&yv| [0.0, 0.0, 0.0, yv]).collect();
        let mut chroma = vec![[0.0; 4]; w * h];
        process_region(
            &mut pixels,
            &mut chroma,
            &linear,
            &luminance,
            w,
            (0, 0, w, h),
            vec![1.0; w * h],
            None,
        );
        let mut before = 0.0f32;
        let mut after = 0.0f32;
        let mut luma_shift = 0.0f32;
        for x in 32..224 {
            let i = 64 * w + x;
            let wave = (x as f32 * std::f32::consts::TAU / period).sin();
            let rgb = linear.get_pixel(x as u32, 64).0;
            before += (rgb[0] - y) * wave;
            after += (rgb[0] - y + chroma[i][0]) * wave;
            luma_shift += 0.2126 * chroma[i][0] + 0.7152 * chroma[i][1] + 0.0722 * chroma[i][2];
        }
        assert!(
            after.abs() < before.abs() * 0.92,
            "色度中频应减弱：{after} vs {before}"
        );
        assert!(
            after.abs() <= before.abs(),
            "色度修正不得抬高局部饱和度：{after} vs {before}"
        );
        assert!(
            luma_shift.abs() < 0.02,
            "色度修正应近似保亮度：{luma_shift}"
        );
    }

    #[test]
    fn face_contour_template_avoids_features_and_profile_keeps_residual() {
        let (w, h) = (128, 128);
        let frontal = synthetic_face_pose(true);
        let profile = synthetic_face_pose(false);
        let mask: Vec<f32> = (0..w * h)
            .map(|i| {
                face_mask(
                    &frontal,
                    frontal
                        .roi
                        .unproject((i % w) as f32 + 0.5, (i / w) as f32 + 0.5),
                )
            })
            .collect();
        let log: Vec<f32> = (0..w * h)
            .map(|i| {
                let x = (i % w) as f32;
                -2.0 + 0.35 * (x * std::f32::consts::TAU / 48.0).sin()
            })
            .collect();
        let luminance: Vec<f32> = log.iter().map(|l| l.exp2()).collect();
        let run = |face: &FaceAnalysis| {
            let mut pixels: Vec<_> = luminance.iter().map(|y| [0.0, 0.0, 0.0, *y]).collect();
            let linear = gray_linear(w, h, &luminance);
            let mut chroma = vec![[0.0; 4]; w * h];
            process_region(
                &mut pixels,
                &mut chroma,
                &linear,
                &luminance,
                w,
                (0, 0, w, h),
                mask.clone(),
                Some(face),
            );
            (pixels, chroma)
        };
        let (front, front_chroma) = run(&frontal);
        let (side, _) = run(&profile);
        assert!(front.iter().any(|p| p[1] > 0.01), "五官通道应有正向 EV");
        assert!(front.iter().any(|p| p[2] < -0.01), "轮廓通道应有负向 EV");
        for uv in [
            [0.33, 0.41],
            [0.67, 0.41],
            [0.33, 0.31],
            [0.67, 0.31],
            [0.5, 0.68],
        ] {
            let xy = frontal.roi.project(uv[0], uv[1]);
            let mut hole = false;
            for dy in -2..=2 {
                for dx in -2..=2 {
                    let x = (xy[0] as i32 + dx).clamp(0, w as i32 - 1) as usize;
                    let y = (xy[1] as i32 + dy).clamp(0, h as i32 - 1) as usize;
                    let p = front[y * w + x];
                    if p[0] == 0.0 && p[1] == 0.0 && p[2] == 0.0 {
                        hole = true;
                    }
                }
            }
            assert!(hole, "五官附近应保持零保护 {uv:?}");
        }
        assert!(
            side.iter().any(|p| p[2].abs() > 1e-4),
            "侧脸应退回光影残差而不是全零"
        );
        // balance 已把蒙版乘进场值并清零加权和；直接对场求和即可。
        let _ = front_chroma;
        let front_mean: f64 = front.iter().map(|p| (p[0] + p[1] + p[2]) as f64).sum();
        assert!(
            front_mean.abs() < 1e-3,
            "立体场加权均值应接近 0：{front_mean}"
        );
        // 鼻梁中线五官通道峰值应为正；颞侧轮廓通道应更负。
        let sample = |uv: [f32; 2]| {
            let xy = frontal.roi.project(uv[0], uv[1]);
            let x = xy[0].round().clamp(0.0, (w - 1) as f32) as usize;
            let y = xy[1].round().clamp(0.0, (h - 1) as f32) as usize;
            front[y * w + x]
        };
        let nose_peak = [0.46f32, 0.48, 0.50, 0.52, 0.54]
            .into_iter()
            .map(|y| sample([0.5, y])[1])
            .fold(f32::NEG_INFINITY, f32::max);
        let temple = sample([0.18, 0.42]);
        assert!(
            nose_peak > 0.01,
            "鼻梁五官通道峰值应为正：features={nose_peak}"
        );
        assert!(
            temple[2] < -0.005,
            "颞侧应以轮廓 Burn 为主：outline={}",
            temple[2]
        );
    }

    #[test]
    fn crop_mapping_matches_pixel_rounding() {
        let image = Rgb32FImage::from_pixel(32, 24, Rgb([0.5; 3]));
        let (analysis_raw, _) = build_fields(
            &image,
            &vec![0.2; 32 * 24],
            Segmentation {
                body: vec![0.0; 256 * 256],
                faces: vec![],
                skipped_faces: 0,
                rejected_faces: vec![],
            },
        );
        let analysis = Arc::new(analysis_raw);
        assert_eq!(
            analysis
                .for_render(&json!({"crop":{"x":8.2,"y":6.1,"width":16,"height":12}}))
                .uv_rect,
            [0.25, 0.25, 0.5, 0.5]
        );
        assert_eq!(
            analysis
                .for_render(&json!({"crop":{"x":40,"y":0,"width":2,"height":2}}))
                .uv_rect,
            [0.0, 0.0, 1.0, 1.0]
        );
    }

    #[test]
    fn cache_lru_and_key_exclude_strength_but_include_patch_contents() {
        let sample = |key: &str| {
            Arc::new(PortraitAnalysis {
                summary: PortraitSummary {
                    key: key.into(),
                    face_count: 0,
                    skipped_faces: 0,
                    has_skin: false,
                },
                width: 1,
                height: 1,
                full_dimensions: (1, 1),
                pixels: Arc::from(vec![[0.0; 4]]),
                frequency_pixels: Arc::from(vec![[0.0; 4]]),
                frequency_detail_pixels: Arc::from(vec![[0.0; 4]]),
                hydrated_pixels: Arc::from(vec![[0.0; 4]]),
                even_pixels: Arc::from(vec![[0.0; 4]]),
                whitening_pixels: Arc::from(vec![[0.0; 4]]),
                evenness_chroma_pixels: Arc::from(vec![[0.0; 4]]),
                blemish_pixels: Arc::from(vec![[0.0; 4]]),
                texture_pixels: Arc::from(vec![[0.0; 4]]),
                shine_pixels: Arc::from(vec![[0.0; 4]]),
                eye_white_pixels: Arc::from(vec![[0.0; 4]]),
            })
        };
        let mut cache = PortraitCache::default();
        cache.insert("a".into(), sample("a"));
        cache.insert("b".into(), sample("b"));
        assert!(cache.get("a").is_some());
        assert_eq!(cache.entries.front().unwrap().0, "b");
        assert_eq!(cache.bytes, 2 * 11 * std::mem::size_of::<[f32; 4]>());
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("image.jpg");
        std::fs::write(&path, b"source").unwrap();
        let path = path.to_str().unwrap();
        let settings = crate::AppSettings::default();
        let a = json!({"aiPatches":[{"patchData":"aaaa"}],"portrait":{"neutralGrayEvenness":20}});
        let mut b = a.clone();
        b["portrait"]["neutralGrayEvenness"] = json!(100);
        b["portrait"]["frequencySeparationFace"] = json!(60);
        b["portrait"]["frequencySeparationBody"] = json!(80);
        b["portrait"]["blemishRemoval"] = json!(90);
        b["portrait"]["textureRetention"] = json!(70);
        b["portrait"]["faceShineRemoval"] = json!(85);
        b["sectionVisibility"] = json!({"portraitSkin": false, "portraitBlemishes": false});
        assert_eq!(
            analysis_key(path, &base_adjustments(&a), &settings).unwrap(),
            analysis_key(path, &base_adjustments(&b), &settings).unwrap()
        );
        b["aiPatches"][0]["patchData"] = json!("bbbb");
        assert_ne!(
            analysis_key(path, &base_adjustments(&a), &settings).unwrap(),
            analysis_key(path, &base_adjustments(&b), &settings).unwrap()
        );
        assert_ne!(
            analysis_key(path, &a, &settings).unwrap(),
            analysis_key(&format!("{path}?vc=test"), &a, &settings).unwrap()
        );
    }

    #[test]
    fn raw_and_srgb_analysis_agree() {
        let srgb = Rgb32FImage::from_pixel(16, 16, Rgb([0.5; 3]));
        let raw = Rgb32FImage::from_pixel(16, 16, Rgb([srgb_to_linear(0.5); 3]));
        let (_, linear_a, a) = analysis_image(&DynamicImage::ImageRgb32F(srgb), false);
        let (_, linear_b, b) = analysis_image(&DynamicImage::ImageRgb32F(raw), true);
        assert!(
            linear_a
                .as_raw()
                .iter()
                .zip(linear_b.as_raw())
                .all(|(a, b)| (a - b).abs() < 1e-6)
        );
        assert!(a.iter().zip(b).all(|(a, b)| (a - b).abs() < 1e-6));
    }

}
