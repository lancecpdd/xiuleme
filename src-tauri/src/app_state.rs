use std::collections::{HashMap, HashSet, VecDeque};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicUsize};
use std::sync::mpsc::Sender;
use std::sync::{Arc, Condvar, Mutex, MutexGuard};

use image::{DynamicImage, GrayImage};
use serde::{Deserialize, Serialize};
use sysinfo::Disks;
use tokio::sync::Mutex as TokioMutex;
use tokio::task::JoinHandle;
use wgpu::{Texture, TextureView};

use crate::ai_processing::AiState;
use crate::cache_utils::DecodedImageCache;
use crate::camera_tethering::CameraSession;
use crate::gpu_processing::GpuProcessor;
use crate::image_processing::GpuContext;
use crate::launch_request::ExternalEditSession;
use crate::lens_correction::LensDatabase;
use crate::lut_processing::Lut;
use crate::white_balance::WhiteBalance;

#[derive(Serialize, Deserialize)]
pub struct WindowState {
    pub width: u32,
    pub height: u32,
    pub x: i32,
    pub y: i32,
    pub maximized: bool,
    pub fullscreen: bool,
}

#[derive(Clone)]
pub struct LoadedImage {
    pub path: String,
    pub image: Arc<DynamicImage>,
    pub is_raw: bool,
    pub as_shot_white_balance: WhiteBalance,
}

#[derive(Clone)]
pub struct CachedPreview {
    pub image: Arc<DynamicImage>,
    pub small_image: Arc<DynamicImage>,
    pub transform_hash: u64,
    pub scale: f32,
    pub unscaled_crop_offset: (f32, f32),
    pub preview_dim: u32,
    pub interactive_divisor: f32,
}

pub struct GpuImageCache {
    pub texture: Texture,
    pub texture_view: TextureView,
    pub gf_coeffs_texture: Texture,
    pub gf_coeffs_view: TextureView,
    pub gf_dehaze_texture: Texture,
    pub gf_dehaze_view: TextureView,
    pub is_raw: u32,
    pub width: u32,
    pub height: u32,
    pub transform_hash: u64,
}

pub struct GpuProcessorState {
    pub processor: GpuProcessor,
    pub width: u32,
    pub height: u32,
}

pub struct PreviewJob {
    pub epoch: usize,
    pub request_id: u64,
    pub source_revision: u64,
    pub loaded_image: LoadedImage,
    pub adjustments: serde_json::Value,
    pub is_interactive: bool,
    pub target_resolution: Option<u32>,
    pub roi: Option<(f32, f32, f32, f32)>,
    pub request_analytics: bool,
    /// When true, also compute a pre-levels (identity levels) histogram for the Levels panel.
    pub request_levels_input_analytics: bool,
    pub compute_waveform: bool,
    pub active_waveform_channel: Option<String>,
    pub responder: tokio::sync::oneshot::Sender<Result<Vec<u8>, String>>,
}

pub struct AnalyticsJob {
    pub path: String,
    pub stamp: crate::portrait_render::RenderStamp,
    pub editor_epoch: u64,
    pub image: Arc<DynamicImage>,
    pub compute_waveform: bool,
    pub active_waveform_channel: Option<String>,
    /// Precomputed Capture One–style levels input histogram (identity levels). None = omit unless reuse flag.
    pub levels_input_histogram: Option<crate::image_processing::HistogramData>,
    /// When true and levels_input_histogram is None, copy final-image histogram into levelsInputHistogram.
    pub reuse_final_as_levels_input: bool,
}

pub struct AnalyticsConfig {
    pub path: String,
    pub stamp: crate::portrait_render::RenderStamp,
    pub editor_epoch: u64,
    pub compute_waveform: bool,
    pub active_waveform_channel: Option<String>,
    pub sender: Sender<AnalyticsJob>,
    pub levels_input_histogram: Option<crate::image_processing::HistogramData>,
    pub reuse_final_as_levels_input: bool,
}

pub struct ThumbnailProgressTracker {
    pub total: usize,
    pub completed: usize,
}

pub struct ThumbnailManager {
    pub queue: Mutex<VecDeque<String>>,
    pub cvar: Condvar,
    pub processing_now: Mutex<HashSet<String>>,
    pub rotational_disk: AtomicBool,
    pub io_gate: Mutex<()>,
}

impl ThumbnailManager {
    pub fn new() -> Arc<Self> {
        Arc::new(Self {
            queue: Mutex::new(VecDeque::new()),
            cvar: Condvar::new(),
            processing_now: Mutex::new(HashSet::new()),
            rotational_disk: AtomicBool::new(false),
            io_gate: Mutex::new(()),
        })
    }
}

pub struct PendingMetadata {
    pub virtual_path: String,
    pub image_path: PathBuf,
    pub sidecar_path: PathBuf,
}

pub struct MetadataManager {
    pub queue: Mutex<VecDeque<PendingMetadata>>,
    pub cvar: Condvar,
    pub pending: Mutex<HashSet<PathBuf>>,
}

impl MetadataManager {
    pub fn new() -> Arc<Self> {
        Arc::new(Self {
            queue: Mutex::new(VecDeque::new()),
            cvar: Condvar::new(),
            pending: Mutex::new(HashSet::new()),
        })
    }
}

pub type ThumbnailGeometryEntry = (u64, Arc<DynamicImage>, f32);
pub type TransformedImageCache = (u64, Arc<DynamicImage>, (f32, f32));

pub struct AppState {
    pub window_setup_complete: AtomicBool,
    pub gpu_crash_flag_path: Mutex<Option<PathBuf>>,
    pub original_image: Mutex<Option<LoadedImage>>,
    pub cached_preview: Mutex<Option<CachedPreview>>,
    pub gpu_context: Mutex<Option<GpuContext>>,
    pub gpu_image_cache: Mutex<Option<GpuImageCache>>,
    pub gpu_processor: Mutex<Option<GpuProcessorState>>,
    pub ai_state: Mutex<Option<AiState>>,
    pub ai_init_lock: TokioMutex<()>,
    pub portrait_state: crate::portrait_processing::PortraitState,
    pub reshape_analysis: crate::portrait_analysis_cache::ReshapeAnalysisState,
    pub portrait_render: crate::portrait_render::PortraitRenderState,
    pub export_task_token: Arc<Mutex<Option<Arc<AtomicBool>>>>,
    pub hdr_result: Arc<Mutex<Option<DynamicImage>>>,
    pub panorama_result: Arc<Mutex<Option<DynamicImage>>>,
    pub focus_stack_result: Arc<Mutex<Option<DynamicImage>>>,
    pub denoise_result: Arc<Mutex<Option<DynamicImage>>>,
    pub indexing_task_handle: Mutex<Option<JoinHandle<()>>>,
    pub lut_cache: Mutex<HashMap<String, Arc<Lut>>>,
    pub initial_file_path: Mutex<Option<String>>,
    pub pending_edit_session: Mutex<Option<ExternalEditSession>>,
    pub thumbnail_cancellation_token: Arc<AtomicBool>,
    pub thumbnail_progress: Mutex<ThumbnailProgressTracker>,
    pub preview_worker_tx: Mutex<Option<Sender<PreviewJob>>>,
    pub analytics_worker_tx: Mutex<Option<Sender<AnalyticsJob>>>,
    pub mask_cache: Mutex<HashMap<u64, GrayImage>>,
    pub patch_cache: Mutex<HashMap<String, serde_json::Value>>,
    pub geometry_cache: Mutex<HashMap<u64, DynamicImage>>,
    pub thumbnail_geometry_cache: Mutex<HashMap<String, ThumbnailGeometryEntry>>,
    pub lens_db: Mutex<Option<Arc<LensDatabase>>>,
    pub load_image_generation: Arc<AtomicUsize>,
    /// 切图、接受新参数和原生画面发布共用临界区，禁止旧任务覆盖新照片。
    pub preview_epoch: Mutex<usize>,
    pub full_warped_cache: Mutex<Option<(u64, Arc<DynamicImage>)>>,
    pub patched_warped_cache: Mutex<Option<(u64, Arc<DynamicImage>)>>,
    pub full_transformed_cache: Mutex<Option<TransformedImageCache>>,
    pub decoded_image_cache: Mutex<DecodedImageCache>,
    pub thumbnail_manager: Arc<ThumbnailManager>,
    pub metadata_manager: Arc<MetadataManager>,
    pub disks_cache: Mutex<Option<Disks>>,
    pub disks_cache_refreshing: AtomicBool,
    pub camera_session: Mutex<CameraSession>,
}

impl AppState {
    /// 预览代次临界区；若先前 GPU panic 毒化了锁，恢复后继续，避免永久「预览任务锁异常」。
    pub fn lock_preview_epoch(&self) -> MutexGuard<'_, usize> {
        self.preview_epoch.lock().unwrap_or_else(|poisoned| {
            log::warn!("预览代次锁曾被 panic 毒化，已恢复继续");
            poisoned.into_inner()
        })
    }
}
