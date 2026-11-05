//! 崩溃与异常日志：本地落盘待上报文件，联网后上报到 paileme-server。
//!
//! - Rust panic：panic hook 同步写报告（含调用栈与日志尾部）。
//! - 原生崩溃 / 进程被强制结束：panic hook 无法感知，靠 session.lock 会话标记在下次启动时补报。
//! - 前端未捕获错误：经 `report_frontend_error` 命令写入。

use std::backtrace::Backtrace;
use std::fs;
use std::io::{Read, Seek, SeekFrom};
use std::panic::PanicHookInfo;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};
use tauri::{AppHandle, Manager};

use crate::qr_login::ApiEnvelope;

pub const LOG_FILE_NAME: &str = "app.log";
const PREV_LOG_FILE_NAME: &str = "app.prev.log";
const REPORT_DIR_NAME: &str = "crash_reports";
const SESSION_FILE_NAME: &str = "session.lock";
const DEVICE_ID_FILE_NAME: &str = "device_id";

const MAX_PENDING_REPORTS: usize = 20;
const MAX_REPORT_AGE: Duration = Duration::from_secs(30 * 24 * 3600);
const MAX_PANIC_REPORTS_PER_SESSION: usize = 5;
const MAX_MESSAGE_BYTES: usize = 2048;
const MAX_LOCATION_BYTES: usize = 512;
const MAX_STACK_BYTES: usize = 16 * 1024;
const MAX_LOG_TAIL_BYTES: usize = 64 * 1024;
const MAX_LOG_TAIL_LINES: usize = 200;

const ENABLED_CONFIG_KEY: &str = "app.xiuleme.crash_report.enabled";
/// debug 构建默认不上报，设置为 1 时才上报（联调用）
const DEV_UPLOAD_ENV: &str = "XIULEME_CRASH_REPORT_DEV";
const INITIAL_FLUSH_DELAY: Duration = Duration::from_secs(15);
const FLUSH_INTERVAL: Duration = Duration::from_secs(10 * 60);

const TYPE_RUST_PANIC: &str = "RUST_PANIC";
const TYPE_ABNORMAL_EXIT: &str = "ABNORMAL_EXIT";
const TYPE_FRONTEND_ERROR: &str = "FRONTEND_ERROR";
const TYPE_FRONTEND_FATAL: &str = "FRONTEND_FATAL";

struct CrashContext {
    log_dir: PathBuf,
    session_id: String,
    device_id: String,
    app_version: String,
}

static CONTEXT: OnceLock<CrashContext> = OnceLock::new();
static PANIC_REPORT_COUNT: AtomicUsize = AtomicUsize::new(0);
static FLUSHING: AtomicBool = AtomicBool::new(false);

#[derive(Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct SessionMarker {
    session_id: String,
    started_at: i64,
    app_version: String,
    #[serde(default)]
    panicked: bool,
}

#[derive(Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct CrashReport {
    session_id: String,
    device_id: String,
    report_type: String,
    message: String,
    #[serde(default)]
    location: Option<String>,
    #[serde(default)]
    stack: Option<String>,
    #[serde(default)]
    log_tail: Option<String>,
    app_version: String,
    #[serde(default)]
    os_info: Option<String>,
    arch: String,
    occurred_at: i64,
}

fn now_millis() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0)
}

/// 按字节上限截断，保证落在字符边界
fn truncate_utf8(text: &str, max_bytes: usize) -> String {
    if text.len() <= max_bytes {
        return text.to_string();
    }
    let mut end = max_bytes;
    while end > 0 && !text.is_char_boundary(end) {
        end -= 1;
    }
    text[..end].to_string()
}

fn non_empty(text: Option<String>, max_bytes: usize) -> Option<String> {
    text.map(|t| t.trim().to_string())
        .filter(|t| !t.is_empty())
        .map(|t| truncate_utf8(&t, max_bytes))
}

fn report_dir(log_dir: &Path) -> PathBuf {
    log_dir.join(REPORT_DIR_NAME)
}

fn session_path(log_dir: &Path) -> PathBuf {
    log_dir.join(SESSION_FILE_NAME)
}

/// 读取日志文件最后 MAX_LOG_TAIL_LINES 行（不超过 MAX_LOG_TAIL_BYTES）
fn read_log_tail(path: &Path) -> Option<String> {
    let mut file = fs::File::open(path).ok()?;
    let len = file.metadata().ok()?.len();
    let start = len.saturating_sub(MAX_LOG_TAIL_BYTES as u64);
    file.seek(SeekFrom::Start(start)).ok()?;
    let mut bytes = Vec::new();
    file.read_to_end(&mut bytes).ok()?;
    let text = String::from_utf8_lossy(&bytes);
    let mut lines: Vec<&str> = text.lines().collect();
    // 从中间截取时第一行可能不完整
    if start > 0 && !lines.is_empty() {
        lines.remove(0);
    }
    let skip = lines.len().saturating_sub(MAX_LOG_TAIL_LINES);
    let tail = lines[skip..].join("\n");
    non_empty(Some(tail), MAX_LOG_TAIL_BYTES)
}

fn load_or_create_device_id(app_handle: &AppHandle) -> String {
    let Ok(dir) = app_handle.path().app_data_dir() else {
        return uuid::Uuid::new_v4().to_string();
    };
    let path = dir.join(DEVICE_ID_FILE_NAME);
    if let Ok(saved) = fs::read_to_string(&path) {
        let saved = saved.trim();
        if uuid::Uuid::parse_str(saved).is_ok() {
            return saved.to_string();
        }
    }
    let device_id = uuid::Uuid::new_v4().to_string();
    if let Err(e) = fs::create_dir_all(&dir).and_then(|_| fs::write(&path, &device_id)) {
        log::warn!("[crash-report] 保存设备标识失败: {}", e);
    }
    device_id
}

fn write_marker(log_dir: &Path, marker: &SessionMarker) {
    if let Ok(json) = serde_json::to_string(marker) {
        let _ = fs::write(session_path(log_dir), json);
    }
}

fn read_marker(log_dir: &Path) -> Option<SessionMarker> {
    let content = fs::read_to_string(session_path(log_dir)).ok()?;
    serde_json::from_str(&content).ok()
}

/// 删除过期报告，并只保留最新的 MAX_PENDING_REPORTS 个
fn prune_reports(log_dir: &Path) {
    let mut files = pending_report_files(log_dir);
    let now = SystemTime::now();
    files.retain(|path| {
        let expired = fs::metadata(path)
            .and_then(|m| m.modified())
            .ok()
            .and_then(|modified| now.duration_since(modified).ok())
            .is_some_and(|age| age > MAX_REPORT_AGE);
        if expired {
            let _ = fs::remove_file(path);
        }
        !expired
    });
    if files.len() > MAX_PENDING_REPORTS {
        for path in &files[..files.len() - MAX_PENDING_REPORTS] {
            let _ = fs::remove_file(path);
        }
    }
}

/// 待上报文件按文件名（发生时间毫秒前缀）升序排列
fn pending_report_files(log_dir: &Path) -> Vec<PathBuf> {
    let Ok(entries) = fs::read_dir(report_dir(log_dir)) else {
        return Vec::new();
    };
    let mut files: Vec<PathBuf> = entries
        .filter_map(|entry| entry.ok().map(|e| e.path()))
        .filter(|path| path.extension().is_some_and(|ext| ext == "json"))
        .collect();
    files.sort();
    files
}

fn write_report(log_dir: &Path, report: &CrashReport) {
    let dir = report_dir(log_dir);
    if fs::create_dir_all(&dir).is_err() {
        return;
    }
    let file_name = format!("{}-{}.json", report.occurred_at, uuid::Uuid::new_v4());
    if let Ok(json) = serde_json::to_string(report) {
        let _ = fs::write(dir.join(file_name), json);
    }
    prune_reports(log_dir);
}

fn build_report(
    ctx: &CrashContext,
    report_type: &str,
    message: &str,
    location: Option<String>,
    stack: Option<String>,
) -> CrashReport {
    CrashReport {
        session_id: ctx.session_id.clone(),
        device_id: ctx.device_id.clone(),
        report_type: report_type.to_string(),
        message: truncate_utf8(message.trim(), MAX_MESSAGE_BYTES),
        location: non_empty(location, MAX_LOCATION_BYTES),
        stack: non_empty(stack, MAX_STACK_BYTES),
        log_tail: read_log_tail(&ctx.log_dir.join(LOG_FILE_NAME)),
        app_version: ctx.app_version.clone(),
        os_info: sysinfo::System::long_os_version(),
        arch: std::env::consts::ARCH.to_string(),
        occurred_at: now_millis(),
    }
}

/// 启动时调用（须在打开新 app.log 之前）：补报上次异常退出、保留上次日志、写入本次会话标记。
pub fn on_startup(app_handle: &AppHandle, log_dir: &Path) {
    let ctx = CrashContext {
        log_dir: log_dir.to_path_buf(),
        session_id: uuid::Uuid::new_v4().to_string(),
        device_id: load_or_create_device_id(app_handle),
        app_version: app_handle.package_info().version.to_string(),
    };
    let log_path = log_dir.join(LOG_FILE_NAME);

    if let Some(previous) = read_marker(log_dir)
        && !previous.panicked
    {
        // 上次日志最后写入时间最接近崩溃时间
        let occurred_at = fs::metadata(&log_path)
            .and_then(|m| m.modified())
            .ok()
            .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
            .map(|d| d.as_millis() as i64)
            .unwrap_or(previous.started_at);
        let report = CrashReport {
            session_id: previous.session_id,
            device_id: ctx.device_id.clone(),
            report_type: TYPE_ABNORMAL_EXIT.to_string(),
            message: "上次运行未正常退出（可能为原生崩溃或进程被强制结束）".to_string(),
            location: None,
            stack: None,
            log_tail: read_log_tail(&log_path),
            app_version: previous.app_version,
            os_info: sysinfo::System::long_os_version(),
            arch: std::env::consts::ARCH.to_string(),
            occurred_at,
        };
        write_report(log_dir, &report);
    }

    if log_path.exists() {
        let _ = fs::rename(&log_path, log_dir.join(PREV_LOG_FILE_NAME));
    }

    write_marker(
        log_dir,
        &SessionMarker {
            session_id: ctx.session_id.clone(),
            started_at: now_millis(),
            app_version: ctx.app_version.clone(),
            panicked: false,
        },
    );
    prune_reports(log_dir);
    let _ = CONTEXT.set(ctx);
}

/// panic hook 中调用：同步写报告，并标记本次会话已上报 panic（避免下次启动重复补报异常退出）。
pub fn record_panic(info: &PanicHookInfo<'_>, message: &str) {
    let Some(ctx) = CONTEXT.get() else {
        return;
    };
    if PANIC_REPORT_COUNT.fetch_add(1, Ordering::SeqCst) >= MAX_PANIC_REPORTS_PER_SESSION {
        return;
    }
    let location = info
        .location()
        .map(|loc| format!("{}:{}:{}", loc.file(), loc.line(), loc.column()));
    let thread = std::thread::current();
    let stack = format!(
        "thread '{}'\n{}",
        thread.name().unwrap_or("<unnamed>"),
        Backtrace::force_capture()
    );
    let report = build_report(ctx, TYPE_RUST_PANIC, message, location, Some(stack));
    write_report(&ctx.log_dir, &report);

    if let Some(mut marker) = read_marker(&ctx.log_dir)
        && marker.session_id == ctx.session_id
    {
        marker.panicked = true;
        write_marker(&ctx.log_dir, &marker);
    }
}

/// 正常退出：删除会话标记
pub fn mark_clean_exit() {
    if let Some(ctx) = CONTEXT.get() {
        let _ = fs::remove_file(session_path(&ctx.log_dir));
    }
}

/// 记录前端未捕获错误，并立即尝试上报
#[tauri::command]
pub async fn report_frontend_error(
    kind: String,
    message: String,
    stack: Option<String>,
    source: Option<String>,
) -> Result<(), String> {
    if kind != TYPE_FRONTEND_ERROR && kind != TYPE_FRONTEND_FATAL {
        return Err(format!("不支持的错误类型: {}", kind));
    }
    let Some(ctx) = CONTEXT.get() else {
        return Ok(());
    };
    // debug 构建未开启联调上报时不落盘，避免开发期噪音积压
    if !upload_allowed() {
        return Ok(());
    }
    let message = if message.trim().is_empty() {
        "Unknown frontend error".to_string()
    } else {
        message
    };
    let report = build_report(ctx, &kind, &message, source, stack);
    write_report(&ctx.log_dir, &report);
    tauri::async_runtime::spawn(flush_pending());
    Ok(())
}

fn upload_allowed() -> bool {
    !cfg!(debug_assertions) || std::env::var(DEV_UPLOAD_ENV).is_ok_and(|v| v == "1")
}

/// 启动后延迟首次上报（等待登录校验完成），之后定期重试
pub fn spawn_background_uploader() {
    tauri::async_runtime::spawn(async {
        tokio::time::sleep(INITIAL_FLUSH_DELAY).await;
        loop {
            flush_pending().await;
            tokio::time::sleep(FLUSH_INTERVAL).await;
        }
    });
}

struct FlushGuard;

impl Drop for FlushGuard {
    fn drop(&mut self) {
        FLUSHING.store(false, Ordering::SeqCst);
    }
}

/// 逐个上报待上报文件；成功即删除，失败保留到下次重试
async fn flush_pending() {
    if !upload_allowed() {
        return;
    }
    let Some(ctx) = CONTEXT.get() else {
        return;
    };
    if FLUSHING
        .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
        .is_err()
    {
        return;
    }
    let _guard = FlushGuard;

    let files = pending_report_files(&ctx.log_dir);
    if files.is_empty() {
        return;
    }

    match crate::sys_config::fetch_sys_config_batch(&[ENABLED_CONFIG_KEY]).await {
        Ok(map) if map.get(ENABLED_CONFIG_KEY).cloned().flatten().as_deref() == Some("true") => {}
        Ok(_) => return,
        Err(e) => {
            log::warn!("[crash-report] 读取上报开关失败: {}", e);
            return;
        }
    }

    let client = crate::qr_login::http_client();
    let url = format!(
        "{}/wx/xiuleme/crash/report",
        crate::app_config::api_base_url().trim_end_matches('/')
    );

    for path in files {
        let report = fs::read_to_string(&path)
            .ok()
            .and_then(|content| serde_json::from_str::<CrashReport>(&content).ok());
        let Some(report) = report else {
            let _ = fs::remove_file(&path);
            continue;
        };
        match send_report(&client, &url, &report).await {
            Ok(()) => {
                let _ = fs::remove_file(&path);
            }
            Err(SendError::Rejected(e)) => {
                // 服务端明确拒收（请求体非法或过大），重试无意义，删除以免堵住队列
                log::warn!("[crash-report] 报告被拒收，已丢弃: {}", e);
                let _ = fs::remove_file(&path);
            }
            Err(SendError::Retry(e)) => {
                // 多为网络不可用或被限频，停止本轮，避免连续失败
                log::warn!("[crash-report] 上报失败: {}", e);
                break;
            }
        }
    }
}

async fn post_report(
    client: &reqwest::Client,
    url: &str,
    report: &CrashReport,
    token: Option<&str>,
) -> Result<reqwest::Response, String> {
    let mut request = client
        .post(url)
        .header("User-Agent", "XiuLeMe-App")
        .header("source_client", "miniapp")
        .json(report);
    if let Some(token) = token {
        request = request.header("Authorization", token);
    }
    request.send().await.map_err(|e| e.to_string())
}

enum SendError {
    /// 网络异常、限频或服务端暂不可用，保留文件下次重试
    Retry(String),
    /// 服务端拒收该报告，重试也不会成功
    Rejected(String),
}

async fn send_report(
    client: &reqwest::Client,
    url: &str,
    report: &CrashReport,
) -> Result<(), SendError> {
    let token = crate::qr_login::saved_token();
    let mut response = post_report(client, url, report, token.as_deref())
        .await
        .map_err(SendError::Retry)?;
    // token 失效时退化为匿名上报；登录态的失效处理交给业务请求，这里不清除
    if response.status() == reqwest::StatusCode::UNAUTHORIZED && token.is_some() {
        response = post_report(client, url, report, None)
            .await
            .map_err(SendError::Retry)?;
    }
    let status = response.status();
    if status == reqwest::StatusCode::BAD_REQUEST
        || status == reqwest::StatusCode::PAYLOAD_TOO_LARGE
    {
        return Err(SendError::Rejected(format!("HTTP {}", status)));
    }
    if !status.is_success() {
        return Err(SendError::Retry(format!("HTTP {}", status)));
    }
    let envelope: ApiEnvelope<serde_json::Value> = response
        .json()
        .await
        .map_err(|e| SendError::Retry(e.to_string()))?;
    if envelope.code != 1 {
        return Err(SendError::Retry(
            envelope.msg.unwrap_or_else(|| "上报失败".to_string()),
        ));
    }
    Ok(())
}
