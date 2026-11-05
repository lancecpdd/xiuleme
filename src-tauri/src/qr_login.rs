use std::fs;
use std::path::PathBuf;
use std::sync::Mutex;
use std::time::Duration;

use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use tauri::{AppHandle, Emitter, Manager};

/// 二维码内容前缀，拍了么小程序端据此识别登录二维码
pub const QR_LOGIN_PREFIX: &str = "xiuleme-login:";

/// 登录态失效（服务端返回 401）时通知前端回到登录页
pub const AUTH_EXPIRED_EVENT: &str = "auth-expired";

const AUTH_FILE_NAME: &str = "auth.json";

const STATUS_CONFIRMED: &str = "CONFIRMED";
const STATUS_EXPIRED: &str = "EXPIRED";
const STATUS_CANCELED: &str = "CANCELED";

const HTTP_TIMEOUT: Duration = Duration::from_secs(15);

/// 本次运行中经服务端确认有效的登录 token，是所有需登录功能的唯一可信来源。
/// 仅由扫码确认（CONFIRMED）或启动时在线校验成功写入；auth.json 只用于跨重启恢复，不直接被信任。
static VERIFIED_TOKEN: Mutex<Option<String>> = Mutex::new(None);

fn set_verified_token(token: Option<String>) {
    *VERIFIED_TOKEN.lock().unwrap_or_else(|e| e.into_inner()) = token;
}

/// 扫码登录会话（后端 /wx/qr-login/session 返回）
#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(rename_all = "camelCase")]
pub struct QrLoginSession {
    pub qr_token: String,
    #[serde(default)]
    pub expires_in: Option<u64>,
}

/// 登录用户信息（后端 CONFIRMED 时下发）
#[derive(Serialize, Deserialize, Debug, Clone, Default)]
#[serde(rename_all = "camelCase")]
pub struct QrLoginUser {
    #[serde(default)]
    pub user_id: Option<i64>,
    #[serde(default)]
    pub user_no: Option<String>,
    #[serde(default)]
    pub nickname: Option<String>,
    #[serde(default)]
    pub avatar_url: Option<String>,
    #[serde(default)]
    pub token: Option<String>,
}

impl QrLoginUser {
    /// 返回给前端的副本：token 只留在 Rust 端
    fn without_token(&self) -> Self {
        Self {
            token: None,
            ..self.clone()
        }
    }
}

/// 轮询结果
#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(rename_all = "camelCase")]
pub struct QrLoginPollResult {
    pub status: String,
    #[serde(default)]
    pub user: Option<QrLoginUser>,
}

/// 后端 /wx/qr-login/status 返回的数据结构
#[derive(Deserialize, Debug)]
#[serde(rename_all = "camelCase")]
struct QrLoginStatusData {
    status: String,
    #[serde(default)]
    user_id: Option<i64>,
    #[serde(default)]
    user_no: Option<String>,
    #[serde(default)]
    nickname: Option<String>,
    #[serde(default)]
    avatar_url: Option<String>,
    #[serde(default)]
    token: Option<String>,
}

/// 后端 /wx/xiuleme/auth/verify 返回的数据结构
#[derive(Deserialize, Debug)]
#[serde(rename_all = "camelCase")]
struct AuthVerifyData {
    #[serde(default)]
    user_id: Option<i64>,
    #[serde(default)]
    user_no: Option<String>,
    #[serde(default)]
    nickname: Option<String>,
    #[serde(default)]
    avatar_url: Option<String>,
}

/// 后端统一响应包装 Result<T>（code=1 成功）
#[derive(Deserialize, Debug)]
pub(crate) struct ApiEnvelope<T> {
    pub(crate) code: i32,
    pub(crate) msg: Option<String>,
    pub(crate) data: Option<T>,
}

pub(crate) fn http_client() -> reqwest::Client {
    reqwest::Client::builder()
        .timeout(HTTP_TIMEOUT)
        .build()
        .unwrap_or_default()
}

/// 携带登录 token 的请求（请求头 source_client 为 miniapp）
fn authorized_request(method: reqwest::Method, url: &str, token: &str) -> reqwest::RequestBuilder {
    http_client()
        .request(method, url)
        .header("User-Agent", "XiuLeMe-App")
        .header(
            crate::app_config::SOURCE_CLIENT_HEADER,
            crate::app_config::SOURCE_CLIENT_MINIAPP,
        )
        .header("Authorization", token)
}

/// 向业务后端发送携带桌面端登录凭证的 JSON POST 请求。
pub(crate) async fn authorized_post_json<B: Serialize + ?Sized>(
    app_handle: &AppHandle,
    token: &str,
    url: String,
    body: &B,
    err_label: &str,
) -> Result<(), String> {
    authorized_post_json_data::<serde_json::Value, B>(app_handle, token, url, body, err_label)
        .await
        .map(|_| ())
}

/// 向业务后端发送携带桌面端登录凭证的 JSON POST 请求，并解析统一响应包装中的 data。
pub(crate) async fn authorized_post_json_data<T: DeserializeOwned, B: Serialize + ?Sized>(
    app_handle: &AppHandle,
    token: &str,
    url: String,
    body: &B,
    err_label: &str,
) -> Result<Option<T>, String> {
    let response = authorized_request(reqwest::Method::POST, &url, token)
        .json(body)
        .send()
        .await
        .map_err(|e| format!("{}失败: {}", err_label, e))?;

    if response.status() == reqwest::StatusCode::UNAUTHORIZED {
        invalidate(app_handle);
        return Err("登录已过期，请重新登录".to_string());
    }
    if !response.status().is_success() {
        return Err(format!("{}失败: HTTP {}", err_label, response.status()));
    }

    let envelope: ApiEnvelope<T> = response
        .json()
        .await
        .map_err(|e| format!("解析{}响应失败: {}", err_label, e))?;
    if envelope.code != 1 {
        return Err(envelope.msg.unwrap_or_else(|| format!("{}失败", err_label)));
    }

    Ok(envelope.data)
}

fn get_auth_path(app_handle: &AppHandle) -> Result<PathBuf, String> {
    let auth_dir = app_handle.path().app_data_dir().map_err(|e| e.to_string())?;

    if !auth_dir.exists() {
        fs::create_dir_all(&auth_dir).map_err(|e| e.to_string())?;
    }

    Ok(auth_dir.join(AUTH_FILE_NAME))
}

fn save_auth_user(app_handle: &AppHandle, user: &QrLoginUser) -> Result<(), String> {
    let path = get_auth_path(app_handle)?;
    let json_string = serde_json::to_string_pretty(user).map_err(|e| e.to_string())?;
    fs::write(path, json_string).map_err(|e| e.to_string())
}

/// 读取本地持久化的登录用户（未经服务端校验，不可直接信任）
fn read_saved_user(app_handle: &AppHandle) -> Result<Option<QrLoginUser>, String> {
    let path = get_auth_path(app_handle)?;

    if !path.exists() {
        return Ok(None);
    }

    let content = fs::read_to_string(&path).map_err(|e| e.to_string())?;
    // 文件损坏时视为未登录
    Ok(serde_json::from_str::<QrLoginUser>(&content).ok())
}

fn clear_auth_file(app_handle: &AppHandle) -> Result<(), String> {
    let path = get_auth_path(app_handle)?;

    if path.exists() {
        fs::remove_file(path).map_err(|e| e.to_string())?;
    }

    Ok(())
}

/// 已通过服务端校验的登录 token（未登录或未校验时返回 None）
pub(crate) fn saved_token() -> Option<String> {
    VERIFIED_TOKEN
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .clone()
        .filter(|token| !token.is_empty())
}

/// 要求已登录（经服务端校验），返回登录 token；用于所有需要登录才能使用的命令入口
pub(crate) fn require_login() -> Result<String, String> {
    saved_token().ok_or_else(|| crate::pixel_billing::ERR_NOT_LOGGED_IN.to_string())
}

/// 登录态被服务端判定失效：清除内存与本地登录态，并通知前端变为 guest（保留图库）
pub(crate) fn invalidate(app_handle: &AppHandle) {
    set_verified_token(None);
    if let Err(e) = clear_auth_file(app_handle) {
        log::error!("Failed to clear saved auth: {}", e);
    }
    if let Err(e) = app_handle.emit(AUTH_EXPIRED_EVENT, ()) {
        log::error!("Failed to emit {} event: {}", AUTH_EXPIRED_EVENT, e);
    }
}

/// 在线校验本地保存的登录态。
/// 校验通过时写入已验证 token 并回写最新用户信息；服务端返回 401 时清除本地登录态并返回 None；
/// 网络或服务异常返回 Err（严格模式：视为未登录，由调用方提示重试）。
pub(crate) async fn verify_saved(app_handle: &AppHandle) -> Result<Option<QrLoginUser>, String> {
    set_verified_token(None);

    let Some(saved) = read_saved_user(app_handle)? else {
        return Ok(None);
    };
    let Some(token) = saved.token.clone().filter(|token| !token.is_empty()) else {
        clear_auth_file(app_handle)?;
        return Ok(None);
    };

    let url = format!("{}/wx/xiuleme/auth/verify", crate::app_config::api_base_url());
    let response = authorized_request(reqwest::Method::GET, &url, &token)
        .send()
        .await
        .map_err(|e| format!("无法连接拍了么服务: {}", e))?;

    if response.status() == reqwest::StatusCode::UNAUTHORIZED {
        clear_auth_file(app_handle)?;
        return Ok(None);
    }
    if !response.status().is_success() {
        return Err(format!("校验登录状态失败: HTTP {}", response.status()));
    }

    let envelope: ApiEnvelope<AuthVerifyData> = response
        .json()
        .await
        .map_err(|e| format!("解析登录校验响应失败: {}", e))?;

    if envelope.code != 1 {
        return Err(envelope.msg.unwrap_or_else(|| "校验登录状态失败".to_string()));
    }

    let data = envelope
        .data
        .ok_or_else(|| "登录校验响应缺少数据".to_string())?;

    let user = QrLoginUser {
        user_id: data.user_id.or(saved.user_id),
        user_no: data.user_no,
        nickname: data.nickname,
        avatar_url: data.avatar_url,
        token: Some(token.clone()),
    };
    if let Err(e) = save_auth_user(app_handle, &user) {
        log::error!("Failed to update saved auth: {}", e);
    }
    set_verified_token(Some(token));

    Ok(Some(user))
}

/// 创建扫码登录会话，返回用于生成二维码的凭证
#[tauri::command]
pub async fn qr_login_create_session() -> Result<QrLoginSession, String> {
    let url = format!("{}/wx/qr-login/session", crate::app_config::api_base_url());

    let response = http_client()
        .post(url)
        .header("User-Agent", "XiuLeMe-App")
        .header(
            crate::app_config::SOURCE_CLIENT_HEADER,
            crate::app_config::SOURCE_CLIENT_MINIAPP,
        )
        .send()
        .await
        .map_err(|e| format!("无法连接拍了么服务: {}", e))?;

    let envelope: ApiEnvelope<QrLoginSession> = response
        .json()
        .await
        .map_err(|e| format!("解析登录会话响应失败: {}", e))?;

    if envelope.code != 1 {
        return Err(envelope.msg.unwrap_or_else(|| "创建登录会话失败".to_string()));
    }

    envelope
        .data
        .ok_or_else(|| "登录会话响应缺少数据".to_string())
}

/// 轮询扫码登录状态。
/// 返回 CONFIRMED 时携带用户信息，并持久化到本地 auth.json。
#[tauri::command]
pub async fn qr_login_poll(
    qr_token: String,
    app_handle: AppHandle,
) -> Result<QrLoginPollResult, String> {
    let url = reqwest::Url::parse_with_params(
        &format!("{}/wx/qr-login/status", crate::app_config::api_base_url()),
        &[("qrToken", qr_token.as_str())],
    )
    .map_err(|e| format!("登录服务地址无效: {}", e))?;

    let response = http_client()
        .get(url)
        .header("User-Agent", "XiuLeMe-App")
        .header(
            crate::app_config::SOURCE_CLIENT_HEADER,
            crate::app_config::SOURCE_CLIENT_MINIAPP,
        )
        .send()
        .await
        .map_err(|e| format!("无法连接拍了么服务: {}", e))?;

    let envelope: ApiEnvelope<QrLoginStatusData> = response
        .json()
        .await
        .map_err(|e| format!("解析登录状态响应失败: {}", e))?;

    if envelope.code != 1 {
        return Err(envelope.msg.unwrap_or_else(|| "查询登录状态失败".to_string()));
    }

    let data = envelope
        .data
        .ok_or_else(|| "登录状态响应缺少数据".to_string())?;

    if data.status == STATUS_CONFIRMED {
        let token = data
            .token
            .filter(|token| !token.is_empty())
            .ok_or_else(|| "登录响应缺少凭证".to_string())?;
        let user = QrLoginUser {
            user_id: data.user_id,
            user_no: data.user_no,
            nickname: data.nickname,
            avatar_url: data.avatar_url,
            token: Some(token.clone()),
        };
        save_auth_user(&app_handle, &user)?;
        set_verified_token(Some(token));

        return Ok(QrLoginPollResult {
            status: data.status,
            user: Some(user.without_token()),
        });
    }

    if data.status != STATUS_EXPIRED && data.status != STATUS_CANCELED && data.status != "WAITING" {
        // 未知状态按等待处理，避免前端误判
        return Ok(QrLoginPollResult {
            status: "WAITING".to_string(),
            user: None,
        });
    }

    Ok(QrLoginPollResult {
        status: data.status,
        user: None,
    })
}

/// 启动时恢复登录态：在线校验本地保存的 token，通过才视为已登录
#[tauri::command]
pub async fn qr_login_verify_saved(app_handle: AppHandle) -> Result<Option<QrLoginUser>, String> {
    verify_saved(&app_handle)
        .await
        .map(|user| user.map(|user| user.without_token()))
}

/// 退出登录：通知服务端吊销 token（失败不阻断），并清除本地登录态
#[tauri::command]
pub async fn qr_login_logout(app_handle: AppHandle) -> Result<(), String> {
    let token = saved_token().or_else(|| {
        read_saved_user(&app_handle)
            .ok()
            .flatten()
            .and_then(|user| user.token)
            .filter(|token| !token.is_empty())
    });
    set_verified_token(None);

    if let Some(token) = token {
        let url = format!("{}/wx/xiuleme/auth/logout", crate::app_config::api_base_url());
        match authorized_request(reqwest::Method::POST, &url, &token).send().await {
            Ok(response) if !response.status().is_success() => {
                log::warn!("Remote logout returned HTTP {}", response.status());
            }
            Ok(_) => {}
            Err(e) => log::warn!("Remote logout failed: {}", e),
        }
    }

    clear_auth_file(&app_handle)
}

/// 像素积分明细项（后端 PixelDetailItemVO）
#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(rename_all = "camelCase")]
pub struct PixelDetailItem {
    #[serde(default)]
    pub id: String,
    #[serde(default, rename = "type")]
    pub kind: String,
    #[serde(default)]
    pub title: Option<String>,
    #[serde(default)]
    pub description: Option<String>,
    #[serde(default)]
    pub amount: i64,
    #[serde(default)]
    pub icon_type: Option<String>,
    #[serde(default)]
    pub create_time: Option<String>,
}

/// 按日期分组的积分明细（后端 PixelDetailGroupVO）
#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(rename_all = "camelCase")]
pub struct PixelDetailGroup {
    #[serde(default)]
    pub date: String,
    #[serde(default)]
    pub details: Vec<PixelDetailItem>,
}

/// 积分明细分页结果（后端 PixelDetailPageVO）
#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(rename_all = "camelCase")]
pub struct PixelDetailPage {
    #[serde(default)]
    pub groups: Vec<PixelDetailGroup>,
    #[serde(default)]
    pub total: i64,
    #[serde(default)]
    pub has_more: bool,
}

/// 携带登录 token 发起 GET 请求并解析统一响应包装；服务端判定登录失效时清除登录态
pub(crate) async fn authorized_get<T: DeserializeOwned>(
    app_handle: &AppHandle,
    token: String,
    url: String,
    err_label: &str,
) -> Result<Option<T>, String> {
    let response = authorized_request(reqwest::Method::GET, &url, &token)
        .send()
        .await
        .map_err(|e| format!("无法连接拍了么服务: {}", e))?;

    if response.status() == reqwest::StatusCode::UNAUTHORIZED {
        invalidate(app_handle);
        return Err("登录已过期，请重新登录".to_string());
    }
    if !response.status().is_success() {
        return Err(format!("查询{}失败: HTTP {}", err_label, response.status()));
    }

    let envelope: ApiEnvelope<T> = response
        .json()
        .await
        .map_err(|e| format!("解析{}响应失败: {}", err_label, e))?;

    if envelope.code != 1 {
        return Err(envelope.msg.unwrap_or_else(|| format!("查询{}失败", err_label)));
    }

    Ok(envelope.data)
}

/// 查询当前登录用户的像素积分余额（未登录或无 token 时返回 None）
#[tauri::command]
pub async fn fetch_pixel_balance(app_handle: AppHandle) -> Result<Option<i64>, String> {
    let Some(token) = saved_token() else {
        return Ok(None);
    };

    let url = format!("{}/wx/user/pixel/balance", crate::app_config::api_base_url());
    authorized_get::<i64>(&app_handle, token, url, "像素积分").await
}

/// 分页查询当前登录用户的像素积分消耗明细（未登录或无 token 时返回 None）
#[tauri::command]
pub async fn fetch_pixel_details(
    app_handle: AppHandle,
    page: u32,
    page_size: u32,
) -> Result<Option<PixelDetailPage>, String> {
    let Some(token) = saved_token() else {
        return Ok(None);
    };

    let url = format!(
        "{}/wx/user/pixel/detail_list?filter=expense&page={}&pageSize={}",
        crate::app_config::api_base_url(),
        page.max(1),
        page_size.max(1)
    );
    authorized_get::<PixelDetailPage>(&app_handle, token, url, "消耗明细").await
}
