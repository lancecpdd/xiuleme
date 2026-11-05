use serde::de::Deserializer;
use serde::{Deserialize, Serialize};
use serde_json::Value;

const PUBLIC_STATUSES: [&str; 4] = ["accepted", "implementing", "rejected", "done"];
const MAX_TITLE_LEN: usize = 100;
const MAX_CONTENT_LEN: usize = 2000;

/// 后端会序列化 null 字段（如功能关闭时），null 按默认值处理
fn null_as_default<'de, D, T>(deserializer: D) -> Result<T, D::Error>
where
    D: Deserializer<'de>,
    T: Default + Deserialize<'de>,
{
    Ok(Option::<T>::deserialize(deserializer)?.unwrap_or_default())
}

/// 点赞标记兼容 JSON boolean 和 0/1
fn flexible_bool<'de, D>(deserializer: D) -> Result<bool, D::Error>
where
    D: Deserializer<'de>,
{
    let value = Value::deserialize(deserializer)?;
    Ok(match value {
        Value::Bool(flag) => flag,
        Value::Number(number) => number.as_i64().unwrap_or(0) != 0,
        _ => false,
    })
}

#[derive(Deserialize, Debug)]
#[serde(rename_all = "camelCase")]
struct RemoteRequirement {
    id: i64,
    #[serde(default)]
    title: Option<String>,
    #[serde(default)]
    content: Option<String>,
    #[serde(default)]
    status: Option<String>,
    #[serde(default)]
    like_count: Option<i64>,
    #[serde(default, deserialize_with = "flexible_bool")]
    liked: bool,
    #[serde(default)]
    create_time: Option<String>,
    #[serde(default)]
    nickname: Option<String>,
    #[serde(default)]
    avatar_url: Option<String>,
}

#[derive(Deserialize, Debug)]
#[serde(rename_all = "camelCase")]
struct RemoteSubmitResult {
    id: i64,
}

#[derive(Deserialize, Debug)]
#[serde(rename_all = "camelCase")]
struct RemoteLikeResult {
    id: i64,
    #[serde(default)]
    like_count: Option<i64>,
    #[serde(default, deserialize_with = "flexible_bool")]
    liked: bool,
}

/// 需求池开关（后端 RequirementInfoVO）
#[derive(Serialize, Deserialize, Debug, Clone, Default)]
#[serde(rename_all = "camelCase")]
pub struct RequirementInfo {
    #[serde(default, deserialize_with = "null_as_default")]
    pub enabled: bool,
}

/// 下发给前端的需求
#[derive(Serialize, Debug, Clone)]
#[serde(rename_all = "camelCase")]
pub struct Requirement {
    pub id: String,
    pub title: String,
    pub content: String,
    pub status: String,
    pub like_count: i64,
    pub liked: bool,
    pub create_time: String,
    pub nickname: String,
    pub avatar_url: String,
}

#[derive(Serialize, Debug, Clone)]
#[serde(rename_all = "camelCase")]
pub struct RequirementLikeResult {
    pub id: String,
    pub like_count: i64,
    pub liked: bool,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct SubmitRequest<'a> {
    title: &'a str,
    content: &'a str,
}

fn into_requirement(remote: RemoteRequirement) -> Requirement {
    Requirement {
        id: remote.id.to_string(),
        title: remote.title.unwrap_or_default(),
        content: remote.content.unwrap_or_default(),
        status: remote.status.unwrap_or_default(),
        like_count: remote.like_count.unwrap_or(0),
        liked: remote.liked,
        create_time: remote
            .create_time
            .unwrap_or_default()
            .replacen(' ', "T", 1),
        nickname: remote.nickname.unwrap_or_default(),
        avatar_url: remote.avatar_url.unwrap_or_default(),
    }
}

fn normalize_status(status: Option<String>) -> Result<Option<String>, String> {
    let Some(status) = status
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
    else {
        return Ok(None);
    };
    if !PUBLIC_STATUSES.contains(&status.as_str()) {
        return Err("状态无效".to_string());
    }
    Ok(Some(status))
}

fn list_url(status: Option<&str>) -> String {
    let base = format!(
        "{}/wx/xiuleme/requirement/list",
        crate::app_config::api_base_url()
    );
    match status {
        Some(status) => format!("{base}?status={status}"),
        None => base,
    }
}

async fn fetch_public_list(
    app_handle: &tauri::AppHandle,
    status: Option<&str>,
) -> Result<Vec<Requirement>, String> {
    let url = list_url(status);
    let items = if let Some(token) = crate::qr_login::saved_token() {
        crate::qr_login::authorized_get::<Vec<RemoteRequirement>>(
            app_handle,
            token,
            url,
            "需求池",
        )
        .await?
    } else {
        let response = crate::qr_login::http_client()
            .get(&url)
            .header("User-Agent", "XiuLeMe-App")
            .header(
                crate::app_config::SOURCE_CLIENT_HEADER,
                crate::app_config::SOURCE_CLIENT_MINIAPP,
            )
            .send()
            .await
            .map_err(|err| format!("无法连接拍了么服务: {}", err))?;
        if !response.status().is_success() {
            return Err(format!("查询需求池失败: HTTP {}", response.status()));
        }
        let envelope: crate::qr_login::ApiEnvelope<Vec<RemoteRequirement>> = response
            .json()
            .await
            .map_err(|err| format!("解析需求池响应失败: {}", err))?;
        if envelope.code != 1 {
            return Err(envelope
                .msg
                .unwrap_or_else(|| "查询需求池失败".to_string()));
        }
        envelope.data
    };

    Ok(items.unwrap_or_default().into_iter().map(into_requirement).collect())
}

/// 查询需求池开关（未登录可调用；失败由前端按关闭处理）
#[tauri::command]
pub async fn fetch_requirement_info() -> Result<RequirementInfo, String> {
    let url = format!(
        "{}/wx/xiuleme/requirement/info",
        crate::app_config::api_base_url()
    );
    let response = crate::qr_login::http_client()
        .get(&url)
        .header("User-Agent", "XiuLeMe-App")
        .header(
            crate::app_config::SOURCE_CLIENT_HEADER,
            crate::app_config::SOURCE_CLIENT_MINIAPP,
        )
        .send()
        .await
        .map_err(|err| format!("无法连接拍了么服务: {}", err))?;
    if !response.status().is_success() {
        return Err(format!("查询需求池配置失败: HTTP {}", response.status()));
    }
    let envelope: crate::qr_login::ApiEnvelope<RequirementInfo> = response
        .json()
        .await
        .map_err(|err| format!("解析需求池配置失败: {}", err))?;
    if envelope.code != 1 {
        return Err(envelope
            .msg
            .unwrap_or_else(|| "查询需求池配置失败".to_string()));
    }
    Ok(envelope.data.unwrap_or_default())
}

/// 已公开需求。已登录时带上 token，用于返回是否已赞。
#[tauri::command]
pub async fn fetch_requirements(
    status: Option<String>,
    app_handle: tauri::AppHandle,
) -> Result<Vec<Requirement>, String> {
    let status = normalize_status(status)?;
    fetch_public_list(&app_handle, status.as_deref()).await
}

/// 提交需求。待审核期间用户不能再改。
#[tauri::command]
pub async fn submit_requirement(
    title: String,
    content: String,
    app_handle: tauri::AppHandle,
) -> Result<String, String> {
    let title = title.trim().split_whitespace().collect::<Vec<_>>().join(" ");
    let content = content.trim().to_string();
    if title.is_empty() {
        return Err("请填写标题".to_string());
    }
    if title.chars().count() > MAX_TITLE_LEN {
        return Err("标题不能超过 100 个字".to_string());
    }
    if content.is_empty() {
        return Err("请填写描述".to_string());
    }
    if content.chars().count() > MAX_CONTENT_LEN {
        return Err("描述不能超过 2000 个字".to_string());
    }

    let token = crate::qr_login::require_login()?;
    let url = format!(
        "{}/wx/xiuleme/requirement/submit",
        crate::app_config::api_base_url()
    );
    let body = SubmitRequest {
        title: &title,
        content: &content,
    };
    let result = crate::qr_login::authorized_post_json_data::<RemoteSubmitResult, _>(
        &app_handle,
        &token,
        url,
        &body,
        "提交需求",
    )
    .await?
    .ok_or_else(|| "提交需求失败".to_string())?;
    Ok(result.id.to_string())
}

/// 点赞或取消点赞
#[tauri::command]
pub async fn toggle_requirement_like(
    id: String,
    app_handle: tauri::AppHandle,
) -> Result<RequirementLikeResult, String> {
    let id = id.trim();
    if id.is_empty() || !id.chars().all(|ch| ch.is_ascii_digit()) {
        return Err("需求不存在".to_string());
    }
    let token = crate::qr_login::require_login()?;
    let url = format!(
        "{}/wx/xiuleme/requirement/{}/like",
        crate::app_config::api_base_url(),
        id
    );
    let result = crate::qr_login::authorized_post_json_data::<RemoteLikeResult, _>(
        &app_handle,
        &token,
        url,
        &serde_json::json!({}),
        "点赞",
    )
    .await?
    .ok_or_else(|| "点赞失败".to_string())?;
    Ok(RequirementLikeResult {
        id: result.id.to_string(),
        like_count: result.like_count.unwrap_or(0),
        liked: result.liked,
    })
}
