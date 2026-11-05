use serde::{Deserialize, Deserializer, Serialize};
use tauri::AppHandle;

/// 后端会序列化 null 字段（如功能关闭时），null 按默认值处理
fn null_as_default<'de, D, T>(deserializer: D) -> Result<T, D::Error>
where
    D: Deserializer<'de>,
    T: Default + Deserialize<'de>,
{
    Ok(Option::<T>::deserialize(deserializer)?.unwrap_or_default())
}

/// Bug 反馈配置信息（后端 BugFeedbackInfoVO）
#[derive(Serialize, Deserialize, Debug, Clone, Default)]
#[serde(rename_all = "camelCase")]
pub struct BugFeedbackInfo {
    #[serde(default, deserialize_with = "null_as_default")]
    pub enabled: bool,
    #[serde(default, deserialize_with = "null_as_default")]
    pub reward: i64,
    #[serde(default, deserialize_with = "null_as_default")]
    pub daily_submit_limit: i64,
    #[serde(default, deserialize_with = "null_as_default")]
    pub today_submit_count: i64,
    #[serde(default, deserialize_with = "null_as_default")]
    pub remaining_submit_count: i64,
    #[serde(default, deserialize_with = "null_as_default")]
    pub min_length: i64,
}

/// 提交 Bug 反馈结果（后端 BugFeedbackSubmitVO）
#[derive(Serialize, Deserialize, Debug, Clone, Default)]
#[serde(rename_all = "camelCase")]
pub struct BugFeedbackSubmitResult {
    #[serde(default, deserialize_with = "null_as_default")]
    pub id: i64,
    #[serde(default, deserialize_with = "null_as_default")]
    pub reward: i64,
    #[serde(default, deserialize_with = "null_as_default")]
    pub remaining_submit_count: i64,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct BugFeedbackSubmitRequest<'a> {
    content: &'a str,
    app_version: &'a str,
    os_info: Option<&'a str>,
}

/// 查询当前登录用户的 Bug 反馈配置（功能关闭时 enabled=false）
#[tauri::command]
pub async fn fetch_bug_feedback_info(app_handle: AppHandle) -> Result<BugFeedbackInfo, String> {
    let token = crate::qr_login::require_login()?;
    let url = format!(
        "{}/wx/xiuleme/bug/info",
        crate::app_config::api_base_url()
    );
    let info = crate::qr_login::authorized_get::<BugFeedbackInfo>(
        &app_handle,
        token,
        url,
        "Bug反馈配置",
    )
    .await?;
    Ok(info.unwrap_or_default())
}

/// 提交 Bug 反馈（仅入库，审核通过后发放积分）
#[tauri::command]
pub async fn submit_bug_feedback(
    content: String,
    app_handle: AppHandle,
) -> Result<BugFeedbackSubmitResult, String> {
    let content = content.trim().to_string();
    if content.is_empty() {
        return Err("问题描述不能为空".to_string());
    }

    let token = crate::qr_login::require_login()?;
    let url = format!(
        "{}/wx/xiuleme/bug/submit",
        crate::app_config::api_base_url()
    );
    let app_version = app_handle.package_info().version.to_string();
    let os_info = sysinfo::System::long_os_version();
    let body = BugFeedbackSubmitRequest {
        content: &content,
        app_version: &app_version,
        os_info: os_info.as_deref(),
    };
    crate::qr_login::authorized_post_json_data::<BugFeedbackSubmitResult, _>(
        &app_handle,
        &token,
        url,
        &body,
        "提交Bug反馈",
    )
    .await?
    .ok_or_else(|| "提交 Bug 反馈失败".to_string())
}
