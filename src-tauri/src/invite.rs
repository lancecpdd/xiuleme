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

/// 邀请好友信息（后端 InviteInfoVO）
#[derive(Serialize, Deserialize, Debug, Clone, Default)]
#[serde(rename_all = "camelCase")]
pub struct InviteInfo {
    #[serde(default, deserialize_with = "null_as_default")]
    pub enabled: bool,
    #[serde(default)]
    pub invite_code: Option<String>,
    #[serde(default, deserialize_with = "null_as_default")]
    pub invited_count: i64,
    #[serde(default, deserialize_with = "null_as_default")]
    pub total_earned: i64,
    #[serde(default, deserialize_with = "null_as_default")]
    pub inviter_reward: i64,
    #[serde(default, deserialize_with = "null_as_default")]
    pub invitee_reward: i64,
    #[serde(default, deserialize_with = "null_as_default")]
    pub inviter_limit: i64,
    #[serde(default, deserialize_with = "null_as_default")]
    pub bound: bool,
    #[serde(default)]
    pub inviter_nickname: Option<String>,
    #[serde(default)]
    pub download_url: Option<String>,
}

/// 填写邀请码结果（后端 InviteBindVO）
#[derive(Serialize, Deserialize, Debug, Clone, Default)]
#[serde(rename_all = "camelCase")]
pub struct InviteBindResult {
    #[serde(default, deserialize_with = "null_as_default")]
    pub reward: i64,
    #[serde(default, deserialize_with = "null_as_default")]
    pub balance: i64,
    #[serde(default)]
    pub inviter_nickname: Option<String>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct InviteBindRequest<'a> {
    invite_code: &'a str,
}

/// 查询当前登录用户的邀请信息（功能关闭时 enabled=false）
#[tauri::command]
pub async fn fetch_invite_info(app_handle: AppHandle) -> Result<InviteInfo, String> {
    let token = crate::qr_login::require_login()?;
    let url = format!(
        "{}/wx/xiuleme/invite/info",
        crate::app_config::api_base_url()
    );
    let info =
        crate::qr_login::authorized_get::<InviteInfo>(&app_handle, token, url, "邀请信息").await?;
    Ok(info.unwrap_or_default())
}

/// 填写好友邀请码，成功后双方获得像素积分
#[tauri::command]
pub async fn bind_invite_code(
    invite_code: String,
    app_handle: AppHandle,
) -> Result<InviteBindResult, String> {
    let invite_code = invite_code.trim().to_uppercase();
    if invite_code.is_empty() {
        return Err("邀请码不能为空".to_string());
    }

    let token = crate::qr_login::require_login()?;
    let url = format!(
        "{}/wx/xiuleme/invite/bind",
        crate::app_config::api_base_url()
    );
    let body = InviteBindRequest {
        invite_code: &invite_code,
    };
    crate::qr_login::authorized_post_json_data::<InviteBindResult, _>(
        &app_handle,
        &token,
        url,
        &body,
        "填写邀请码",
    )
    .await?
    .ok_or_else(|| "填写邀请码失败".to_string())
}
