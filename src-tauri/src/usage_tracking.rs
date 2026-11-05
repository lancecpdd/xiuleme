use serde::Serialize;
use tauri::AppHandle;

const MAX_ACTIVE_SECONDS_PER_REPORT: u32 = 60;

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct DesktopUsageHeartbeat {
    active_seconds: u32,
}

/// 上报当前登录用户的一段前台活跃时长。
#[tauri::command]
pub async fn report_desktop_usage(
    active_seconds: u32,
    app_handle: AppHandle,
) -> Result<(), String> {
    if active_seconds > MAX_ACTIVE_SECONDS_PER_REPORT {
        return Err(format!(
            "单次活跃秒数不能超过 {}",
            MAX_ACTIVE_SECONDS_PER_REPORT
        ));
    }

    let token = crate::qr_login::require_login()?;
    let url = format!(
        "{}/wx/xiuleme/usage/heartbeat",
        crate::app_config::api_base_url()
    );
    let heartbeat = DesktopUsageHeartbeat { active_seconds };

    crate::qr_login::authorized_post_json(&app_handle, &token, url, &heartbeat, "上报使用统计")
        .await
}
