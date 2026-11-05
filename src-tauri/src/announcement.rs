use std::time::Duration;

use regex::{Captures, Regex};
use serde::{Deserialize, Serialize};

const LEVEL_NORMAL: &str = "normal";
const LEVELS: [&str; 3] = [LEVEL_NORMAL, "important", "urgent"];

/// 拍了么管理后台统一响应包装 AjaxResult（code=200 成功）
#[derive(Deserialize, Debug)]
struct AjaxEnvelope<T> {
    code: i32,
    msg: Option<String>,
    data: Option<T>,
}

/// 管理后台 /xiuleme/announcement/public/list 返回的公告
#[derive(Deserialize, Debug)]
#[serde(rename_all = "camelCase")]
struct RemoteAnnouncement {
    id: i64,
    #[serde(default)]
    title: Option<String>,
    #[serde(default)]
    content: Option<String>,
    #[serde(default)]
    level: Option<String>,
    #[serde(default)]
    is_pinned: Option<i32>,
    #[serde(default)]
    published_at: Option<String>,
}

/// 下发给前端的公告（字段与 src/types/announcement.ts 对齐）
#[derive(Serialize, Debug, Clone)]
#[serde(rename_all = "camelCase")]
pub struct Announcement {
    pub id: String,
    pub title: String,
    pub content: String,
    pub level: String,
    pub is_pinned: bool,
    pub published_at: String,
}

/// 管理后台静态资源源站（去掉 /prod-api 等 API 前缀）。
/// 线上 nginx 用 `location /profile/` 直接托管上传文件；走 `/prod-api/profile/` 会被 Spring Security 拦成 401。
fn admin_asset_origin(admin_api_base_url: &str) -> String {
    let trimmed = admin_api_base_url.trim_end_matches('/');
    for prefix in ["/prod-api", "/dev-api", "/stage-api"] {
        if let Some(origin) = trimmed.strip_suffix(prefix) {
            return origin.trim_end_matches('/').to_string();
        }
    }
    trimmed.to_string()
}

/// 富文本中由管理后台上传的图片为相对路径（如 /prod-api/profile/upload/...），补全为可直连的绝对地址
fn resolve_profile_urls(content: &str, admin_api_base_url: &str) -> String {
    let origin = admin_asset_origin(admin_api_base_url);
    let re = Regex::new(r#"(src\s*=\s*["'])(?:/[^"'/]*)?/profile/"#).expect("valid regex");
    re.replace_all(content, |caps: &Captures| {
        format!("{}{}/profile/", &caps[1], origin)
    })
    .into_owned()
}

fn into_announcement(remote: RemoteAnnouncement, admin_api_base_url: &str) -> Announcement {
    let level = remote
        .level
        .filter(|level| LEVELS.contains(&level.as_str()))
        .unwrap_or_else(|| LEVEL_NORMAL.to_string());

    Announcement {
        id: remote.id.to_string(),
        title: remote.title.unwrap_or_default(),
        content: resolve_profile_urls(&remote.content.unwrap_or_default(), admin_api_base_url),
        level,
        is_pinned: remote.is_pinned.unwrap_or(0) == 1,
        // "yyyy-MM-dd HH:mm:ss" 转为 ISO 格式，保证各平台 WebView 的 Date 都能解析
        published_at: remote
            .published_at
            .unwrap_or_default()
            .replacen(' ', "T", 1),
    }
}

/// 获取拍了么管理后台发布的公告（免登录接口）
#[tauri::command]
pub async fn fetch_announcements() -> Result<Vec<Announcement>, String> {
    let base_url = crate::app_config::admin_api_base_url();
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(10))
        .build()
        .map_err(|e| format!("初始化网络请求失败: {}", e))?;
    let url = format!("{}/xiuleme/announcement/public/list", base_url);

    let response = client
        .get(url)
        .header("User-Agent", "XiuLeMe-App")
        .header(
            crate::app_config::SOURCE_CLIENT_HEADER,
            crate::app_config::SOURCE_CLIENT_MINIAPP,
        )
        .send()
        .await
        .map_err(|e| format!("无法连接拍了么管理后台: {}", e))?;

    let envelope: AjaxEnvelope<Vec<RemoteAnnouncement>> = response
        .json()
        .await
        .map_err(|e| format!("解析公告响应失败: {}", e))?;

    if envelope.code != 200 {
        return Err(envelope.msg.unwrap_or_else(|| "获取公告失败".to_string()));
    }

    Ok(envelope
        .data
        .unwrap_or_default()
        .into_iter()
        .map(|remote| into_announcement(remote, &base_url))
        .collect())
}

#[cfg(test)]
mod tests {
    use super::{admin_asset_origin, resolve_profile_urls};

    #[test]
    fn strips_prod_api_prefix_for_profile_assets() {
        assert_eq!(
            admin_asset_origin("https://admin.paileya.com/prod-api"),
            "https://admin.paileya.com"
        );
        assert_eq!(
            admin_asset_origin("https://admin.paileya.com/prod-api/"),
            "https://admin.paileya.com"
        );
        assert_eq!(
            admin_asset_origin("http://192.168.31.66:8081"),
            "http://192.168.31.66:8081"
        );
    }

    #[test]
    fn resolves_relative_profile_src_to_nginx_alias() {
        let html = r#"<p>公告<img src="/prod-api/profile/upload/2026/a.jpg"></p>"#;
        let out = resolve_profile_urls(html, "https://admin.paileya.com/prod-api");
        assert!(out.contains(r#"src="https://admin.paileya.com/profile/upload/2026/a.jpg""#));
        assert!(!out.contains("/prod-api/profile/"));
    }
}
