use std::collections::HashMap;
use std::time::Duration;

use crate::qr_login::ApiEnvelope;

/// 通过 paileme-server 公开接口 /api/system/config/batch 批量读取 sys_config 配置。
/// 未配置的 key 不会出现在返回的 Map 中，值也可能为 None。
pub async fn fetch_sys_config_batch(
    keys: &[&str],
) -> Result<HashMap<String, Option<String>>, String> {
    let base_url = crate::app_config::api_base_url();
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(10))
        .build()
        .map_err(|e| format!("初始化网络请求失败: {e}"))?;
    let url = reqwest::Url::parse_with_params(
        &format!("{}/api/system/config/batch", base_url.trim_end_matches('/')),
        &[("keys", keys.join(","))],
    )
    .map_err(|e| format!("系统配置接口地址无效: {e}"))?;
    let response = client
        .get(url)
        .header("User-Agent", "XiuLeMe-App")
        .header(
            crate::app_config::SOURCE_CLIENT_HEADER,
            crate::app_config::SOURCE_CLIENT_MINIAPP,
        )
        .send()
        .await
        .map_err(|e| format!("获取系统配置失败: {e}"))?
        .error_for_status()
        .map_err(|e| format!("系统配置接口请求失败: {e}"))?;
    let envelope: ApiEnvelope<HashMap<String, Option<String>>> = response
        .json()
        .await
        .map_err(|e| format!("解析系统配置失败: {e}"))?;
    if envelope.code != 1 {
        return Err(envelope
            .msg
            .unwrap_or_else(|| "获取系统配置失败".to_string()));
    }
    envelope.data.ok_or_else(|| "系统配置响应缺失".to_string())
}
