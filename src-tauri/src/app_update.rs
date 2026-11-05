use serde::Serialize;

const LATEST_VERSION_CONFIG_KEY: &str = "app.xiuleme.latest_version";
const MIN_VERSION_CONFIG_KEY: &str = "app.xiuleme.min_version";

/// 服务端 sys_config 下发的版本策略；未配置或格式错误的字段为 None
#[derive(Serialize, Debug, Default)]
#[serde(rename_all = "camelCase")]
pub struct AppVersionPolicy {
    latest_version: Option<String>,
    min_version: Option<String>,
}

/// 规范化版本号：去掉首尾空格与 v/V 前缀，要求 1~4 段纯数字
fn normalize_version(value: Option<&str>) -> Option<String> {
    let value = value?.trim();
    let value = value
        .strip_prefix('v')
        .or_else(|| value.strip_prefix('V'))
        .unwrap_or(value);
    let segments: Vec<&str> = value.split('.').collect();
    let is_valid = (1..=4).contains(&segments.len())
        && segments
            .iter()
            .all(|segment| !segment.is_empty() && segment.chars().all(|c| c.is_ascii_digit()));
    is_valid.then(|| value.to_string())
}

/// 从 sys_config 读取桌面端最新版本号与最低可用版本号
#[tauri::command]
pub async fn fetch_app_version_policy() -> Result<AppVersionPolicy, String> {
    let configs = crate::sys_config::fetch_sys_config_batch(&[
        LATEST_VERSION_CONFIG_KEY,
        MIN_VERSION_CONFIG_KEY,
    ])
    .await?;
    let read = |key: &str| normalize_version(configs.get(key).and_then(|value| value.as_deref()));
    Ok(AppVersionPolicy {
        latest_version: read(LATEST_VERSION_CONFIG_KEY),
        min_version: read(MIN_VERSION_CONFIG_KEY),
    })
}

#[cfg(test)]
mod tests {
    use super::normalize_version;

    #[test]
    fn normalizes_valid_versions() {
        assert_eq!(
            normalize_version(Some(" v1.6.5 ")),
            Some("1.6.5".to_string())
        );
        assert_eq!(normalize_version(Some("V2")), Some("2".to_string()));
        assert_eq!(
            normalize_version(Some("1.2.3.4")),
            Some("1.2.3.4".to_string())
        );
    }

    #[test]
    fn rejects_invalid_versions() {
        assert_eq!(normalize_version(None), None);
        assert_eq!(normalize_version(Some("")), None);
        assert_eq!(normalize_version(Some("1..2")), None);
        assert_eq!(normalize_version(Some("1.6.5-beta")), None);
        assert_eq!(normalize_version(Some("1.2.3.4.5")), None);
    }
}
