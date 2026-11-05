// 错误码由前端映射为本地化文案，格式：CODE 或 CODE:附加信息
pub const ERR_NOT_LOGGED_IN: &str = "PIXEL_NOT_LOGGED_IN";
#[allow(dead_code)]
pub const ERR_SESSION_EXPIRED: &str = "PIXEL_SESSION_EXPIRED";

const OFFICIAL_ACCOUNT_CONFIG_KEY: &str = "app.about.follow.official.account";

fn official_account_qr_url(value: Option<&str>) -> Result<Option<String>, String> {
    let Some(value) = value.map(str::trim).filter(|value| !value.is_empty()) else {
        return Ok(None);
    };
    let url = reqwest::Url::parse(value).map_err(|_| "公众号二维码地址无效".to_string())?;
    if !matches!(url.scheme(), "http" | "https") || url.host_str().is_none() {
        return Err("公众号二维码地址必须为 HTTP(S) 图片地址".to_string());
    }
    Ok(Some(url.to_string()))
}

/// 复用小程序公开配置接口，从 sys_config 获取公众号二维码图片地址。
#[tauri::command]
pub async fn fetch_pixel_earn_qr_code() -> Result<Option<String>, String> {
    let configs = crate::sys_config::fetch_sys_config_batch(&[OFFICIAL_ACCOUNT_CONFIG_KEY]).await?;
    official_account_qr_url(
        configs
            .get(OFFICIAL_ACCOUNT_CONFIG_KEY)
            .and_then(|value| value.as_deref()),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn official_account_qr_accepts_http_images_and_trims_whitespace() {
        for value in [
            "https://example.com/qr.png",
            "http://example.com/qr.jpg?version=2",
        ] {
            assert_eq!(
                official_account_qr_url(Some(&format!(" {value} \n"))).unwrap(),
                Some(value.to_string())
            );
        }
    }

    #[test]
    fn official_account_qr_missing_or_blank_is_not_configured() {
        for value in [None, Some(""), Some(" \n ")] {
            assert_eq!(official_account_qr_url(value).unwrap(), None);
        }
    }

    #[test]
    fn official_account_qr_rejects_non_http_or_invalid_addresses() {
        for value in [
            "file:///qr.png",
            "data:image/png;base64,AAAA",
            "javascript:alert(1)",
            "not-a-url",
        ] {
            assert!(official_account_qr_url(Some(value)).is_err());
        }
    }
}
