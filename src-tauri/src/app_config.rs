use std::fs;
use std::path::PathBuf;

use serde::{Deserialize, Serialize};

/// 外部配置文件名称（随可执行文件分发，上线时可直接修改，无需重新编译）。
const CONFIG_FILE_NAME: &str = "config.json";

/// 环境变量：指定配置文件路径（优先级高于可执行文件同目录的 config.json）。
const CONFIG_PATH_ENV: &str = "XIULEME_CONFIG";

/// 请求头：标识调用来源。服务端目前只接受小程序类型 `miniapp`。
pub const SOURCE_CLIENT_HEADER: &str = "source_client";
pub const SOURCE_CLIENT_MINIAPP: &str = "miniapp";
/// 拍了么后端 API 基地址（例如 开发环境：http://192.168.31.66:8080，生产环境：https://api.paileya.com）。
fn default_api_base_url() -> String {
    "https://api.paileya.com".to_string()
}

/// 拍了么管理后台 API 基地址（公告等桌面端运营数据，例如 开发环境：http://192.168.31.66:8081，生产环境：https://admin.paileya.com/prod-api）。
fn default_admin_api_base_url() -> String {
    "https://admin.paileya.com/prod-api".to_string()
}

/// 应用全局运行时配置。
///
/// 读取顺序：
/// 1. 环境变量 `XIULEME_CONFIG` 指向的文件；
/// 2. 可执行文件同目录下的 `config.json`。
/// 文件不存在或解析失败时回退到默认值。
#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(rename_all = "camelCase")]
pub struct AppConfig {
    /// 拍了么后端 API 基地址（例如 开发环境：http://192.168.31.66:8080，生产环境：https://api.paileya.com）。
    #[serde(default = "default_api_base_url")]
    pub api_base_url: String,
    /// 拍了么管理后台 API 基地址（若后台经反向代理带路径前缀，需一并写入）。
    #[serde(default = "default_admin_api_base_url")]
    pub admin_api_base_url: String,
}

impl Default for AppConfig {
    fn default() -> Self {
        Self {
            api_base_url: default_api_base_url(),
            admin_api_base_url: default_admin_api_base_url(),
        }
    }
}

/// 解析配置文件路径（不保证文件存在）。
pub fn resolve_config_path() -> PathBuf {
    if let Ok(custom) = std::env::var(CONFIG_PATH_ENV) {
        let trimmed = custom.trim();
        if !trimmed.is_empty() {
            return PathBuf::from(trimmed);
        }
    }

    std::env::current_exe()
        .ok()
        .and_then(|exe| exe.parent().map(|dir| dir.join(CONFIG_FILE_NAME)))
        .unwrap_or_else(|| PathBuf::from(CONFIG_FILE_NAME))
}

/// 读取配置；文件缺失或解析失败时回退默认值。
pub fn load_config() -> AppConfig {
    let path = resolve_config_path();
    match fs::read_to_string(&path) {
        Ok(content) => serde_json::from_str(&content).unwrap_or_else(|e| {
            log::warn!("配置文件解析失败，改用默认配置：{} ({})", path.display(), e);
            AppConfig::default()
        }),
        Err(_) => AppConfig::default(),
    }
}

/// 获取拍了么后端 API 基地址。
pub fn api_base_url() -> String {
    load_config().api_base_url
}

/// 获取拍了么管理后台 API 基地址（去掉末尾斜杠）。
pub fn admin_api_base_url() -> String {
    load_config()
        .admin_api_base_url
        .trim_end_matches('/')
        .to_string()
}

/// 首次运行时在可执行文件同目录生成一份默认配置文件（失败仅记录日志，不阻断启动）。
pub fn ensure_default_config_file() {
    let path = resolve_config_path();
    if path.exists() {
        return;
    }

    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        let _ = fs::create_dir_all(parent);
    }

    if let Ok(json) = serde_json::to_string_pretty(&AppConfig::default()) {
        match fs::write(&path, format!("{}\n", json)) {
            Ok(_) => log::info!("已生成默认配置文件：{}", path.display()),
            Err(e) => log::warn!("生成默认配置文件失败：{} ({})", path.display(), e),
        }
    }
}
