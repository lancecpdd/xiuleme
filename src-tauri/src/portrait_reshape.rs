//! 重塑参数契约；不接入 GPU，也不改变旧人像效果的启用判定。
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::collections::HashSet;

/// 已移除的网格轮廓键；读参时清零／剥离，不再参与执行。
pub const KEYS: [&str; 4] = ["slimFace", "smallFace", "cheekbones", "jawline"];
/// 已移除的网格扩展键；读参时剥离。
/// 读参迁移时剥离；快照解码等遗留路径仍引用，恒为零。
pub fn read_extended_strengths(_face: Option<&Value>, _reshape_vis: bool) -> [f64; 8] {
    [0.0; 8]
}

pub const EXTENDED_KEYS: [&str; 8] = [
    "noseWidth",
    "noseLength",
    "mouthSize",
    "lipThickness",
    "eyeAperture",
    "eyeDistance",
    "forehead",
    "chinProjection",
];

/// 经典 UV 存储键；`uvSlimFace` 与历史网格 `slimFace` 隔离。
#[allow(dead_code)]
pub const UV_KEYS: [&str; 2] = ["uvSlimFace", "bigEye"];
/// 已下线的网格引擎；仅用于旧 `.rrdata` 读参迁移，不再执行。
pub const ENGINE_3DMM: &str = "3dmm-tddfa-v1";
pub const ENGINE_ARAP: &str = "mesh-arap-v1";
pub const ENGINE_UV: &str = "gpupixel-uv-v1";
/// 产品默认：原版 GPUPixel 动态库整图 bake。
pub const ENGINE_RUNTIME: &str = "gpupixel-runtime-v1";

/// 执行层支持 runtime bake；旧 UV／网格引擎在 `read_parameters` 中迁入。
#[allow(dead_code)]
pub fn is_supported_engine(engine: &str) -> bool {
    engine == ENGINE_RUNTIME || engine == ENGINE_UV
}

/// 可迁入 runtime 的引擎；删除 3DMM／ARAP 会让旧编辑的照片读参失败。
fn is_migratable_engine(engine: &str) -> bool {
    matches!(
        engine,
        ENGINE_RUNTIME | ENGINE_UV | ENGINE_3DMM | ENGINE_ARAP
    )
}

#[allow(dead_code)]
pub fn is_uv_engine(engine: &str) -> bool {
    engine == ENGINE_UV
}

#[allow(dead_code)]
pub fn is_runtime_engine(engine: &str) -> bool {
    engine == ENGINE_RUNTIME
}

pub fn reshape_section_visible(adjustments: &Value) -> bool {
    adjustments["sectionVisibility"]["portraitReshape"] != false
}

pub fn eyes_section_visible(adjustments: &Value) -> bool {
    adjustments["sectionVisibility"]["portraitEyes"] != false
}

pub fn effective_uv_slim(face: Option<&Value>, portrait_default: u32, reshape_visible: bool) -> u32 {
    if !reshape_visible {
        return 0;
    }
    face.and_then(|f| f["overrides"].get("uvSlimFace"))
        .map(amount)
        .unwrap_or(portrait_default)
}

pub fn effective_big_eye(face: Option<&Value>, portrait_default: u32, eyes_visible: bool) -> u32 {
    if !eyes_visible {
        return 0;
    }
    face.and_then(|f| f["overrides"].get("bigEye"))
        .map(amount)
        .unwrap_or(portrait_default)
}

#[allow(dead_code)]
#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(rename_all = "kebab-case")]
pub enum ReshapeReason {
    NoFace,
    ModelError,
    LowConfidence,
    UnsupportedPose,
    ProtectionLimited,
    UnmatchedFace,
}

#[allow(dead_code)]
#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(rename_all = "camelCase")]
pub struct ReshapeSourceAnchor {
    pub source_base_hash: String,
    pub bounds: [f64; 4],
}

#[allow(dead_code)]
#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(rename_all = "camelCase")]
pub struct ReshapeFaceSummary {
    pub face_id: String,
    pub source_anchor: ReshapeSourceAnchor,
    pub processable: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reason: Option<ReshapeReason>,
}

pub struct ReshapeReadResult {
    pub status: &'static str,
    // 未知／损坏版本保留原对象，只允许 supported 的值交给执行层。
    pub value: Value,
}

fn number(value: &Value, min: f64, max: f64) -> bool {
    value
        .as_f64()
        .is_some_and(|n| n.is_finite() && n >= min && n <= max)
}

fn hash(value: &Value) -> bool {
    value.as_str().is_some_and(|s| {
        s.len() == 64
            && s.bytes()
                .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
    })
}

fn id(value: &Value) -> bool {
    value.as_str().is_some_and(|s| {
        !s.is_empty()
            && s.len() <= 128
            && s.bytes()
                .all(|b| b.is_ascii_alphanumeric() || b"_.:-".contains(&b))
    })
}

fn vector(value: &Value, len: usize, min: f64, max: f64) -> bool {
    value
        .as_array()
        .is_some_and(|v| v.len() == len && v.iter().all(|n| number(n, min, max)))
}

pub fn amount(value: &Value) -> u32 {
    value
        .as_f64()
        .filter(|n| n.is_finite())
        .map(|n| n.clamp(0.0, 100.0).round() as u32)
        .unwrap_or(0)
}

pub fn normalize_strengths(value: &Value) -> Value {
    let mut result = serde_json::Map::new();
    for key in KEYS {
        result.insert(key.to_owned(), json!(amount(&value[key])));
    }
    Value::Object(result)
}

fn normalize_face(value: &Value) -> Option<Value> {
    let mut face = value.as_object()?.clone();
    if !id(&value["faceId"]) {
        return None;
    }
    let anchor = &value["sourceAnchor"];
    if !hash(&anchor["sourceBaseHash"]) || !vector(&anchor["bounds"], 4, 0.0, 1.0) {
        return None;
    }
    let b = &anchor["bounds"];
    let (x, y, w, h) = (
        b[0].as_f64()?,
        b[1].as_f64()?,
        b[2].as_f64()?,
        b[3].as_f64()?,
    );
    if w <= 0.0 || h <= 0.0 || x + w > 1.0 || y + h > 1.0 {
        return None;
    }
    let mut protection = match value.get("protection") {
        Some(p) => p.as_object()?.clone(),
        None => serde_json::Map::new(),
    };
    let strokes = match protection.get("strokes") {
        Some(s) => s.as_array()?.clone(),
        None => Vec::new(),
    };
    if strokes.len() > 32 {
        return None;
    }
    for stroke in &strokes {
        if !matches!(stroke["mode"].as_str(), Some("protect" | "erase"))
            || !number(&stroke["radius"], f64::from_bits(1), 1.0)
        {
            return None;
        }
        let points = stroke["points"].as_array()?;
        if points.is_empty() || points.len() > 256 || !points.iter().all(|p| vector(p, 2, 0.0, 1.0))
        {
            return None;
        }
    }
    if value.get("overrides").is_some_and(|v| !v.is_object()) {
        return None;
    }
    let mut overrides = serde_json::Map::new();
    if let Some(raw) = value["overrides"].as_object() {
        let slim = amount(raw.get("slimFace").unwrap_or(&Value::Null));
        let uv = amount(raw.get("uvSlimFace").unwrap_or(&Value::Null));
        let merged = slim.max(uv);
        if merged > 0 {
            overrides.insert("uvSlimFace".into(), json!(merged));
        }
        if let Some(v) = raw.get("bigEye") {
            overrides.insert("bigEye".into(), json!(amount(v)));
        }
    }
    let hair = protection.get("hair") != Some(&Value::Bool(false));
    let background = protection.get("background") != Some(&Value::Bool(false));
    protection.insert("hair".into(), json!(hair));
    protection.insert("background".into(), json!(background));
    protection.insert("strokes".into(), Value::Array(strokes));
    face.insert("protection".into(), Value::Object(protection));
    face.insert("overrides".into(), Value::Object(overrides));
    Some(Value::Object(face))
}

pub fn read_parameters(input: &Value) -> ReshapeReadResult {
    let result = |status, value| ReshapeReadResult { status, value };
    if input.is_null() {
        return result("legacy", Value::Null);
    }
    let Some(object) = input.as_object() else {
        return result("invalid", Value::Null);
    };
    if !object.contains_key("schemaVersion") {
        return result("legacy", input.clone());
    }
    let engine = input["engineVersion"].as_str().unwrap_or("");
    if input["schemaVersion"].as_f64() != Some(1.0) || !is_migratable_engine(engine) {
        return result("unsupported", input.clone());
    }
    let empty = Vec::new();
    let raw_faces = match object.get("faces") {
        None => &empty,
        Some(Value::Array(faces)) => faces,
        _ => return result("invalid", input.clone()),
    };
    if raw_faces.len() > 32 {
        return result("invalid", input.clone());
    }
    let mut faces = Vec::new();
    let mut ids = HashSet::new();
    for raw in raw_faces {
        let Some(face) = normalize_face(raw) else {
            return result("invalid", input.clone());
        };
        if !ids.insert(face["faceId"].as_str().unwrap_or_default().to_owned()) {
            return result("invalid", input.clone());
        }
        faces.push(face);
    }
    // 产品默认 runtime bake；旧 UV／网格读入后迁到 runtime。剥离网格快照。
    let mut value = json!({
        "schemaVersion": 1,
        "engineVersion": ENGINE_RUNTIME,
        "enabled": input["enabled"] == true,
        "faces": faces,
    });
    let _ = value.as_object_mut().map(|o| o.remove("snapshot"));
    result("supported", value)
}

/// 目标必须由当前照片、当前基底的匹配／质量门控结果提供；不使用 UI 选择。
pub fn request_enabled(adjustments: &Value, processable_face_ids: &[String]) -> bool {
    let parsed = read_parameters(&adjustments["portraitReshape"]);
    let reshape_vis = reshape_section_visible(adjustments);
    let eyes_vis = eyes_section_visible(adjustments);
    if parsed.status != "supported"
        || parsed.value["enabled"] != true
        || (!reshape_vis && !eyes_vis)
    {
        return false;
    }
    let uv_slim_default = amount(&adjustments["portrait"]["uvSlimFace"])
        .max(amount(&adjustments["portrait"]["slimFace"]));
    let big_eye_default = amount(&adjustments["portrait"]["bigEye"]);
    let Some(faces) = parsed.value["faces"].as_array() else {
        return false;
    };
    processable_face_ids.iter().any(|id| {
        let face = faces
            .iter()
            .find(|f| f["faceId"].as_str() == Some(id.as_str()));
        let slim = effective_uv_slim(face, uv_slim_default, reshape_vis);
        let big_eye = effective_big_eye(face, big_eye_default, eyes_vis);
        slim > 0 || big_eye > 0
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shared_json_contract() {
        let cases: Value =
            serde_json::from_str(include_str!("../../bench/portrait-reshape-cases.json")).unwrap();
        for case in cases.as_array().unwrap() {
            let parsed = read_parameters(&case["input"]["portraitReshape"]);
            assert_eq!(
                parsed.status,
                case["status"].as_str().unwrap(),
                "{}",
                case["name"]
            );
            assert_eq!(parsed.value, case["normalized"], "{}", case["name"]);
            let ids: Vec<String> = serde_json::from_value(case["targets"].clone()).unwrap();
            assert_eq!(
                request_enabled(&case["input"], &ids),
                case["enabled"].as_bool().unwrap(),
                "{}",
                case["name"]
            );
            assert_eq!(
                normalize_strengths(&case["input"]["portrait"]),
                case["strengths"],
                "{}",
                case["name"]
            );
        }
    }

    #[test]
    fn rejects_corrupt_protection_and_duplicate_identity() {
        let face = json!({"faceId":"face-1", "sourceAnchor":{"sourceBaseHash":"a".repeat(64),"bounds":[0.1,0.1,0.3,0.3]}});
        let mut config = json!({"schemaVersion":1,"engineVersion":"mesh-arap-v1","enabled":true,"faces":[face.clone(),face.clone()]});
        assert_eq!(read_parameters(&config).status, "invalid");
        config["faces"] = json!([face]);
        config["faces"][0]["protection"] =
            json!({"strokes":[{"mode":"protect","radius":0,"points":[[0.5,0.5]]}]});
        assert_eq!(read_parameters(&config).status, "invalid");
        assert_eq!(read_parameters(&config).value, config);
    }

    #[test]
    fn mesh_engine_migrates_to_uv_and_strips_snapshot() {
        let config = json!({
            "schemaVersion":1,"engineVersion":"mesh-arap-v1","enabled":true,
            "faces":[], "snapshot":{"version":"reshape-snapshot-v1","faces":[]}
        });
        let parsed = read_parameters(&config);
        assert_eq!(parsed.status, "supported");
        assert_eq!(parsed.value["engineVersion"], ENGINE_UV);
        assert!(parsed.value.get("snapshot").is_none());

        let mut unknown = json!({"schemaVersion":1,"engineVersion":"mesh-arap-v2","enabled":true});
        assert_eq!(read_parameters(&unknown).status, "unsupported");
        unknown["engineVersion"] = json!(ENGINE_UV);
        assert_eq!(read_parameters(&unknown).status, "supported");

        assert_eq!(amount(&json!("NaN")), 0);
        assert_eq!(amount(&json!(12.5)), 13);
    }
}
