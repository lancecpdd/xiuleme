//! 11：场景级重塑快照。序列化冻结的分析输入（关键点、保护图、约束）与已接受网格，
//! 保证保存、关闭、重开后同一参数直接重现；参数改变时用快照输入重求解。
//! 概率图以"和保持"u8 量化后采用，解码前校验维度、展开上限、长度与哈希，禁止无界解压。
//! 快照只写 `.rrdata` sidecar，不写回源图；重放不重新推理模型、不重新检测人脸。
#![allow(dead_code)]
use crate::face_reshape::{
    LANDMARK_VERSION, MESH_TRIANGLES, MESH_VERTICES, PreparedFace, Strengths,
};
use crate::portrait_adapters::ProtectionMaps;
use crate::portrait_analysis::{AnalyzedFace, ReshapeAnalysis};
use crate::portrait_coordinates::{Affine, CoordinateMap};
use crate::portrait_protection::{
    LimitReason, OcclusionEvidence, ProtectedRegion, ProtectionPlan, QualityGate, quality_gate,
};
use crate::portrait_reshape::{
    KEYS, ReshapeFaceSummary, ReshapeReason, ReshapeSourceAnchor, normalize_strengths,
    read_parameters,
};
use crate::portrait_safety::{FaceRequest, PreparedSafety, SafeScene, SceneRequest};
use base64::{Engine as _, engine::general_purpose::STANDARD as BASE64};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};

pub const SNAPSHOT_VERSION: &str = "reshape-snapshot-v1";
const PARAMETER_HASH_VERSION: &str = "reshape-params-ext-v1";
const MAX_FACES: usize = 32;
const MAX_MAP_SIDE: usize = 256;
const MAX_ENCODED_RUNS: usize = 65536;
const MAX_RAW_BYTES: usize = 4 * MAX_MAP_SIDE * MAX_MAP_SIDE;
// RLE 每条 6 字节（count-1 小端 u16 + RGBA 四通道），条数与展开量都有界。
const MAX_RLE_BYTES: usize = 6 * MAX_ENCODED_RUNS;
// base64(393216 字节) 的字符上限；解码前先按该长度拒绝，避免无界分配。
const MAX_MAP_BASE64: usize = MAX_RLE_BYTES.div_ceil(3) * 4;

/// 保护计划的固定集与软约束权重；`SafeScene` 报告只保留区域，不含这两项。
pub struct FrozenPlan {
    pub fixed: Vec<bool>,
    pub weights: Vec<f64>,
}

fn number(value: &Value, min: f64, max: f64) -> bool {
    value
        .as_f64()
        .is_some_and(|n| n.is_finite() && n >= min && n <= max)
}

fn integer(value: &Value, min: f64, max: f64) -> bool {
    number(value, min, max) && value.as_f64().is_some_and(|n| n.fract() == 0.0)
}

fn hash(value: &Value) -> bool {
    value.as_str().is_some_and(hash_string)
}

fn hash_string(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
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

fn valid_reason(value: &Value) -> bool {
    match value {
        Value::Null => true,
        other => serde_json::from_value::<ReshapeReason>(other.clone()).is_ok(),
    }
}

// LimitReason 位掩码；位序与枚举声明一致，未知位一律拒绝。
const LIMIT_REASONS: [LimitReason; 16] = [
    LimitReason::Hair,
    LimitReason::Background,
    LimitReason::UncertainSegmentation,
    LimitReason::UserProtection,
    LimitReason::BackgroundLine,
    LimitReason::ImageBoundary,
    LimitReason::Overlap,
    LimitReason::GroupQualityLimited,
    LimitReason::LowConfidence,
    LimitReason::PoseLimited,
    LimitReason::UnsupportedPose,
    LimitReason::OcclusionUnverified,
    LimitReason::KnownOcclusion,
    LimitReason::AnalysisRejected,
    LimitReason::SafetyReduced,
    LimitReason::SafetyBypass,
];
const REASONS_MASK_MAX: f64 = (1u32 << LIMIT_REASONS.len()) as f64 - 1.0;

fn reasons_mask(reasons: &BTreeSet<LimitReason>) -> u32 {
    LIMIT_REASONS
        .iter()
        .enumerate()
        .filter(|(_, reason)| reasons.contains(*reason))
        .fold(0, |mask, (bit, _)| mask | (1u32 << bit))
}

fn reasons_from_mask(mask: u32) -> BTreeSet<LimitReason> {
    LIMIT_REASONS
        .iter()
        .enumerate()
        .filter(|(bit, _)| mask & (1 << bit) != 0)
        .map(|(_, reason)| *reason)
        .collect()
}

fn valid_mask(value: &Value) -> bool {
    integer(value, 0.0, REASONS_MASK_MAX)
}

fn valid_map(value: &Value) -> bool {
    let encoding = value["encoding"].as_str();
    if !matches!(encoding, Some("rle-u8-v1" | "raw-u8-v1")) {
        return false;
    }
    if !integer(&value["width"], 1.0, MAX_MAP_SIDE as f64)
        || !integer(&value["height"], 1.0, MAX_MAP_SIDE as f64)
        || !hash(&value["sha256"])
    {
        return false;
    }
    let Some(data) = value["data"].as_str() else {
        return false;
    };
    let width = value["width"].as_f64().unwrap_or(0.0) as usize;
    let height = value["height"].as_f64().unwrap_or(0.0) as usize;
    let limit = match encoding {
        Some("raw-u8-v1") => (4 * width * height).div_ceil(3) * 4,
        // RLE 每条 6 字节；展开量在解码时再校验，这里只限制包装长度。
        _ => MAX_MAP_BASE64,
    };
    data.len() <= limit
}

fn valid_face(value: &Value) -> bool {
    if !id(&value["faceId"]) || !vector(&value["bounds"], 4, 0.0, 1.0) {
        return false;
    }
    let b = &value["bounds"];
    let (x, y, w, h) = (
        b[0].as_f64().unwrap_or(0.0),
        b[1].as_f64().unwrap_or(0.0),
        b[2].as_f64().unwrap_or(0.0),
        b[3].as_f64().unwrap_or(0.0),
    );
    if w <= 0.0 || h <= 0.0 || x + w > 1.0 || y + h > 1.0 {
        return false;
    }
    // roi 直接存 Affine::roi 的 6 个矩阵元素，重放位相同，不从角度／边长反推。
    if !vector(&value["roiMatrix"], 6, -1e9, 1e9) {
        return false;
    }
    if !number(&value["detectionConfidence"], 0.0, 1.0)
        || !number(&value["landmarkConfidence"], 0.0, 1.0)
        || !value["refined"].is_boolean()
        || !value["processable"].is_boolean()
        || !valid_reason(&value["reason"])
    {
        return false;
    }
    if !value["landmarks"]
        .as_array()
        .is_some_and(|points| points.len() == 478 && points.iter().all(|p| vector(p, 3, -4.0, 4.0)))
    {
        return false;
    }
    if !value["localProtectionMap"].is_null() && !valid_map(&value["localProtectionMap"]) {
        return false;
    }
    let plan = &value["plan"];
    let Some(fixed) = plan["fixed"].as_str() else {
        return false;
    };
    if fixed.len() != MESH_VERTICES || !fixed.bytes().all(|b| b == b'0' || b == b'1') {
        return false;
    }
    let Some(weights) = plan["weights"].as_array() else {
        return false;
    };
    if weights.len() > MESH_VERTICES {
        return false;
    }
    let mut previous = None;
    for entry in weights {
        if entry.as_array().is_none_or(|v| v.len() != 2)
            || entry[0].as_u64().is_none_or(|i| i as usize >= MESH_VERTICES)
            // 与 06 软约束契约一致：权重必须落在 (0, 10000]。
            || entry[1]
                .as_f64()
                .is_none_or(|w| !w.is_finite() || w <= 0.0 || w > 10000.0)
        {
            return false;
        }
        let index = entry[0].as_u64().unwrap_or(0);
        if previous.is_some_and(|p| p >= index) {
            return false;
        }
        previous = Some(index);
    }
    let Some(regions) = plan["regions"].as_array() else {
        return false;
    };
    if regions.len() > MESH_TRIANGLES {
        return false;
    }
    let mut previous = None;
    for region in regions {
        if region.as_array().is_none_or(|v| v.len() != 4)
            || region[0]
                .as_u64()
                .is_none_or(|t| t as usize >= MESH_TRIANGLES)
            || !number(&region[1], 0.0, 1.0)
            || !region[2].is_boolean()
            || !valid_mask(&region[3])
        {
            return false;
        }
        let triangle = region[0].as_u64().unwrap_or(0);
        if previous.is_some_and(|p| p >= triangle) {
            return false;
        }
        previous = Some(triangle);
    }
    let mesh = &value["mesh"];
    if !mesh["target"]
        .as_array()
        .is_some_and(|t| t.len() == MESH_VERTICES && t.iter().all(|p| vector(p, 2, -1e9, 1e9)))
        || !vector(&mesh["requested"], 4, 0.0, 100.0)
        || !vector(&mesh["effective"], 4, 0.0, 1.0)
    {
        return false;
    }
    if !number(&value["safetyScale"], 0.0, 1.0)
        || !integer(&value["backoffs"], 0.0, 16.0)
        || !integer(&value["detectedLines"], 0.0, 32.0)
        || !valid_mask(&value["reasons"])
    {
        return false;
    }
    true
}

/// 静态格式闸：只做有界结构检查，不解码大数组；损坏或超大快照在此拒绝。
pub fn valid_snapshot(value: &Value) -> bool {
    if !value.is_object()
        || value["version"] != SNAPSHOT_VERSION
        || !hash(&value["sourceBaseHash"])
        || !hash(&value["analysisKey"])
        || !hash(&value["parameterHash"])
        || value["meshVersion"] != LANDMARK_VERSION
    {
        return false;
    }
    let Some(hashes) = value["modelHashes"].as_object() else {
        return false;
    };
    if hashes.is_empty()
        || hashes.len() > 8
        || !hashes
            .iter()
            .all(|(key, value)| id(&json!(key)) && hash(value))
    {
        return false;
    }
    let Some(dimensions) = value["baseDimensions"].as_array() else {
        return false;
    };
    if dimensions.len() != 2
        || !integer(&dimensions[0], 1.0, 32768.0)
        || !integer(&dimensions[1], 1.0, 32768.0)
    {
        return false;
    }
    let (w, h) = (
        dimensions[0].as_f64().unwrap_or(0.0),
        dimensions[1].as_f64().unwrap_or(0.0),
    );
    if w * h > 64_000_000.0 {
        return false;
    }
    if !value["protectionMap"].is_null() && !valid_map(&value["protectionMap"]) {
        return false;
    }
    let Some(faces) = value["faces"].as_array() else {
        return false;
    };
    if faces.len() > MAX_FACES {
        return false;
    }
    let mut ids = BTreeSet::new();
    let mut previous_id: Option<&str> = None;
    for face in faces {
        if !valid_face(face) {
            return false;
        }
        let current = face["faceId"].as_str().unwrap_or_default();
        if previous_id.is_some_and(|p| p >= current) {
            return false;
        }
        previous_id = Some(current);
        if !ids.insert(current) {
            return false;
        }
    }
    let Some(groups) = value["groups"].as_array() else {
        return false;
    };
    if groups.len() > MAX_FACES {
        return false;
    }
    // 组必须构成 faces 的一次划分：每个索引恰好出现一次、组内严格升序。
    let mut seen = vec![false; faces.len()];
    for group in groups {
        let Some(indices) = group.as_array() else {
            return false;
        };
        if indices.is_empty() || indices.len() > faces.len() {
            return false;
        }
        let mut previous: Option<usize> = None;
        for index in indices {
            let Some(i) = index.as_u64() else {
                return false;
            };
            let i = i as usize;
            if i >= faces.len() || seen[i] || previous.is_some_and(|p| p >= i) {
                return false;
            }
            seen[i] = true;
            previous = Some(i);
        }
    }
    seen.iter().all(|s| *s)
}

/// 和保持 u8 量化：每像素四通道整数和精确等于 255，解码 v/255 后概率和保持 1。
/// 朴素四舍五入会破坏 `valid_protection` 的 1e-4 和校验，因此采用最大余数法分配。
pub fn quantize_map(map: &ProtectionMaps) -> Option<ProtectionMaps> {
    if map.width == 0 || map.height == 0 || map.probabilities.len() != map.width * map.height {
        return None;
    }
    let mut probabilities = Vec::with_capacity(map.probabilities.len());
    for pixel in &map.probabilities {
        if pixel
            .iter()
            .any(|v| !v.is_finite() || !(0.0..=1.0).contains(v))
            || (pixel.iter().sum::<f32>() - 1.0).abs() > 1e-4
        {
            return None;
        }
        let scaled: [f64; 4] = std::array::from_fn(|c| pixel[c] as f64 * 255.0);
        let mut quantized: [u8; 4] =
            std::array::from_fn(|c| scaled[c].floor().clamp(0.0, 255.0) as u8);
        let sum: i32 = quantized.iter().map(|v| i32::from(*v)).sum();
        let need = 255 - sum;
        if !(0..=4).contains(&need) {
            return None;
        }
        let mut order: Vec<usize> = (0..4).collect();
        let fraction = |c: usize| scaled[c] - scaled[c].floor();
        order.sort_by(|a, b| fraction(*b).total_cmp(&fraction(*a)).then(a.cmp(b)));
        for &c in order.iter().take(need as usize) {
            if quantized[c] == 255 {
                return None;
            }
            quantized[c] += 1;
        }
        probabilities.push(std::array::from_fn(|c| f32::from(quantized[c]) / 255.0));
    }
    Some(ProtectionMaps {
        width: map.width,
        height: map.height,
        probabilities,
    })
}

/// 首次采用：把分析的保护图切换为和保持量化版本；这是渲染与快照共用的唯一输入。
/// 失败返回 None（保留原图与缓存对象，不污染分析缓存）。
pub fn quantized_analysis(analysis: ReshapeAnalysis) -> Option<ReshapeAnalysis> {
    let protection = match &analysis.protection {
        Some(map) => Some(quantize_map(map)?),
        None => None,
    };
    let mut faces = Vec::with_capacity(analysis.faces.len());
    for face in analysis.faces {
        let local = match &face.protection {
            Some(map) => Some(quantize_map(map)?),
            None => None,
        };
        faces.push(AnalyzedFace {
            summary: face.summary,
            roi_to_base: face.roi_to_base,
            roi_points: face.roi_points,
            source_points: face.source_points,
            detection_confidence: face.detection_confidence,
            landmark_confidence: face.landmark_confidence,
            refined: face.refined,
            protection: local,
        });
    }
    Some(ReshapeAnalysis {
        key: analysis.key,
        source_base_hash: analysis.source_base_hash,
        model_hashes: analysis.model_hashes,
        coordinates: analysis.coordinates,
        coarse_dimensions: analysis.coarse_dimensions,
        faces,
        protection,
        rejected_faces: analysis.rejected_faces,
    })
}

fn digest_bytes(bytes: &[u8]) -> String {
    hex::encode(Sha256::digest(bytes))
}

/// 展开像素 → 6 字节/条 RLE（count-1 小端 u16 + RGBA）；相邻相同四元组合并。
fn rle_encode(pixels: &[[u8; 4]]) -> Vec<u8> {
    let mut bytes = Vec::new();
    let mut run: Option<([u8; 4], usize)> = None;
    let flush = |bytes: &mut Vec<u8>, value: [u8; 4], count: usize| {
        let count = (count as u64 - 1) as u16;
        bytes.extend_from_slice(&[count.to_le_bytes()[0], count.to_le_bytes()[1]]);
        bytes.extend_from_slice(&value);
    };
    for pixel in pixels {
        match &mut run {
            Some((value, count)) if *value == *pixel && *count < MAX_ENCODED_RUNS => *count += 1,
            Some((value, count)) => {
                flush(&mut bytes, *value, *count);
                *value = *pixel;
                *count = 1;
            }
            None => run = Some((*pixel, 1)),
        }
    }
    if let Some((value, count)) = run {
        flush(&mut bytes, value, count);
    }
    bytes
}

/// RLE 解码：长度与展开量先行校验，不做无界解压。
fn rle_decode(bytes: &[u8], pixels: usize) -> Option<Vec<[u8; 4]>> {
    if bytes.len() % 6 != 0 || bytes.len() / 6 > MAX_ENCODED_RUNS {
        return None;
    }
    let mut output = Vec::with_capacity(pixels);
    for run in bytes.chunks_exact(6) {
        let count = usize::from(u16::from_le_bytes([run[0], run[1]])) + 1;
        if output.len() + count > pixels {
            return None;
        }
        output.resize(output.len() + count, [run[2], run[3], run[4], run[5]]);
    }
    (output.len() == pixels).then_some(output)
}

/// 量化图 → 保护图 JSON：二选一编码后取更小者；哈希覆盖解码后的字节流。
fn encode_map(map: &ProtectionMaps) -> Result<Value, String> {
    if map.width == 0 || map.height == 0 || map.probabilities.len() != map.width * map.height {
        return Err("保护图维度与像素数不一致".into());
    }
    let mut quantized = Vec::with_capacity(map.probabilities.len());
    for pixel in &map.probabilities {
        let mut q = [0u8; 4];
        for c in 0..4 {
            let value = pixel[c];
            if !value.is_finite() || !(0.0..=1.0).contains(&value) {
                return Err("保护图包含非法概率".into());
            }
            q[c] = (f64::from(value) * 255.0).round().clamp(0.0, 255.0) as u8;
        }
        if q.iter().map(|v| u32::from(*v)).sum::<u32>() != 255 {
            return Err("保护图不是和保持量化输入，拒绝写快照".into());
        }
        quantized.push(q);
    }
    let raw: Vec<u8> = quantized.iter().flatten().copied().collect();
    let runs = rle_encode(&quantized);
    let (encoding, bytes) = if runs.len() < raw.len() {
        ("rle-u8-v1", runs)
    } else {
        ("raw-u8-v1", raw)
    };
    Ok(json!({
        "encoding": encoding,
        "width": map.width,
        "height": map.height,
        "sha256": digest_bytes(&bytes),
        "data": BASE64.encode(&bytes),
    }))
}

/// 保护图 JSON → 概率图：base64 长度 → 解码长度 → 哈希 → 展开量 → 逐像素和校验。
fn decode_map(value: &Value) -> Result<ProtectionMaps, String> {
    if !valid_map(value) {
        return Err("保护图编码非法".into());
    }
    let width = value["width"].as_f64().unwrap_or(0.0) as usize;
    let height = value["height"].as_f64().unwrap_or(0.0) as usize;
    let pixels = width * height;
    let data = value["data"].as_str().unwrap_or_default();
    let bytes = BASE64
        .decode(data)
        .map_err(|_| "保护图 base64 解码失败".to_string())?;
    let encoding = value["encoding"].as_str().unwrap_or_default();
    let quantized = match encoding {
        "raw-u8-v1" => {
            if bytes.len() != pixels * 4 {
                return Err("保护图原始数组长度与维度不符".into());
            }
            bytes
                .chunks_exact(4)
                .map(|p| [p[0], p[1], p[2], p[3]])
                .collect::<Vec<_>>()
        }
        _ => {
            if bytes.len() > MAX_RLE_BYTES {
                return Err("保护图 RLE 超出展开上限".into());
            }
            rle_decode(&bytes, pixels).ok_or("保护图 RLE 展开量不合法")?
        }
    };
    if digest_bytes(&bytes) != value["sha256"].as_str().unwrap_or_default() {
        return Err("保护图内容哈希不匹配".into());
    }
    let mut probabilities = Vec::with_capacity(pixels);
    for pixel in &quantized {
        let sum: u32 = pixel.iter().map(|v| u32::from(*v)).sum();
        if sum != 255 {
            return Err("保护图量化和不满足和保持约束".into());
        }
        probabilities.push(std::array::from_fn(|c| f32::from(pixel[c]) / 255.0));
    }
    Ok(ProtectionMaps {
        width,
        height,
        probabilities,
    })
}

pub struct SnapshotFace {
    pub face_id: String,
    pub bounds: [f64; 4],
    pub roi_matrix: [f64; 6],
    pub detection_confidence: f32,
    pub landmark_confidence: f32,
    pub refined: bool,
    pub processable: bool,
    pub reason: Option<ReshapeReason>,
    pub landmarks: Vec<[f32; 3]>,
    pub local_protection: Option<ProtectionMaps>,
    pub fixed: Vec<bool>,
    pub weights: Vec<f64>,
    pub regions: Vec<ProtectedRegion>,
    pub target: Vec<[f64; 2]>,
    pub requested: [f64; 4],
    pub effective_response: [f64; 4],
    pub safety_scale: f64,
    pub backoffs: usize,
    pub detected_lines: usize,
    pub reasons: BTreeSet<LimitReason>,
}

pub struct Snapshot {
    pub source_base_hash: String,
    pub analysis_key: String,
    pub model_hashes: BTreeMap<String, String>,
    pub base_dimensions: [u32; 2],
    pub parameter_hash: String,
    pub protection: Option<ProtectionMaps>,
    pub groups: Vec<Vec<usize>>,
    pub faces: Vec<SnapshotFace>,
}

/// 轻量绑定查询：不解析数组，供重放前的源／分析键比对使用。
pub fn source_hash(value: &Value) -> Option<&str> {
    value["sourceBaseHash"].as_str()
}

pub fn analysis_key_of(value: &Value) -> Option<&str> {
    value["analysisKey"].as_str()
}

pub fn parameter_hash_of(value: &Value) -> Option<&str> {
    value["parameterHash"].as_str()
}

pub fn model_hashes_of(value: &Value) -> Option<BTreeMap<String, String>> {
    value["modelHashes"].as_object().map(|hashes| {
        hashes
            .iter()
            .map(|(key, value)| (key.clone(), value.as_str().unwrap_or_default().to_owned()))
            .collect()
    })
}

fn face_f64(value: &Value, index: usize) -> f64 {
    value[index].as_f64().unwrap_or(0.0)
}

/// 完整解析：格式闸之后解码保护图与全部数组；任何损坏在有界范围内拒绝。
pub fn parse(value: &Value) -> Result<Snapshot, String> {
    if !valid_snapshot(value) {
        return Err("重塑快照格式非法或超出上限".into());
    }
    let protection = if value["protectionMap"].is_null() {
        None
    } else {
        Some(decode_map(&value["protectionMap"])?)
    };
    let model_hashes = model_hashes_of(value).ok_or("重塑快照缺少模型哈希")?;
    let dimensions = &value["baseDimensions"];
    let mut faces = Vec::new();
    for face in value["faces"].as_array().ok_or("重塑快照缺少脸列表")? {
        let local_protection = if face["localProtectionMap"].is_null() {
            None
        } else {
            Some(decode_map(&face["localProtectionMap"])?)
        };
        let plan = &face["plan"];
        let fixed: Vec<bool> = plan["fixed"]
            .as_str()
            .unwrap_or_default()
            .bytes()
            .map(|b| b == b'1')
            .collect();
        let mut weights = vec![0.0; MESH_VERTICES];
        for entry in plan["weights"].as_array().ok_or("重塑快照缺少软约束")? {
            let index = entry[0].as_u64().unwrap_or(0) as usize;
            weights[index] = entry[1].as_f64().unwrap_or(0.0);
        }
        let mut regions = Vec::new();
        for region in plan["regions"].as_array().ok_or("重塑快照缺少保护区域")? {
            regions.push(ProtectedRegion {
                triangle: region[0].as_u64().unwrap_or(0) as usize,
                strength: face_f64(region, 1),
                hard: region[2].as_bool().unwrap_or(false),
                reasons: reasons_from_mask(region[3].as_u64().unwrap_or(0) as u32),
            });
        }
        let mesh = &face["mesh"];
        let target = mesh["target"]
            .as_array()
            .ok_or("重塑快照缺少网格")?
            .iter()
            .map(|p| [face_f64(p, 0), face_f64(p, 1)])
            .collect();
        faces.push(SnapshotFace {
            face_id: face["faceId"].as_str().unwrap_or_default().to_owned(),
            bounds: std::array::from_fn(|i| face_f64(&face["bounds"], i)),
            roi_matrix: std::array::from_fn(|i| face_f64(&face["roiMatrix"], i)),
            detection_confidence: face["detectionConfidence"].as_f64().unwrap_or(0.0) as f32,
            landmark_confidence: face["landmarkConfidence"].as_f64().unwrap_or(0.0) as f32,
            refined: face["refined"].as_bool().unwrap_or(false),
            processable: face["processable"].as_bool().unwrap_or(false),
            reason: serde_json::from_value(face["reason"].clone()).ok(),
            landmarks: face["landmarks"]
                .as_array()
                .ok_or("重塑快照缺少关键点")?
                .iter()
                .map(|p| std::array::from_fn(|c| p[c].as_f64().unwrap_or(0.0) as f32))
                .collect(),
            local_protection,
            fixed,
            weights,
            regions,
            target,
            requested: std::array::from_fn(|i| face_f64(&mesh["requested"], i)),
            effective_response: std::array::from_fn(|i| face_f64(&mesh["effective"], i)),
            safety_scale: face["safetyScale"].as_f64().unwrap_or(1.0),
            backoffs: face["backoffs"].as_u64().unwrap_or(0) as usize,
            detected_lines: face["detectedLines"].as_u64().unwrap_or(0) as usize,
            reasons: reasons_from_mask(face["reasons"].as_u64().unwrap_or(0) as u32),
        });
    }
    let groups = value["groups"]
        .as_array()
        .ok_or("重塑快照缺少分组")?
        .iter()
        .map(|group| {
            group
                .as_array()
                .map(|indices| {
                    indices
                        .iter()
                        .map(|i| i.as_u64().unwrap_or(0) as usize)
                        .collect()
                })
                .unwrap_or_default()
        })
        .collect();
    Ok(Snapshot {
        source_base_hash: value["sourceBaseHash"]
            .as_str()
            .unwrap_or_default()
            .to_owned(),
        analysis_key: value["analysisKey"].as_str().unwrap_or_default().to_owned(),
        model_hashes,
        base_dimensions: [
            dimensions[0].as_u64().unwrap_or(0) as u32,
            dimensions[1].as_u64().unwrap_or(0) as u32,
        ],
        parameter_hash: value["parameterHash"]
            .as_str()
            .unwrap_or_default()
            .to_owned(),
        protection,
        groups,
        faces,
    })
}

/// 参数哈希：绑定分析键、逐脸轮廓／扩展强度与全部引擎／拓扑／安全版本。
/// 求解结果完全由这些输入决定，与求解时刻无关，因此重开时可安全比对。
pub fn parameter_hash(analysis_key: &str, faces: &[(String, [f64; 4], [f64; 8])]) -> String {
    let faces: Vec<Value> = faces
        .iter()
        .map(|(id, strengths, extended)| json!([id, strengths, extended]))
        .collect();
    crate::portrait_render::content_key(&json!([
        PARAMETER_HASH_VERSION,
        analysis_key,
        faces,
        "unknown-occlusion",
        crate::face_reshape::ENGINE_VERSION,
        crate::face_reshape::TOPOLOGY_VERSION,
        LANDMARK_VERSION,
        "default-arap-pcg-v1",
        crate::portrait_safety::SAFETY_VERSION,
        crate::portrait_tiles::FIELD_VERSION
    ]))
}

fn fixed_string(fixed: &[bool]) -> String {
    fixed.iter().map(|v| if *v { '1' } else { '0' }).collect()
}

fn sparse_weights(weights: &[f64]) -> Vec<Value> {
    weights
        .iter()
        .enumerate()
        .filter(|(_, w)| **w > 0.0)
        .map(|(i, w)| json!([i, w]))
        .collect()
}

fn regions_value(regions: &[ProtectedRegion]) -> Vec<Value> {
    let mut sorted = regions.iter().collect::<Vec<_>>();
    sorted.sort_by_key(|r| r.triangle);
    sorted
        .into_iter()
        .map(|r| json!([r.triangle, r.strength, r.hard, reasons_mask(&r.reasons)]))
        .collect()
}

/// 从冻结资产构建快照：分析（量化版）＋已接受场景＋求解输入（fixed／weights／groups）。
pub fn build_snapshot(
    analysis: &ReshapeAnalysis,
    scene: &SafeScene,
    plans: &BTreeMap<String, FrozenPlan>,
    groups: &[Vec<usize>],
    parameter_hash: &str,
) -> Result<Value, String> {
    if !hash_string(parameter_hash) {
        return Err("参数哈希非法".into());
    }
    if analysis.key != scene.analysis_key() {
        return Err("冻结场与分析键不一致".into());
    }
    let reports: Vec<_> = scene.reports().collect();
    let meshes: Vec<_> = scene.meshes().collect();
    if reports.len() != meshes.len() || reports.len() > MAX_FACES {
        return Err("冻结场脸数量非法".into());
    }
    let by_id: BTreeMap<&str, &AnalyzedFace> = analysis
        .faces
        .iter()
        .map(|face| (face.summary.face_id.as_str(), face))
        .collect();
    let protection = match &analysis.protection {
        Some(map) => encode_map(map)?,
        None => Value::Null,
    };
    let mut faces = Vec::with_capacity(reports.len());
    for (report, mesh) in reports.iter().zip(&meshes) {
        let face = by_id
            .get(report.face_id.as_str())
            .ok_or("冻结脸在分析中缺少来源")?;
        let plan = plans.get(&report.face_id).ok_or("冻结脸缺少保护计划")?;
        if plan.fixed.len() != MESH_VERTICES || plan.weights.len() != MESH_VERTICES {
            return Err("保护计划尺寸与网格不符".into());
        }
        if mesh.target_vertices().len() != MESH_VERTICES || mesh.triangles().len() != MESH_TRIANGLES
        {
            return Err("冻结网格与拓扑版本不符".into());
        }
        let local_protection = match &face.protection {
            Some(map) => encode_map(map)?,
            None => Value::Null,
        };
        let m = &face.roi_to_base.0;
        faces.push(json!({
            "faceId": report.face_id,
            "bounds": face.summary.source_anchor.bounds,
            "roiMatrix": [m[(0, 0)], m[(0, 1)], m[(0, 2)], m[(1, 0)], m[(1, 1)], m[(1, 2)]],
            "detectionConfidence": face.detection_confidence,
            "landmarkConfidence": face.landmark_confidence,
            "refined": face.refined,
            "processable": face.summary.processable,
            "reason": face.summary.reason,
            "landmarks": face.roi_points,
            "localProtectionMap": local_protection,
            "plan": {
                "fixed": fixed_string(&plan.fixed),
                "weights": sparse_weights(&plan.weights),
                "regions": regions_value(&report.regions),
            },
            "mesh": {
                "target": mesh.target_vertices(),
                "requested": report.requested.0,
                "effective": mesh.effective_response,
            },
            "safetyScale": report.safety_scale,
            "backoffs": report.backoffs,
            "detectedLines": report.detected_lines,
            "reasons": reasons_mask(&report.reasons),
        }));
    }
    let snapshot = json!({
        "version": SNAPSHOT_VERSION,
        "sourceBaseHash": analysis.source_base_hash,
        "analysisKey": analysis.key,
        "modelHashes": analysis.model_hashes,
        "meshVersion": LANDMARK_VERSION,
        "baseDimensions": [analysis.coordinates.base_dimensions[0], analysis.coordinates.base_dimensions[1]],
        "parameterHash": parameter_hash,
        "protectionMap": protection,
        "groups": groups,
        "faces": faces,
    });
    if !valid_snapshot(&snapshot) {
        return Err("构建的重塑快照未通过格式校验".into());
    }
    Ok(snapshot)
}

/// 从快照恢复分析记录（量化版）：坐标映射由调用方用当前基底重建并已比对分析键。
pub fn rebuild_analysis(snapshot: &Snapshot, coordinates: CoordinateMap) -> ReshapeAnalysis {
    let faces = snapshot
        .faces
        .iter()
        .map(|face| analyzed_face(face, &snapshot.source_base_hash, Some(&coordinates)))
        .collect();
    ReshapeAnalysis {
        key: snapshot.analysis_key.clone(),
        source_base_hash: snapshot.source_base_hash.clone(),
        model_hashes: snapshot.model_hashes.clone(),
        coordinates,
        // 粗检尺寸只参与缓存计账，重放路径不使用；用基底尺寸占位。
        coarse_dimensions: snapshot.base_dimensions,
        faces,
        protection: snapshot.protection.clone(),
        rejected_faces: 0,
    }
}

fn analyzed_face(
    face: &SnapshotFace,
    source_base_hash: &str,
    coordinates: Option<&CoordinateMap>,
) -> AnalyzedFace {
    let roi_to_base = Affine::new(face.roi_matrix);
    let source_points = coordinates
        .map(|map| {
            face.landmarks
                .iter()
                .map(|p| {
                    roi_to_base
                        .map([p[0] as f64, p[1] as f64])
                        .and_then(|xy| map.base_to_source(xy))
                })
                .collect::<Option<Vec<_>>>()
                .unwrap_or_default()
        })
        .unwrap_or_default();
    AnalyzedFace {
        summary: ReshapeFaceSummary {
            face_id: face.face_id.clone(),
            source_anchor: ReshapeSourceAnchor {
                source_base_hash: source_base_hash.to_owned(),
                bounds: face.bounds,
            },
            processable: face.processable,
            reason: face.reason.clone(),
        },
        roi_to_base,
        roi_points: face.landmarks.clone(),
        source_points,
        detection_confidence: face.detection_confidence,
        landmark_confidence: face.landmark_confidence,
        refined: face.refined,
        protection: face.local_protection.clone(),
    }
}

fn restore_plan(face: &SnapshotFace) -> Result<ProtectionPlan, String> {
    ProtectionPlan::restore(
        face.fixed.clone(),
        face.weights.clone(),
        face.regions.clone(),
    )
}

fn rebuild_prepared(face: &SnapshotFace) -> Result<PreparedFace, String> {
    let roi_to_base = Affine::new(face.roi_matrix);
    let points = face
        .landmarks
        .iter()
        .map(|p| roi_to_base.map([p[0] as f64, p[1] as f64]))
        .collect::<Option<Vec<_>>>()
        .ok_or("重塑快照关键点无法映射到基底")?;
    PreparedFace::from_landmarks(&points, LANDMARK_VERSION)
        .map_err(|e| format!("重塑快照关键点无法重建网格：{e:?}"))
}

fn quality_of(face: &SnapshotFace) -> Result<QualityGate, String> {
    let analyzed = analyzed_face(face, "", None);
    quality_gate(&analyzed, OcclusionEvidence::Unknown)
        .map_err(|e| format!("重塑快照质量门控重算失败：{e:?}"))
}

/// 从快照与当前参数构建场景请求；快照脸集合是权威，参数不得引用不存在的脸。
pub fn scene_request(snapshot: &Snapshot, adjustments: &Value) -> Result<SceneRequest, String> {
    let parameters = read_parameters(&adjustments["portraitReshape"]);
    if parameters.status != "supported" {
        return Err("重塑参数版本不可执行".into());
    }
    let saved = parameters.value["faces"]
        .as_array()
        .ok_or("缺少重塑脸列表")?;
    let known: BTreeSet<&str> = snapshot.faces.iter().map(|f| f.face_id.as_str()).collect();
    for entry in saved {
        let id = entry["faceId"].as_str().unwrap_or_default();
        if !id.is_empty() && !known.contains(id) {
            return Err("重塑参数引用了快照中不存在的人脸，请重新分析并选择人脸".into());
        }
    }
    let mut faces = Vec::with_capacity(snapshot.faces.len());
    for face in &snapshot.faces {
        let entry = saved.iter().find(|f| f["faceId"] == face.face_id);
        let strengths = KEYS.map(|key| {
            entry
                .and_then(|f| f["overrides"].get(key))
                .and_then(|v| v.as_f64())
                .unwrap_or(0.0)
        });
        let reshape_vis = crate::portrait_reshape::reshape_section_visible(adjustments);
        let extended = crate::face_reshape::ExtendedStrengths(
            crate::portrait_reshape::read_extended_strengths(entry, reshape_vis),
        );
        faces.push(FaceRequest {
            face_id: face.face_id.clone(),
            strengths: Strengths(strengths),
            extended,
            occlusion: OcclusionEvidence::Unknown,
        });
    }
    Ok(SceneRequest {
        analysis_key: snapshot.analysis_key.clone(),
        faces,
        strokes: Vec::new(),
    })
}

/// 快速重现：直接从快照的已接受网格重建场景，并重新通过 06／07 安全检查。
pub fn restore_scene(snapshot: &Snapshot, current: &dyn Fn() -> bool) -> Result<SafeScene, String> {
    let mut faces = Vec::with_capacity(snapshot.faces.len());
    for face in &snapshot.faces {
        let geometry = rebuild_prepared(face)?
            .restore_solution(
                face.target.clone(),
                Strengths(face.requested),
                face.effective_response,
            )
            .map_err(|e| format!("重塑快照网格未通过恢复校验：{e:?}"))?;
        faces.push(crate::portrait_safety::RestoredSafeFace {
            id: face.face_id.clone(),
            geometry,
            plan: restore_plan(face)?,
            quality: quality_of(face)?,
            requested: Strengths(face.requested),
            safety_scale: face.safety_scale,
            backoffs: face.backoffs,
            reasons: face.reasons.clone(),
            detected_lines: face.detected_lines,
        });
    }
    SafeScene::restore(
        snapshot.analysis_key.clone(),
        snapshot.base_dimensions,
        faces,
        &snapshot.groups,
        current,
    )
}

/// 参数改变时的重求解：用快照输入恢复冻结保护与质量门控，再走同一求解入口。
pub fn restore_safety(
    snapshot: &Snapshot,
    request: &SceneRequest,
    current: &dyn Fn() -> bool,
) -> Result<PreparedSafety, String> {
    if request.analysis_key != snapshot.analysis_key || request.faces.len() != snapshot.faces.len()
    {
        return Err("重塑快照与请求不一致".into());
    }
    let mut faces = Vec::with_capacity(snapshot.faces.len());
    for (face, requested) in snapshot.faces.iter().zip(&request.faces) {
        if face.face_id != requested.face_id {
            return Err("重塑快照与请求脸顺序不一致".into());
        }
        faces.push(crate::portrait_safety::RestoredPreparedFace {
            id: face.face_id.clone(),
            geometry: rebuild_prepared(face)?,
            plan: restore_plan(face)?,
            quality: quality_of(face)?,
            requested: requested.strengths,
            extended: requested.extended,
            detected_lines: face.detected_lines,
        });
    }
    PreparedSafety::restore(
        snapshot.analysis_key.clone(),
        snapshot.base_dimensions,
        faces,
        snapshot.groups.clone(),
        current,
    )
}

#[cfg(test)]
#[path = "portrait_snapshot_tests.rs"]
mod tests;
