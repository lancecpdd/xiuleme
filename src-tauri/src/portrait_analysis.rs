//! 独立基础分析：不生成修肤场、不求解形变、不包含滑块或 UI 选中脸。
use crate::portrait_adapters::{
    DetectionTile, LandmarkSet, ProtectionMaps, ReshapeDetector, detect_tiles,
};
use crate::portrait_coordinates::{Affine, CoordinateMap};
use crate::portrait_models::{FaceRoi, ReshapeModels, sample_rgb};
use crate::portrait_reshape::{
    ReshapeFaceSummary, ReshapeReason, ReshapeSourceAnchor, read_parameters,
};
use image::Rgb32FImage;
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, HashSet};

pub const ANALYSIS_VERSION: &str = "reshape-analysis-v1";
pub const COARSE_EDGE: u32 = 2048;
pub const MAX_FACES: usize = 32;
pub const CANCELLED: &str = "重塑分析请求已过期";

pub trait AnalysisModels: ReshapeDetector {
    fn landmarks(&mut self, image: &Rgb32FImage, roi: FaceRoi) -> Result<LandmarkSet, String>;
    fn protection(&mut self, image: &Rgb32FImage) -> Result<ProtectionMaps, String>;
}

impl AnalysisModels for ReshapeModels {
    fn landmarks(&mut self, image: &Rgb32FImage, roi: FaceRoi) -> Result<LandmarkSet, String> {
        ReshapeModels::landmarks(self, image, roi)
    }
    fn protection(&mut self, image: &Rgb32FImage) -> Result<ProtectionMaps, String> {
        ReshapeModels::protection(self, image)
    }
}

// 完整数据仅留后端，后续几何求解／导出通过 Arc 引用冻结结果。
#[allow(dead_code)]
pub struct AnalyzedFace {
    pub summary: ReshapeFaceSummary,
    pub roi_to_base: Affine,
    pub roi_points: Vec<[f32; 3]>,
    pub source_points: Vec<[f64; 2]>,
    pub detection_confidence: f32,
    pub landmark_confidence: f32,
    pub refined: bool,
    pub protection: Option<ProtectionMaps>,
}

#[allow(dead_code)]
pub struct ReshapeAnalysis {
    pub key: String,
    pub source_base_hash: String,
    pub model_hashes: BTreeMap<String, String>,
    pub coordinates: CoordinateMap,
    pub coarse_dimensions: [u32; 2],
    pub faces: Vec<AnalyzedFace>,
    pub protection: Option<ProtectionMaps>,
    pub rejected_faces: usize,
}

impl ReshapeAnalysis {
    pub fn cache_size(&self) -> Option<usize> {
        // 保留部分成功摘要，但模型／保护失败不能永久命中缓存并阻止用户重试。
        if (!self.faces.is_empty() && self.protection.is_none())
            || self.faces.iter().any(|face| {
                matches!(
                    face.summary.reason,
                    Some(ReshapeReason::ModelError | ReshapeReason::ProtectionLimited)
                )
            })
        {
            return None;
        }
        Some(self.byte_size())
    }

    pub fn byte_size(&self) -> usize {
        let protection_bytes = |map: &Option<ProtectionMaps>| {
            map.as_ref().map_or(0, |m| {
                m.probabilities.capacity() * std::mem::size_of::<[f32; 4]>()
            })
        };
        // 包括关键点、逐脸保护、字符串、变换元数据；不缓存源像素或 sidecar 快照。
        std::mem::size_of::<Self>()
            + self.key.capacity()
            + self.source_base_hash.capacity()
            + self
                .model_hashes
                .iter()
                .map(|(k, v)| k.capacity() + v.capacity() + 128)
                .sum::<usize>()
            + protection_bytes(&self.protection)
            + self.faces.capacity() * std::mem::size_of::<AnalyzedFace>()
            + self
                .faces
                .iter()
                .map(|f| {
                    f.roi_points.capacity() * 12
                        + f.source_points.capacity() * 16
                        + protection_bytes(&f.protection)
                        + f.summary.face_id.capacity()
                        + f.summary.source_anchor.source_base_hash.capacity()
                })
                .sum::<usize>()
            + 16 * 1024 // 有界几何 JSON 与容器开销的保守余量。
    }
}

pub fn check_current(current: &dyn Fn() -> bool) -> Result<(), String> {
    if current() {
        Ok(())
    } else {
        Err(CANCELLED.into())
    }
}

fn valid_protection(map: &ProtectionMaps) -> bool {
    map.width == 256
        && map.height == 256
        && map.probabilities.len() == 256 * 256
        && map.probabilities.iter().all(|p| {
            p.iter().all(|v| v.is_finite() && (0.0..=1.0).contains(v))
                && (p.iter().sum::<f32>() - 1.0).abs() <= 1e-4
        })
}

fn source_bounds(map: &CoordinateMap, b: [f64; 4]) -> Option<[f64; 4]> {
    let mut low = [1.0f64; 2];
    let mut high = [0.0f64; 2];
    // 对框边界作采样，不将非线性变换后两角的连线当作原框。
    for i in 0..=8 {
        let t = i as f64 / 8.0;
        for p in [
            [b[0] + (b[2] - b[0]) * t, b[1]],
            [b[0] + (b[2] - b[0]) * t, b[3]],
            [b[0], b[1] + (b[3] - b[1]) * t],
            [b[2], b[1] + (b[3] - b[1]) * t],
        ] {
            let uv = map.base_to_source(p)?;
            for c in 0..2 {
                low[c] = low[c].min(uv[c]);
                high[c] = high[c].max(uv[c]);
            }
        }
    }
    (high[0] > low[0] && high[1] > low[1]).then_some([
        low[0],
        low[1],
        high[0] - low[0],
        high[1] - low[1],
    ])
}

pub fn analyze_base(
    image: &Rgb32FImage,
    coordinates: CoordinateMap,
    source_base_hash: String,
    model_hashes: BTreeMap<String, String>,
    models: &mut impl AnalysisModels,
    current: &dyn Fn() -> bool,
) -> Result<ReshapeAnalysis, String> {
    check_current(current)?;
    if image.width() == 0
        || image.height() == 0
        || coordinates.base_dimensions != [image.width(), image.height()]
        || image
            .as_raw()
            .iter()
            .any(|v| !v.is_finite() || !(0.0..=1.0).contains(v))
    {
        return Err("重塑分析基底必须是匹配坐标尺寸的有限 RGB 0–1 图像".into());
    }
    let factor = (COARSE_EDGE as f64 / image.width().max(image.height()) as f64).min(1.0);
    let (w, h) = (
        (image.width() as f64 * factor).round().max(1.0) as u32,
        (image.height() as f64 * factor).round().max(1.0) as u32,
    );
    let coarse = image::imageops::resize(image, w, h, image::imageops::FilterType::Triangle);
    check_current(current)?;
    let detections = detect_tiles(
        models,
        &coarse,
        &[DetectionTile {
            rect: [0.0, 0.0, w as f32, h as f32],
            input_edge: 256,
        }],
        0.35,
    )?;
    check_current(current)?;
    if detections.len() > MAX_FACES {
        return Err("重塑分析超过 32 脸安全上限".into());
    }
    let protection = if detections.is_empty() {
        None
    } else {
        let result = models.protection(&coarse).ok().filter(valid_protection);
        check_current(current)?;
        result
    };
    let scale = [
        image.width() as f64 / w as f64,
        image.height() as f64 / h as f64,
    ];
    let mut faces = Vec::with_capacity(detections.len());
    let mut rejected_faces = 0;
    let mut ids = HashSet::new();
    for detection in detections {
        check_current(current)?;
        let b = [
            detection.bounds[0] as f64 * scale[0],
            detection.bounds[1] as f64 * scale[1],
            detection.bounds[2] as f64 * scale[0],
            detection.bounds[3] as f64 * scale[1],
        ];
        let Some(bounds) = source_bounds(&coordinates, b) else {
            rejected_faces += 1;
            continue;
        };
        // 不猜检测器关键点次序：先轴对齐，再用 478 点的双眼旋正一次。
        let mut roi = FaceRoi {
            center: [((b[0] + b[2]) * 0.5) as f32, ((b[1] + b[3]) * 0.5) as f32],
            side: ((b[2] - b[0]).max(b[3] - b[1]) * 1.5) as f32,
            angle: 0.0,
        };
        let refined = (detection.bounds[2] - detection.bounds[0])
            .max(detection.bounds[3] - detection.bounds[1])
            < 256.0
            && factor < 1.0;
        // 原图取样保证小脸不会被粗检缩图永久限制；大脸也使用同一高分辨率基底。
        let mut points = match models.landmarks(image, roi) {
            Ok(points) => points,
            Err(_) => {
                check_current(current)?;
                LandmarkSet {
                    topology: "mediapipe-478-v1",
                    model_hash: String::new(),
                    points: Vec::new(),
                    confidence: 0.0,
                }
            }
        };
        check_current(current)?;
        let mut reason = if detection.confidence < 0.5 || points.confidence < 0.5 {
            Some(ReshapeReason::LowConfidence)
        } else {
            None
        };
        let valid_points = |p: &LandmarkSet| {
            p.topology == "mediapipe-478-v1"
                && p.points.len() == 478
                && model_hashes.get("landmarks") == Some(&p.model_hash)
                && p.confidence.is_finite()
                && (0.0..=1.0).contains(&p.confidence)
                && p.points
                    .iter()
                    .all(|v| v.iter().all(|n| n.is_finite() && n.abs() <= 4.0))
        };
        if valid_points(&points) && reason.is_none() {
            let left = points.points[33];
            let right = points.points[263];
            let angle = (right[1] - left[1]).atan2(right[0] - left[0]);
            if (right[0] - left[0]).hypot(right[1] - left[1]) < 0.05 || angle.abs() > 0.7 {
                reason = Some(ReshapeReason::UnsupportedPose);
            } else if angle.abs() > 0.01 {
                roi.angle = angle;
                check_current(current)?;
                match models.landmarks(image, roi) {
                    Ok(value) => points = value,
                    Err(_) => reason = Some(ReshapeReason::ModelError),
                }
                check_current(current)?;
            }
        }
        if !valid_points(&points) {
            reason = Some(ReshapeReason::ModelError);
        }
        if points.confidence < 0.5 && reason.is_none() {
            reason = Some(ReshapeReason::LowConfidence);
        }
        let roi_to_base = Affine::roi(roi);
        let source_points = if valid_points(&points) {
            points
                .points
                .iter()
                .map(|p| {
                    roi_to_base
                        .map([p[0] as f64, p[1] as f64])
                        .and_then(|xy| coordinates.base_to_source(xy))
                })
                .collect::<Option<Vec<_>>>()
        } else {
            None
        };
        if source_points.is_none() && reason.is_none() {
            reason = Some(ReshapeReason::ProtectionLimited);
        }
        let local_protection =
            if roi.side < image.width().max(image.height()) as f32 * 0.25 && reason.is_none() {
                let crop = Rgb32FImage::from_fn(256, 256, |x, y| {
                    let xy = roi.project((x as f32 + 0.5) / 256.0, (y as f32 + 0.5) / 256.0);
                    image::Rgb(sample_rgb(image, xy[0], xy[1]))
                });
                check_current(current)?;
                let result = models.protection(&crop).ok().filter(valid_protection);
                check_current(current)?;
                if result.is_none() {
                    reason = Some(ReshapeReason::ProtectionLimited);
                }
                result
            } else {
                None
            };
        if protection.is_none() && reason.is_none() {
            reason = Some(ReshapeReason::ProtectionLimited);
        }
        let quantized = bounds.map(|v| (v * 100_000.0).round() as i64);
        let face_id = format!(
            "face-{}",
            hex::encode(Sha256::digest(
                format!("{source_base_hash}:{quantized:?}").as_bytes()
            ))
        );
        if !ids.insert(face_id.clone()) {
            return Err("人脸锚点无法唯一标识".into());
        }
        faces.push(AnalyzedFace {
            summary: ReshapeFaceSummary {
                face_id,
                source_anchor: ReshapeSourceAnchor {
                    source_base_hash: source_base_hash.clone(),
                    bounds,
                },
                processable: reason.is_none(),
                reason,
            },
            roi_to_base,
            roi_points: points.points,
            source_points: source_points.unwrap_or_default(),
            detection_confidence: detection.confidence,
            landmark_confidence: points.confidence,
            refined,
            protection: local_protection,
        });
    }
    check_current(current)?;
    Ok(ReshapeAnalysis {
        key: String::new(),
        source_base_hash,
        model_hashes,
        coordinates,
        coarse_dimensions: [w, h],
        faces,
        protection,
        rejected_faces,
    })
}

fn anchor_matches(a: &ReshapeSourceAnchor, b: &ReshapeSourceAnchor) -> bool {
    if a.source_base_hash != b.source_base_hash {
        return false;
    }
    let (a, b) = (a.bounds, b.bounds);
    let intersection = (a[0] + a[2]).min(b[0] + b[2]) - a[0].max(b[0]);
    let height = (a[1] + a[3]).min(b[1] + b[3]) - a[1].max(b[1]);
    let area = intersection.max(0.0) * height.max(0.0);
    let iou = area / (a[2] * a[3] + b[2] * b[3] - area).max(1e-12);
    let distance =
        (a[0] + a[2] * 0.5 - b[0] - b[2] * 0.5).hypot(a[1] + a[3] * 0.5 - b[1] - b[3] * 0.5);
    let size = a[2].hypot(a[3]).min(b[2].hypot(b[3]));
    let aspect_ratio = (a[2] / a[3]) / (b[2] / b[3]);
    iou >= 0.45 && distance <= size * 0.2 && (0.65..=1.55).contains(&aspect_ratio)
}

/// 只匹配照片内源锚点；不按数组下标关联，也不使用身份识别。
/// 匹配放在缓存外，因此保存的 overrides／保护笔刷永远不会进入基础分析缓存。
pub fn match_faces(
    current: &[ReshapeFaceSummary],
    parameters: &Value,
) -> Result<Vec<ReshapeFaceSummary>, String> {
    let parsed = read_parameters(parameters);
    if parsed.status != "supported" {
        return Err("重塑参数版本不可执行".into());
    }
    let saved = parsed.value["faces"].as_array().ok_or("缺少重塑脸列表")?;
    let anchors: Vec<ReshapeSourceAnchor> = saved
        .iter()
        .map(|f| serde_json::from_value(f["sourceAnchor"].clone()).map_err(|e| e.to_string()))
        .collect::<Result<_, _>>()?;
    let candidates: Vec<Vec<usize>> = anchors
        .iter()
        .map(|anchor| {
            current
                .iter()
                .enumerate()
                .filter_map(|(i, face)| anchor_matches(anchor, &face.source_anchor).then_some(i))
                .collect()
        })
        .collect();
    let mut owners = vec![0usize; current.len()];
    for list in &candidates {
        for i in list {
            owners[*i] += 1;
        }
    }
    let mut result = current.to_vec();
    for (index, list) in candidates.iter().enumerate() {
        if list.len() == 1 && owners[list[0]] == 1 {
            result[list[0]].face_id = saved[index]["faceId"]
                .as_str()
                .ok_or("缺少人脸标识")?
                .into();
        } else {
            for i in list {
                result[*i].processable = false;
                result[*i].reason = Some(ReshapeReason::UnmatchedFace);
            }
            result.push(ReshapeFaceSummary {
                face_id: saved[index]["faceId"]
                    .as_str()
                    .ok_or("缺少人脸标识")?
                    .into(),
                source_anchor: anchors[index].clone(),
                processable: false,
                reason: Some(ReshapeReason::UnmatchedFace),
            });
        }
    }
    // 旧标识与新生成标识碰撞时宁可阻断，不允许一个 override 命中两张脸。
    let mut seen = HashSet::new();
    if result.iter().any(|f| !seen.insert(f.face_id.clone())) {
        return Err("人脸匹配标识冲突".into());
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::portrait_adapters::ReshapeDetection;
    use serde_json::json;

    fn summary(id: &str, x: f64) -> ReshapeFaceSummary {
        ReshapeFaceSummary {
            face_id: id.into(),
            source_anchor: ReshapeSourceAnchor {
                source_base_hash: "a".repeat(64),
                bounds: [x, 0.2, 0.15, 0.3],
            },
            processable: true,
            reason: None,
        }
    }

    #[test]
    fn reordered_faces_and_ambiguous_anchors() {
        let a = summary("a", 0.1);
        let b = summary("b", 0.65);
        let settings = json!({"schemaVersion":1,"engineVersion":"mesh-arap-v1","enabled":true,
            "faces":[{"faceId":"saved-a","sourceAnchor":a.source_anchor},
                {"faceId":"saved-b","sourceAnchor":b.source_anchor}]});
        let result = match_faces(&[b.clone(), a.clone()], &settings).unwrap();
        assert_eq!(result[0].face_id, "saved-b");
        assert_eq!(result[1].face_id, "saved-a");
        let result = match_faces(&[a.clone(), summary("near-a", 0.105)], &settings).unwrap();
        assert!(result.iter().all(|f| !f.processable));
        assert!(result.iter().any(|f| f.face_id == "saved-b"));
        let mut foreign = a;
        foreign.source_anchor.source_base_hash = "b".repeat(64);
        let result = match_faces(&[foreign], &settings).unwrap();
        assert_eq!(result[0].face_id, "a");
        assert!(result.iter().skip(1).all(|f| !f.processable));
    }

    #[derive(Default)]
    struct FakeModels {
        calls: [usize; 3],
        empty: bool,
        small: bool,
        fail_landmarks: bool,
        fail_protection: bool,
        landmark_dimensions: Vec<[u32; 2]>,
    }
    impl ReshapeDetector for FakeModels {
        fn detect_tile(
            &mut self,
            _: &Rgb32FImage,
            _: DetectionTile,
        ) -> Result<Vec<ReshapeDetection>, String> {
            self.calls[0] += 1;
            if self.empty {
                return Ok(vec![]);
            }
            Ok(vec![ReshapeDetection {
                bounds: if self.small {
                    [0.45, 0.4, 0.5, 0.5]
                } else {
                    [0.35, 0.3, 0.6, 0.65]
                },
                keypoints: vec![],
                confidence: 0.9,
            }])
        }
    }
    impl AnalysisModels for FakeModels {
        fn landmarks(&mut self, image: &Rgb32FImage, _: FaceRoi) -> Result<LandmarkSet, String> {
            self.calls[1] += 1;
            self.landmark_dimensions
                .push([image.width(), image.height()]);
            if self.fail_landmarks {
                return Err("合成关键点失败".into());
            }
            let mut points = vec![[0.5, 0.5, 0.0]; 478];
            points[33] = [0.3, 0.4, 0.0];
            points[263] = [0.7, 0.4, 0.0];
            Ok(LandmarkSet {
                topology: "mediapipe-478-v1",
                model_hash: "c".repeat(64),
                points,
                confidence: 0.9,
            })
        }
        fn protection(&mut self, _: &Rgb32FImage) -> Result<ProtectionMaps, String> {
            self.calls[2] += 1;
            if self.fail_protection {
                return Err("合成保护失败".into());
            }
            Ok(ProtectionMaps {
                width: 256,
                height: 256,
                probabilities: vec![[0.1, 0.1, 0.7, 0.1]; 256 * 256],
            })
        }
    }

    #[test]
    fn small_faces_refine_from_full_resolution_and_keep_local_protection() {
        let image = Rgb32FImage::from_pixel(2400, 400, image::Rgb([0.5; 3]));
        let mut models = FakeModels {
            small: true,
            ..Default::default()
        };
        let map = CoordinateMap::new([2400, 400], &json!({})).unwrap();
        let analysis = analyze_base(
            &image,
            map,
            "a".repeat(64),
            BTreeMap::from([("landmarks".into(), "c".repeat(64))]),
            &mut models,
            &|| true,
        )
        .unwrap();
        assert_eq!(analysis.coarse_dimensions[0], COARSE_EDGE);
        assert_eq!(models.landmark_dimensions, vec![[2400, 400]]);
        assert!(analysis.faces[0].refined);
        assert!(analysis.faces[0].protection.is_some());
        assert!(analysis.faces[0].summary.processable);
        assert!(analysis.cache_size().is_some());
    }

    #[test]
    fn no_face_skips_other_models_and_transient_failures_remain_retryable() {
        let image = Rgb32FImage::from_pixel(160, 100, image::Rgb([0.5; 3]));
        let map = CoordinateMap::new([160, 100], &json!({})).unwrap();
        let hashes = BTreeMap::from([("landmarks".into(), "c".repeat(64))]);
        let mut models = FakeModels {
            empty: true,
            ..Default::default()
        };
        let empty = analyze_base(
            &image,
            map.clone(),
            "a".repeat(64),
            hashes.clone(),
            &mut models,
            &|| true,
        )
        .unwrap();
        assert!(empty.faces.is_empty());
        assert_eq!(models.calls, [1, 0, 0]);
        assert!(empty.cache_size().is_some());
        models.empty = false;
        models.fail_protection = true;
        let partial = analyze_base(
            &image,
            map.clone(),
            "a".repeat(64),
            hashes.clone(),
            &mut models,
            &|| true,
        )
        .unwrap();
        assert!(!partial.faces[0].summary.processable);
        assert!(matches!(
            partial.faces[0].summary.reason,
            Some(ReshapeReason::ProtectionLimited)
        ));
        assert!(partial.cache_size().is_none());
        models.fail_protection = false;
        models.fail_landmarks = true;
        let partial =
            analyze_base(&image, map, "a".repeat(64), hashes, &mut models, &|| true).unwrap();
        assert!(matches!(
            partial.faces[0].summary.reason,
            Some(ReshapeReason::ModelError)
        ));
        assert!(partial.cache_size().is_none());
    }

    #[test]
    fn synthetic_analysis_freezes_full_data_and_cancels_before_inference() {
        let image = Rgb32FImage::from_pixel(160, 100, image::Rgb([0.5; 3]));
        let mut models = FakeModels::default();
        let map = CoordinateMap::new([160, 100], &json!({})).unwrap();
        let hashes = BTreeMap::from([("landmarks".into(), "c".repeat(64))]);
        let analysis = analyze_base(
            &image,
            map.clone(),
            "a".repeat(64),
            hashes.clone(),
            &mut models,
            &|| true,
        )
        .unwrap();
        assert_eq!(analysis.faces.len(), 1);
        assert!(analysis.faces[0].summary.processable);
        assert_eq!(analysis.faces[0].source_points.len(), 478);
        assert!(analysis.byte_size() > 256 * 256 * 16);
        let calls = models.calls;
        assert!(analyze_base(&image, map, "a".repeat(64), hashes, &mut models, &|| false).is_err());
        assert_eq!(models.calls, calls);
    }
}
