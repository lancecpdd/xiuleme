//! 重塑推理适配：预处理、解码与分块检测契约；真实权重须经 reshape.approvedModels 独立审批。
#![allow(dead_code)] // 步骤 04 才接入分析请求；本阶段只开放内部契约与合成测试。
use crate::portrait_models::{FaceRoi, sample_rgb};
use ndarray::{Array4, ArrayD, ArrayViewD};

pub fn validate_tensor(
    array: ArrayViewD<'_, f32>,
    expected: &[usize],
    name: &str,
) -> Result<(), String> {
    if array.is_empty()
        || array.ndim() != expected.len()
        || array
            .shape()
            .iter()
            .zip(expected)
            .any(|(a, e)| *e != 0 && a != e)
    {
        return Err(format!(
            "{name} 张量形状异常：{:?}，预期 {expected:?}",
            array.shape()
        ));
    }
    if array.iter().any(|v| !v.is_finite()) {
        return Err(format!("{name} 包含 NaN 或无穷数值"));
    }
    Ok(())
}

#[derive(Clone, Copy)]
pub enum RgbInput {
    FaceMesh478,
    SelfieMulticlass,
    // 合成直线拉伸格式；真实检测预处理是 letterbox（见 prepare_blazeface_tile）。
    DetectorNhwcSigned,
}

pub fn prepare_rgb(
    image: &image::Rgb32FImage,
    roi: Option<FaceRoi>,
    edge: usize,
    format: RgbInput,
) -> Result<Array4<f32>, String> {
    if image.width() == 0
        || image.height() == 0
        || !(1..=1024).contains(&edge)
        || image
            .as_raw()
            .iter()
            .any(|v| !v.is_finite() || !(0.0..=1.0).contains(v))
    {
        return Err("重塑输入必须是非空、有限的 RGB 0–1 图像".into());
    }
    if let Some(roi) = roi
        && (!roi.side.is_finite()
            || roi.side <= 0.0
            || !roi.angle.is_finite()
            || roi.center.iter().any(|v| !v.is_finite()))
    {
        return Err("重塑 ROI 非法".into());
    }
    if matches!(format, RgbInput::FaceMesh478 | RgbInput::SelfieMulticlass) && edge != 256 {
        return Err("关键点／保护分割要求 256×256 输入".into());
    }
    let nchw = matches!(format, RgbInput::FaceMesh478);
    let mut input = Array4::zeros(if nchw {
        (1, 3, edge, edge)
    } else {
        (1, edge, edge, 3)
    });
    for y in 0..edge {
        for x in 0..edge {
            let (u, v) = (
                (x as f32 + 0.5) / edge as f32,
                (y as f32 + 0.5) / edge as f32,
            );
            let xy = roi
                .map(|r| r.project(u, v))
                .unwrap_or([u * image.width() as f32, v * image.height() as f32]);
            if xy.iter().any(|v| !v.is_finite()) {
                return Err("ROI 投影溢出".into());
            }
            let rgb = sample_rgb(image, xy[0], xy[1]);
            for c in 0..3 {
                let value = if matches!(format, RgbInput::DetectorNhwcSigned) {
                    rgb[c] * 2.0 - 1.0
                } else {
                    rgb[c]
                };
                if nchw {
                    input[[0, c, y, x]] = value
                } else {
                    input[[0, y, x, c]] = value
                }
            }
        }
    }
    Ok(input)
}

#[derive(Clone, Copy)]
pub enum ScoreDomain {
    Logits,
    Probabilities,
}

fn sigmoid(value: f32) -> f32 {
    if value >= 0.0 {
        1.0 / (1.0 + (-value).exp())
    } else {
        let e = value.exp();
        e / (1.0 + e)
    }
}

pub struct LandmarkSet {
    pub topology: &'static str,
    pub model_hash: String,
    pub points: Vec<[f32; 3]>,
    pub confidence: f32,
}

pub fn decode_landmarks(
    points: &ArrayD<f32>,
    score: &ArrayD<f32>,
    model_hash: &str,
) -> Result<LandmarkSet, String> {
    validate_tensor(points.view(), &[1, 478, 3], "Face Mesh landmarks")?;
    validate_tensor(score.view(), &[1, 1], "Face Mesh score")?;
    if model_hash.len() != 64
        || !model_hash
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
    {
        return Err("关键点拓扑必须绑定模型 SHA-256".into());
    }
    let values = points.as_slice().ok_or("关键点输出不连续")?;
    let points: Vec<[f32; 3]> = values
        .as_chunks::<3>()
        .0
        .iter()
        .map(|v| [v[0] / 256.0, v[1] / 256.0, v[2] / 256.0])
        .collect();
    if points.iter().any(|p| p.iter().any(|n| n.abs() > 4.0)) {
        return Err("关键点超出 ROI 契约范围".into());
    }
    Ok(LandmarkSet {
        topology: "mediapipe-478-v1",
        model_hash: model_hash.to_owned(),
        points,
        confidence: sigmoid(*score.iter().next().ok_or("缺少关键点分数")?),
    })
}

#[derive(Clone)]
pub struct ProtectionMaps {
    pub width: usize,
    pub height: usize,
    // 背景、头发、皮肤（身体+面部）、其他（衣物+剩余类别）；不是手／眼镜／耳饰分类器。
    pub probabilities: Vec<[f32; 4]>,
}

pub fn decode_protection(
    output: &ArrayD<f32>,
    domain: ScoreDomain,
) -> Result<ProtectionMaps, String> {
    validate_tensor(output.view(), &[1, 256, 256, 6], "Selfie Multiclass")?;
    let values = output.as_slice().ok_or("保护分割输出不连续")?;
    let mut probabilities = Vec::with_capacity(256 * 256);
    for pixel in values.as_chunks::<6>().0 {
        let mut p = [0.0; 6];
        match domain {
            ScoreDomain::Logits => {
                let max = pixel.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                for c in 0..6 {
                    p[c] = (pixel[c] - max).exp()
                }
                let sum: f32 = p.iter().sum();
                for v in &mut p {
                    *v /= sum
                }
            }
            ScoreDomain::Probabilities => {
                let sum: f32 = pixel.iter().sum();
                if pixel.iter().any(|v| !(0.0..=1.0).contains(v)) || (sum - 1.0).abs() > 1e-4 {
                    return Err("保护概率未归一化；禁止猜测输出域".into());
                }
                p.copy_from_slice(pixel);
            }
        }
        probabilities.push([p[0], p[1], p[2] + p[3], p[4] + p[5]]);
    }
    Ok(ProtectionMaps {
        width: 256,
        height: 256,
        probabilities,
    })
}

#[derive(Clone, Debug)]
pub struct ReshapeDetection {
    pub bounds: [f32; 4], // 当前块归一化 xyxy；汇总后为全图像素 xyxy。
    pub keypoints: Vec<[f32; 2]>,
    pub confidence: f32,
}

// anchors、scale、分数域必须来自确切上游配置；不复用 RetinaFace anchors 或 decoder。
pub struct BlazeFaceDecodeContract {
    pub anchors: Vec<[f32; 4]>, // 归一化 cx,cy,w,h
    pub scales: [f32; 4],
    pub keypoint_count: usize,
    pub score_domain: ScoreDomain,
}

pub fn decode_blazeface(
    boxes: &ArrayD<f32>,
    scores: &ArrayD<f32>,
    contract: &BlazeFaceDecodeContract,
    threshold: f32,
) -> Result<Vec<ReshapeDetection>, String> {
    let count = contract.anchors.len();
    if count == 0
        || count > 10000
        || contract.keypoint_count > 16
        || contract.scales.iter().any(|s| !s.is_finite() || *s <= 0.0)
        || contract
            .anchors
            .iter()
            .any(|a| a.iter().any(|v| !v.is_finite()) || a[2] <= 0.0 || a[3] <= 0.0)
        || !threshold.is_finite()
        || !(0.0..=1.0).contains(&threshold)
    {
        return Err("BlazeFace 解码配置非法或缺少上游 anchors".into());
    }
    let columns = 4 + contract.keypoint_count * 2;
    validate_tensor(boxes.view(), &[1, count, columns], "BlazeFace boxes")?;
    validate_tensor(scores.view(), &[1, count, 1], "BlazeFace scores")?;
    let boxes = boxes.as_slice().ok_or("检测框输出不连续")?;
    let scores = scores.as_slice().ok_or("检测分数输出不连续")?;
    let mut detections = Vec::new();
    for (i, raw) in boxes.chunks_exact(columns).enumerate() {
        let confidence = match contract.score_domain {
            ScoreDomain::Logits => sigmoid(scores[i]),
            ScoreDomain::Probabilities if (0.0..=1.0).contains(&scores[i]) => scores[i],
            ScoreDomain::Probabilities => return Err("检测概率越界".into()),
        };
        if confidence < threshold {
            continue;
        }
        let a = contract.anchors[i];
        let s = contract.scales;
        let cx = raw[0] / s[0] * a[2] + a[0];
        let cy = raw[1] / s[1] * a[3] + a[1];
        let w = raw[2] / s[2] * a[2];
        let h = raw[3] / s[3] * a[3];
        // 上游 ConvertToDetections 过滤负宽高／NaN 框（b/171391719）；0 宽高无下游语义也跳过。
        if w.is_nan() || h.is_nan() || w <= 0.0 || h <= 0.0 {
            continue;
        }
        let detection = ReshapeDetection {
            bounds: [cx - w * 0.5, cy - h * 0.5, cx + w * 0.5, cy + h * 0.5],
            keypoints: raw[4..]
                .as_chunks::<2>()
                .0
                .iter()
                .map(|v| [v[0] / s[0] * a[2] + a[0], v[1] / s[1] * a[3] + a[1]])
                .collect(),
            confidence,
        };
        validate_detection(&detection)?;
        detections.push(detection);
    }
    Ok(detections)
}

#[derive(Clone, Copy)]
pub struct DetectionTile {
    pub rect: [f32; 4], // 全图像素 x,y,width,height。
    pub input_edge: usize,
}

/// MediaPipe BlazeFace full range 的固有输入边长（上游 output_tensor_width/height = 192）。
pub const BLAZEFACE_INPUT_EDGE: usize = 192;
/// 上游 full_range 的 48×48 单锚框网格：stride 4、anchor_offset 0.5、fixed_anchor_size。
pub const BLAZEFACE_ANCHOR_GRID: usize = 48;
/// 上游 full_range pbtxt 的 min_score_thresh（作用于 sigmoid 后的概率）。
pub const BLAZEFACE_MIN_SCORE: f32 = 0.6;

/// 上游 SsdAnchorsCalculator 对 full range 生成的锚框（归一化 cx,cy,w,h）。
/// 顺序与模型输出逐位对应：y 外层、x 内层，i = y * 48 + x；fixed_anchor_size → w = h = 1。
pub fn blazeface_anchors() -> &'static [[f32; 4]] {
    static ANCHORS: std::sync::OnceLock<Vec<[f32; 4]>> = std::sync::OnceLock::new();
    ANCHORS.get_or_init(|| {
        let grid = BLAZEFACE_ANCHOR_GRID;
        let mut anchors = Vec::with_capacity(grid * grid);
        for y in 0..grid {
            for x in 0..grid {
                anchors.push([
                    (x as f32 + 0.5) / grid as f32,
                    (y as f32 + 0.5) / grid as f32,
                    1.0,
                    1.0,
                ]);
            }
        }
        anchors
    })
}

/// tile → MediaPipe keep_aspect_ratio 的方化 ROI（中心不变、边长 = max(w, h)）。
fn blazeface_letterbox_roi(tile: DetectionTile) -> [f32; 4] {
    let [x, y, w, h] = tile.rect;
    let side = w.max(h);
    [x + (w - side) * 0.5, y + (h - side) * 0.5, side, side]
}

/// 采样一个邻点：越界取 border 0（uint8 域）→ signed 域 -1；合法点取 2v-1。
#[inline]
fn signed_border_pixel(
    image: &image::Rgb32FImage,
    x: i64,
    y: i64,
    channel: usize,
    size: (i64, i64),
) -> f32 {
    if x < 0 || y < 0 || x >= size.0 || y >= size.1 {
        -1.0
    } else {
        image.get_pixel(x as u32, y as u32)[channel] * 2.0 - 1.0
    }
}

/// 复刻 MediaPipe ImageToTensor（OpenCV converter）对单个轴对齐 tile 的预处理：
/// keep_aspect_ratio 方化 + warpPerspective(INTER_LINEAR, BORDER_CONSTANT) + 范围 [-1,1]。
/// dst 整数坐标角对齐映射到 src 连续坐标（cv2 探针实测），越界邻点取 border 0 → -1。
pub fn prepare_blazeface_tile(
    image: &image::Rgb32FImage,
    tile: DetectionTile,
) -> Result<Array4<f32>, String> {
    let [x, y, w, h] = tile.rect;
    if image.width() == 0
        || image.height() == 0
        || tile.rect.iter().any(|v| !v.is_finite())
        || w <= 0.0
        || h <= 0.0
        || x < 0.0
        || y < 0.0
        || x + w > image.width() as f32
        || y + h > image.height() as f32
    {
        return Err("BlazeFace 检测分块非法".into());
    }
    if image.as_raw().iter().any(|v| !v.is_finite()) {
        return Err("BlazeFace 检测图像含非法数值".into());
    }
    let [x0, y0, side, _] = blazeface_letterbox_roi(tile);
    let edge = BLAZEFACE_INPUT_EDGE;
    let size = (image.width() as i64, image.height() as i64);
    let mut input = Array4::zeros((1, edge, edge, 3));
    for row in 0..edge {
        let sy = y0 + row as f32 * side / edge as f32;
        let sy0 = sy.floor();
        let fy = sy - sy0;
        for col in 0..edge {
            let sx = x0 + col as f32 * side / edge as f32;
            let sx0 = sx.floor();
            let fx = sx - sx0;
            let (xi, yi) = (sx0 as i64, sy0 as i64);
            for channel in 0..3 {
                let top = signed_border_pixel(image, xi, yi, channel, size) * (1.0 - fx)
                    + signed_border_pixel(image, xi + 1, yi, channel, size) * fx;
                let bottom = signed_border_pixel(image, xi, yi + 1, channel, size) * (1.0 - fx)
                    + signed_border_pixel(image, xi + 1, yi + 1, channel, size) * fx;
                input[[0, row, col, channel]] = top * (1.0 - fy) + bottom * fy;
            }
        }
    }
    Ok(input)
}

/// 方化 letterbox 的逆映射：模型归一化坐标 → 块归一化坐标。
/// 由 src = roi0 + u * side 与 roi0 = tile 中心 - side / 2 推得中心锚定缩放：
/// u' = u * side / tile_w + (1 - side / tile_w) / 2。
pub fn blazeface_tile_from_model(tile: DetectionTile, u: f32, v: f32) -> [f32; 2] {
    let side = tile.rect[2].max(tile.rect[3]);
    let scale_x = side / tile.rect[2];
    let scale_y = side / tile.rect[3];
    [
        u * scale_x + (1.0 - scale_x) * 0.5,
        v * scale_y + (1.0 - scale_y) * 0.5,
    ]
}

pub trait ReshapeDetector {
    // 返回块归一化结果；模型 letterbox padding 必须先由具体适配器移除。
    fn detect_tile(
        &mut self,
        image: &image::Rgb32FImage,
        tile: DetectionTile,
    ) -> Result<Vec<ReshapeDetection>, String>;
}

fn validate_detection(d: &ReshapeDetection) -> Result<(), String> {
    if d.bounds
        .iter()
        .chain(d.keypoints.iter().flatten())
        .any(|v| !v.is_finite())
        || d.bounds[2] <= d.bounds[0]
        || d.bounds[3] <= d.bounds[1]
        || !d.confidence.is_finite()
        || !(0.0..=1.0).contains(&d.confidence)
    {
        return Err("人脸检测输出坐标或分数非法".into());
    }
    Ok(())
}

pub fn detect_tiles(
    detector: &mut impl ReshapeDetector,
    image: &image::Rgb32FImage,
    tiles: &[DetectionTile],
    nms_threshold: f32,
) -> Result<Vec<ReshapeDetection>, String> {
    if image.width() == 0
        || image.height() == 0
        || image.as_raw().iter().any(|v| !v.is_finite())
        || tiles.is_empty()
        || tiles.len() > 64
        || !nms_threshold.is_finite()
        || !(0.0..=1.0).contains(&nms_threshold)
    {
        return Err("检测图像、分块计划或 NMS 阈值非法".into());
    }
    let mut candidates = Vec::new();
    for tile in tiles {
        let [x, y, w, h] = tile.rect;
        if tile.rect.iter().any(|v| !v.is_finite())
            || x < 0.0
            || y < 0.0
            || w <= 0.0
            || h <= 0.0
            || x + w > image.width() as f32
            || y + h > image.height() as f32
            || !(1..=1024).contains(&tile.input_edge)
        {
            return Err("检测分块超出照片范围".into());
        }
        for mut detection in detector.detect_tile(image, *tile)? {
            validate_detection(&detection)?;
            detection.bounds = [
                x + detection.bounds[0] * w,
                y + detection.bounds[1] * h,
                x + detection.bounds[2] * w,
                y + detection.bounds[3] * h,
            ];
            for point in &mut detection.keypoints {
                *point = [x + point[0] * w, y + point[1] * h]
            }
            validate_detection(&detection)?;
            candidates.push(detection);
            if candidates.len() > 10000 {
                return Err("检测候选超出预算".into());
            }
        }
    }
    candidates.sort_by(|a, b| b.confidence.total_cmp(&a.confidence));
    let mut selected: Vec<ReshapeDetection> = Vec::new();
    for d in candidates {
        let overlaps = selected.iter().any(|other| {
            let a = d.bounds.map(f64::from);
            let b = other.bounds.map(f64::from);
            let area = |r: [f64; 4]| (r[2] - r[0]) * (r[3] - r[1]);
            let intersection = (a[2].min(b[2]) - a[0].max(b[0])).max(0.0)
                * (a[3].min(b[3]) - a[1].max(b[1])).max(0.0);
            intersection / (area(a) + area(b) - intersection).max(1e-12) > f64::from(nms_threshold)
        });
        if !overlaps {
            selected.push(d)
        }
    }
    Ok(selected)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array, IxDyn};

    #[test]
    fn rgb_layout_range_and_empty_inputs() {
        let image = image::Rgb32FImage::from_pixel(2, 2, image::Rgb([1.0, 0.25, 0.0]));
        let mesh = prepare_rgb(&image, None, 256, RgbInput::FaceMesh478).unwrap();
        assert_eq!(
            [mesh[[0, 0, 0, 0]], mesh[[0, 1, 0, 0]], mesh[[0, 2, 0, 0]]],
            [1.0, 0.25, 0.0]
        );
        let detector = prepare_rgb(&image, None, 2, RgbInput::DetectorNhwcSigned).unwrap();
        assert_eq!(
            [
                detector[[0, 0, 0, 0]],
                detector[[0, 0, 0, 1]],
                detector[[0, 0, 0, 2]]
            ],
            [1.0, -0.5, -1.0]
        );
        assert!(
            prepare_rgb(
                &image::Rgb32FImage::new(0, 0),
                None,
                256,
                RgbInput::FaceMesh478
            )
            .is_err()
        );
        assert!(prepare_rgb(&image, None, 0, RgbInput::SelfieMulticlass).is_err());
        for value in [f32::NAN, f32::INFINITY] {
            let invalid = Array::from_elem(IxDyn(&[1, 1]), value);
            assert!(validate_tensor(invalid.view(), &[1, 1], "test").is_err());
        }
        assert!(validate_tensor(Array::zeros(IxDyn(&[1, 0])).view(), &[1, 0], "test").is_err());
    }

    #[test]
    fn landmark_topology_units_and_logit_score() {
        let points = Array::from_elem(IxDyn(&[1, 478, 3]), 128.0);
        let score = Array::from_elem(IxDyn(&[1, 1]), 0.0);
        let result = decode_landmarks(&points, &score, &"a".repeat(64)).unwrap();
        assert_eq!(result.points[0], [0.5; 3]);
        assert_eq!(result.confidence, 0.5);
        assert_eq!(result.topology, "mediapipe-478-v1");
        assert!(
            decode_landmarks(&Array::zeros(IxDyn(&[1, 468, 3])), &score, &"a".repeat(64)).is_err()
        );
    }

    #[test]
    fn segmentation_domain_and_all_protection_channels() {
        let logits = Array::zeros(IxDyn(&[1, 256, 256, 6]));
        let maps = decode_protection(&logits, ScoreDomain::Logits).unwrap();
        let expected = [1.0 / 6.0, 1.0 / 6.0, 2.0 / 6.0, 2.0 / 6.0];
        for (a, b) in maps.probabilities[0].iter().zip(expected) {
            assert!((a - b).abs() < 1e-6)
        }
        let probabilities = Array::from_elem(IxDyn(&[1, 256, 256, 6]), 1.0 / 6.0);
        assert!(decode_protection(&probabilities, ScoreDomain::Probabilities).is_ok());
        assert!(decode_protection(&logits, ScoreDomain::Probabilities).is_err());
        assert!(
            decode_protection(&Array::zeros(IxDyn(&[1, 6, 256, 256])), ScoreDomain::Logits)
                .is_err()
        );
    }

    #[test]
    fn blazeface_uses_supplied_anchors_and_scales() {
        let contract = BlazeFaceDecodeContract {
            anchors: vec![[0.5, 0.5, 1.0, 1.0]],
            scales: [100.0; 4],
            keypoint_count: 1,
            score_domain: ScoreDomain::Logits,
        };
        let boxes =
            Array::from_shape_vec(IxDyn(&[1, 1, 6]), vec![0.0, 0.0, 20.0, 40.0, 10.0, -10.0])
                .unwrap();
        let scores = Array::from_elem(IxDyn(&[1, 1, 1]), 10.0);
        let faces = decode_blazeface(&boxes, &scores, &contract, 0.9).unwrap();
        assert_eq!(faces.len(), 1);
        assert!((faces[0].bounds[0] - 0.4).abs() < 1e-6);
        assert!((faces[0].keypoints[0][0] - 0.6).abs() < 1e-6);
        assert!(decode_blazeface(&boxes, &scores, &contract, f32::NAN).is_err());
    }

    #[test]
    fn blazeface_negative_box_size_is_skipped_not_fatal() {
        let contract = BlazeFaceDecodeContract {
            anchors: vec![[0.5, 0.5, 1.0, 1.0]; 2],
            scales: [100.0; 4],
            keypoint_count: 1,
            score_domain: ScoreDomain::Logits,
        };
        // 第一框宽为负（上游 ConvertToDetections 过滤）；第二框正常保留。
        let boxes = Array::from_shape_vec(
            IxDyn(&[1, 2, 6]),
            vec![
                0.0, 0.0, -20.0, 40.0, 10.0, -10.0, 0.0, 0.0, 20.0, 40.0, 10.0, -10.0,
            ],
        )
        .unwrap();
        let scores = Array::from_elem(IxDyn(&[1, 2, 1]), 10.0);
        let faces = decode_blazeface(&boxes, &scores, &contract, 0.9).unwrap();
        assert_eq!(faces.len(), 1);
        assert!((faces[0].bounds[0] - 0.4).abs() < 1e-6);
    }

    #[test]
    fn blazeface_anchors_match_upstream_grid_order() {
        let anchors = blazeface_anchors();
        assert_eq!(anchors.len(), 48 * 48);
        assert_eq!(anchors[0], [0.5f32 / 48.0, 0.5 / 48.0, 1.0, 1.0]);
        assert_eq!(anchors[1], [1.5f32 / 48.0, 0.5 / 48.0, 1.0, 1.0]);
        assert_eq!(anchors[48], [0.5f32 / 48.0, 1.5 / 48.0, 1.0, 1.0]);
        assert_eq!(anchors[2303], [47.5f32 / 48.0, 47.5 / 48.0, 1.0, 1.0]);
    }

    #[test]
    fn blazeface_letterbox_sampling_matches_mediapipe_probe() {
        // 4×2 图；signed 编码 R = 2*(x*0.25)-1、G = 2*(y*0.5)-1、B = -0.5。
        let mut image = image::Rgb32FImage::new(4, 2);
        for (x, y, pixel) in image.enumerate_pixels_mut() {
            *pixel = image::Rgb([x as f32 * 0.25, y as f32 * 0.5, 0.25]);
        }
        let tile = DetectionTile {
            rect: [0.0, 0.0, 4.0, 2.0],
            input_edge: 192,
        };
        let input = prepare_blazeface_tile(&image, tile).unwrap();
        assert_eq!(input.shape(), &[1, 192, 192, 3]);
        // 期望值来自 cv2 复刻探针（角对齐 + 逐点 border=-1，同 MediaPipe 链路）。
        let close = |a: f32, b: f32| (a - b).abs() < 1e-6;
        let expect = |row: usize, col: usize, values: [f32; 3]| {
            for c in 0..3 {
                assert!(
                    close(input[[0, row, col, c]], values[c]),
                    "({row},{col},{c}) = {} 期望 {}",
                    input[[0, row, col, c]],
                    values[c]
                );
            }
        };
        expect(0, 0, [-1.0, -1.0, -1.0]); // src(0,-1) 全越界
        expect(24, 0, [-1.0, -1.0, -0.75]); // src(0,-0.5) 半越界
        expect(48, 0, [-1.0, -1.0, -0.5]); // src(0,0)
        expect(96, 0, [-1.0, 0.0, -0.5]); // src(0,1)
        expect(48, 96, [0.0, -1.0, -0.5]); // src(2,0)
        expect(48, 120, [0.25, -1.0, -0.5]); // src(2.5,0) 双线性
        assert!(
            prepare_blazeface_tile(
                &image,
                DetectionTile {
                    rect: [3.0, 0.0, 4.0, 2.0],
                    input_edge: 192,
                }
            )
            .is_err()
        );
        assert!(prepare_blazeface_tile(&image::Rgb32FImage::new(0, 0), tile).is_err());
    }

    #[test]
    fn blazeface_unletterbox_maps_model_space_to_tile() {
        // tile 4×2 → side=4：scale_x=1、scale_y=2 → u'=u、v'=2v-0.5。
        let tile = DetectionTile {
            rect: [0.0, 0.0, 4.0, 2.0],
            input_edge: 192,
        };
        let close = |a: f32, b: f32| (a - b).abs() < 1e-6;
        let mapped = blazeface_tile_from_model(tile, 0.5, 0.5);
        assert!(close(mapped[0], 0.5) && close(mapped[1], 0.5));
        let mapped = blazeface_tile_from_model(tile, 0.0, 0.0);
        assert!(close(mapped[0], 0.0) && close(mapped[1], -0.5));
        let mapped = blazeface_tile_from_model(tile, 1.0, 0.75);
        assert!(close(mapped[0], 1.0) && close(mapped[1], 1.0));
        // 映射只依赖 w/h（ROI 中心与 tile 中心一致），与 tile 位置无关。
        let shifted = DetectionTile {
            rect: [10.0, 5.0, 8.0, 4.0],
            input_edge: 192,
        };
        let mapped = blazeface_tile_from_model(shifted, 0.5, 0.5);
        assert!(close(mapped[0], 0.5) && close(mapped[1], 0.5));
    }

    #[test]
    fn global_coordinates_deduplicate_overlapping_tiles() {
        struct Synthetic;
        impl ReshapeDetector for Synthetic {
            fn detect_tile(
                &mut self,
                _: &image::Rgb32FImage,
                tile: DetectionTile,
            ) -> Result<Vec<ReshapeDetection>, String> {
                let [x, y, w, h] = tile.rect;
                Ok(vec![ReshapeDetection {
                    bounds: [
                        (40.0 - x) / w,
                        (20.0 - y) / h,
                        (60.0 - x) / w,
                        (40.0 - y) / h,
                    ],
                    keypoints: vec![],
                    confidence: 0.95,
                }])
            }
        }
        let image = image::Rgb32FImage::new(100, 100);
        let tiles = [
            DetectionTile {
                rect: [0.0, 0.0, 80.0, 80.0],
                input_edge: 128,
            },
            DetectionTile {
                rect: [20.0, 0.0, 80.0, 80.0],
                input_edge: 256,
            },
        ];
        let faces = detect_tiles(&mut Synthetic, &image, &tiles, 0.4).unwrap();
        assert_eq!(faces.len(), 1);
        assert_eq!(faces[0].bounds, [40.0, 20.0, 60.0, 40.0]);
        assert!(detect_tiles(&mut Synthetic, &image, &[], 0.4).is_err());
    }
}
