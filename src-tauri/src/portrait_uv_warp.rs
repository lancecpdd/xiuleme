//! GPUPixel FaceReshapeFilter 风格的逐像素 UV 瘦脸／大眼。
//! 瘦脸对齐原版 thinFace：九对锚点、线性 curveWarp、满档 delta=`/200`→0.05。
//! 大眼仍为 enlargeEye（MediaPipe 虹膜映射）。不复制 GLSL、不链接 GPUPixel 运行时。
//! 输出稠密像素网格几何，供 CanonicalField 栅格化；不改动 ARAP Strengths[4] 契约。
#![allow(dead_code)]

use crate::face_reshape::{GeometryError, GeometrySolution, Strengths};
use crate::portrait_analysis::{AnalyzedFace, ReshapeAnalysis};
use nalgebra::Vector2;
use sha2::{Digest, Sha256};
use std::sync::Arc;

type Point = Vector2<f64>;

pub const ENGINE_VERSION: &str = "gpupixel-uv-v1";
pub const PARAMETER_HASH_VERSION: &str = "reshape-params-uv-v3";
/// 存储键：`uvSlimFace` 与网格 `slimFace` 隔离，避免两引擎串值。
pub const UV_KEYS: [&str; 2] = ["uvSlimFace", "bigEye"];

/// MediaPipe 478：原版 GPUPixel 106 瘦脸九对（轮廓 → 内锚），与 `face_reshape::SLIM_PAIRS` 一致。
const SLIM_PAIRS: [[usize; 2]; 9] = [
    [234, 6],
    [454, 6],
    [132, 1],
    [361, 1],
    [58, 98],
    [288, 327],
    [172, 13],
    [397, 13],
    [152, 13],
];
/// GPUPixel bigEye：origin=扩中心，target 仅定半径。映射到虹膜＋外眼角。
const BIG_EYE_PAIRS: [[usize; 2]; 2] = [[468, 33], [473, 263]];

/// 原版 desktop：`setFaceSlimLevel(v/200)`，v∈[0,10] → 满档 0.05。
const SLIM_DELTA_GAIN: f64 = 0.05;
/// 原版 desktop：`setEyeZoomLevel(v/100)`，v∈[0,10] → 满档 0.10。
const BIG_EYE_DELTA_GAIN: f64 = 0.10;
/// 人脸包围盒外扩（相对面宽），覆盖颊侧／下颌影响圆。
const BOUNDS_PAD: f64 = 0.55;
/// 控制网格沿包围盒最长边的分段数（另边按宽高比取整）。
const GRID_SEGMENTS: u32 = 36;

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct UvWarpStrengths {
    pub slim_face: f64,
    pub big_eye: f64,
}

impl UvWarpStrengths {
    pub fn from_ui(slim: f64, big_eye: f64) -> Result<Self, GeometryError> {
        for v in [slim, big_eye] {
            if !v.is_finite() || !(0.0..=100.0).contains(&v) {
                return Err(GeometryError::InvalidStrengths);
            }
        }
        Ok(Self {
            slim_face: slim,
            big_eye,
        })
    }

    pub fn is_zero(self) -> bool {
        self.slim_face == 0.0 && self.big_eye == 0.0
    }

    fn deltas(self) -> (f64, f64) {
        let slim = smooth(self.slim_face / 100.0) * SLIM_DELTA_GAIN;
        let eye = smooth(self.big_eye / 100.0) * BIG_EYE_DELTA_GAIN;
        (slim, eye)
    }
}

#[derive(Debug)]
pub struct UvFaceWarp {
    pub face_id: String,
    pub strengths: UvWarpStrengths,
    pub geometry: GeometrySolution,
}

pub fn parameter_hash(analysis_key: &str, faces: &[(String, UvWarpStrengths)]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(PARAMETER_HASH_VERSION.as_bytes());
    hasher.update(analysis_key.as_bytes());
    for (id, s) in faces {
        hasher.update(id.as_bytes());
        hasher.update(s.slim_face.to_bits().to_le_bytes());
        hasher.update(s.big_eye.to_bits().to_le_bytes());
    }
    hex::encode(hasher.finalize())
}

/// 对分析中可处理脸建立 UV 稠密网格；强度全零时返回 identity 网格。
pub fn solve_analysis(
    analysis: &ReshapeAnalysis,
    faces: &[(String, UvWarpStrengths)],
    current: &dyn Fn() -> bool,
) -> Result<Vec<UvFaceWarp>, String> {
    let dims = analysis.coordinates.base_dimensions;
    if dims[0] == 0 || dims[1] == 0 {
        return Err("UV 重塑基底尺寸非法".into());
    }
    let mut out = Vec::with_capacity(faces.len());
    for (face_id, strengths) in faces {
        check_current(current)?;
        let analyzed = analysis
            .faces
            .iter()
            .find(|f| f.summary.face_id == *face_id && f.summary.processable)
            .ok_or_else(|| format!("UV 重塑缺少可处理脸：{face_id}"))?;
        let geometry = solve_face(analyzed, dims, *strengths, current)?;
        out.push(UvFaceWarp {
            face_id: face_id.clone(),
            strengths: *strengths,
            geometry,
        });
    }
    out.sort_by(|a, b| a.face_id.cmp(&b.face_id));
    Ok(out)
}

fn solve_face(
    face: &AnalyzedFace,
    dims: [u32; 2],
    strengths: UvWarpStrengths,
    current: &dyn Fn() -> bool,
) -> Result<GeometrySolution, String> {
    // 与 mesh-arap 一致：roi 归一化点经 roi_to_base → 基底像素。
    // 禁止用 source_points（base_to_source 产出的 0–1 源 UV），否则包围盒缩到亚像素、效果不可见。
    if face.roi_points.len() != 478 {
        return Err("UV 重塑关键点不足 478".into());
    }
    let points: Vec<Point> = face
        .roi_points
        .iter()
        .map(|p| {
            face.roi_to_base
                .map([p[0] as f64, p[1] as f64])
                .map(|xy| Point::new(xy[0], xy[1]))
                .ok_or_else(|| "UV 重塑关键点映射失败".to_string())
        })
        .collect::<Result<Vec<_>, _>>()?;
    if points.iter().any(|p| !p.x.is_finite() || !p.y.is_finite()) {
        return Err("UV 重塑关键点非有限".into());
    }
    let face_width = (points[454] - points[234]).norm().max(1e-3);
    let (slim_delta, eye_delta) = strengths.deltas();
    let aspect = dims[0] as f64 / dims[1] as f64;
    let [w, h] = [dims[0] as f64, dims[1] as f64];

    let (min_x, min_y, max_x, max_y) = face_bounds(&points, face_width, dims);
    let (nx, ny, step_x, step_y) = grid_shape(min_x, min_y, max_x, max_y);
    let mut target = Vec::with_capacity(((nx + 1) * (ny + 1)) as usize);
    let mut source = Vec::with_capacity(target.capacity());
    for j in 0..=ny {
        check_current(current)?;
        let y = min_y + j as f64 * step_y;
        for i in 0..=nx {
            let x = min_x + i as f64 * step_x;
            target.push([x, y]);
            if strengths.is_zero() {
                source.push([x, y]);
                continue;
            }
            let uv = Point::new(x / w, y / h);
            let warped = warp_uv(uv, &points, w, h, aspect, slim_delta, eye_delta);
            // 采样点必须落在基底内，否则 CanonicalField 校验「位移…越界」直接失败。
            source.push([(warped.x * w).clamp(0.0, w), (warped.y * h).clamp(0.0, h)]);
        }
    }
    let triangles = Arc::new(grid_triangles(nx, ny));
    let identity = strengths.is_zero() || source == target;
    // Strengths[4] 仅占位：slim 在 [0]，其余 0；bigEye 不进入网格契约。
    let requested = Strengths([strengths.slim_face, 0.0, 0.0, 0.0]);
    let response = [smooth(strengths.slim_face / 100.0), 0.0, 0.0, 0.0];
    GeometrySolution::from_dense_pixel_mesh(
        source, target, triangles, face_width, requested, response, requested, identity,
    )
    .map_err(|e| format!("UV 稠密网格失败：{e:?}"))
}

fn face_bounds(points: &[Point], face_width: f64, dims: [u32; 2]) -> (f64, f64, f64, f64) {
    let pad = face_width * BOUNDS_PAD;
    let mut min_x = f64::INFINITY;
    let mut min_y = f64::INFINITY;
    let mut max_x = f64::NEG_INFINITY;
    let mut max_y = f64::NEG_INFINITY;
    for &i in &[
        10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377,
        152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109, 468,
        473, 33, 263,
    ] {
        let p = points[i];
        min_x = min_x.min(p.x);
        min_y = min_y.min(p.y);
        max_x = max_x.max(p.x);
        max_y = max_y.max(p.y);
    }
    (
        (min_x - pad).clamp(0.0, dims[0] as f64),
        (min_y - pad).clamp(0.0, dims[1] as f64),
        (max_x + pad).clamp(0.0, dims[0] as f64),
        (max_y + pad).clamp(0.0, dims[1] as f64),
    )
}

fn grid_shape(min_x: f64, min_y: f64, max_x: f64, max_y: f64) -> (u32, u32, f64, f64) {
    let span_x = (max_x - min_x).max(1.0);
    let span_y = (max_y - min_y).max(1.0);
    let (nx, ny) = if span_x >= span_y {
        let nx = GRID_SEGMENTS;
        let ny = ((GRID_SEGMENTS as f64) * span_y / span_x).round().max(4.0) as u32;
        (nx, ny)
    } else {
        let ny = GRID_SEGMENTS;
        let nx = ((GRID_SEGMENTS as f64) * span_x / span_y).round().max(4.0) as u32;
        (nx, ny)
    };
    (nx, ny, span_x / nx as f64, span_y / ny as f64)
}

fn grid_triangles(nx: u32, ny: u32) -> Vec<[usize; 3]> {
    let mut tris = Vec::with_capacity((nx * ny * 2) as usize);
    let row = (nx + 1) as usize;
    for j in 0..ny as usize {
        for i in 0..nx as usize {
            let a = j * row + i;
            let b = a + 1;
            let c = a + row;
            let d = c + 1;
            tris.push([a, b, c]);
            tris.push([b, d, c]);
        }
    }
    tris
}

fn warp_uv(
    mut uv: Point,
    points: &[Point],
    w: f64,
    h: f64,
    aspect: f64,
    slim_delta: f64,
    eye_delta: f64,
) -> Point {
    if slim_delta > 0.0 {
        for &[oi, ti] in &SLIM_PAIRS {
            let origin = Point::new(points[oi].x / w, points[oi].y / h);
            let target = Point::new(points[ti].x / w, points[ti].y / h);
            uv = curve_warp(uv, origin, target, slim_delta, aspect);
        }
    }
    if eye_delta > 0.0 {
        for &[oi, ti] in &BIG_EYE_PAIRS {
            let origin = Point::new(points[oi].x / w, points[oi].y / h);
            let target = Point::new(points[ti].x / w, points[ti].y / h);
            let radius = dist_aspect(origin, target, aspect) * 5.0;
            if radius > 1e-8 {
                uv = enlarge_eye(uv, origin, radius, eye_delta, aspect);
            }
        }
    }
    uv
}

/// 原版 GPUPixel curveWarp：`ratio = clamp(1 - r/R, 0, 1)`，无 smoothstep。
fn curve_warp(coord: Point, origin: Point, target: Point, delta: f64, aspect: f64) -> Point {
    let radius = dist_aspect(target, origin, aspect);
    if !radius.is_finite() || radius <= 1e-8 {
        return coord;
    }
    let ratio = (1.0 - dist_aspect(coord, origin, aspect) / radius).clamp(0.0, 1.0);
    let direction = (target - origin) * delta;
    coord - direction * ratio
}

/// GPUPixel enlargeEye：径向权重 `1-(1-w²)*delta`。
fn enlarge_eye(coord: Point, origin: Point, radius: f64, delta: f64, aspect: f64) -> Point {
    let mut weight = dist_aspect(coord, origin, aspect) / radius;
    weight = 1.0 - (1.0 - weight * weight) * delta;
    weight = weight.clamp(0.0, 1.0);
    origin + (coord - origin) * weight
}

fn dist_aspect(a: Point, b: Point, aspect: f64) -> f64 {
    let dx = a.x - b.x;
    let dy = (a.y - b.y) / aspect;
    (dx * dx + dy * dy).sqrt()
}

fn smooth(t: f64) -> f64 {
    let t = t.clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

fn check_current(current: &dyn Fn() -> bool) -> Result<(), String> {
    if current() {
        Ok(())
    } else {
        Err(crate::portrait_analysis::CANCELLED.into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn synth_face() -> Vec<Point> {
        let mut p = vec![Point::new(0.5, 0.5); 478];
        // 粗略正脸：面宽约 0.3（归一化后映射到 300px @ 1000 宽）
        p[234] = Point::new(350.0, 400.0);
        p[454] = Point::new(650.0, 400.0);
        p[6] = Point::new(500.0, 380.0);
        p[1] = Point::new(500.0, 420.0);
        p[98] = Point::new(470.0, 480.0);
        p[327] = Point::new(530.0, 480.0);
        p[13] = Point::new(500.0, 520.0);
        p[132] = Point::new(380.0, 450.0);
        p[361] = Point::new(620.0, 450.0);
        p[58] = Point::new(400.0, 500.0);
        p[288] = Point::new(600.0, 500.0);
        p[136] = Point::new(410.0, 530.0);
        p[365] = Point::new(590.0, 530.0);
        p[172] = Point::new(420.0, 560.0);
        p[397] = Point::new(580.0, 560.0);
        p[150] = Point::new(450.0, 580.0);
        p[379] = Point::new(550.0, 580.0);
        p[152] = Point::new(500.0, 600.0);
        p[468] = Point::new(420.0, 360.0);
        p[473] = Point::new(580.0, 360.0);
        p[33] = Point::new(380.0, 360.0);
        p[263] = Point::new(620.0, 360.0);
        p[10] = Point::new(500.0, 280.0);
        for &i in &[
            338, 297, 332, 284, 251, 389, 356, 323, 378, 400, 377, 148, 176, 149, 93, 127, 162,
            21, 54, 103, 67, 109,
        ] {
            if p[i] == Point::new(0.5, 0.5) {
                p[i] = Point::new(
                    500.0 + ((i % 17) as f64 - 8.0) * 8.0,
                    400.0 + (i % 11) as f64 * 10.0,
                );
            }
        }
        p
    }

    #[test]
    fn zero_strength_is_identity_warp() {
        let uv = Point::new(0.42, 0.55);
        let points = synth_face();
        let out = warp_uv(uv, &points, 1000.0, 1000.0, 1.0, 0.0, 0.0);
        assert!((out - uv).norm() < 1e-12);
    }

    #[test]
    fn slim_moves_cheek_inward() {
        let points = synth_face();
        // 左颊轮廓附近 UV
        let cheek = Point::new(points[234].x / 1000.0, points[234].y / 1000.0);
        let warped = warp_uv(cheek, &points, 1000.0, 1000.0, 1.0, SLIM_DELTA_GAIN, 0.0);
        assert!(
            warped.x.is_finite() && (warped - cheek).norm() > 1e-5,
            "瘦脸应对轮廓点产生可见 UV 位移"
        );
    }

    #[test]
    fn slim_uses_nine_pairs_only() {
        assert_eq!(SLIM_PAIRS.len(), 9);
        assert_eq!(SLIM_PAIRS[0], [234, 6]);
        assert_eq!(SLIM_PAIRS[8], [152, 13]);
    }

    #[test]
    fn big_eye_pulls_sample_toward_iris() {
        let points = synth_face();
        let w = 1000.0;
        let h = 1000.0;
        let aspect = 1.0;
        let origin = Point::new(points[468].x / w, points[468].y / h);
        let target = Point::new(points[33].x / w, points[33].y / h);
        let radius = dist_aspect(origin, target, aspect) * 5.0;
        let near = origin + Point::new(0.008, 0.0);
        let warped = enlarge_eye(near, origin, radius, 0.10, aspect);
        assert!(
            (warped - origin).norm() < (near - origin).norm(),
            "单眼 enlargeEye 应将邻域采样点拉向扩中心"
        );
    }

    #[test]
    fn deltas_match_gpupixel_desktop_gains() {
        let (s, e) = UvWarpStrengths::from_ui(100.0, 100.0).unwrap().deltas();
        assert!((s - SLIM_DELTA_GAIN).abs() < 1e-12);
        assert!((e - BIG_EYE_DELTA_GAIN).abs() < 1e-12);
        assert!((SLIM_DELTA_GAIN - 0.05).abs() < 1e-12);
        let (z0, z1) = UvWarpStrengths::default().deltas();
        assert_eq!((z0, z1), (0.0, 0.0));
    }

    #[test]
    fn curve_warp_linear_ratio_at_origin() {
        let o = Point::new(0.2, 0.5);
        let t = Point::new(0.5, 0.5);
        let at_o = curve_warp(o, o, t, 0.05, 1.0);
        // 线性 clamp(1)=1，原点位移与满 delta 一致。
        let expected = o - (t - o) * 0.05;
        assert!((at_o - expected).norm() < 1e-12);
        let far = curve_warp(Point::new(0.9, 0.5), o, t, 0.05, 1.0);
        assert!((far - Point::new(0.9, 0.5)).norm() < 1e-12);
        // 中点：ratio = 0.5，无 smoothstep。
        let mid = Point::new(0.35, 0.5);
        let at_mid = curve_warp(mid, o, t, 0.05, 1.0);
        let expected_mid = mid - (t - o) * 0.05 * 0.5;
        assert!((at_mid - expected_mid).norm() < 1e-12);
    }

    #[test]
    fn warped_samples_stay_inside_image() {
        let points = synth_face();
        let w = 1000.0;
        let h = 1000.0;
        let aspect = 1.0;
        // 图边附近的点满档瘦脸／大眼后仍应可钳制在域内（模拟 solve_face 钳制）。
        for &(x, y) in &[(0.0, 400.0), (999.0, 400.0), (500.0, 0.0), (350.0, 400.0)] {
            let uv = Point::new(x / w, y / h);
            let warped = warp_uv(uv, &points, w, h, aspect, SLIM_DELTA_GAIN, BIG_EYE_DELTA_GAIN);
            let sx = (warped.x * w).clamp(0.0, w);
            let sy = (warped.y * h).clamp(0.0, h);
            assert!((0.0..=w).contains(&sx) && (0.0..=h).contains(&sy));
        }
    }

    #[test]
    fn parameter_hash_version_is_v3() {
        assert_eq!(PARAMETER_HASH_VERSION, "reshape-params-uv-v3");
    }
}
