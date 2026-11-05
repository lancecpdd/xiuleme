//! 06 的保护输入：概率图、基底空间保护笔迹、局部直线和保守姿态代理。
//! 不加载模型，不把皮肤标签当作无遮挡证明；源 UV 笔迹重放由步骤 10 负责。
#![allow(dead_code)]
use crate::face_reshape::{GeometryError, PreparedFace, ProtectionConstraint};
use crate::portrait_adapters::ProtectionMaps;
use crate::portrait_analysis::AnalyzedFace;
use crate::portrait_coordinates::Affine;
use crate::portrait_models::sample_rgb;
use image::{GrayImage, Luma, Rgb32FImage};
use nalgebra::Vector2;
use std::collections::BTreeSet;

pub(crate) type P = Vector2<f64>;
pub(crate) type Triangle = [P; 3];
pub(crate) type Result<T> = std::result::Result<T, SafetyError>;

pub(crate) const MAX_PROBABILITY_VISITS_PER_FACE: usize = 2_000_000;
const MAX_PROTECTION_COORDINATE: f64 = 1e9;

#[derive(Clone, Debug, PartialEq)]
pub enum SafetyError {
    InvalidInput,
    StaleAnalysis,
    MissingProtection,
    UnreliableFaceDomain,
    InvalidProbability,
    InvalidProjection,
    WorkLimit,
    Cancelled,
    Geometry(GeometryError),
    UnsafeJacobian,
    ProtectedRegionMoved,
    OverlappingDeformation,
}
impl From<GeometryError> for SafetyError {
    fn from(error: GeometryError) -> Self {
        if error == GeometryError::Cancelled {
            Self::Cancelled
        } else {
            Self::Geometry(error)
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum LimitReason {
    Hair,
    Background,
    UncertainSegmentation,
    UserProtection,
    BackgroundLine,
    ImageBoundary,
    Overlap,
    GroupQualityLimited,
    LowConfidence,
    PoseLimited,
    UnsupportedPose,
    OcclusionUnverified,
    KnownOcclusion,
    AnalysisRejected,
    SafetyReduced,
    SafetyBypass,
}

/// 只接收同一未裁剪基底的像素边界坐标；没有隐式 source UV 或 ROI 转换。
/// erase 在上游合成用户保护层，不得移除模型头发／背景／不确定区保护。
#[derive(Clone, Debug)]
pub struct ProtectedStroke {
    pub points: Vec<[f64; 2]>,
    pub radius_pixels: f64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OcclusionEvidence {
    /// 默认：现有分割无法排除眼镜／耳饰／同肤色的手。
    Unknown,
    /// 由额外可信检测或人工标注报告，不由皮肤类别自动产生。
    KnownOccluded,
}

#[derive(Clone, Debug)]
pub struct QualityGate {
    /// 瘦脸／颧骨／下颌等通道的响应缩放（含遮挡）。
    pub response_scale: f64,
    /// 小脸通道缩放：姿态／置信度／分析拒绝仍生效，**不含**遮挡乘子。
    pub small_face_scale: f64,
    pub yaw_proxy_degrees: f64,
    pub reasons: BTreeSet<LimitReason>,
}

pub fn pose_response(degrees: f64) -> f64 {
    if !degrees.is_finite() {
        return 0.0;
    }
    1.0 - smooth((degrees.abs() - 25.0) / 15.0)
}

pub fn quality_gate(face: &AnalyzedFace, occlusion: OcclusionEvidence) -> Result<QualityGate> {
    if face.roi_points.len() != 478
        || face
            .roi_points
            .iter()
            .flatten()
            .any(|v| !v.is_finite() || v.abs() > 4.0)
        || [face.detection_confidence, face.landmark_confidence]
            .iter()
            .any(|v| !v.is_finite() || !(0.0..=1.0).contains(v))
    {
        return Err(SafetyError::InvalidInput);
    }
    let mut angles = Vec::new();
    for (a, b) in [(33, 263), (234, 454), (172, 397)] {
        let a = face.roi_points[a];
        let b = face.roi_points[b];
        let span = (b[0] - a[0]).hypot(b[1] - a[1]) as f64;
        if span <= 1e-5 {
            return Err(SafetyError::UnreliableFaceDomain);
        }
        angles.push(((b[2] - a[2]) as f64).atan2(span).to_degrees());
    }
    angles.sort_by(f64::total_cmp);
    // 相对 z 仅作弱透视偏航代理；三对估计不一致时不冒充精确姿态。
    let yaw = angles[1].abs();
    let mut reasons = BTreeSet::new();
    // Unknown 不再默认砍半：正脸仍可感知；仍记录 OcclusionUnverified 供面板提示。
    let occlusion_scale = match occlusion {
        OcclusionEvidence::Unknown => 0.85,
        OcclusionEvidence::KnownOccluded => 0.0,
    };
    if occlusion == OcclusionEvidence::Unknown {
        reasons.insert(LimitReason::OcclusionUnverified);
    }
    let mut base = pose_response(yaw);
    if yaw > 25.0 {
        reasons.insert(LimitReason::PoseLimited);
    }
    if yaw >= 40.0 || angles[2] - angles[0] > 20.0 {
        base = 0.0;
        reasons.insert(LimitReason::UnsupportedPose);
    }
    let confidence = face.detection_confidence.min(face.landmark_confidence) as f64;
    base *= smooth((confidence - 0.5) / 0.35);
    if confidence < 0.85 {
        reasons.insert(LimitReason::LowConfidence);
    }
    if !face.summary.processable || face.summary.reason.is_some() {
        base = 0.0;
        reasons.insert(LimitReason::AnalysisRejected);
    }
    let mut response_scale = occlusion_scale * base;
    if occlusion == OcclusionEvidence::KnownOccluded {
        response_scale = 0.0;
        reasons.insert(LimitReason::KnownOcclusion);
    }
    // 小脸整头相似缩：不受遮挡门控；极端姿态／低置信／分析拒绝仍旁路。
    let small_face_scale = base;
    Ok(QualityGate {
        response_scale,
        small_face_scale,
        yaw_proxy_degrees: yaw,
        reasons,
    })
}

pub(crate) fn checked_inverse(transform: &Affine) -> Result<Affine> {
    if transform.0.iter().any(|v| !v.is_finite())
        || transform.0[(2, 0)] != 0.0
        || transform.0[(2, 1)] != 0.0
        || transform.0[(2, 2)] != 1.0
    {
        return Err(SafetyError::InvalidProjection);
    }
    let inverse = transform.inverse().ok_or(SafetyError::InvalidProjection)?;
    if inverse.0.iter().any(|v| !v.is_finite()) {
        return Err(SafetyError::InvalidProjection);
    }
    Ok(inverse)
}

pub(crate) fn validate_strokes(strokes: &[ProtectedStroke]) -> Result<()> {
    if strokes.len() > 32 {
        return Err(SafetyError::WorkLimit);
    }
    for stroke in strokes {
        if stroke.points.is_empty()
            || stroke.points.len() > 256
            || !stroke.radius_pixels.is_finite()
            || !(0.0..=MAX_PROTECTION_COORDINATE).contains(&stroke.radius_pixels)
            || stroke.radius_pixels == 0.0
            || stroke
                .points
                .iter()
                .flatten()
                .any(|v| !v.is_finite() || v.abs() > MAX_PROTECTION_COORDINATE)
        {
            return Err(SafetyError::InvalidInput);
        }
    }
    Ok(())
}

/// 概率像素中心为 (x+.5,y+.5)。全图图层覆盖基底，ROI 图层只增加保护。
pub struct ProbabilityPlane<'a> {
    map: &'a ProtectionMaps,
    base_to_grid: Affine,
    covers_base: bool,
}
impl<'a> ProbabilityPlane<'a> {
    pub fn new(map: &'a ProtectionMaps, unit_to_base: &Affine, covers_base: bool) -> Result<Self> {
        if map.width != 256
            || map.height != 256
            || map.probabilities.len() != 65536
            || map.probabilities.iter().any(|p| {
                p.iter().any(|v| !v.is_finite() || !(0.0..=1.0).contains(v))
                    || (p.iter().sum::<f32>() - 1.0).abs() > 1e-4
            })
        {
            return Err(SafetyError::InvalidProbability);
        }
        let inverse = checked_inverse(unit_to_base)?;
        let base_to_grid = inverse.then(&Affine::new([256.0, 0.0, 0.0, 0.0, 256.0, 0.0]));
        if base_to_grid.0.iter().any(|v| !v.is_finite()) {
            return Err(SafetyError::InvalidProjection);
        }
        Ok(Self {
            map,
            base_to_grid,
            covers_base,
        })
    }

    pub fn sample(&self, point: [f64; 2]) -> Option<[f64; 4]> {
        let p = self.base_to_grid.map(point)?;
        if p.iter().any(|v| !(0.0..=256.0).contains(v)) {
            return None;
        }
        let x = (p[0] - 0.5).clamp(0.0, 255.0);
        let y = (p[1] - 0.5).clamp(0.0, 255.0);
        let (ix, iy) = (x.floor() as usize, y.floor() as usize);
        Some(std::array::from_fn(|c| {
            let at = |x: usize, y: usize| {
                self.map.probabilities[y.min(255) * 256 + x.min(255)][c] as f64
            };
            (at(ix, iy) * (1.0 - x.fract()) + at(ix + 1, iy) * x.fract()) * (1.0 - y.fract())
                + (at(ix, iy + 1) * (1.0 - x.fract()) + at(ix + 1, iy + 1) * x.fract()) * y.fract()
        }))
    }

    fn classify(
        &self,
        triangle: Triangle,
        remaining_visits: &mut usize,
        current: &dyn Fn() -> bool,
    ) -> Result<TriangleRisk> {
        let transformed = triangle.map(|p| self.base_to_grid.map(arr(p)).map(P::from));
        let [Some(a), Some(b), Some(c)] = transformed else {
            return Err(SafetyError::InvalidProjection);
        };
        let t = [a, b, c];
        let bounds = bounds(&t);
        let mut risk = TriangleRisk::default();
        if self.covers_base
            && t.iter()
                .any(|p| p.x < 0.0 || p.y < 0.0 || p.x > 256.0 || p.y > 256.0)
        {
            risk.hard = true;
            risk.strength = 1.0;
            risk.reasons.insert(LimitReason::ImageBoundary);
        }
        // 扫描双线性核的完整支持域，避免只在顶点采样而漏掉三角形内部的一根头发。
        let lo = [
            ((bounds[0] - 1.5).floor() as i32).clamp(0, 255),
            ((bounds[1] - 1.5).floor() as i32).clamp(0, 255),
        ];
        let hi = [
            ((bounds[2] + 0.5).ceil() as i32).clamp(0, 255),
            ((bounds[3] + 0.5).ceil() as i32).clamp(0, 255),
        ];
        if bounds[2] < 0.0 || bounds[3] < 0.0 || bounds[0] > 256.0 || bounds[1] > 256.0 {
            return Ok(risk);
        }
        for y in lo[1]..=hi[1] {
            check(current)?;
            *remaining_visits = remaining_visits
                .checked_sub((hi[0] - lo[0] + 1) as usize)
                .ok_or(SafetyError::WorkLimit)?;
            for x in lo[0]..=hi[0] {
                let square = [
                    P::new(x as f64 - 0.5, y as f64 - 0.5),
                    P::new(x as f64 + 1.5, y as f64 - 0.5),
                    P::new(x as f64 + 1.5, y as f64 + 1.5),
                    P::new(x as f64 - 0.5, y as f64 + 1.5),
                ];
                if !convex_intersects(&t, &square) {
                    continue;
                }
                let p = self.map.probabilities[y as usize * 256 + x as usize].map(f64::from);
                let uncertainty = 1.0 - p.into_iter().fold(0.0, f64::max);
                risk.strength = risk.strength.max(p[0].max(p[1]).max(p[3]).max(uncertainty));
                if p[0] > 0.1 {
                    risk.reasons.insert(LimitReason::Background);
                }
                if p[1] > 0.1 {
                    risk.reasons.insert(LimitReason::Hair);
                }
                if p[3] > 0.1 || uncertainty > 0.2 {
                    risk.reasons.insert(LimitReason::UncertainSegmentation);
                }
                risk.hard |= p[0] >= 0.85 || p[1] >= 0.85 || p[3] >= 0.6 || uncertainty >= 0.5;
            }
        }
        Ok(risk)
    }
}

#[derive(Clone, Debug, Default)]
struct TriangleRisk {
    strength: f64,
    hard: bool,
    reasons: BTreeSet<LimitReason>,
}

#[derive(Clone, Debug)]
pub struct ProtectedRegion {
    pub triangle: usize,
    pub strength: f64,
    pub hard: bool,
    pub reasons: BTreeSet<LimitReason>,
}

#[derive(Clone, Debug)]
pub struct ProtectionPlan {
    pub(crate) fixed: Vec<bool>,
    weights: Vec<f64>,
    pub regions: Vec<ProtectedRegion>,
}
impl ProtectionPlan {
    pub(crate) fn empty(vertices: usize) -> Self {
        Self {
            fixed: vec![false; vertices],
            weights: vec![0.0; vertices],
            regions: Vec::new(),
        }
    }

    pub(crate) fn lock_triangle(&mut self, ids: [usize; 3], triangle: usize, reason: LimitReason) {
        for i in ids {
            self.fixed[i] = true;
        }
        if let Some(region) = self.regions.iter_mut().find(|r| r.triangle == triangle) {
            region.hard = true;
            region.strength = 1.0;
            region.reasons.insert(reason);
        } else {
            self.regions.push(ProtectedRegion {
                triangle,
                strength: 1.0,
                hard: true,
                reasons: BTreeSet::from([reason]),
            });
        }
    }
    pub(crate) fn constraints(&self) -> Vec<ProtectionConstraint> {
        (0..self.fixed.len())
            .filter_map(|i| {
                if self.fixed[i] {
                    Some(ProtectionConstraint::Fixed { vertex: i })
                } else if self.weights[i] > 0.0 {
                    Some(ProtectionConstraint::Soft {
                        vertex: i,
                        weight: self.weights[i],
                    })
                } else {
                    None
                }
            })
            .collect()
    }

    /// 小脸整头缩：释放仅由头发／分割不确定导致的硬锁与对应软权，
    /// 保留背景、画面边界、用户笔迹、背景线、交叠硬区，避免发丝钉死弯折。
    pub(crate) fn release_hair_for_small_face(&self, triangles: &[[usize; 3]]) -> Self {
        let n = self.fixed.len();
        let mut keep_fixed = vec![false; n];
        let mut hair_only_hard = vec![false; n];
        for region in &self.regions {
            if region.triangle >= triangles.len() {
                continue;
            }
            let retain = region.reasons.iter().any(|r| {
                matches!(
                    r,
                    LimitReason::Background
                        | LimitReason::UserProtection
                        | LimitReason::ImageBoundary
                        | LimitReason::BackgroundLine
                        | LimitReason::Overlap
                )
            });
            let hairish = region
                .reasons
                .iter()
                .any(|r| matches!(r, LimitReason::Hair | LimitReason::UncertainSegmentation));
            for &v in &triangles[region.triangle] {
                if v >= n {
                    continue;
                }
                if region.hard && retain {
                    keep_fixed[v] = true;
                }
                if region.hard && hairish && !retain {
                    hair_only_hard[v] = true;
                }
            }
        }
        // 无区域标注但仍固定的顶点（例如外环由求解器另钉）保持；仅清「纯头发硬锁」。
        let mut fixed = self.fixed.clone();
        for i in 0..n {
            if hair_only_hard[i] && !keep_fixed[i] {
                fixed[i] = false;
            }
        }
        // 保留靠近必须硬锁顶点的软权；清掉纯发丝羽化阻尼。
        let mut weights = self.weights.clone();
        for i in 0..n {
            if fixed[i] || keep_fixed[i] {
                continue;
            }
            if hair_only_hard[i] {
                weights[i] = 0.0;
            }
        }
        Self {
            fixed,
            weights,
            regions: self.regions.clone(),
        }
    }

    /// 快照冻结：导出固定集与含 0 占位的软约束权重（安全报告只保留区域，不含这两项）。
    pub(crate) fn frozen(&self) -> (Vec<bool>, Vec<f64>) {
        (self.fixed.clone(), self.weights.clone())
    }

    /// 快照恢复：尺寸与数值域与 `build_plan` 输出一致，非法输入拒绝。
    pub(crate) fn restore(
        fixed: Vec<bool>,
        weights: Vec<f64>,
        regions: Vec<ProtectedRegion>,
    ) -> std::result::Result<Self, String> {
        if fixed.len() != weights.len() || fixed.is_empty() || fixed.len() > 200_000 {
            return Err("保护计划顶点数与网格拓扑不符".into());
        }
        // 与 06 软约束契约一致：权重必须落在 [0, 10000]（0 表示该顶点无软约束）。
        if weights
            .iter()
            .any(|w| !w.is_finite() || !(0.0..=10000.0).contains(w))
        {
            return Err("保护计划软约束权重越界".into());
        }
        for region in &regions {
            if region.triangle >= 400_000
                || !region.strength.is_finite()
                || !(0.0..=1.0).contains(&region.strength)
            {
                return Err("保护计划区域与网格不符".into());
            }
        }
        Ok(Self {
            fixed,
            weights,
            regions,
        })
    }
}

pub(crate) fn build_plan(
    face: &PreparedFace,
    planes: &[ProbabilityPlane<'_>],
    strokes: &[ProtectedStroke],
    lines: &[ProtectedStroke],
    small_face_active: bool,
    current: &dyn Fn() -> bool,
) -> Result<ProtectionPlan> {
    if planes.is_empty() || !planes.iter().any(|p| p.covers_base) {
        return Err(SafetyError::MissingProtection);
    }
    if planes.len() > 2 {
        return Err(SafetyError::WorkLimit);
    }
    validate_strokes(strokes)?;
    validate_strokes(lines)?;
    let mut remaining_visits = MAX_PROBABILITY_VISITS_PER_FACE;
    let vertices: Vec<P> = face.source_vertices().into_iter().map(P::from).collect();
    let mut plan = ProtectionPlan {
        fixed: vec![false; vertices.len()],
        weights: vec![0.0; vertices.len()],
        regions: Vec::new(),
    };
    // 脸内轮廓（颊／颌／下脸）是小脸／瘦脸可见性的关键；分割常把轮廓误标为发／背景／手遮挡。
    // 用户笔迹与画面边界仍硬锁；其余脸内硬保护降为软约束，让轮廓可动。
    // 小脸开启时扩大到头壳半径，发际硬锁降为软约束，避免发缘钉死。
    let interior_radius = if small_face_active { 1.35 } else { 1.10 };
    for (index, ids) in face.triangles().iter().enumerate() {
        check(current)?;
        let triangle = ids.map(|i| vertices[i]);
        let mut risk = TriangleRisk::default();
        for plane in planes {
            let part = plane.classify(triangle, &mut remaining_visits, current)?;
            risk.strength = risk.strength.max(part.strength);
            risk.hard |= part.hard;
            risk.reasons.extend(part.reasons);
        }
        for (input, reason) in [
            (strokes, LimitReason::UserProtection),
            (lines, LimitReason::BackgroundLine),
        ] {
            for stroke in input {
                let hit = if stroke.points.len() == 1 {
                    capsule_hits(
                        triangle,
                        P::from(stroke.points[0]),
                        P::from(stroke.points[0]),
                        stroke.radius_pixels,
                    )
                } else {
                    stroke.points.windows(2).any(|p| {
                        capsule_hits(triangle, P::from(p[0]), P::from(p[1]), stroke.radius_pixels)
                    })
                };
                if hit {
                    risk.hard = true;
                    risk.strength = 1.0;
                    risk.reasons.insert(reason);
                }
            }
        }
        let interior = face.triangle_centroid_radius(*ids) < interior_radius;
        let must_hard = risk.reasons.contains(&LimitReason::UserProtection)
            || risk.reasons.contains(&LimitReason::ImageBoundary);
        if risk.hard && interior && !must_hard {
            risk.hard = false;
            risk.strength = risk.strength.max(0.85);
        }
        if risk.hard {
            for &i in ids {
                plan.fixed[i] = true;
            }
        }
        if risk.strength > 0.1 || risk.hard {
            plan.regions.push(ProtectedRegion {
                triangle: index,
                strength: risk.strength,
                hard: risk.hard,
                reasons: risk.reasons,
            });
        }
    }
    feather_constraints(face, &mut plan, current)?;
    Ok(plan)
}

pub(crate) fn feather_constraints(
    face: &PreparedFace,
    plan: &mut ProtectionPlan,
    current: &dyn Fn() -> bool,
) -> Result<()> {
    let vertices: Vec<P> = face.source_vertices().into_iter().map(P::from).collect();
    // 先在几何约束上按面宽铺开过渡带，不对求解后的位移场乘陡峭掩膜。
    let feather = face.face_width() * 0.08;
    for (i, &point) in vertices.iter().enumerate() {
        check(current)?;
        if plan.fixed[i] {
            continue;
        }
        for region in &plan.regions {
            let triangle = face.triangles()[region.triangle].map(|i| vertices[i]);
            let distance = point_triangle_distance(point, triangle);
            if distance < feather {
                let risk = if region.hard {
                    1.0
                } else {
                    smooth((region.strength - 0.1) / 0.75)
                };
                // 硬保护区外羽化仍高权；软保护区（含脸内降级）权重量级接近五官手柄，避免把轮廓钉死。
                let ceiling = if region.hard { 10000.0 } else { 160.0 };
                plan.weights[i] =
                    plan.weights[i].max(ceiling * risk * smooth(1.0 - distance / feather));
            }
        }
    }
    Ok(())
}

/// 仅在本脸影响域内生成至多 256² 灰度图。8 位仅用于找边，不用于图像输出。
pub(crate) fn background_lines(
    image: &Rgb32FImage,
    face: &PreparedFace,
    global: &ProbabilityPlane<'_>,
    current: &dyn Fn() -> bool,
) -> Result<Vec<ProtectedStroke>> {
    let vertices: Vec<P> = face.source_vertices().into_iter().map(P::from).collect();
    let b = bounds(&vertices);
    let lo = P::new(b[0].max(0.0), b[1].max(0.0));
    let hi = P::new(
        b[2].min(image.width() as f64),
        b[3].min(image.height() as f64),
    );
    if hi.x <= lo.x || hi.y <= lo.y {
        return Ok(Vec::new());
    }
    let step = ((hi.x - lo.x).max(hi.y - lo.y) / 256.0).max(1.0);
    let w = (((hi.x - lo.x) / step).ceil() as u32).min(256);
    let h = (((hi.y - lo.y) / step).ceil() as u32).min(256);
    if w < 8 || h < 8 {
        return Ok(Vec::new());
    }
    let mut gray = GrayImage::new(w, h);
    for y in 0..h {
        check(current)?;
        for x in 0..w {
            let p = lo + P::new(x as f64 + 0.5, y as f64 + 0.5) * step;
            let rgb = sample_rgb(image, p.x as f32, p.y as f32);
            if rgb
                .iter()
                .any(|v| !v.is_finite() || !(0.0..=1.0).contains(v))
            {
                return Err(SafetyError::InvalidInput);
            }
            gray.put_pixel(
                x,
                y,
                Luma([
                    (255.0 * (0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2])).round() as u8,
                ]),
            );
        }
    }
    let mut edges = imageproc::edges::canny(&gray, 25.0, 60.0);
    let mut samples = Vec::new();
    for y in 0..h {
        check(current)?;
        for x in 0..w {
            let p = lo + P::new(x as f64 + 0.5, y as f64 + 0.5) * step;
            if global.sample(arr(p)).is_none_or(|v| v[0] < 0.4) {
                edges.put_pixel(x, y, Luma([0]));
            } else if edges.get_pixel(x, y)[0] != 0 {
                samples.push(P::new(x as f64, y as f64));
            }
        }
    }
    let minimum = (face.face_width() * 0.35 / step).max(12.0);
    let lines = imageproc::hough::detect_lines(
        &edges,
        imageproc::hough::LineDetectionOptions {
            vote_threshold: minimum.ceil() as u32,
            suppression_radius: 8,
        },
    );
    check(current)?;
    if lines.len() > 64 {
        return Err(SafetyError::WorkLimit);
    }
    let mut output = Vec::new();
    for line in lines {
        check(current)?;
        let angle = (line.angle_in_degrees as f64).to_radians();
        let normal = P::new(angle.cos(), angle.sin());
        let tangent = P::new(-normal.y, normal.x);
        let mut support: Vec<f64> = samples
            .iter()
            .filter(|p| (p.dot(&normal) - line.r as f64).abs() <= 1.5)
            .map(|p| p.dot(&tangent))
            .collect();
        support.sort_by(f64::total_cmp);
        let mut start = 0;
        for end in 1..=support.len() {
            if end < support.len() && support[end] - support[end - 1] <= 3.0 {
                continue;
            }
            if end > start && support[end - 1] - support[start] >= minimum {
                let position =
                    |t| arr(lo + (normal * line.r as f64 + tangent * t + P::repeat(0.5)) * step);
                output.push(ProtectedStroke {
                    points: vec![position(support[start]), position(support[end - 1])],
                    radius_pixels: (step * 2.0).max(face.face_width() * 0.01),
                });
                if output.len() > 32 {
                    return Err(SafetyError::WorkLimit);
                }
            }
            start = end;
        }
    }
    Ok(output)
}

pub(crate) fn check(current: &dyn Fn() -> bool) -> Result<()> {
    if current() {
        Ok(())
    } else {
        Err(SafetyError::Cancelled)
    }
}
pub(crate) fn smooth(t: f64) -> f64 {
    let t = t.clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}
pub(crate) fn arr(p: P) -> [f64; 2] {
    [p.x, p.y]
}
pub(crate) fn bounds(points: &[P]) -> [f64; 4] {
    points.iter().fold(
        [
            f64::INFINITY,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::NEG_INFINITY,
        ],
        |b, p| [b[0].min(p.x), b[1].min(p.y), b[2].max(p.x), b[3].max(p.y)],
    )
}
pub(crate) fn convex_intersects(a: &[P], b: &[P]) -> bool {
    let x = bounds(a);
    let y = bounds(b);
    if x[0] > y[2] || y[0] > x[2] || x[1] > y[3] || y[1] > x[3] {
        return false;
    }
    for polygon in [a, b] {
        for i in 0..polygon.len() {
            let edge = polygon[(i + 1) % polygon.len()] - polygon[i];
            let normal = P::new(-edge.y, edge.x);
            let range = |points: &[P]| {
                points
                    .iter()
                    .fold([f64::INFINITY, f64::NEG_INFINITY], |r, p| {
                        let t = p.dot(&normal);
                        [r[0].min(t), r[1].max(t)]
                    })
            };
            let x = range(a);
            let y = range(b);
            if x[0] > y[1] + 1e-10 || y[0] > x[1] + 1e-10 {
                return false;
            }
        }
    }
    true
}
fn point_segment_distance(p: P, a: P, b: P) -> f64 {
    let d = b - a;
    let length = d.norm_squared();
    if length <= 1e-20 {
        return (p - a).norm();
    }
    (p - a - d * ((p - a).dot(&d) / length).clamp(0.0, 1.0)).norm()
}
fn inside(p: P, t: Triangle) -> bool {
    let cross = |a: P, b: P| a.x * b.y - a.y * b.x;
    let signs: Vec<_> = (0..3)
        .map(|i| cross(t[(i + 1) % 3] - t[i], p - t[i]))
        .collect();
    signs.iter().all(|&s| s >= -1e-10) || signs.iter().all(|&s| s <= 1e-10)
}
fn point_triangle_distance(p: P, t: Triangle) -> f64 {
    if inside(p, t) {
        0.0
    } else {
        (0..3)
            .map(|i| point_segment_distance(p, t[i], t[(i + 1) % 3]))
            .fold(f64::INFINITY, f64::min)
    }
}
fn capsule_hits(t: Triangle, a: P, b: P, radius: f64) -> bool {
    if point_triangle_distance(a, t) <= radius || point_triangle_distance(b, t) <= radius {
        return true;
    }
    if convex_intersects(&t, &[a, b]) {
        return true;
    }
    t.iter().any(|&p| point_segment_distance(p, a, b) <= radius)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;

    #[test]
    fn probability_scan_budget_and_row_cancellation_fail_closed() {
        let map = ProtectionMaps {
            width: 256,
            height: 256,
            probabilities: vec![[0.0, 0.0, 1.0, 0.0]; 65536],
        };
        let plane =
            ProbabilityPlane::new(&map, &Affine::new([256.0, 0.0, 0.0, 0.0, 256.0, 0.0]), true)
                .unwrap();
        let triangle = [P::new(10.0, 10.0), P::new(20.0, 10.0), P::new(10.0, 20.0)];
        let mut remaining = 0;
        assert!(matches!(
            plane.classify(triangle, &mut remaining, &|| true),
            Err(SafetyError::WorkLimit)
        ));
        let mut remaining = MAX_PROBABILITY_VISITS_PER_FACE;
        let calls = Cell::new(0);
        assert!(matches!(
            plane.classify(triangle, &mut remaining, &|| {
                calls.set(calls.get() + 1);
                calls.get() < 3
            }),
            Err(SafetyError::Cancelled)
        ));
        assert_eq!(calls.get(), 3);
        assert!(remaining < MAX_PROBABILITY_VISITS_PER_FACE);
        let result = plane.classify(triangle, &mut remaining, &|| true).unwrap();
        assert!(!result.hard);
        assert_eq!(result.strength, 0.0);
    }
}
