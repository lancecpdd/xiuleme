//! 05：纯几何原型。只消费冻结分析，不调用模型、GPU、文件或旧修肤管线。
//! 控制网格是本项目编写的径向拓扑，不是 MediaPipe 的官方面部三角网。
//! 语义索引参考 MediaPipe v0.10.21 face_mesh_connections.py（Apache-2.0）；
//! 索引来源版权：Copyright 2021 The MediaPipe Authors.
//! 许可入口：https://www.apache.org/licenses/LICENSE-2.0
//! 来源、许可边界及数值约定见 docs/portrait-reshape/05-geometry-engine.md。
#![allow(dead_code)] // 06–08 尚未接入；不为消除告警提前启用渲染。

use crate::portrait_analysis::AnalyzedFace;
use nalgebra::Vector2;
use std::collections::BTreeSet;
use std::sync::{Arc, OnceLock};

type Point = Vector2<f64>;
pub const ENGINE_VERSION: &str = "mesh-arap-v1";
pub const TOPOLOGY_VERSION: &str = "mp478-semantic-rings-v1";
pub const LANDMARK_VERSION: &str = "mediapipe-478-v1";
const RINGS: [f64; 6] = [0.22, 0.45, 0.70, 1.0, 1.25, 1.60];
// 从额顶开始，沿模型索引的同一侧绕到下巴及另一侧；镜像不交换索引。
const OVAL: [usize; 36] = [
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152,
    148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109,
];
const FEATURES: [[usize; 4]; 4] = [
    [33, 133, 159, 145],
    [263, 362, 386, 374],
    [168, 1, 98, 327],
    [61, 291, 13, 14],
];
/// MediaPipe 478 上对应 GPUPixel 106 瘦脸 9 对（轮廓 → 内锚），顺序与其 `thinFace` 循环一致。
/// 算法为自写重述，不复制其 GLSL；距离在面部局部坐标中度量（与像素各向同性距离成正比）。
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
/// 将 0–1 响应映到 GPUPixel 式 `delta`；满滑块应可见，联合求解仍受 `BUDGET` 同比缩放。
const SLIM_WARP_GAIN: f64 = 0.45;
/// 小脸满响应相似缩小比例（相对缩心）；仍受 `BUDGET` 同比缩放。
const SMALL_AMOUNT: f64 = 0.09;
/// 头壳有效半径（面宽单位，相对眼中点）：覆盖脸盘＋额＋侧发，颏下另衰减。
const SMALL_HEAD_RADIUS: f64 = 1.35;
/// 控制网格尺寸（公开契约）：顶点 = 中心 + 环数×表盘；三角 = 表盘×(顶环 + 内环×2)。
pub const MESH_VERTICES: usize = 1 + RINGS.len() * OVAL.len();
pub const MESH_TRIANGLES: usize = OVAL.len() * (1 + (RINGS.len() - 1) * 2);
const BUDGET: f64 = 0.14;
const AREA_EPS: f64 = 1e-9;
const QUERY_EPS: f64 = 1e-9;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum GeometryError {
    UnsupportedTopology,
    UnprocessableFace,
    InvalidLandmarks,
    InvalidStrengths,
    InvalidOptions,
    InvalidConstraint,
    DegenerateMesh,
    Foldover,
    SelfIntersection,
    BoundaryMoved,
    OutsideMesh,
    NonFinite,
    Cancelled,
    NotConverged {
        stage: &'static str,
        iterations: usize,
    },
}

type Result<T> = std::result::Result<T, GeometryError>;

/// 顺序与 portrait_reshape::KEYS 一致；调用前已完成全局／逐脸值的合并。
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Strengths(pub [f64; 4]);

impl Strengths {
    pub(crate) fn response(self) -> Result<[f64; 4]> {
        if self
            .0
            .iter()
            .any(|v| !v.is_finite() || !(0.0..=100.0).contains(v))
        {
            return Err(GeometryError::InvalidStrengths);
        }
        Ok(self.0.map(|v| smooth(v / 100.0)))
    }

    /// 在响应域统一减弱整个目标组，避免对已求位移逐点乘蒙版。
    pub fn scaled_response(self, scale: f64) -> Result<Self> {
        if !scale.is_finite() || !(0.0..=1.0).contains(&scale) {
            return Err(GeometryError::InvalidStrengths);
        }
        Ok(Self(self.response()?.map(|v| inverse_response(v * scale))))
    }

    /// 四通道各自缩放；`scales[1]` 用于小脸（可不受遮挡门控）。
    pub fn scaled_response_channels(self, scales: [f64; 4]) -> Result<Self> {
        if scales
            .iter()
            .any(|s| !s.is_finite() || !(0.0..=1.0).contains(s))
        {
            return Err(GeometryError::InvalidStrengths);
        }
        let response = self.response()?;
        Ok(Self(std::array::from_fn(|i| {
            inverse_response(response[i] * scales[i])
        })))
    }
}

/// 顺序与 portrait_reshape::EXTENDED_KEYS 一致；几何现阶段消费 noseWidth／noseLength／mouthSize。
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ExtendedStrengths(pub [f64; 8]);

impl ExtendedStrengths {
    pub fn nose_width(self) -> f64 {
        self.0[0]
    }

    pub fn nose_length(self) -> f64 {
        self.0[1]
    }

    pub fn mouth_size(self) -> f64 {
        self.0[2]
    }

    pub fn validate(self) -> Result<()> {
        if self
            .0
            .iter()
            .any(|v| !v.is_finite() || !(0.0..=100.0).contains(v))
        {
            return Err(GeometryError::InvalidStrengths);
        }
        Ok(())
    }

    pub fn response_nose_width(self) -> Result<f64> {
        self.validate()?;
        Ok(smooth(self.0[0] / 100.0))
    }

    pub fn response_nose_length(self) -> Result<f64> {
        self.validate()?;
        Ok(smooth(self.0[1] / 100.0))
    }

    pub fn response_mouth_size(self) -> Result<f64> {
        self.validate()?;
        Ok(smooth(self.0[2] / 100.0))
    }

    pub fn is_inactive(self) -> Result<bool> {
        self.validate()?;
        Ok(self.0.iter().all(|&v| v == 0.0))
    }

    /// 在响应域统一缩放全部扩展键（与轮廓四键共用质量／回退门控）。
    pub fn scaled(self, scale: f64) -> Result<Self> {
        if !scale.is_finite() || !(0.0..=1.0).contains(&scale) {
            return Err(GeometryError::InvalidStrengths);
        }
        self.validate()?;
        Ok(Self(std::array::from_fn(|i| {
            inverse_response(smooth(self.0[i] / 100.0) * scale)
        })))
    }
}

#[derive(Clone, Copy, Debug)]
pub struct SolverOptions {
    pub arap_iterations: usize,
    pub pcg_iterations: usize,
    pub arap_tolerance: f64,
    pub pcg_tolerance: f64,
}

impl Default for SolverOptions {
    fn default() -> Self {
        Self {
            arap_iterations: 48,
            pcg_iterations: 256,
            arap_tolerance: 1e-6,
            pcg_tolerance: 1e-9,
        }
    }
}

/// 06 的保护接口：硬约束为零位移；软约束权重必须为 (0, 10000]。
/// 索引绑定本 PreparedFace 的控制顶点，不是模型的关键点索引。
#[derive(Clone, Copy, Debug)]
pub enum ProtectionConstraint {
    Fixed { vertex: usize },
    Soft { vertex: usize, weight: f64 },
}

#[derive(Clone, Debug, Default)]
pub struct Diagnostics {
    pub arap_iterations: usize,
    pub pcg_iterations: usize,
    pub pcg_relative_residual: f64,
    pub last_step: f64,
    /// 距离均以面宽为单位；不是像素，也不是百分比。
    pub max_displacement: f64,
    pub target_rms_error: f64,
    pub feature_max_error: f64,
    pub min_area_ratio: f64,
    pub budget_scale: f64,
}

#[derive(Debug)]
struct Topology {
    triangles: Vec<[usize; 3]>,
    edges: Vec<(usize, usize)>,
    boundary: Vec<usize>,
}

fn vertex(ring: usize, segment: usize) -> usize {
    1 + ring * OVAL.len() + segment % OVAL.len()
}

fn topology() -> Arc<Topology> {
    static TOPOLOGY: OnceLock<Arc<Topology>> = OnceLock::new();
    TOPOLOGY
        .get_or_init(|| {
            let mut triangles = Vec::new();
            for j in 0..OVAL.len() {
                triangles.push([0, vertex(0, j), vertex(0, j + 1)]);
                for r in 1..RINGS.len() {
                    let (a, b, c, d) = (
                        vertex(r - 1, j),
                        vertex(r, j),
                        vertex(r, j + 1),
                        vertex(r - 1, j + 1),
                    );
                    triangles.extend([[a, b, c], [a, c, d]]);
                }
            }
            let mut edges = BTreeSet::new();
            for &[a, b, c] in &triangles {
                for (i, j) in [(a, b), (b, c), (c, a)] {
                    edges.insert((i.min(j), i.max(j)));
                }
            }
            Arc::new(Topology {
                triangles,
                edges: edges.into_iter().collect(),
                boundary: (0..OVAL.len())
                    .map(|j| vertex(RINGS.len() - 1, j))
                    .collect(),
            })
        })
        .clone()
}

#[derive(Clone, Debug)]
struct Frame {
    origin: Point,
    x: Point,
    y: Point,
    width: f64,
}

impl Frame {
    fn local(&self, p: Point) -> Point {
        let d = (p - self.origin) / self.width;
        Point::new(d.dot(&self.x), d.dot(&self.y))
    }
    fn world(&self, p: Point) -> Point {
        self.origin + self.width * (self.x * p.x + self.y * p.y)
    }
}

#[derive(Clone, Debug)]
struct Binding {
    vertices: [usize; 3],
    weights: [f64; 3],
}

#[derive(Debug)]
pub struct PreparedFace {
    topology: Arc<Topology>,
    frame: Frame,
    source: Vec<Point>,
    feature_bindings: Vec<(Binding, Point, usize)>,
    feature_centers: [Point; 4],
    /// 小脸缩心：局部坐标下两眼中点。
    shrink_center: Point,
    top: f64,
    chin: f64,
    cheek: f64,
    jaw: f64,
    slim_pairs: [(Point, Point); 9],
}

impl PreparedFace {
    /// 输入是 04 未裁剪基底的像素边界坐标，绝不把 source UV 当成像素。
    pub fn from_analysis(face: &AnalyzedFace) -> Result<Self> {
        if !face.summary.processable || face.summary.reason.is_some() {
            return Err(GeometryError::UnprocessableFace);
        }
        if face.roi_points.len() != 478
            || face
                .roi_points
                .iter()
                .flatten()
                .any(|v| !v.is_finite() || v.abs() > 4.0)
        {
            return Err(GeometryError::InvalidLandmarks);
        }
        let points = face
            .roi_points
            .iter()
            .map(|p| {
                face.roi_to_base
                    .map([p[0] as f64, p[1] as f64])
                    .ok_or(GeometryError::InvalidLandmarks)
            })
            .collect::<Result<Vec<_>>>()?;
        Self::from_landmarks(&points, LANDMARK_VERSION)
    }

    /// 纯几何入口供合成测试使用；真实入口仍须经过模型审批和 04 质量门控。
    pub fn from_landmarks(points: &[[f64; 2]], version: &str) -> Result<Self> {
        if version != LANDMARK_VERSION {
            return Err(GeometryError::UnsupportedTopology);
        }
        if points.len() != 478 || points.iter().flatten().any(|v| !v.is_finite()) {
            return Err(GeometryError::InvalidLandmarks);
        }
        let points: Vec<Point> = points.iter().map(|p| Point::new(p[0], p[1])).collect();
        let eye_a = (points[33] + points[133]) * 0.5;
        let eye_b = (points[263] + points[362]) * 0.5;
        let eye_delta = eye_b - eye_a;
        let eye_distance = eye_delta.norm();
        if !eye_distance.is_finite() || eye_distance <= 1e-8 {
            return Err(GeometryError::InvalidLandmarks);
        }
        let x = eye_delta / eye_distance;
        let mut y = Point::new(-x.y, x.x);
        let down = (points[1] + (points[61] + points[291]) * 0.5) * 0.5 - (eye_a + eye_b) * 0.5;
        if down.dot(&y) < 0.0 {
            y = -y;
        }
        let width = (points[454] - points[234]).dot(&x);
        if !width.is_finite() || width <= 1e-8 || !(0.2..=0.85).contains(&(eye_distance / width)) {
            return Err(GeometryError::InvalidLandmarks);
        }
        let frame = Frame {
            origin: points[168],
            x,
            y,
            width,
        };
        let p: Vec<Point> = points.iter().map(|v| frame.local(*v)).collect();
        if p.iter().any(|v| !finite(*v) || v.norm() > 4.0) {
            return Err(GeometryError::InvalidLandmarks);
        }
        // 重复投影点不进入求解；闭眼等退化输入也保守拒绝，不能声称支持所有姿态。
        for i in 0..p.len() {
            if p[..i].iter().any(|q| (p[i] - q).norm_squared() < 1e-16) {
                return Err(GeometryError::InvalidLandmarks);
            }
        }
        let top = p[10].y;
        let chin = p[152].y;
        if !(0.6..=2.0).contains(&(chin - top))
            || top >= -0.05
            || chin <= 0.2
            || p[1].y <= p[168].y
            || (p[61].y + p[291].y) * 0.5 <= p[1].y
        {
            return Err(GeometryError::InvalidLandmarks);
        }
        let topology = topology();
        let mut source = vec![Point::zeros()];
        for radius in RINGS {
            source.extend(OVAL.iter().map(|&i| p[i] * radius));
        }
        validate_mesh(&source, &source, &topology)?;
        let mut feature_bindings = Vec::new();
        let feature_centers = FEATURES.map(|ids| ids.iter().map(|&i| p[i]).sum::<Point>() / 4.0);
        for (group, ids) in FEATURES.iter().enumerate() {
            for &i in ids {
                feature_bindings.push((locate(p[i], &source, &topology)?, p[i], group));
            }
        }
        let eye_mid_local = (frame.local(eye_a) + frame.local(eye_b)) * 0.5;
        Ok(Self {
            topology,
            frame,
            source,
            feature_bindings,
            feature_centers,
            shrink_center: eye_mid_local,
            top,
            chin,
            cheek: (p[234].y + p[454].y) * 0.5,
            jaw: (p[172].y + p[397].y) * 0.5,
            slim_pairs: SLIM_PAIRS.map(|[o, t]| (p[o], p[t])),
        })
    }

    pub fn engine_version(&self) -> &'static str {
        ENGINE_VERSION
    }

    pub fn vertex_count(&self) -> usize {
        self.source.len()
    }
    pub fn face_width(&self) -> f64 {
        self.frame.width
    }

    /// 三角形重心相对鼻根的面宽归一化半径；用于保护是否侵入可重塑脸内区的判定。
    pub fn triangle_centroid_radius(&self, ids: [usize; 3]) -> f64 {
        ((self.source[ids[0]] + self.source[ids[1]] + self.source[ids[2]]) / 3.0).norm()
    }

    pub fn source_vertices(&self) -> Vec<[f64; 2]> {
        self.source
            .iter()
            .map(|p| array(self.frame.world(*p)))
            .collect()
    }
    pub fn triangles(&self) -> &[[usize; 3]] {
        &self.topology.triangles
    }

    /// 快照重放：用已接受的基底像素网格重建解，不重跑 ARAP；预算与网格合法性重新验证。
    /// 网格经 `frame.world → frame.local` 往返只有 ~1e-16 相对误差，恒等网格吸附回源顶点，
    /// 保持与原始求解一致的 identity 语义（否则重放会把零位移误判为微小位移）。
    pub(crate) fn restore_solution(
        &self,
        target: Vec<[f64; 2]>,
        requested: Strengths,
        effective_response: [f64; 4],
    ) -> Result<GeometrySolution> {
        if self.source.len() != MESH_VERTICES {
            return Err(GeometryError::UnsupportedTopology);
        }
        if target.len() != self.source.len()
            || target
                .iter()
                .flatten()
                .any(|v| !v.is_finite() || v.abs() > 1e9)
        {
            return Err(GeometryError::InvalidLandmarks);
        }
        if requested
            .0
            .iter()
            .any(|v| !v.is_finite() || !(0.0..=100.0).contains(v))
            || effective_response
                .iter()
                .any(|v| !v.is_finite() || !(0.0..=1.0).contains(v))
        {
            return Err(GeometryError::InvalidStrengths);
        }
        let mut local_target: Vec<Point> = target
            .iter()
            .map(|p| self.frame.local(Point::new(p[0], p[1])))
            .collect();
        // 边界在首次 ARAP 求解中绝对固定；世界→局部往返仅会给这些点引入舍入误差。
        // 不能因场内顶点已变形而跳过吸附，否则严格边界契约会把已接受快照误判为篡改。
        for &index in &self.topology.boundary {
            local_target[index] = self.source[index];
        }
        if local_target
            .iter()
            .zip(&self.source)
            .all(|(a, b)| (*a - *b).norm() <= 1e-9)
        {
            local_target.clone_from(&self.source);
        }
        let max_displacement = local_target
            .iter()
            .zip(&self.source)
            .map(|(a, b)| (a - b).norm())
            .fold(0.0, f64::max);
        // 预算按面宽比例；接受网格由 post_scale 收敛到 BUDGET，容差只吸收往返误差。
        if !max_displacement.is_finite() || max_displacement > BUDGET * (1.0 + 1e-6) {
            return Err(GeometryError::InvalidConstraint);
        }
        let min_area_ratio = validate_mesh(&self.source, &local_target, &self.topology)?;
        let diagnostics = Diagnostics {
            max_displacement,
            min_area_ratio,
            // 两层回退（质量／安全／外部请求）与预算比例不逐项保存在快照中；
            // 报告使用快照冻结的比例与原因集，此处只重建可精确恢复的网格诊断。
            budget_scale: 1.0,
            ..Diagnostics::default()
        };
        Ok(self.solution(local_target, requested, effective_response, diagnostics))
    }

    fn small_delta(&self, p: Point) -> Point {
        // 整头均匀相似缩小：缩心为眼中点，xy 同比例；发际随动，颏下／颈衰减到 0。
        let rel = p - self.shrink_center;
        let radial = smooth((SMALL_HEAD_RADIUS - rel.norm()) / 0.45);
        let neck = 1.0 - smooth((p.y - self.chin) / 0.22);
        let amount = SMALL_AMOUNT * radial * neck;
        -rel * amount
    }

    /// 成对局部 curveWarp 的顺向一阶目标：颊轮廓点沿指向内锚的向量内收。
    /// 衰减用 smooth(1−r/R)，减轻多圆线性叠加的折痕；外环仍由 `fade` 收到零。
    fn slim_delta(&self, p: Point, slim: f64) -> Point {
        if slim <= 0.0 {
            return Point::zeros();
        }
        let delta = slim * SLIM_WARP_GAIN;
        let mut coord = p;
        for &(origin, target) in &self.slim_pairs {
            coord = curve_warp(coord, origin, target, delta);
        }
        coord - p
    }

    fn target_delta(&self, p: Point, radius: f64, s: [f64; 4]) -> Point {
        let side = smooth((p.x.abs() - 0.08) / 0.36);
        let cheek = -p.x.signum()
            * 0.085
            * smooth((p.x.abs() - 0.18) / 0.24)
            * bump((p.y - self.cheek) / 0.26);
        let jaw = -p.x.signum() * 0.10 * side * bump((p.y - self.jaw) / 0.28);
        // 瘦脸／颧骨／下颌乘环 fade；小脸不乘，否则外过渡环学不到「头发跟缩」。
        let fade = smooth((1.6 - radius) / 0.75);
        (self.slim_delta(p, s[0]) + Point::new(s[2] * cheek + s[3] * jaw, 0.0)) * fade
            + s[1] * self.small_delta(p)
    }

    pub fn solve(
        &self,
        strengths: Strengths,
        extended: ExtendedStrengths,
        protection: &[ProtectionConstraint],
        options: SolverOptions,
        current: &dyn Fn() -> bool,
    ) -> Result<GeometrySolution> {
        if options.arap_iterations > 128
            || options.pcg_iterations > 1024
            || !options.arap_tolerance.is_finite()
            || !(1e-12..=1e-4).contains(&options.arap_tolerance)
            || !options.pcg_tolerance.is_finite()
            || !(1e-14..=1e-6).contains(&options.pcg_tolerance)
        {
            return Err(GeometryError::InvalidOptions);
        }
        let _ = extended;
        self.solve_arap_only(strengths, protection, options, current)
    }

    /// ARAP 求解（不接收扩展强度），供连续门失败兜底。
    pub fn solve_arap_only(
        &self,
        strengths: Strengths,
        protection: &[ProtectionConstraint],
        options: SolverOptions,
        current: &dyn Fn() -> bool,
    ) -> Result<GeometrySolution> {
        let response = strengths.response()?;
        if options.arap_iterations > 128
            || options.pcg_iterations > 1024
            || !options.arap_tolerance.is_finite()
            || !(1e-12..=1e-4).contains(&options.arap_tolerance)
            || !options.pcg_tolerance.is_finite()
            || !(1e-14..=1e-6).contains(&options.pcg_tolerance)
        {
            return Err(GeometryError::InvalidOptions);
        }
        check_current(current)?;
        let n = self.source.len();
        let mut fixed = vec![false; n];
        for &i in &self.topology.boundary {
            fixed[i] = true;
        }
        let mut soft = Vec::new();
        let mut seen = BTreeSet::new();
        if protection.len() > n {
            return Err(GeometryError::InvalidConstraint);
        }
        for constraint in protection {
            let i = match *constraint {
                ProtectionConstraint::Fixed { vertex }
                | ProtectionConstraint::Soft { vertex, .. } => vertex,
            };
            if i >= n || !seen.insert(i) {
                return Err(GeometryError::InvalidConstraint);
            }
            match *constraint {
                ProtectionConstraint::Fixed { .. } => fixed[i] = true,
                ProtectionConstraint::Soft { weight, .. }
                    if weight.is_finite() && weight > 0.0 && weight <= 10000.0 =>
                {
                    soft.push((i, weight))
                }
                _ => return Err(GeometryError::InvalidConstraint),
            }
        }
        if response == [0.0; 4] {
            return Ok(self.solution(
                self.source.clone(),
                strengths,
                response,
                Diagnostics {
                    budget_scale: 1.0,
                    min_area_ratio: 1.0,
                    ..Diagnostics::default()
                },
            ));
        }
        let mut delta: Vec<Point> = self
            .source
            .iter()
            .enumerate()
            .map(|(i, p)| {
                let radius = if i == 0 {
                    0.0
                } else {
                    RINGS[(i - 1) / OVAL.len()]
                };
                self.target_delta(*p, radius, response)
            })
            .collect();
        let feature_delta = self
            .feature_centers
            .map(|p| self.small_delta(p) * response[1]);
        let peak = delta
            .iter()
            .chain(feature_delta.iter())
            .map(|v| v.norm())
            .fold(0.0, f64::max);
        let scale = (BUDGET / peak.max(BUDGET)).min(1.0);
        for d in &mut delta {
            *d *= scale;
        }
        let mut handles: Vec<Handle> = delta
            .iter()
            .enumerate()
            .map(|(i, d)| Handle {
                binding: Binding {
                    vertices: [i, i, i],
                    weights: [1.0, 0.0, 0.0],
                },
                target: self.source[i] + d,
                weight: 4.0,
            })
            .collect();
        for (binding, p, group) in &self.feature_bindings {
            handles.push(Handle {
                binding: binding.clone(),
                target: p + feature_delta[*group] * scale,
                weight: 80.0,
            });
        }
        for (i, weight) in soft {
            handles.push(Handle {
                binding: Binding {
                    vertices: [i, i, i],
                    weights: [1.0, 0.0, 0.0],
                },
                target: self.source[i],
                weight,
            });
        }
        let mut known = vec![Point::zeros(); n];
        for i in 0..n {
            if fixed[i] {
                known[i] = self.source[i];
            }
        }
        // 下巴的两个坐标硬锁至小脸目标；其他三项绝不通过拉长下巴补偿收窄。
        let chin_vertex = vertex(3, 18);
        if !fixed[chin_vertex] {
            fixed[chin_vertex] = true;
            known[chin_vertex] = self.source[chin_vertex]
                + self.small_delta(self.source[chin_vertex]) * response[1] * scale;
        }
        let system = System::new(&self.topology.edges, handles, fixed);
        let mut q = self.source.clone();
        for i in 0..n {
            if system.fixed[i] {
                q[i] = known[i];
            }
        }
        let mut diagnostics = Diagnostics {
            budget_scale: scale,
            ..Diagnostics::default()
        };
        let mut converged = false;
        for outer in 0..options.arap_iterations {
            check_current(current)?;
            let mut cosine = vec![0.0; n];
            let mut sine = vec![0.0; n];
            for &(i, j) in &self.topology.edges {
                let rest = self.source[i] - self.source[j];
                let deformed = q[i] - q[j];
                let c = rest.dot(&deformed);
                let s = cross(rest, deformed);
                cosine[i] += c;
                cosine[j] += c;
                sine[i] += s;
                sine[j] += s;
            }
            let mut rotations = Vec::with_capacity(n);
            for i in 0..n {
                let norm = cosine[i].hypot(sine[i]);
                if !norm.is_finite() || norm <= 1e-14 {
                    return Err(GeometryError::DegenerateMesh);
                }
                rotations.push([cosine[i] / norm, sine[i] / norm]);
            }
            let mut rhs = vec![Point::zeros(); n];
            for &(i, j) in &self.topology.edges {
                let rest = self.source[i] - self.source[j];
                let d = (rotate(rest, rotations[i]) + rotate(rest, rotations[j])) * 0.5;
                rhs[i] += d;
                rhs[j] -= d;
            }
            for h in &system.handles {
                for k in 0..3 {
                    rhs[h.binding.vertices[k]] += h.target * (h.weight * h.binding.weights[k]);
                }
            }
            let previous = q.clone();
            for axis in 0..2 {
                let fixed_axis: Vec<f64> = known.iter().map(|p| p[axis]).collect();
                let fixed_rhs = system.apply(&fixed_axis, false);
                let b: Vec<f64> = (0..n)
                    .map(|i| {
                        if system.fixed[i] {
                            0.0
                        } else {
                            rhs[i][axis] - fixed_rhs[i]
                        }
                    })
                    .collect();
                let initial: Vec<f64> = (0..n)
                    .map(|i| if system.fixed[i] { 0.0 } else { q[i][axis] })
                    .collect();
                let (x, count, residual) = pcg(&system, &b, initial, options, current)?;
                diagnostics.pcg_iterations += count;
                diagnostics.pcg_relative_residual = diagnostics.pcg_relative_residual.max(residual);
                for i in 0..n {
                    q[i][axis] = x[i] + fixed_axis[i];
                }
            }
            diagnostics.arap_iterations = outer + 1;
            diagnostics.last_step = q
                .iter()
                .zip(&previous)
                .map(|(a, b)| (a - b).norm())
                .fold(0.0, f64::max);
            if q.iter().any(|p| !finite(*p)) {
                return Err(GeometryError::NonFinite);
            }
            if diagnostics.last_step <= options.arap_tolerance {
                converged = true;
                break;
            }
        }
        if !converged {
            return Err(GeometryError::NotConverged {
                stage: "arap",
                iterations: options.arap_iterations,
            });
        }
        let max_displacement = q
            .iter()
            .zip(&self.source)
            .map(|(a, b)| (a - b).norm())
            .fold(0.0, f64::max);
        let post_scale = BUDGET / max_displacement.max(BUDGET);
        // 全组统一预算缩放，零位移保护仍精确固定；缩放后的最终网格必须重新验证。
        if post_scale < 1.0 {
            for (p, source) in q.iter_mut().zip(&self.source) {
                *p = source + (*p - source) * post_scale;
            }
        }
        diagnostics.budget_scale *= post_scale;
        diagnostics.max_displacement = max_displacement * post_scale;
        diagnostics.target_rms_error = (q
            .iter()
            .enumerate()
            .map(|(i, p)| (p - self.source[i] - delta[i] * post_scale).norm_squared())
            .sum::<f64>()
            / n as f64)
            .sqrt();
        diagnostics.feature_max_error = self
            .feature_bindings
            .iter()
            .map(|(b, p, group)| {
                let actual = (0..3)
                    .map(|k| q[b.vertices[k]] * b.weights[k])
                    .sum::<Point>();
                (actual - p - feature_delta[*group] * diagnostics.budget_scale).norm()
            })
            .fold(0.0, f64::max);
        diagnostics.min_area_ratio = validate_mesh(&self.source, &q, &self.topology)?;
        check_current(current)?;
        Ok(self.solution(
            q,
            strengths,
            response.map(|s| s * diagnostics.budget_scale),
            diagnostics,
        ))
    }

    fn solution(
        &self,
        target: Vec<Point>,
        requested: Strengths,
        effective_response: [f64; 4],
        diagnostics: Diagnostics,
    ) -> GeometrySolution {
        let identity = target == self.source;
        let effective_response = if identity {
            [0.0; 4]
        } else {
            effective_response
        };
        GeometrySolution {
            identity,
            source: self.source_vertices(),
            target: target.iter().map(|p| array(self.frame.world(*p))).collect(),
            local_source: self.source.clone(),
            local_target: target,
            frame: self.frame.clone(),
            topology: Arc::clone(&self.topology),
            requested,
            effective_response,
            // 表示预算后的等效驱动值，不把软约束下的实际像素响应冒充滑块精确比例。
            actual_strengths: Strengths(effective_response.map(inverse_response)),
            diagnostics,
        }
    }
}

#[derive(Debug)]
pub struct GeometrySolution {
    identity: bool,
    source: Vec<[f64; 2]>,
    target: Vec<[f64; 2]>,
    local_source: Vec<Point>,
    local_target: Vec<Point>,
    frame: Frame,
    topology: Arc<Topology>,
    pub requested: Strengths,
    pub effective_response: [f64; 4],
    pub actual_strengths: Strengths,
    pub diagnostics: Diagnostics,
}

impl GeometrySolution {
    /// 稠密像素网格（UV 形变）：不做 ARAP 边界／O(E²) 自交检查。
    pub(crate) fn from_dense_pixel_mesh(
        source: Vec<[f64; 2]>,
        target: Vec<[f64; 2]>,
        triangles: Arc<Vec<[usize; 3]>>,
        face_width: f64,
        requested: Strengths,
        effective_response: [f64; 4],
        actual_strengths: Strengths,
        identity: bool,
    ) -> Result<Self> {
        if source.len() != target.len()
            || source.is_empty()
            || !face_width.is_finite()
            || face_width <= 0.0
            || source
                .iter()
                .chain(target.iter())
                .flatten()
                .any(|v| !v.is_finite())
        {
            return Err(GeometryError::DegenerateMesh);
        }
        let n = source.len();
        let mut kept = Vec::with_capacity(triangles.len());
        for tri in triangles.iter() {
            if tri.iter().any(|&i| i >= n) {
                return Err(GeometryError::DegenerateMesh);
            }
            let a = Point::new(source[tri[0]][0], source[tri[0]][1]);
            let b = Point::new(source[tri[1]][0], source[tri[1]][1]);
            let c = Point::new(source[tri[2]][0], source[tri[2]][1]);
            let ta = Point::new(target[tri[0]][0], target[tri[0]][1]);
            let tb = Point::new(target[tri[1]][0], target[tri[1]][1]);
            let tc = Point::new(target[tri[2]][0], target[tri[2]][1]);
            let before = cross(b - a, c - a);
            let after = cross(tb - ta, tc - ta);
            // 投影后侧视三角可退化；塑形后偶发翻折——从场中剔除，不整网失败。
            if before <= AREA_EPS || (!identity && after <= AREA_EPS) {
                continue;
            }
            kept.push(*tri);
        }
        if kept.is_empty() {
            return Err(GeometryError::DegenerateMesh);
        }
        let local_source: Vec<Point> = source.iter().map(|p| Point::new(p[0], p[1])).collect();
        let local_target: Vec<Point> = target.iter().map(|p| Point::new(p[0], p[1])).collect();
        let mut edges = BTreeSet::new();
        for &[a, b, c] in &kept {
            for (u, v) in [(a, b), (b, c), (c, a)] {
                edges.insert((u.min(v), u.max(v)));
            }
        }
        let topology = Arc::new(Topology {
            triangles: kept,
            edges: edges.into_iter().collect(),
            boundary: Vec::new(),
        });
        // 稠密像素网：frame.width 存真实面宽，供诊断 max_disp_px 与保护半径使用。
        let frame = Frame {
            origin: Point::zeros(),
            x: Point::new(1.0, 0.0),
            y: Point::new(0.0, 1.0),
            width: face_width,
        };
        let mut max_displacement = 0.0_f64;
        for (s, t) in local_source.iter().zip(&local_target) {
            max_displacement = max_displacement.max((t - s).norm());
        }
        Ok(Self {
            identity,
            source,
            target,
            local_source,
            local_target,
            frame,
            topology,
            requested,
            effective_response,
            actual_strengths,
            diagnostics: Diagnostics {
                max_displacement: max_displacement / face_width.max(1e-8),
                min_area_ratio: 1.0,
                budget_scale: 1.0,
                ..Diagnostics::default()
            },
        })
    }

    pub(crate) fn byte_size(&self) -> usize {
        std::mem::size_of::<Self>()
            + (self.source.capacity()
                + self.target.capacity()
                + self.local_source.capacity()
                + self.local_target.capacity())
                * 16
            + self.topology.triangles.capacity() * std::mem::size_of::<[usize; 3]>()
            + self.topology.edges.capacity() * std::mem::size_of::<(usize, usize)>()
            + self.topology.boundary.capacity() * std::mem::size_of::<usize>()
    }

    pub fn is_identity(&self) -> bool {
        self.identity
    }

    /// 在最终预算缩放后复查网格；06 另检查逆映射奇异值与多人合成。
    pub fn validate_final(&self) -> Result<()> {
        // 稠密像素网：构网时已剔除退化／翻折三角，这里只做有限性复查，避免重复判 Foldover。
        if self.topology.boundary.is_empty() && self.source.len() != MESH_VERTICES {
            if self
                .local_source
                .iter()
                .chain(self.local_target.iter())
                .any(|p| !p.x.is_finite() || !p.y.is_finite())
            {
                return Err(GeometryError::NonFinite);
            }
            if self.topology.triangles.is_empty() {
                return Err(GeometryError::DegenerateMesh);
            }
            return Ok(());
        }
        validate_mesh(&self.local_source, &self.local_target, &self.topology).map(|_| ())
    }

    pub fn face_width(&self) -> f64 {
        self.frame.width
    }

    pub fn source_vertices(&self) -> &[[f64; 2]] {
        &self.source
    }
    pub fn target_vertices(&self) -> &[[f64; 2]] {
        &self.target
    }
    pub fn triangles(&self) -> &[[usize; 3]] {
        &self.topology.triangles
    }
    pub fn forward(&self, source: [f64; 2]) -> Result<[f64; 2]> {
        self.query(source, false)
    }
    pub fn inverse(&self, target: [f64; 2]) -> Result<[f64; 2]> {
        self.query(target, true)
    }
    fn query(&self, p: [f64; 2], inverse: bool) -> Result<[f64; 2]> {
        let (from, to) = if inverse {
            (&self.local_target, &self.local_source)
        } else {
            (&self.local_source, &self.local_target)
        };
        let binding = locate(
            self.frame.local(Point::new(p[0], p[1])),
            from,
            &self.topology,
        )?;
        if self.identity {
            return Ok(p);
        }
        let q = (0..3)
            .map(|k| to[binding.vertices[k]] * binding.weights[k])
            .sum::<Point>();
        let result = self.frame.world(q);
        if !finite(result) {
            return Err(GeometryError::NonFinite);
        }
        Ok(array(result))
    }
}

#[derive(Debug)]
struct Handle {
    binding: Binding,
    target: Point,
    weight: f64,
}

/// A = 图 Laplacian + Σ λ b bᵀ；不建立 N×N 稠密矩阵。
struct System<'a> {
    edges: &'a [(usize, usize)],
    handles: Vec<Handle>,
    fixed: Vec<bool>,
    diagonal: Vec<f64>,
}

impl<'a> System<'a> {
    fn new(edges: &'a [(usize, usize)], handles: Vec<Handle>, fixed: Vec<bool>) -> Self {
        let mut diagonal = vec![0.0; fixed.len()];
        for &(i, j) in edges {
            diagonal[i] += 1.0;
            diagonal[j] += 1.0;
        }
        for h in &handles {
            for k in 0..3 {
                diagonal[h.binding.vertices[k]] += h.weight * h.binding.weights[k].powi(2);
            }
        }
        for (i, &is_fixed) in fixed.iter().enumerate() {
            if is_fixed {
                diagonal[i] = 1.0;
            }
        }
        Self {
            edges,
            handles,
            fixed,
            diagonal,
        }
    }
    fn apply(&self, x: &[f64], eliminate: bool) -> Vec<f64> {
        let at = |i: usize| {
            if eliminate && self.fixed[i] {
                0.0
            } else {
                x[i]
            }
        };
        let mut result = vec![0.0; x.len()];
        for &(i, j) in self.edges {
            let d = at(i) - at(j);
            result[i] += d;
            result[j] -= d;
        }
        for h in &self.handles {
            let sum = (0..3)
                .map(|k| at(h.binding.vertices[k]) * h.binding.weights[k])
                .sum::<f64>()
                * h.weight;
            for k in 0..3 {
                result[h.binding.vertices[k]] += sum * h.binding.weights[k];
            }
        }
        if eliminate {
            for i in 0..x.len() {
                if self.fixed[i] {
                    result[i] = x[i];
                }
            }
        }
        result
    }
}

fn pcg(
    system: &System<'_>,
    b: &[f64],
    mut x: Vec<f64>,
    options: SolverOptions,
    current: &dyn Fn() -> bool,
) -> Result<(Vec<f64>, usize, f64)> {
    let product = system.apply(&x, true);
    let mut r: Vec<f64> = b.iter().zip(product).map(|(b, a)| b - a).collect();
    let denominator = dot(b, b).sqrt().max(1.0);
    let tolerance = options.pcg_tolerance * denominator;
    let mut z: Vec<f64> = r.iter().zip(&system.diagonal).map(|(r, d)| r / d).collect();
    let mut direction = z.clone();
    let mut rz = dot(&r, &z);
    for iteration in 0..=options.pcg_iterations {
        if iteration % 16 == 0 {
            check_current(current)?;
        }
        if dot(&r, &r).sqrt() <= tolerance {
            let actual = system.apply(&x, true);
            let residual = actual
                .iter()
                .zip(b)
                .map(|(a, b)| (a - b).powi(2))
                .sum::<f64>()
                .sqrt()
                / denominator;
            if residual.is_finite() && residual <= options.pcg_tolerance {
                return Ok((x, iteration, residual));
            }
            return Err(GeometryError::NotConverged {
                stage: "pcg-residual",
                iterations: iteration,
            });
        }
        if iteration == options.pcg_iterations {
            break;
        }
        let ad = system.apply(&direction, true);
        let dad = dot(&direction, &ad);
        if !dad.is_finite() || dad <= 0.0 || !rz.is_finite() || rz <= 0.0 {
            return Err(GeometryError::NonFinite);
        }
        let alpha = rz / dad;
        for i in 0..x.len() {
            x[i] += alpha * direction[i];
            r[i] -= alpha * ad[i];
            z[i] = r[i] / system.diagonal[i];
        }
        let next = dot(&r, &z);
        let beta = next / rz;
        for i in 0..x.len() {
            direction[i] = z[i] + beta * direction[i];
        }
        rz = next;
    }
    Err(GeometryError::NotConverged {
        stage: "pcg",
        iterations: options.pcg_iterations,
    })
}

fn locate(p: Point, vertices: &[Point], topology: &Topology) -> Result<Binding> {
    if !finite(p) {
        return Err(GeometryError::NonFinite);
    }
    for &ids in &topology.triangles {
        let [a, b, c] = ids.map(|i| vertices[i]);
        let area = cross(b - a, c - a);
        if !area.is_finite() || area <= AREA_EPS {
            return Err(GeometryError::DegenerateMesh);
        }
        let v = cross(p - a, c - a) / area;
        let w = cross(b - a, p - a) / area;
        let u = 1.0 - v - w;
        if [u, v, w]
            .iter()
            .all(|x| *x >= -QUERY_EPS && *x <= 1.0 + QUERY_EPS)
        {
            let mut weights = [u.max(0.0), v.max(0.0), w.max(0.0)];
            let sum = weights.iter().sum::<f64>();
            for weight in &mut weights {
                *weight /= sum;
            }
            return Ok(Binding {
                vertices: ids,
                weights,
            });
        }
    }
    Err(GeometryError::OutsideMesh)
}

fn validate_mesh(source: &[Point], target: &[Point], topology: &Topology) -> Result<f64> {
    if source.len() != 1 + OVAL.len() * RINGS.len() || target.len() != source.len() {
        return Err(GeometryError::DegenerateMesh);
    }
    if source.iter().chain(target).any(|p| !finite(*p)) {
        return Err(GeometryError::NonFinite);
    }
    if topology.boundary.iter().any(|&i| source[i] != target[i]) {
        return Err(GeometryError::BoundaryMoved);
    }
    let mut min_ratio = f64::INFINITY;
    for &[a, b, c] in &topology.triangles {
        let before = cross(source[b] - source[a], source[c] - source[a]);
        let after = cross(target[b] - target[a], target[c] - target[a]);
        if before <= AREA_EPS {
            return Err(GeometryError::DegenerateMesh);
        }
        if after <= AREA_EPS {
            return Err(GeometryError::Foldover);
        }
        min_ratio = min_ratio.min(after / before);
    }
    // 包括非相邻边相切／共线重叠；邻边共享同一顶点，插值沿边天然连续。
    for (index, &(a, b)) in topology.edges.iter().enumerate() {
        for &(c, d) in &topology.edges[index + 1..] {
            if a == c || a == d || b == c || b == d {
                continue;
            }
            if intersects(target[a], target[b], target[c], target[d]) {
                return Err(GeometryError::SelfIntersection);
            }
        }
    }
    Ok(min_ratio)
}

fn intersects(a: Point, b: Point, c: Point, d: Point) -> bool {
    for axis in 0..2 {
        if a[axis].max(b[axis]) + 1e-12 < c[axis].min(d[axis])
            || c[axis].max(d[axis]) + 1e-12 < a[axis].min(b[axis])
        {
            return false;
        }
    }
    let (ab_c, ab_d, cd_a, cd_b) = (
        cross(b - a, c - a),
        cross(b - a, d - a),
        cross(d - c, a - c),
        cross(d - c, b - c),
    );
    let straddles = |x: f64, y: f64| (x <= 1e-12 && y >= -1e-12) || (y <= 1e-12 && x >= -1e-12);
    straddles(ab_c, ab_d) && straddles(cd_a, cd_b)
}

fn check_current(current: &dyn Fn() -> bool) -> Result<()> {
    if current() {
        Ok(())
    } else {
        Err(GeometryError::Cancelled)
    }
}
fn smooth(t: f64) -> f64 {
    let t = t.clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}
fn bump(t: f64) -> f64 {
    smooth(1.0 - t.abs())
}

/// 局部影响圆：半径为锚点距，权重 `smooth(1−r/R)`。
/// 顺向把查询点推向内锚（与片段着色器 `uv -= offset` 的一阶顶点目标等价）。
fn curve_warp(coord: Point, origin: Point, target: Point, delta: f64) -> Point {
    let radius = (target - origin).norm();
    if !radius.is_finite() || radius <= 1e-8 {
        return coord;
    }
    let ratio = smooth(1.0 - (coord - origin).norm() / radius);
    coord + (target - origin) * (delta * ratio)
}
fn cross(a: Point, b: Point) -> f64 {
    a.x * b.y - a.y * b.x
}
fn finite(p: Point) -> bool {
    p.x.is_finite() && p.y.is_finite()
}
fn array(p: Point) -> [f64; 2] {
    [p.x, p.y]
}
fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(a, b)| a * b).sum()
}
fn rotate(p: Point, r: [f64; 2]) -> Point {
    Point::new(r[0] * p.x - r[1] * p.y, r[1] * p.x + r[0] * p.y)
}
pub(crate) fn inverse_response_pub(s: f64) -> f64 {
    inverse_response(s)
}

fn inverse_response(s: f64) -> f64 {
    if s == 0.0 {
        return 0.0;
    }
    if s == 1.0 {
        return 100.0;
    }
    let (mut low, mut high) = (0.0, 1.0);
    for _ in 0..48 {
        let middle = (low + high) * 0.5;
        if smooth(middle) < s {
            low = middle;
        } else {
            high = middle;
        }
    }
    (low + high) * 50.0
}

#[cfg(test)]
#[path = "face_reshape_tests.rs"]
mod tests;
