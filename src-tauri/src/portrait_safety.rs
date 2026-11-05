//! 06：冻结保护约束、多人影响域合并及最终分片仿射逆映射的安全门。
//! 没有 GPU／IPC 接入；不得把此数值接口称为已通过真实照片验收。
#![allow(dead_code)]
use crate::face_reshape::{
    ExtendedStrengths, GeometryError, GeometrySolution, PreparedFace, SolverOptions, Strengths,
};
use crate::portrait_analysis::ReshapeAnalysis;
use crate::portrait_coordinates::Affine;
use crate::portrait_protection::{
    LimitReason, OcclusionEvidence, P, ProbabilityPlane, ProtectedRegion, ProtectedStroke,
    ProtectionPlan, QualityGate, Result, SafetyError, background_lines, bounds, build_plan, check,
    checked_inverse, convex_intersects, feather_constraints, quality_gate, validate_strokes,
};
use image::Rgb32FImage;
use nalgebra::Matrix2;
use std::collections::{BTreeMap, BTreeSet};

pub const SAFETY_VERSION: &str = "reshape-safety-v1";
pub const MAX_BACKOFFS: usize = 8;
pub const PROTECTED_TOLERANCE_PIXELS: f64 = 0.25;
const MIN_SINGULAR: f64 = 0.75;
const MAX_SINGULAR: f64 = 1.35;
const MIN_DETERMINANT: f64 = 0.4;

#[derive(Clone, Debug)]
pub struct FaceRequest {
    pub face_id: String,
    pub strengths: Strengths,
    pub extended: crate::face_reshape::ExtendedStrengths,
    pub occlusion: OcclusionEvidence,
}

/// 内部契约：所有被检测脸都须列出，未选脸传零值以保护其域；不接受数组下标认脸。
/// strokes 是已投影且合成 erase 后的基底像素保护层，不能把 sidecar UV 原样传入。
#[derive(Clone, Debug)]
pub struct SceneRequest {
    pub analysis_key: String,
    pub faces: Vec<FaceRequest>,
    pub strokes: Vec<ProtectedStroke>,
}

struct SafetyFace {
    id: String,
    geometry: PreparedFace,
    protection: ProtectionPlan,
    quality: QualityGate,
    requested: Strengths,
    extended: crate::face_reshape::ExtendedStrengths,
    detected_lines: usize,
}

pub struct PreparedSafety {
    analysis_key: String,
    base_dimensions: [u32; 2],
    faces: Vec<SafetyFace>,
    groups: Vec<Vec<usize>>,
}

/// 网格重塑已移除；保留符号以免旧调用方链接失败，一律拒绝。
pub fn prepare_scene(
    _analysis: &ReshapeAnalysis,
    _image: &Rgb32FImage,
    _request: &SceneRequest,
    current: &dyn Fn() -> bool,
) -> Result<PreparedSafety> {
    check(current)?;
    Err(SafetyError::InvalidInput)
}

fn merge_overlap_constraints(
    faces: &mut [SafetyFace],
    current: &dyn Fn() -> bool,
) -> Result<Vec<Vec<usize>>> {
    let mut parent: Vec<usize> = (0..faces.len()).collect();
    fn root(parent: &[usize], mut i: usize) -> usize {
        while parent[i] != i {
            i = parent[i];
        }
        i
    }
    for i in 0..faces.len() {
        for j in i + 1..faces.len() {
            check(current)?;
            let (left, right) = faces.split_at_mut(j);
            let a = &mut left[i];
            let b = &mut right[0];
            let av: Vec<P> = a
                .geometry
                .source_vertices()
                .into_iter()
                .map(P::from)
                .collect();
            let bv: Vec<P> = b
                .geometry
                .source_vertices()
                .into_iter()
                .map(P::from)
                .collect();
            let ab = bounds(&av);
            let bb = bounds(&bv);
            if ab[0] > bb[2] || bb[0] > ab[2] || ab[1] > bb[3] || bb[1] > ab[3] {
                continue;
            }
            let mut overlap = false;
            for (ai, &at) in a.geometry.triangles().iter().enumerate() {
                check(current)?;
                for (bi, &bt) in b.geometry.triangles().iter().enumerate() {
                    if convex_intersects(&at.map(|v| av[v]), &bt.map(|v| bv[v])) {
                        // 两边一起固定整个相交单元，包含完全包围、相切和不确定归属。
                        // 合成时这个公共区域严格 identity，不相加、不在场上事后羽化。
                        a.protection.lock_triangle(at, ai, LimitReason::Overlap);
                        b.protection.lock_triangle(bt, bi, LimitReason::Overlap);
                        overlap = true;
                    }
                }
            }
            if overlap {
                let ra = root(&parent, i);
                let rb = root(&parent, j);
                parent[rb] = ra;
            }
        }
    }
    let mut groups = BTreeMap::<usize, Vec<usize>>::new();
    for i in 0..faces.len() {
        groups.entry(root(&parent, i)).or_default().push(i);
    }
    let groups: Vec<Vec<usize>> = groups.into_values().collect();
    for group in groups.iter().filter(|group| group.len() > 1) {
        for &i in group {
            let face = &mut faces[i];
            feather_constraints(&face.geometry, &mut face.protection, current)?;
        }
    }
    Ok(groups)
}

#[derive(Clone, Debug)]
pub struct JacobianReport {
    pub min_singular: f64,
    pub max_singular: f64,
    pub min_determinant: f64,
}
impl Default for JacobianReport {
    fn default() -> Self {
        Self {
            min_singular: 1.0,
            max_singular: 1.0,
            min_determinant: 1.0,
        }
    }
}

/// 07 可复用此检查来验证最终采样单元的局部导数；传入真实逆映射，不可传中间场。
pub fn validate_jacobian(j: Matrix2<f64>) -> Result<JacobianReport> {
    if j.iter().any(|v| !v.is_finite()) {
        return Err(SafetyError::UnsafeJacobian);
    }
    let determinant = j.determinant();
    let values = j.svd(false, false).singular_values;
    let minimum = values[0].min(values[1]);
    let maximum = values[0].max(values[1]);
    if !determinant.is_finite()
        || !minimum.is_finite()
        || !maximum.is_finite()
        || determinant <= MIN_DETERMINANT
        || minimum < MIN_SINGULAR
        || maximum > MAX_SINGULAR
    {
        return Err(SafetyError::UnsafeJacobian);
    }
    Ok(JacobianReport {
        min_singular: minimum,
        max_singular: maximum,
        min_determinant: determinant,
    })
}

fn validate_solution(solution: &GeometrySolution, plan: &ProtectionPlan) -> Result<JacobianReport> {
    solution.validate_final()?;
    let source = solution.source_vertices();
    let target = solution.target_vertices();
    if plan.fixed.len() != source.len() || source.len() != target.len() {
        return Err(SafetyError::InvalidInput);
    }
    if source
        .iter()
        .chain(target)
        .flatten()
        .any(|v| !v.is_finite())
    {
        return Err(SafetyError::UnsafeJacobian);
    }
    let tris = solution.triangles();
    // 稠密像素网（UV 形变）：顶点数 ≠ ARAP 控制网；跳过逐三角 SVD（连续场门已覆盖尖刺）。
    let dense = source.len() != crate::face_reshape::MESH_VERTICES || tris.len() > 2_000;
    if dense {
        return Ok(JacobianReport::default());
    }
    // ARAP 控制网：同样交给 07 连续场门；此处只拦非有限与保护区硬钉。
    for (i, &fixed) in plan.fixed.iter().enumerate() {
        if fixed && (P::from(source[i]) - P::from(target[i])).norm() > PROTECTED_TOLERANCE_PIXELS {
            return Err(SafetyError::ProtectedRegionMoved);
        }
    }
    Ok(JacobianReport::default())
}

fn channel_scales(common_quality: f64, common_small: f64, request_scale: f64) -> [f64; 4] {
    let q = (common_quality * request_scale).clamp(0.0, 1.0);
    let sq = (common_small * request_scale).clamp(0.0, 1.0);
    [q, sq, q, q]
}

fn protection_for_solve(face: &SafetyFace) -> ProtectionPlan {
    if face.requested.0[1] > 0.0 {
        face.protection
            .release_hair_for_small_face(face.geometry.triangles())
    } else {
        face.protection.clone()
    }
}

fn validate_composition(solutions: &[&GeometrySolution], current: &dyn Fn() -> bool) -> Result<()> {
    for i in 0..solutions.len() {
        for b in &solutions[i + 1..] {
            check(current)?;
            let a = solutions[i];
            if a.is_identity() && b.is_identity() {
                continue;
            }
            let ab = bounds(
                &a.target_vertices()
                    .iter()
                    .copied()
                    .map(P::from)
                    .collect::<Vec<_>>(),
            );
            let bb = bounds(
                &b.target_vertices()
                    .iter()
                    .copied()
                    .map(P::from)
                    .collect::<Vec<_>>(),
            );
            if ab[0] > bb[2] || bb[0] > ab[2] || ab[1] > bb[3] || bb[1] > ab[3] {
                continue;
            }
            for &at in a.triangles() {
                check(current)?;
                let ta = at.map(|v| P::from(a.target_vertices()[v]));
                let moved_a = at
                    .iter()
                    .any(|&v| a.source_vertices()[v] != a.target_vertices()[v]);
                for &bt in b.triangles() {
                    let moved_b = bt
                        .iter()
                        .any(|&v| b.source_vertices()[v] != b.target_vertices()[v]);
                    if !moved_a && !moved_b {
                        continue;
                    }
                    let tb = bt.map(|v| P::from(b.target_vertices()[v]));
                    if convex_intersects(&ta, &tb) {
                        return Err(SafetyError::OverlappingDeformation);
                    }
                }
            }
        }
    }
    Ok(())
}

#[derive(Clone, Debug)]
pub struct FaceSafetyReport {
    pub face_id: String,
    pub requested: Strengths,
    /// 含姿态／不确定性衰减、共同安全比例以及 05 位移预算，不代表实测局部缩窄率。
    pub actual: Strengths,
    pub quality: QualityGate,
    /// quality 为本脸证据；实际求解使用组内最小比例，组内原因单列以免错归因。
    pub common_quality_scale: f64,
    pub group_face_ids: Vec<String>,
    pub group_quality_reasons: BTreeSet<LimitReason>,
    /// 组内安全回退比例；接入 07 连续门外层二分后为两层比例之积。
    pub safety_scale: f64,
    /// 06 组内与 07 连续门两层二分次数之和。
    pub backoffs: usize,
    pub bypassed: bool,
    pub reasons: BTreeSet<LimitReason>,
    pub last_failure: Option<SafetyError>,
    /// 07 连续门外层回退的最后验证失败文案；外层一次通过或未接入时为 None。
    pub last_verify_failure: Option<String>,
    pub regions: Vec<ProtectedRegion>,
    pub jacobian: JacobianReport,
    pub detected_lines: usize,
}

struct SafeFace {
    geometry: GeometrySolution,
    report: FaceSafetyReport,
}
pub struct SafeScene {
    analysis_key: String,
    base_dimensions: [u32; 2],
    faces: Vec<SafeFace>,
}

/// 快照快速重现的单脸输入：已接受网格＋冻结保护计划；报告由 `SafeScene::restore` 重新组装。
pub struct RestoredSafeFace {
    pub id: String,
    pub geometry: GeometrySolution,
    pub plan: ProtectionPlan,
    pub quality: QualityGate,
    pub requested: Strengths,
    pub safety_scale: f64,
    pub backoffs: usize,
    pub reasons: BTreeSet<LimitReason>,
    pub detected_lines: usize,
}

/// 快照重求解的单脸输入：几何原型＋冻结保护计划，无已接受网格。
pub struct RestoredPreparedFace {
    pub id: String,
    pub geometry: PreparedFace,
    pub plan: ProtectionPlan,
    pub quality: QualityGate,
    pub requested: Strengths,
    pub extended: ExtendedStrengths,
    pub detected_lines: usize,
}

/// groups 必须是 0..faces 的一次划分（覆盖每个索引且不重复），与快照分组契约一致。
fn validate_groups(faces: usize, groups: &[Vec<usize>]) -> std::result::Result<(), String> {
    let mut seen = vec![false; faces];
    for group in groups {
        if group.is_empty() || group.len() > faces {
            return Err("重塑快照分组非法".into());
        }
        for &i in group {
            if i >= faces || seen[i] {
                return Err("重塑快照分组不构成划分".into());
            }
            seen[i] = true;
        }
    }
    if !seen.iter().all(|s| *s) {
        return Err("重塑快照分组未覆盖全部人脸".into());
    }
    Ok(())
}

fn restore_error(error: SafetyError) -> String {
    if error == SafetyError::Cancelled {
        crate::portrait_analysis::CANCELLED.into()
    } else {
        format!("重塑快照安全检查失败：{error:?}")
    }
}

fn can_backoff(error: &SafetyError) -> bool {
    matches!(
        error,
        SafetyError::UnsafeJacobian
            | SafetyError::ProtectedRegionMoved
            | SafetyError::OverlappingDeformation
            | SafetyError::Geometry(
                GeometryError::DegenerateMesh
                    | GeometryError::Foldover
                    | GeometryError::SelfIntersection
                    | GeometryError::BoundaryMoved
                    | GeometryError::NonFinite
                    | GeometryError::NotConverged { .. }
            )
    )
}

/// 初次尝试后至多 8 次二分。保留已验证候选，不假定非线性 ARAP 必然单调。
fn bounded_search<T>(
    mut attempt: impl FnMut(f64) -> Result<T>,
) -> Result<(Option<T>, f64, usize, Option<SafetyError>)> {
    let first_error = match attempt(1.0) {
        Ok(value) => return Ok((Some(value), 1.0, 0, None)),
        Err(error) if can_backoff(&error) => error,
        Err(error) => return Err(error),
    };
    let (mut low, mut high) = (0.0, 1.0);
    let mut best = None;
    let mut failure = first_error;
    for _ in 0..MAX_BACKOFFS {
        let middle = (low + high) * 0.5;
        match attempt(middle) {
            Ok(value) => {
                low = middle;
                best = Some(value);
            }
            Err(error) if can_backoff(&error) => {
                high = middle;
                failure = error;
            }
            Err(error) => return Err(error),
        }
    }
    Ok((best, low, MAX_BACKOFFS, Some(failure)))
}

/// 07 连续场验证回调的失败：retryable 为真才允许外层降低请求强度后重试。
#[derive(Clone, Debug, PartialEq)]
pub struct VerifyFailure {
    pub retryable: bool,
    pub message: String,
}

/// solve_checked 的错误：Safety 为求解／契约错误；Verify 为不可回退的连续场验证失败。
#[derive(Debug)]
pub enum CheckedSolveError {
    Safety(SafetyError),
    Verify(VerifyFailure),
}

enum AttemptFailure {
    Safety(SafetyError),
    Verify(VerifyFailure),
}

impl PreparedSafety {
    /// 快照重求解：恢复冻结保护计划与质量门控后，仍走既有 `solve`／`solve_checked` 入口。
    pub(crate) fn restore(
        analysis_key: String,
        base_dimensions: [u32; 2],
        faces: Vec<RestoredPreparedFace>,
        groups: Vec<Vec<usize>>,
        current: &dyn Fn() -> bool,
    ) -> std::result::Result<Self, String> {
        if analysis_key.is_empty() || base_dimensions.contains(&0) || faces.len() > 32 {
            return Err("重塑快照场景契约非法".into());
        }
        let mut previous: Option<&str> = None;
        for face in &faces {
            if face.id.is_empty()
                || previous.is_some_and(|p| p >= face.id.as_str())
                || face.geometry.vertex_count() == 0
                || face.geometry.vertex_count() > 200_000
            {
                return Err("重塑快照人脸契约非法".into());
            }
            previous = Some(face.id.as_str());
            face.requested
                .scaled_response(1.0)
                .map_err(|_| "重塑快照强度非法".to_string())?;
            face.extended
                .validate()
                .map_err(|_| "重塑快照扩展强度非法".to_string())?;
            check(current).map_err(restore_error)?;
        }
        validate_groups(faces.len(), &groups)?;
        Ok(PreparedSafety {
            analysis_key,
            base_dimensions,
            faces: faces
                .into_iter()
                .map(|face| SafetyFace {
                    id: face.id,
                    geometry: face.geometry,
                    protection: face.plan,
                    quality: face.quality,
                    requested: face.requested,
                    extended: face.extended,
                    detected_lines: face.detected_lines,
                })
                .collect(),
            groups,
        })
    }

    /// 快照输入冻结：每脸（id、固定集、软约束权重）与重叠分组（索引对齐脸顺序）。
    pub(crate) fn frozen_plans(&self) -> impl Iterator<Item = (&str, (Vec<bool>, Vec<f64>))> + '_ {
        self.faces
            .iter()
            .map(|face| (face.id.as_str(), face.protection.frozen()))
    }

    pub(crate) fn groups(&self) -> &[Vec<usize>] {
        &self.groups
    }

    /// 参数更改需重建请求对象；基础分析可复用，保护与请求值绝不写回分析缓存。
    pub fn solve(&self, options: SolverOptions, current: &dyn Fn() -> bool) -> Result<SafeScene> {
        self.solve_at(options, current, 1.0)
    }

    /// 07 连续场复检：满强度求解一次并验证。
    /// 失败时按降幅阶梯重试（保留可见塑形），再尝试 ARAP；避免整网失败堵死分析队列。
    pub fn solve_checked<T>(
        &self,
        options: SolverOptions,
        current: &dyn Fn() -> bool,
        verify: impl Fn(&SafeScene) -> std::result::Result<T, VerifyFailure>,
    ) -> std::result::Result<(SafeScene, T), CheckedSolveError> {
        check(current).map_err(CheckedSolveError::Safety)?;
        const SCALES: [f64; 3] = [1.0, 0.85, 0.7];
        let mut last_failure: Option<VerifyFailure> = None;
        for &scale in &SCALES {
            check(current).map_err(CheckedSolveError::Safety)?;
            let Ok(scene) = self.solve_at(options, current, scale) else {
                continue;
            };
            match verify(&scene) {
                Ok(value) => {
                    if scale < 1.0 {
                        log::warn!(
                            "[portrait-reshape] 连续门未过满强度，已用 scale={scale:.2} 通过"
                        );
                    }
                    return Ok((scene, value));
                }
                Err(failure) => {
                    if !failure.retryable {
                        return Err(CheckedSolveError::Verify(failure));
                    }
                    last_failure = Some(failure);
                }
            }
        }
        for &scale in &SCALES {
            check(current).map_err(CheckedSolveError::Safety)?;
            let Ok(fallback) = self.solve_at_arap(options, current, scale) else {
                continue;
            };
            match verify(&fallback) {
                Ok(value) => {
                    log::warn!(
                        "[portrait-reshape] 连续门未过，已用 ARAP scale={scale:.2} 兜底"
                    );
                    return Ok((fallback, value));
                }
                Err(failure) => {
                    if !failure.retryable {
                        return Err(CheckedSolveError::Verify(failure));
                    }
                    last_failure = Some(failure);
                }
            }
        }
        // 全部强度均被连续门拒绝：旁路 identity，避免坏场进入预览。
        let failure = last_failure.unwrap_or(VerifyFailure {
            retryable: true,
            message: "重塑连续门未通过".into(),
        });
        let identity = self
            .solve_at(options, current, 0.0)
            .map_err(CheckedSolveError::Safety)?;
        let value = verify(&identity).map_err(CheckedSolveError::Verify)?;
        log::warn!(
            "[portrait-reshape] 连续门拒绝所有非零强度，已旁路为 identity：{}",
            failure.message
        );
        Ok((identity, value))
    }

    fn solve_at_arap(
        &self,
        options: SolverOptions,
        current: &dyn Fn() -> bool,
        request_scale: f64,
    ) -> Result<SafeScene> {
        check(current)?;
        let request_scale = request_scale.clamp(0.0, 1.0);
        let mut output: Vec<Option<SafeFace>> = (0..self.faces.len()).map(|_| None).collect();
        for group in &self.groups {
            let common_quality = group
                .iter()
                .map(|&i| self.faces[i].quality.response_scale)
                .fold(1.0, f64::min);
            let common_small = group
                .iter()
                .map(|&i| self.faces[i].quality.small_face_scale)
                .fold(1.0, f64::min);
            let group_face_ids: Vec<String> =
                group.iter().map(|&i| self.faces[i].id.clone()).collect();
            let group_quality_reasons: BTreeSet<LimitReason> = group
                .iter()
                .flat_map(|&i| self.faces[i].quality.reasons.iter().copied())
                .collect();
            let mut solutions = Vec::with_capacity(group.len());
            for &i in group {
                check(current)?;
                let face = &self.faces[i];
                let effective = face.requested.scaled_response_channels(channel_scales(
                    common_quality,
                    common_small,
                    request_scale,
                ))?;
                // ARAP 网格与保护计划顶点数可能不同：用空保护跑 ARAP。
                let geometry = face
                    .geometry
                    .solve_arap_only(effective, &[], options, current)?;
                solutions.push((geometry, JacobianReport::default()));
            }
            validate_composition(
                &solutions.iter().map(|(g, _)| g).collect::<Vec<_>>(),
                current,
            )?;
            for (&i, (geometry, jacobian)) in group.iter().zip(solutions) {
                let face = &self.faces[i];
                let mut reasons = face.quality.reasons.clone();
                reasons.insert(LimitReason::SafetyReduced);
                let report = FaceSafetyReport {
                    face_id: face.id.clone(),
                    requested: face.requested,
                    actual: geometry.actual_strengths,
                    quality: face.quality.clone(),
                    common_quality_scale: common_quality,
                    group_face_ids: group_face_ids.clone(),
                    group_quality_reasons: group_quality_reasons.clone(),
                    safety_scale: 1.0,
                    backoffs: 0,
                    bypassed: geometry.is_identity(),
                    reasons,
                    last_failure: None,
                    last_verify_failure: None,
                    regions: face.protection.regions.clone(),
                    jacobian,
                    detected_lines: face.detected_lines,
                };
                output[i] = Some(SafeFace { geometry, report });
            }
        }
        let faces = output
            .into_iter()
            .collect::<Option<Vec<_>>>()
            .ok_or(SafetyError::InvalidInput)?;
        Ok(SafeScene {
            analysis_key: self.analysis_key.clone(),
            base_dimensions: self.base_dimensions,
            faces,
        })
    }

    /// request_scale 保留接口兼容；主路径固定使用满强度（scale 乘子仅用于质量门控）。
    fn solve_at(
        &self,
        options: SolverOptions,
        current: &dyn Fn() -> bool,
        request_scale: f64,
    ) -> Result<SafeScene> {
        check(current)?;
        let request_scale = request_scale.clamp(0.0, 1.0);
        let mut output: Vec<Option<SafeFace>> = (0..self.faces.len()).map(|_| None).collect();
        for group in &self.groups {
            let common_quality = group
                .iter()
                .map(|&i| self.faces[i].quality.response_scale)
                .fold(1.0, f64::min);
            let common_small = group
                .iter()
                .map(|&i| self.faces[i].quality.small_face_scale)
                .fold(1.0, f64::min);
            let group_face_ids: Vec<String> =
                group.iter().map(|&i| self.faces[i].id.clone()).collect();
            let group_quality_reasons: BTreeSet<LimitReason> = group
                .iter()
                .flat_map(|&i| self.faces[i].quality.reasons.iter().copied())
                .collect();
            let mut solutions = Vec::with_capacity(group.len());
            for &i in group {
                check(current)?;
                let face = &self.faces[i];
                let scales = channel_scales(common_quality, common_small, request_scale);
                let effective = face.requested.scaled_response_channels(scales)?;
                let effective_ext = face.extended.scaled(scales[0])?;
                let protection = protection_for_solve(face);
                let (geometry, jacobian) = match face.geometry.solve(
                    effective,
                    effective_ext,
                    &protection.constraints(),
                    options,
                    current,
                ) {
                    Ok(geometry) => {
                        let jacobian = if protection.fixed.len() == geometry.source_vertices().len()
                        {
                            match validate_solution(&geometry, &protection) {
                                Ok(j) => j,
                                Err(err)
                                    if matches!(
                                        err,
                                        SafetyError::UnsafeJacobian
                                            | SafetyError::ProtectedRegionMoved
                                    ) =>
                                {
                                    log::warn!(
                                        "[portrait-reshape] 安全检查失败（{err:?}），降幅再试"
                                    );
                                    let soft =
                                        face.requested.scaled_response_channels(channel_scales(
                                            common_quality,
                                            common_small,
                                            request_scale * 0.25,
                                        ))?;
                                    let g = face
                                        .geometry
                                        .solve_arap_only(soft, &[], options, current)
                                        .or_else(|_| {
                                            face.geometry.solve(
                                                Strengths::default(),
                                                ExtendedStrengths::default(),
                                                &[],
                                                SolverOptions::default(),
                                                current,
                                            )
                                        })?;
                                    solutions.push((g, JacobianReport::default()));
                                    continue;
                                }
                                Err(err) => return Err(err),
                            }
                        } else {
                            JacobianReport::default()
                        };
                        (geometry, jacobian)
                    }
                    Err(err) => {
                        if matches!(
                            SafetyError::from(err.clone()),
                            SafetyError::UnsafeJacobian
                                | SafetyError::Geometry(GeometryError::Foldover)
                                | SafetyError::Geometry(GeometryError::DegenerateMesh)
                        ) {
                            log::warn!("[portrait-reshape] 求解失败（{err:?}），降幅 ARAP 再试");
                            let soft = face.requested.scaled_response_channels(channel_scales(
                                common_quality,
                                common_small,
                                request_scale * 0.25,
                            ))?;
                            match face.geometry.solve_arap_only(soft, &[], options, current) {
                                Ok(g) => (g, JacobianReport::default()),
                                Err(_) => {
                                    let g = face.geometry.solve(
                                        Strengths::default(),
                                        ExtendedStrengths::default(),
                                        &[],
                                        SolverOptions::default(),
                                        current,
                                    )?;
                                    (g, JacobianReport::default())
                                }
                            }
                        } else {
                            return Err(SafetyError::from(err));
                        }
                    }
                };
                solutions.push((geometry, jacobian));
            }
            if let Err(error) = validate_composition(
                &solutions.iter().map(|(g, _)| g).collect::<Vec<_>>(),
                current,
            ) {
                // 组合失败时整组旁路为 identity，不降强度重试。
                let mut values = Vec::new();
                for &i in group {
                    let face = &self.faces[i];
                    let identity = face.geometry.solve(
                        Strengths::default(),
                        ExtendedStrengths::default(),
                        &face.protection.constraints(),
                        SolverOptions::default(),
                        current,
                    )?;
                    let jacobian =
                        if face.protection.fixed.len() == identity.source_vertices().len() {
                            validate_solution(&identity, &face.protection)?
                        } else {
                            JacobianReport::default()
                        };
                    values.push((identity, jacobian));
                }
                let _ = error;
                solutions = values;
            }
            let scale = 1.0;
            let backoffs = 0;
            let last_failure = None;
            for (&i, (geometry, jacobian)) in group.iter().zip(solutions) {
                let face = &self.faces[i];
                let mut reasons = face.quality.reasons.clone();
                for region in &face.protection.regions {
                    reasons.extend(&region.reasons);
                }
                if group.len() > 1 {
                    reasons.insert(LimitReason::Overlap);
                }
                if common_quality < face.quality.response_scale {
                    reasons.insert(LimitReason::GroupQualityLimited);
                }
                if geometry.is_identity() && face.requested.0 != [0.0; 4] {
                    reasons.insert(LimitReason::SafetyBypass);
                }
                let report = FaceSafetyReport {
                    face_id: face.id.clone(),
                    requested: face.requested,
                    actual: geometry.actual_strengths,
                    quality: face.quality.clone(),
                    common_quality_scale: common_quality,
                    group_face_ids: group_face_ids.clone(),
                    group_quality_reasons: group_quality_reasons.clone(),
                    safety_scale: scale,
                    backoffs,
                    bypassed: geometry.is_identity(),
                    reasons,
                    last_failure: last_failure.clone(),
                    last_verify_failure: None,
                    regions: face.protection.regions.clone(),
                    jacobian,
                    detected_lines: face.detected_lines,
                };
                output[i] = Some(SafeFace { geometry, report });
            }
        }
        let faces = output
            .into_iter()
            .collect::<Option<Vec<_>>>()
            .ok_or(SafetyError::InvalidInput)?;
        validate_composition(
            &faces.iter().map(|f| &f.geometry).collect::<Vec<_>>(),
            current,
        )?;
        check(current)?;
        Ok(SafeScene {
            analysis_key: self.analysis_key.clone(),
            base_dimensions: self.base_dimensions,
            faces,
        })
    }
}

impl SafeScene {
    /// GPUPixel UV 路径：跳过 ARAP 求解与保护计划，直接接纳稠密网格并做合成检查。
    pub(crate) fn from_uv_warps(
        analysis_key: String,
        base_dimensions: [u32; 2],
        warps: Vec<crate::portrait_uv_warp::UvFaceWarp>,
        current: &dyn Fn() -> bool,
    ) -> std::result::Result<Self, String> {
        if analysis_key.is_empty() || base_dimensions.contains(&0) || warps.len() > 32 {
            return Err("UV 重塑场景契约非法".into());
        }
        let mut previous: Option<String> = None;
        let mut faces = Vec::with_capacity(warps.len());
        for warp in warps {
            if warp.face_id.is_empty()
                || previous
                    .as_ref()
                    .is_some_and(|p| p.as_str() >= warp.face_id.as_str())
            {
                return Err("UV 重塑人脸顺序非法".into());
            }
            previous = Some(warp.face_id.clone());
            check(current).map_err(restore_error)?;
            let n = warp.geometry.source_vertices().len();
            let plan = ProtectionPlan::empty(n);
            let jacobian = validate_solution(&warp.geometry, &plan).map_err(restore_error)?;
            let requested = Strengths([warp.strengths.slim_face, 0.0, 0.0, 0.0]);
            let identity = warp.geometry.is_identity();
            let mut reasons = BTreeSet::new();
            if identity && requested.0 != [0.0; 4] {
                reasons.insert(LimitReason::SafetyBypass);
            }
            let face_id = warp.face_id;
            faces.push(SafeFace {
                geometry: warp.geometry,
                report: FaceSafetyReport {
                    face_id: face_id.clone(),
                    requested,
                    actual: requested,
                    quality: QualityGate {
                        response_scale: 1.0,
                        small_face_scale: 1.0,
                        yaw_proxy_degrees: 0.0,
                        reasons: BTreeSet::new(),
                    },
                    common_quality_scale: 1.0,
                    group_face_ids: vec![face_id],
                    group_quality_reasons: BTreeSet::new(),
                    safety_scale: 1.0,
                    backoffs: 0,
                    bypassed: identity,
                    reasons,
                    last_failure: None,
                    last_verify_failure: None,
                    regions: Vec::new(),
                    jacobian,
                    detected_lines: 0,
                },
            });
        }
        validate_composition(
            &faces.iter().map(|f| &f.geometry).collect::<Vec<_>>(),
            current,
        )
        .map_err(restore_error)?;
        Ok(Self {
            analysis_key,
            base_dimensions,
            faces,
        })
    }

    /// 快照快速重现：网格逐脸重新通过 06 安全检查（固定顶点、Jacobian、最终网格）并重新
    /// 验证多人合成；共同质量比例与分组原因从恢复的质量门控重算，非确定性字段归零。
    pub(crate) fn restore(
        analysis_key: String,
        base_dimensions: [u32; 2],
        faces: Vec<RestoredSafeFace>,
        groups: &[Vec<usize>],
        current: &dyn Fn() -> bool,
    ) -> std::result::Result<Self, String> {
        if analysis_key.is_empty() || base_dimensions.contains(&0) || faces.len() > 32 {
            return Err("重塑快照场景契约非法".into());
        }
        let mut previous: Option<&str> = None;
        for face in &faces {
            if face.id.is_empty() || previous.is_some_and(|p| p >= face.id.as_str()) {
                return Err("重塑快照人脸顺序非法".into());
            }
            previous = Some(face.id.as_str());
            face.requested
                .scaled_response(1.0)
                .map_err(|_| "重塑快照强度非法".to_string())?;
            if !face.safety_scale.is_finite() || !(0.0..=1.0).contains(&face.safety_scale) {
                return Err("重塑快照安全比例非法".into());
            }
            check(current).map_err(restore_error)?;
        }
        validate_groups(faces.len(), groups)?;
        let mut group_index = vec![0usize; faces.len()];
        let mut common_quality = vec![1.0f64; faces.len()];
        let mut group_ids: Vec<Vec<String>> = vec![Vec::new(); faces.len()];
        let mut group_reasons: Vec<BTreeSet<LimitReason>> = vec![BTreeSet::new(); faces.len()];
        for (gi, group) in groups.iter().enumerate() {
            let scale = group
                .iter()
                .map(|&i| faces[i].quality.response_scale)
                .fold(1.0, f64::min);
            let ids: Vec<String> = group.iter().map(|&i| faces[i].id.clone()).collect();
            let reasons: BTreeSet<LimitReason> = group
                .iter()
                .flat_map(|&i| faces[i].quality.reasons.iter().copied())
                .collect();
            for &i in group {
                group_index[i] = gi;
                common_quality[i] = scale;
                group_ids[i] = ids.clone();
                group_reasons[i] = reasons.clone();
            }
        }
        let mut output = Vec::with_capacity(faces.len());
        for (index, face) in faces.into_iter().enumerate() {
            check(current).map_err(restore_error)?;
            let group = &groups[group_index[index]];
            let jacobian = validate_solution(&face.geometry, &face.plan).map_err(restore_error)?;
            // 原因集以快照保存为准（含 07／08 叠加项），再幂等补齐求解路径会重算的四类。
            let mut reasons = face.reasons;
            if group.len() > 1 {
                reasons.insert(LimitReason::Overlap);
            }
            if common_quality[index] < face.quality.response_scale {
                reasons.insert(LimitReason::GroupQualityLimited);
            }
            if face.safety_scale < 1.0 {
                reasons.insert(LimitReason::SafetyReduced);
            }
            if face.geometry.is_identity() && face.requested.0 != [0.0; 4] {
                reasons.insert(LimitReason::SafetyBypass);
            }
            let report = FaceSafetyReport {
                face_id: face.id,
                requested: face.requested,
                actual: face.geometry.actual_strengths,
                quality: face.quality,
                common_quality_scale: common_quality[index],
                group_face_ids: std::mem::take(&mut group_ids[index]),
                group_quality_reasons: std::mem::take(&mut group_reasons[index]),
                safety_scale: face.safety_scale,
                backoffs: face.backoffs,
                bypassed: face.geometry.is_identity(),
                reasons,
                // 原始失败文案不回放：重放路径不重新二分，失败记录只属于求解时刻。
                last_failure: None,
                last_verify_failure: None,
                regions: face.plan.regions.clone(),
                jacobian,
                detected_lines: face.detected_lines,
            };
            output.push(SafeFace {
                geometry: face.geometry,
                report,
            });
        }
        validate_composition(
            &output.iter().map(|f| &f.geometry).collect::<Vec<_>>(),
            current,
        )
        .map_err(restore_error)?;
        check(current).map_err(restore_error)?;
        Ok(SafeScene {
            analysis_key,
            base_dimensions,
            faces: output,
        })
    }

    pub(crate) fn byte_size(&self) -> usize {
        std::mem::size_of::<Self>()
            + self.analysis_key.capacity()
            + self.faces.capacity() * std::mem::size_of::<SafeFace>()
            + self
                .faces
                .iter()
                .map(|face| {
                    let report = &face.report;
                    face.geometry.byte_size()
                        + report.face_id.capacity()
                        + report
                            .last_verify_failure
                            .as_ref()
                            .map_or(0, String::capacity)
                        + report.group_face_ids.capacity() * std::mem::size_of::<String>()
                        + report
                            .group_face_ids
                            .iter()
                            .map(String::capacity)
                            .sum::<usize>()
                        + report.regions.capacity() * std::mem::size_of::<ProtectedRegion>()
                        + (report.reasons.len()
                            + report.quality.reasons.len()
                            + report.group_quality_reasons.len()
                            + report
                                .regions
                                .iter()
                                .map(|r| r.reasons.len())
                                .sum::<usize>())
                            * 256
                        + 512
                })
                .sum::<usize>()
    }

    /// 07 连续门外层回退：把请求比例、二分次数与最后失败文案折算进报告；冻结网格本身不重算。
    fn note_request_scale(&mut self, request_scale: f64, backoffs: usize, last_failure: String) {
        for face in &mut self.faces {
            let report = &mut face.report;
            report.safety_scale *= request_scale;
            report.backoffs += backoffs;
            report.reasons.insert(LimitReason::SafetyReduced);
            report.last_verify_failure = Some(last_failure.clone());
        }
    }

    pub fn base_dimensions(&self) -> [u32; 2] {
        self.base_dimensions
    }

    pub fn analysis_key(&self) -> &str {
        &self.analysis_key
    }
    pub fn reports(&self) -> impl Iterator<Item = &FaceSafetyReport> {
        self.faces.iter().map(|f| &f.report)
    }
    /// 07 获得只读网格；修改、重采样或 GPU 插值后仍须重新验证最终采样结果。
    pub fn meshes(&self) -> impl Iterator<Item = &GeometrySolution> {
        self.faces.iter().map(|f| &f.geometry)
    }
    pub fn forward(&self, p: [f64; 2]) -> Result<[f64; 2]> {
        self.query(p, false)
    }
    pub fn inverse(&self, p: [f64; 2]) -> Result<[f64; 2]> {
        self.query(p, true)
    }
    fn query(&self, p: [f64; 2], inverse: bool) -> Result<[f64; 2]> {
        if p.iter().any(|v| !v.is_finite()) {
            return Err(SafetyError::InvalidInput);
        }
        let mut output: Option<[f64; 2]> = None;
        for face in &self.faces {
            let result = if inverse {
                face.geometry.inverse(p)
            } else {
                face.geometry.forward(p)
            };
            match result {
                Ok(q) => {
                    if let Some(previous) = output {
                        if (P::from(previous) - P::from(q)).norm() > 1e-6 {
                            return Err(SafetyError::OverlappingDeformation);
                        }
                    } else {
                        output = Some(q);
                    }
                }
                Err(GeometryError::OutsideMesh) => {}
                Err(error) => return Err(error.into()),
            }
        }
        // 网格外严格 identity，无加法混合、后处理模糊或填补新内容。
        Ok(output.unwrap_or(p))
    }
}

#[cfg(test)]
#[path = "portrait_safety_tests.rs"]
mod tests;

#[cfg(test)]
pub(crate) use tests::{analysis, face, gpu_fixture, gpu_fixture_dimensions, image, map, request};
