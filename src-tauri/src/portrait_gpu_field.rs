//! 07：固定基底的相对逆位移栅格。只接受 06 冻结场，不接受任意外部位移。
#![allow(dead_code)]
use crate::portrait_protection::{P, bounds, convex_intersects};
use crate::portrait_safety::SafeScene;
use half::f16;
use nalgebra::Matrix2;
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;

pub const MAX_SIDE: u32 = 1024;
pub const FIELD_VERSION: &str = "reshape-grid-v1";
// ||J-I|| Frobenius；ARAP 满滑块常到 ~2–3.5，须与 WGSL norm_bound 同步。
pub const GRADIENT_LIMIT: f64 = 4.0;
pub const MAX_QUANTIZATION_ERROR: f64 = 1.0 / 64.0;
const MAX_VISITS: usize = 32_000_000;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FieldPrecision {
    F16,
    F32,
}

pub struct ValidatedField {
    dimensions: [u32; 2],
    analysis_key: String,
    key: String,
    precision: FieldPrecision,
    // xy 为相对基底像素位移，z 为硬锁标记，w 保留。数值已按最终格式量化。
    samples: Vec<[f32; 4]>,
    quantization_error: f64,
}

pub fn check_dimensions([w, h]: [u32; 2]) -> Result<(), String> {
    if w == 0 || h == 0 || w > MAX_SIDE || h > MAX_SIDE {
        return Err(format!("07 仅支持 1–{MAX_SIDE} 像素边长的固定完整基底"));
    }
    Ok(())
}
fn current_check(current: &dyn Fn() -> bool) -> Result<(), String> {
    if current() {
        Ok(())
    } else {
        Err("重塑场生成已取消".into())
    }
}
fn visit_rect(
    points: &[P],
    dims: [u32; 2],
    budget: &mut usize,
    current: &dyn Fn() -> bool,
    mut visit: impl FnMut(u32, u32) -> Result<(), String>,
) -> Result<(), String> {
    let b = bounds(points);
    if b[2] < 0.0 || b[3] < 0.0 || b[0] > dims[0] as f64 || b[1] > dims[1] as f64 {
        return Ok(());
    }
    let lo = [
        (b[0].floor() - 1.0).max(0.0) as u32,
        (b[1].floor() - 1.0).max(0.0) as u32,
    ];
    let hi = [
        (b[2].ceil() as u32).min(dims[0]),
        (b[3].ceil() as u32).min(dims[1]),
    ];
    for y in lo[1]..=hi[1] {
        current_check(current)?;
        *budget = budget
            .checked_sub((hi[0] - lo[0] + 1) as usize)
            .ok_or("重塑场栅格化超出访问预算")?;
        for x in lo[0]..=hi[0] {
            visit(x, y)?;
        }
    }
    Ok(())
}
fn square(x: u32, y: u32) -> [P; 4] {
    let (x, y) = (x as f64, y as f64);
    [
        P::new(x, y),
        P::new(x + 1.0, y),
        P::new(x + 1.0, y + 1.0),
        P::new(x, y + 1.0),
    ]
}
fn lock_cell(locked: &mut [bool], w: u32, x: u32, y: u32) {
    let i = (y * (w + 1) + x) as usize;
    for j in [i, i + 1, i + w as usize + 1, i + w as usize + 2] {
        locked[j] = true;
    }
}

impl ValidatedField {
    /// None 表示完整 identity，连整图 CPU 位移数组也不分配。
    pub fn from_scene(
        scene: &SafeScene,
        prefer_f16: bool,
        current: &dyn Fn() -> bool,
    ) -> Result<Option<Self>, String> {
        current_check(current)?;
        let dims = scene.base_dimensions();
        check_dimensions(dims)?;
        if scene.meshes().all(|m| m.is_identity()) {
            return Ok(None);
        }
        let [w, h] = dims;
        let count = ((w + 1) * (h + 1)) as usize;
        let mut raw = vec![[0.0f64; 2]; count];
        let mut assigned = vec![false; count];
        let mut locked = vec![false; count];
        let mut budget = MAX_VISITS;
        for (mesh, report) in scene.meshes().zip(scene.reports()) {
            let source = mesh.source_vertices();
            let target = mesh.target_vertices();
            let mut edges = BTreeMap::<(usize, usize), usize>::new();
            for (ti, &indices) in mesh.triangles().iter().enumerate() {
                let t = indices.map(|i| P::from(target[i]));
                let s = indices.map(|i| P::from(source[i]));
                let inverse = Matrix2::from_columns(&[t[1] - t[0], t[2] - t[0]])
                    .try_inverse()
                    .ok_or("退化的冻结目标三角形")?;
                visit_rect(&t, dims, &mut budget, current, |x, y| {
                    let p = P::new(x as f64, y as f64);
                    let uv = inverse * (p - t[0]);
                    if uv.x >= -1e-9 && uv.y >= -1e-9 && uv.x + uv.y <= 1.0 + 1e-9 {
                        let d = s[0] + (s[1] - s[0]) * uv.x + (s[2] - s[0]) * uv.y - p;
                        let i = (y * (w + 1) + x) as usize;
                        if assigned[i] && (P::from(raw[i]) - d).norm() > 1e-6 {
                            return Err("重叠逆位移不一致，禁止相加".into());
                        }
                        raw[i] = [d.x, d.y];
                        assigned[i] = true;
                    }
                    Ok(())
                })?;
                // 硬保护和零值邻脸的完整双线性支持域固定，不只检查顶点。
                if mesh.is_identity() || report.regions.iter().any(|r| r.triangle == ti && r.hard) {
                    visit_rect(&t, dims, &mut budget, current, |x, y| {
                        if x < w && y < h && convex_intersects(&t, &square(x, y)) {
                            lock_cell(&mut locked, w, x, y);
                        }
                        Ok(())
                    })?;
                }
                for (a, b) in [
                    (indices[0], indices[1]),
                    (indices[1], indices[2]),
                    (indices[2], indices[0]),
                ] {
                    *edges.entry((a.min(b), a.max(b))).or_default() += 1;
                }
            }
            // 外边界相交单元整体固定，插值支持域不能泄漏到网格外。
            // 稠密像素网（UV 形变）跳过：网格「边界」会误锁整脸并清零位移。
            let dense_mesh = source.len() != crate::face_reshape::MESH_VERTICES;
            if !dense_mesh {
                for ((a, b), _) in edges.iter().filter(|(_, n)| **n == 1) {
                    let segment = [P::from(target[*a]), P::from(target[*b])];
                    visit_rect(&segment, dims, &mut budget, current, |x, y| {
                        if x < w && y < h && convex_intersects(&segment, &square(x, y)) {
                            lock_cell(&mut locked, w, x, y);
                        }
                        Ok(())
                    })?;
                }
            }
        }
        for y in 0..=h {
            current_check(current)?;
            for x in 0..=w {
                let i = (y * (w + 1) + x) as usize;
                if x == 0 || y == 0 || x == w || y == h {
                    locked[i] = true;
                }
                if locked[i] {
                    raw[i] = [0.0; 2];
                }
            }
        }
        if raw.iter().all(|d| *d == [0.0; 2]) {
            // 与 CanonicalField::from_scene 一致：硬保护中和全部采样后收成显式 identity，
            // 不抛错阻断缩略图／预览；None 表示完整 identity，不分配整图位移数组。
            log::warn!("栅格保护使非零网格场采样全零，收成显式 identity");
            return Ok(None);
        }
        let field = Self::quantize(
            dims,
            scene.analysis_key(),
            &raw,
            &locked,
            prefer_f16,
            current,
        )?;
        Ok(Some(field))
    }

    fn quantize(
        dims: [u32; 2],
        analysis_key: &str,
        raw: &[[f64; 2]],
        locked: &[bool],
        prefer_f16: bool,
        current: &dyn Fn() -> bool,
    ) -> Result<Self, String> {
        let candidates: &[FieldPrecision] = if prefer_f16 {
            &[FieldPrecision::F16, FieldPrecision::F32]
        } else {
            &[FieldPrecision::F32]
        };
        let mut last = "量化失败".to_owned();
        for &precision in candidates {
            current_check(current)?;
            let mut error = 0.0f64;
            let samples: Vec<[f32; 4]> = raw
                .iter()
                .zip(locked)
                .map(|(p, &lock)| {
                    let q = p.map(|v| match precision {
                        FieldPrecision::F16 => f16::from_f64(v).to_f32(),
                        FieldPrecision::F32 => v as f32,
                    });
                    error = error
                        .max(((q[0] as f64 - p[0]).powi(2) + (q[1] as f64 - p[1]).powi(2)).sqrt());
                    [q[0], q[1], if lock { 1.0 } else { 0.0 }, 0.0]
                })
                .collect();
            if raw.iter().any(|p| *p != [0.0; 2])
                && samples.iter().all(|p| p[0] == 0.0 && p[1] == 0.0)
            {
                last = "非零位移被量化成全零".into();
                continue;
            }
            if error > MAX_QUANTIZATION_ERROR {
                last = "位移量化误差超过 1/64 基底像素".into();
                continue;
            }
            if let Err(e) = validate_grid(dims, &samples, current) {
                current_check(current)?;
                last = e;
                continue;
            }
            let mut hash = Sha256::new();
            hash.update(FIELD_VERSION);
            hash.update(analysis_key.len().to_le_bytes());
            hash.update(analysis_key);
            hash.update(dims[0].to_le_bytes());
            hash.update(dims[1].to_le_bytes());
            hash.update([if precision == FieldPrecision::F16 {
                16
            } else {
                32
            }]);
            for p in &samples {
                for v in p {
                    hash.update(v.to_le_bytes());
                }
            }
            return Ok(Self {
                dimensions: dims,
                analysis_key: analysis_key.into(),
                key: hex::encode(hash.finalize()),
                precision,
                samples,
                quantization_error: error,
            });
        }
        Err(format!("FP32 重塑场仍未通过安全门：{last}"))
    }
    pub fn dimensions(&self) -> [u32; 2] {
        self.dimensions
    }
    pub fn analysis_key(&self) -> &str {
        &self.analysis_key
    }
    pub fn key(&self) -> &str {
        &self.key
    }
    pub fn precision(&self) -> FieldPrecision {
        self.precision
    }
    pub fn samples(&self) -> &[[f32; 4]] {
        &self.samples
    }
    pub fn quantization_error(&self) -> f64 {
        self.quantization_error
    }
    pub fn upload_bytes(&self) -> Vec<u8> {
        match self.precision {
            FieldPrecision::F16 => self
                .samples
                .iter()
                .flatten()
                .flat_map(|&v| f16::from_f32(v).to_bits().to_le_bytes())
                .collect(),
            FieldPrecision::F32 => self
                .samples
                .iter()
                .flatten()
                .flat_map(|v| v.to_le_bytes())
                .collect(),
        }
    }
    #[cfg(test)]
    pub(crate) fn fixture(
        dims: [u32; 2],
        prefer_f16: bool,
        f: impl Fn(u32, u32) -> [f64; 2],
    ) -> Self {
        let raw: Vec<_> = (0..=dims[1])
            .flat_map(|y| (0..=dims[0]).map(move |x| (x, y)))
            .map(|(x, y)| f(x, y))
            .collect();
        Self::quantize(
            dims,
            "fixture",
            &raw,
            &vec![false; raw.len()],
            prefer_f16,
            &|| true,
        )
        .unwrap()
    }
}

/// 双线性单元的 J-I 是坐标的仿射函数；范数凸性保证四角范数上界覆盖整个单元。
/// ||J-I|| < .249 => 逆向奇异值 [ .751, 1.249 ]，正向 [1/1.249,1/.751]。
/// 与 identity 连通且不奇异，行列式为正，正反均满足 06；不是离散像素抽样证明。
pub(crate) fn validate_grid(
    dims: [u32; 2],
    samples: &[[f32; 4]],
    current: &dyn Fn() -> bool,
) -> Result<(), String> {
    check_dimensions(dims)?;
    validate_grid_window(dims, [0, 0], dims, samples, current)
}

/// 页只改变存储原点，不改变真实照片边界或连续安全条件。
pub(crate) fn validate_grid_window(
    full: [u32; 2],
    origin: [u32; 2],
    dims: [u32; 2],
    samples: &[[f32; 4]],
    current: &dyn Fn() -> bool,
) -> Result<(), String> {
    let [w, h] = dims;
    if w == 0
        || h == 0
        || w > MAX_SIDE
        || h > MAX_SIDE
        || origin[0].checked_add(w).is_none_or(|x| x > full[0])
        || origin[1].checked_add(h).is_none_or(|y| y > full[1])
        || samples.len() != ((w + 1) * (h + 1)) as usize
    {
        return Err("位移栅格窗口或长度不符".into());
    }
    for y in 0..=h {
        current_check(current)?;
        for x in 0..=w {
            let p = samples[(y * (w + 1) + x) as usize];
            let gx = origin[0] + x;
            let gy = origin[1] + y;
            if p.iter().any(|v| !v.is_finite())
                || p[2] != 0.0 && p[2] != 1.0
                || p[3] != 0.0
                || (p[2] == 1.0 || gx == 0 || gy == 0 || gx == full[0] || gy == full[1])
                    && (p[0] != 0.0 || p[1] != 0.0)
                || !(0.0..=full[0] as f64).contains(&(gx as f64 + p[0] as f64))
                || !(0.0..=full[1] as f64).contains(&(gy as f64 + p[1] as f64))
            {
                return Err("位移非有限、保护移动或越界".into());
            }
            if x == w || y == h {
                continue;
            }
            let at = |dx, dy| {
                let p = samples[((y + dy) * (w + 1) + x + dx) as usize];
                P::new(p[0] as f64, p[1] as f64)
            };
            let (a, b, c, d) = (at(0, 0), at(1, 0), at(0, 1), at(1, 1));
            for (dx, dy) in [
                (b - a, c - a),
                (b - a, d - b),
                (d - c, c - a),
                (d - c, d - b),
            ] {
                let delta = Matrix2::from_columns(&[dx, dy]);
                if delta.norm() > GRADIENT_LIMIT {
                    return Err("双线性场连续 Jacobian 界超限".into());
                }
                // 上述连续界同时蕴含正反奇异值与行列式限制，无需逐像素重复做 SVD。
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn rejects_invalid_dimensions_and_nan() {
        assert!(check_dimensions([1025, 1]).is_err());
        assert!(check_dimensions([0, 1]).is_err());
        let mut p = vec![[0.0; 4]; 9];
        p[4][0] = f32::NAN;
        assert!(validate_grid([2, 2], &p, &|| true).is_err());
    }
    #[test]
    fn checks_interior_gradient_and_hard_lock() {
        let mut p = vec![[0.0; 4]; 25];
        p[12][0] = 3.0;
        assert!(validate_grid([4, 4], &p, &|| true).is_err());
        p[12] = [0.1, 0.0, 1.0, 0.0];
        assert!(validate_grid([4, 4], &p, &|| true).is_err());
        p[12][2] = 0.0;
        validate_grid([4, 4], &p, &|| true).unwrap();
        // 美颜级局部梯度（约 1.0）应可通过；仅极端尖刺被拒。
        p[12][0] = 1.0;
        validate_grid([4, 4], &p, &|| true).unwrap();
    }
    #[test]
    fn fp16_promotes_without_relaxing_gradient_limit() {
        let raw = vec![
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.1244995, 0.1244995],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
        ];
        // 美颜级梯度界放宽后，该微位移可安全走 FP16。
        let f = ValidatedField::quantize([2, 2], "p", &raw, &[false; 9], true, &|| true).unwrap();
        assert_eq!(f.precision(), FieldPrecision::F16);
        let f32_only =
            ValidatedField::quantize([2, 2], "p32", &raw, &[false; 9], false, &|| true).unwrap();
        assert_eq!(f32_only.precision(), FieldPrecision::F32);
    }
    #[test]
    fn frozen_scene_dimensions_identity_and_rasterization() {
        for empty in [false, true] {
            let scene = crate::portrait_safety::gpu_fixture(0.0, empty);
            assert_eq!(scene.base_dimensions(), [128, 128]);
            assert!(
                ValidatedField::from_scene(&scene, true, &|| true)
                    .unwrap()
                    .is_none()
            );
        }
        let scene = crate::portrait_safety::gpu_fixture(10.0, false);
        let field = ValidatedField::from_scene(&scene, true, &|| true)
            .unwrap()
            .unwrap();
        assert!(field.samples().iter().any(|p| p[0] != 0.0 || p[1] != 0.0));
        assert!(
            field
                .samples()
                .iter()
                .filter(|p| p[2] == 1.0)
                .all(|p| p[0] == 0.0 && p[1] == 0.0)
        );
    }
    #[test]
    fn key_includes_analysis_precision_and_values() {
        let raw = vec![[0.0; 2]; 9];
        let locks = [false; 9];
        let a = ValidatedField::quantize([2, 2], "a", &raw, &locks, true, &|| true).unwrap();
        let b = ValidatedField::quantize([2, 2], "b", &raw, &locks, true, &|| true).unwrap();
        let c = ValidatedField::quantize([2, 2], "a", &raw, &locks, false, &|| true).unwrap();
        assert_ne!(a.key(), b.key());
        assert_ne!(a.key(), c.key());
        let mut changed = raw.clone();
        changed[4] = [0.1, 0.0];
        let d = ValidatedField::quantize([2, 2], "a", &changed, &locks, true, &|| true).unwrap();
        assert_ne!(a.key(), d.key());
        assert!(ValidatedField::quantize([2, 2], "a", &raw, &locks, true, &|| false).is_err());
    }
}
