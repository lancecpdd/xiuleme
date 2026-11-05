//! 08：规范绝对坐标、FP32 场页、逆采样足迹和有界资源规划；不依赖 Tauri。
#![allow(dead_code)]
use crate::portrait_analysis::check_current;
use crate::portrait_gpu_field::{MAX_QUANTIZATION_ERROR, validate_grid_window};
use crate::portrait_protection::{P, bounds, convex_intersects};
use crate::portrait_safety::SafeScene;
use nalgebra::Matrix2;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, VecDeque};
use std::sync::{Arc, Mutex, OnceLock};

pub const FIELD_VERSION: &str = "reshape-native-fp32-pages-v1";
pub const PAGE_SIDE: u32 = 256;
pub const PAGE_BUDGET: usize = 64 * 1024 * 1024;
pub const GPU_BUDGET: u64 = 256 * 1024 * 1024;
pub const MAX_SIDE: u32 = 32768;
pub const MAX_PIXELS: u64 = 64_000_000;

pub fn check_dimensions([w, h]: [u32; 2]) -> Result<(), String> {
    if w == 0 || h == 0 || w > MAX_SIDE || h > MAX_SIDE || w as u64 * h as u64 > MAX_PIXELS {
        return Err("重塑基底必须为非空图像，单轴不超过 32768、总像素不超过 6400 万".into());
    }
    Ok(())
}

/// 半开像素边界矩形；扩展视图允许负原点，内部块边界不代表真实图像边界。
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct PixelRect {
    pub x: i32,
    pub y: i32,
    pub width: u32,
    pub height: u32,
}
impl PixelRect {
    pub fn new(x: i32, y: i32, width: u32, height: u32) -> Result<Self, String> {
        if width == 0
            || height == 0
            || i64::from(x) + i64::from(width) > i64::from(i32::MAX)
            || i64::from(y) + i64::from(height) > i64::from(i32::MAX)
        {
            return Err("像素窗口为空或坐标溢出".into());
        }
        Ok(Self {
            x,
            y,
            width,
            height,
        })
    }
    pub fn full(dims: [u32; 2]) -> Result<Self, String> {
        check_dimensions(dims)?;
        Self::new(0, 0, dims[0], dims[1])
    }
    pub fn right(self) -> i64 {
        i64::from(self.x) + i64::from(self.width)
    }
    pub fn bottom(self) -> i64 {
        i64::from(self.y) + i64::from(self.height)
    }
    pub fn intersect(self, other: Self) -> Option<Self> {
        let (x, y) = (self.x.max(other.x), self.y.max(other.y));
        let (r, b) = (
            self.right().min(other.right()),
            self.bottom().min(other.bottom()),
        );
        if r <= i64::from(x) || b <= i64::from(y) {
            return None;
        }
        Self::new(x, y, (r - i64::from(x)) as u32, (b - i64::from(y)) as u32).ok()
    }
    pub fn contains(self, other: Self) -> bool {
        self.width > 0
            && self.height > 0
            && other.width > 0
            && other.height > 0
            && self.x <= other.x
            && self.y <= other.y
            && self.right() >= other.right()
            && self.bottom() >= other.bottom()
    }
    pub fn expand(self, radius: u32) -> Result<Self, String> {
        let r = i32::try_from(radius).map_err(|_| "窗口扩张溢出")?;
        Self::new(
            self.x.checked_sub(r).ok_or("窗口扩张溢出")?,
            self.y.checked_sub(r).ok_or("窗口扩张溢出")?,
            self.width
                .checked_add(radius.checked_mul(2).ok_or("窗口扩张溢出")?)
                .ok_or("窗口扩张溢出")?,
            self.height
                .checked_add(radius.checked_mul(2).ok_or("窗口扩张溢出")?)
                .ok_or("窗口扩张溢出")?,
        )
    }
    pub fn from_bounds(b: [f64; 4]) -> Result<Self, String> {
        if b.iter().any(|v| !v.is_finite() || v.abs() > 1e8) {
            return Err("采样足迹非有限或溢出".into());
        }
        let (x, y, r, bottom) = (
            b[0].floor() as i32,
            b[1].floor() as i32,
            b[2].ceil() as i32,
            b[3].ceil() as i32,
        );
        if r <= x || bottom <= y {
            return Err("采样足迹为空".into());
        }
        Self::new(x, y, (r - x) as u32, (bottom - y) as u32)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct OutputWindow {
    pub crop: PixelRect,
    pub output_dimensions: [u32; 2],
    pub roi: PixelRect,
    /// 输出像素边界到未裁剪 B 的仿射矩阵；未裁剪 U 使用 U→B。
    pub output_to_base: [f64; 6],
}
impl OutputWindow {
    pub fn new(
        full: [u32; 2],
        crop: &Value,
        output: [u32; 2],
        roi: Option<PixelRect>,
    ) -> Result<Self, String> {
        let crop = effective_crop(full, crop)?;
        let output_rect = PixelRect::full(output)?;
        let roi = roi.unwrap_or(output_rect);
        if !output_rect.contains(roi) {
            return Err("ROI 超出完整输出窗口".into());
        }
        Ok(Self {
            crop,
            output_dimensions: output,
            roi,
            output_to_base: [
                crop.width as f64 / output[0] as f64,
                0.0,
                crop.x as f64,
                0.0,
                crop.height as f64 / output[1] as f64,
                crop.y as f64,
            ],
        })
    }
    pub fn base_support(&self, tile: PixelRect, kernel: f64) -> Result<PixelRect, String> {
        if !self.roi.contains(tile) || !kernel.is_finite() || kernel < 0.0 {
            return Err("输出核或块窗口非法".into());
        }
        let [a, b, c, d, e, f] = self.output_to_base;
        if [a, b, c, d, e, f].iter().any(|v| !v.is_finite()) || (a * e - b * d).abs() < 1e-12 {
            return Err("输出视图矩阵非法".into());
        }
        let mut bounds = [
            f64::INFINITY,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::NEG_INFINITY,
        ];
        // 核宽随降采样比例扩张；不把 ROI 当作新的原点。
        for x in [tile.x as f64 - kernel, tile.right() as f64 + kernel] {
            for y in [tile.y as f64 - kernel, tile.bottom() as f64 + kernel] {
                let (u, v) = (a * x + b * y + c, d * x + e * y + f);
                bounds[0] = bounds[0].min(u);
                bounds[1] = bounds[1].min(v);
                bounds[2] = bounds[2].max(u);
                bounds[3] = bounds[3].max(v);
            }
        }
        PixelRect::from_bounds(bounds)
    }
}

/// 与 apply_crop 共用 round 和求交语义；非法或空裁剪返回完整图。
pub fn effective_crop(dims: [u32; 2], crop: &Value) -> Result<PixelRect, String> {
    let full = PixelRect::full(dims)?;
    let Some(values) = ["x", "y", "width", "height"]
        .map(|key| crop[key].as_f64())
        .into_iter()
        .collect::<Option<Vec<_>>>()
    else {
        return Ok(full);
    };
    if values.iter().any(|v| !v.is_finite()) {
        return Ok(full);
    }
    let v: Vec<u32> = values.iter().map(|v| v.round() as u32).collect();
    if v[2] == 0 || v[3] == 0 || v[0] >= dims[0] || v[1] >= dims[1] {
        return Ok(full);
    }
    PixelRect::new(
        v[0] as i32,
        v[1] as i32,
        v[2].min(dims[0] - v[0]),
        v[3].min(dims[1] - v[1]),
    )
}

struct Triangle {
    source: [P; 3],
    target: [P; 3],
    inverse: Matrix2<f64>,
    hard: bool,
}
enum Primitive {
    Triangle(Triangle),
    Boundary([P; 2]),
}
impl Primitive {
    fn points(&self) -> &[P] {
        match self {
            Self::Triangle(t) => &t.target,
            Self::Boundary(edge) => edge,
        }
    }
}

pub struct FieldPage {
    pub origin: [u32; 2],
    pub dimensions: [u32; 2],
    samples: Vec<[f32; 4]>,
    error: f64,
}
impl FieldPage {
    pub fn samples(&self) -> &[[f32; 4]] {
        &self.samples
    }
    pub fn node(&self, x: u32, y: u32) -> Result<[f32; 4], String> {
        if x < self.origin[0]
            || y < self.origin[1]
            || x > self.origin[0] + self.dimensions[0]
            || y > self.origin[1] + self.dimensions[1]
        {
            return Err("位移节点未包含在输入页内".into());
        }
        Ok(self.samples
            [((y - self.origin[1]) * (self.dimensions[0] + 1) + x - self.origin[0]) as usize])
    }
    fn bytes(&self) -> usize {
        self.samples.capacity() * 16 + std::mem::size_of::<Self>() + 256
    }
}

#[derive(Default)]
struct PageCache {
    entries: VecDeque<(String, [u32; 2], Arc<FieldPage>)>,
    bytes: usize,
}
// 所有资产共用一个 64 MiB 场页预算，不能每个资产各自占满一次预算。
static PAGES: OnceLock<Mutex<PageCache>> = OnceLock::new();

pub struct CanonicalField {
    dimensions: [u32; 2],
    primitives: Vec<Primitive>,
    index: BTreeMap<[u32; 2], Vec<usize>>,
    key: String,
    identity: bool,
    index_bytes: usize,
    quantization_error: f64,
    // 未裁剪 U 视图的共轭源，或 mesh∘UV 逆位移链；None 表示直接场。
    derived: Option<Box<DerivedView>>,
}

/// B 基底到 U 基底的逆旋转视图：U 节点映射到 B 坐标采样位移，再反变换位移向量。
/// B 是应用自由旋转后的同尺寸画布；旋转绕边界坐标半像素中心，与 CoordinateMap 一致。
struct ConjugateView {
    base: Arc<CanonicalField>,
    cos: f64,
    sin: f64,
    center: [f64; 2],
}

/// 逆位移链：src = inv_primary(inv_secondary(p))，即 d(p)=d_sec(p)+sample(primary, p+d_sec)。
/// 用于网格塑形后再叠 UV 大眼（先 secondary=UV，再 primary=mesh）。
struct InverseChainView {
    primary: Arc<CanonicalField>,
    secondary: Arc<CanonicalField>,
}

enum DerivedView {
    Conjugate(ConjugateView),
    InverseChain(InverseChainView),
}

/// 07 连续门失败中随请求强度降低可复检通过的类别：连续 Jacobian 界、位移越界、
/// 量化误差与跨脸逆位移冲突；其余（维度、窗口、索引、退化单元与存储预算）
/// 是契约错误，降强度重试不能掩盖。文案来源：本文件 generate_page 与 portrait_gpu_field 的最终场检查。
/// 硬保护把非零网格采样清成全零时，`from_scene` 直接收成显式 identity，不再作为错误抛出。
pub fn lower_strength_retryable(error: &str) -> bool {
    matches!(
        error,
        "双线性场连续 Jacobian 界超限"
            | "位移非有限、保护移动或越界"
            | "FP32 位移量化误差超限"
            | "共轭场 FP32 位移量化误差超限"
            | "规范场重叠逆位移不一致"
    )
}

impl CanonicalField {
    pub fn from_scene(scene: &SafeScene, current: &dyn Fn() -> bool) -> Result<Self, String> {
        let dimensions = scene.base_dimensions();
        check_dimensions(dimensions)?;
        let mut field = Self {
            dimensions,
            primitives: Vec::new(),
            index: BTreeMap::new(),
            key: String::new(),
            identity: scene.meshes().all(|mesh| mesh.is_identity()),
            index_bytes: std::mem::size_of::<Self>(),
            quantization_error: 0.0,
            derived: None,
        };
        if !field.identity {
            for (mesh, report) in scene.meshes().zip(scene.reports()) {
                let source = mesh.source_vertices();
                let target = mesh.target_vertices();
                let mut edges = BTreeMap::<(usize, usize), usize>::new();
                for (ti, &ids) in mesh.triangles().iter().enumerate() {
                    check_current(current)?;
                    let t = ids.map(|i| P::from(target[i]));
                    let inverse = Matrix2::from_columns(&[t[1] - t[0], t[2] - t[0]])
                        .try_inverse()
                        .ok_or("退化的冻结目标三角形")?;
                    field.insert(Primitive::Triangle(Triangle {
                        source: ids.map(|i| P::from(source[i])),
                        target: t,
                        inverse,
                        hard: mesh.is_identity()
                            || report.regions.iter().any(|r| r.triangle == ti && r.hard),
                    }))?;
                    for (a, b) in [(ids[0], ids[1]), (ids[1], ids[2]), (ids[2], ids[0])] {
                        *edges.entry((a.min(b), a.max(b))).or_default() += 1;
                    }
                }
                // ARAP 控制网需要边界硬锁；稠密像素网（UV 形变）的「边界」
                // 多为空洞，硬锁会把塑形位移清零。
                let dense_mesh = source.len() != crate::face_reshape::MESH_VERTICES;
                if !dense_mesh {
                    for ((a, b), _) in edges.iter().filter(|(_, n)| **n == 1) {
                        field.insert(Primitive::Boundary([
                            P::from(target[*a]),
                            P::from(target[*b]),
                        ]))?;
                    }
                }
            }
        }
        let mut digest = Sha256::new();
        digest.update(FIELD_VERSION);
        digest.update(scene.analysis_key().as_bytes());
        for n in dimensions {
            digest.update(n.to_le_bytes());
        }
        let mut moved = false;
        // 固定顺序流式证明整个支持域；没有索引的单元已知为 identity。
        for &page in field.index.keys() {
            let p = field.generate_page(page, current)?;
            field.quantization_error = field.quantization_error.max(p.error);
            for n in page {
                digest.update(n.to_le_bytes());
            }
            for sample in p.samples() {
                moved |= sample[0] != 0.0 || sample[1] != 0.0;
                for value in sample {
                    digest.update(value.to_le_bytes());
                }
            }
        }
        if !field.identity && !moved {
            // 05：完全硬保护导致输出不变时返回 identity。
            // 禁止外泄「非 identity 网格 + 全零采样场」；显式收成 identity 后入口走旧像素旁路。
            log::warn!("规范栅格保护使非零网格场采样全零，收成显式 identity");
            field.primitives.clear();
            field.index.clear();
            field.identity = true;
            field.index_bytes = std::mem::size_of::<Self>();
            field.quantization_error = 0.0;
            let mut identity_digest = Sha256::new();
            identity_digest.update(FIELD_VERSION);
            identity_digest.update(scene.analysis_key().as_bytes());
            for n in dimensions {
                identity_digest.update(n.to_le_bytes());
            }
            field.key = hex::encode(identity_digest.finalize());
        } else {
            field.key = hex::encode(digest.finalize());
        }
        check_current(current)?;
        Ok(field)
    }

    /// 未裁剪 U 视图共轭场 F_U = R⁻¹∘F_B∘R：复用同一冻结 B 场，不重新分析。
    /// U 节点映射到 B 坐标双线性采样位移，位移向量经逆旋转回到 U 基；
    /// U 画布边界与 B 节点域外（旋转矩形超出同尺寸画布的角落）恒锁零。
    /// rotation 为 B 视图的自由旋转角（度），与 apply_rotation 同向。
    pub fn conjugate_uncropped(
        base: &Arc<Self>,
        rotation_degrees: f64,
        current: &dyn Fn() -> bool,
    ) -> Result<Self, String> {
        check_current(current)?;
        let angle = rotation_degrees.to_radians();
        if !angle.is_finite() {
            return Err("未裁剪共轭旋转角非法".into());
        }
        let dimensions = base.dimensions;
        check_dimensions(dimensions)?;
        let (sin, cos) = angle.sin_cos();
        let mut digest = Sha256::new();
        digest.update("conjugate-uncropped-v1".as_bytes());
        digest.update(base.key.as_bytes());
        digest.update(rotation_degrees.to_le_bytes());
        for n in dimensions {
            digest.update(n.to_le_bytes());
        }
        Ok(Self {
            dimensions,
            primitives: Vec::new(),
            index: BTreeMap::new(),
            key: hex::encode(digest.finalize()),
            identity: base.identity,
            index_bytes: std::mem::size_of::<Self>(),
            quantization_error: base.quantization_error,
            derived: Some(Box::new(DerivedView::Conjugate(ConjugateView {
                base: Arc::clone(base),
                cos,
                sin,
                center: [
                    dimensions[0] as f64 / 2.0 + 0.5,
                    dimensions[1] as f64 / 2.0 + 0.5,
                ],
            }))),
        })
    }

    /// 逆位移链合成：inv_primary ∘ inv_secondary（先 secondary 再 primary）。
    /// 用于网格场上叠加 UV 大眼场。
    pub fn compose_inverse_chain(
        primary: &Arc<Self>,
        secondary: &Arc<Self>,
        current: &dyn Fn() -> bool,
    ) -> Result<Self, String> {
        check_current(current)?;
        if primary.dimensions != secondary.dimensions {
            return Err("位移链合成要求场尺寸一致".into());
        }
        if primary.identity && secondary.identity {
            return Ok(Self {
                dimensions: primary.dimensions,
                primitives: Vec::new(),
                index: BTreeMap::new(),
                key: primary.key.clone(),
                identity: true,
                index_bytes: std::mem::size_of::<Self>(),
                quantization_error: 0.0,
                derived: None,
            });
        }
        let dimensions = primary.dimensions;
        check_dimensions(dimensions)?;
        let mut index = BTreeMap::new();
        let mut index_bytes = std::mem::size_of::<Self>();
        for &page in primary.index.keys().chain(secondary.index.keys()) {
            index_bytes = index_bytes.checked_add(256).ok_or("场索引字节溢出")?;
            if index_bytes > PAGE_BUDGET {
                return Err("冻结场索引超过 64 MiB".into());
            }
            index.entry(page).or_insert_with(Vec::new);
        }
        let mut digest = Sha256::new();
        digest.update(b"inverse-chain-v1");
        digest.update(primary.key.as_bytes());
        digest.update(secondary.key.as_bytes());
        for n in dimensions {
            digest.update(n.to_le_bytes());
        }
        Ok(Self {
            dimensions,
            primitives: Vec::new(),
            index,
            key: hex::encode(digest.finalize()),
            identity: false,
            index_bytes,
            quantization_error: primary.quantization_error.max(secondary.quantization_error),
            derived: Some(Box::new(DerivedView::InverseChain(InverseChainView {
                primary: Arc::clone(primary),
                secondary: Arc::clone(secondary),
            }))),
        })
    }

    fn sample_bilinear(
        &self,
        x: f64,
        y: f64,
        current: &dyn Fn() -> bool,
    ) -> Result<([f64; 2], bool), String> {
        let [bw, bh] = self.dimensions;
        // 链采样偶发越界时钳制到域内，避免贡献突然变零造成梯度尖刺。
        let x = x.clamp(0.0, bw as f64);
        let y = y.clamp(0.0, bh as f64);
        let x0 = x.floor().clamp(0.0, (bw.saturating_sub(1)) as f64) as u32;
        let y0 = y.floor().clamp(0.0, (bh.saturating_sub(1)) as f64) as u32;
        let (tx, ty) = (x - x0 as f64, y - y0 as f64);
        let (n00, n10) = (self.node(x0, y0, current)?, self.node(x0 + 1, y0, current)?);
        let (n01, n11) = (
            self.node(x0, y0 + 1, current)?,
            self.node(x0 + 1, y0 + 1, current)?,
        );
        let blend = |i: usize| {
            let top = n00[i] as f64 * (1.0 - tx) + n10[i] as f64 * tx;
            let bottom = n01[i] as f64 * (1.0 - tx) + n11[i] as f64 * tx;
            top * (1.0 - ty) + bottom * ty
        };
        if blend(2) > 0.5 {
            // 锁定＝该场位移为零，不是整条合成链锁死。
            Ok(([0.0; 2], true))
        } else {
            Ok(([blend(0), blend(1)], false))
        }
    }

    /// 合成后立刻校验全部索引页，避免把非法场交给预览/导出才爆错。
    pub fn validate_indexed_pages(&self, current: &dyn Fn() -> bool) -> Result<(), String> {
        for &page in self.index.keys() {
            check_current(current)?;
            let _ = self.page(page, current)?;
        }
        Ok(())
    }

    fn insert(&mut self, primitive: Primitive) -> Result<(), String> {
        let b = bounds(primitive.points());
        if b.iter().any(|v| !v.is_finite() || v.abs() > 1e8) {
            return Err("场索引坐标非法".into());
        }
        // 扩 2px 包含相邻完整单元及共享边界节点的所有约束。
        let (lo_x, lo_y) = (
            (b[0].floor() - 2.0).max(0.0) as u32,
            (b[1].floor() - 2.0).max(0.0) as u32,
        );
        let (hi_x, hi_y) = (
            (b[2].ceil() + 2.0)
                .max(0.0)
                .min((self.dimensions[0] - 1) as f64) as u32,
            (b[3].ceil() + 2.0)
                .max(0.0)
                .min((self.dimensions[1] - 1) as f64) as u32,
        );
        let id = self.primitives.len();
        if lo_x <= hi_x && lo_y <= hi_y {
            for py in lo_y / PAGE_SIDE..=hi_y / PAGE_SIDE {
                for px in lo_x / PAGE_SIDE..=hi_x / PAGE_SIDE {
                    // 保守计入 BTree 节点、Vec 扩容和引用；准入前检查。
                    self.index_bytes = self.index_bytes.checked_add(256).ok_or("场索引字节溢出")?;
                    if self.index_bytes > PAGE_BUDGET {
                        return Err("冻结场索引超过 64 MiB".into());
                    }
                    self.index.entry([px, py]).or_default().push(id);
                }
            }
        }
        self.index_bytes = self
            .index_bytes
            .checked_add(2 * std::mem::size_of::<Primitive>())
            .ok_or("场索引字节溢出")?;
        if self.index_bytes > PAGE_BUDGET {
            return Err("冻结场索引超过 64 MiB".into());
        }
        self.primitives.push(primitive);
        Ok(())
    }

    fn generate_page(
        &self,
        page: [u32; 2],
        current: &dyn Fn() -> bool,
    ) -> Result<FieldPage, String> {
        let origin = page.map(|v| v.checked_mul(PAGE_SIDE).ok_or("场页坐标溢出"));
        let origin = [origin[0]?, origin[1]?];
        if origin[0] >= self.dimensions[0] || origin[1] >= self.dimensions[1] {
            return Err("场页超出规范基底".into());
        }
        let dims = [
            (self.dimensions[0] - origin[0]).min(PAGE_SIDE),
            (self.dimensions[1] - origin[1]).min(PAGE_SIDE),
        ];
        if let Some(derived) = self.derived.as_ref() {
            return match derived.as_ref() {
                DerivedView::Conjugate(view) => {
                    self.generate_conjugate_page(view, origin, dims, current)
                }
                DerivedView::InverseChain(view) => {
                    self.generate_inverse_chain_page(view, origin, dims, current)
                }
            };
        }
        let count = ((dims[0] + 1) * (dims[1] + 1)) as usize;
        let mut raw = vec![[0.0f64; 2]; count];
        let mut assigned = vec![false; count];
        let mut locked = vec![false; count];
        for &id in self.index.get(&page).into_iter().flatten() {
            let primitive = &self.primitives[id];
            let b = bounds(primitive.points());
            if let Primitive::Triangle(t) = primitive {
                visit_bounds(b, origin, dims, self.dimensions, false, current, |x, y| {
                    let p = P::new(x as f64, y as f64);
                    let uv = t.inverse * (p - t.target[0]);
                    if uv.x >= -1e-9 && uv.y >= -1e-9 && uv.x + uv.y <= 1.0 + 1e-9 {
                        let d = t.source[0]
                            + (t.source[1] - t.source[0]) * uv.x
                            + (t.source[2] - t.source[0]) * uv.y
                            - p;
                        let i = ((y - origin[1]) * (dims[0] + 1) + x - origin[0]) as usize;
                        if assigned[i] && (P::from(raw[i]) - d).norm() > 1e-6 {
                            return Err("规范场重叠逆位移不一致".into());
                        }
                        raw[i] = [d.x, d.y];
                        assigned[i] = true;
                    }
                    Ok(())
                })?;
            }
            if matches!(primitive, Primitive::Boundary(_))
                || matches!(primitive, Primitive::Triangle(t) if t.hard)
            {
                visit_bounds(b, origin, dims, self.dimensions, true, current, |x, y| {
                    let square = [
                        P::new(x as f64, y as f64),
                        P::new((x + 1) as f64, y as f64),
                        P::new((x + 1) as f64, (y + 1) as f64),
                        P::new(x as f64, (y + 1) as f64),
                    ];
                    if convex_intersects(primitive.points(), &square) {
                        for (nx, ny) in [(x, y), (x + 1, y), (x, y + 1), (x + 1, y + 1)] {
                            if nx >= origin[0]
                                && ny >= origin[1]
                                && nx <= origin[0] + dims[0]
                                && ny <= origin[1] + dims[1]
                            {
                                locked[((ny - origin[1]) * (dims[0] + 1) + nx - origin[0])
                                    as usize] = true;
                            }
                        }
                    }
                    Ok(())
                })?;
            }
        }
        let mut error = 0.0f64;
        let mut samples = Vec::with_capacity(count);
        for y in 0..=dims[1] {
            check_current(current)?;
            for x in 0..=dims[0] {
                let i = (y * (dims[0] + 1) + x) as usize;
                let gx = x + origin[0];
                let gy = y + origin[1];
                let lock = locked[i]
                    || gx == 0
                    || gy == 0
                    || gx == self.dimensions[0]
                    || gy == self.dimensions[1];
                let p = if lock { [0.0; 2] } else { raw[i] };
                let q = p.map(|v| v as f32);
                error =
                    error.max(((q[0] as f64 - p[0]).powi(2) + (q[1] as f64 - p[1]).powi(2)).sqrt());
                samples.push([q[0], q[1], if lock { 1.0 } else { 0.0 }, 0.0]);
            }
        }
        if !error.is_finite() || error > MAX_QUANTIZATION_ERROR {
            return Err("FP32 位移量化误差超限".into());
        }
        validate_grid_window(self.dimensions, origin, dims, &samples, current)?;
        Ok(FieldPage {
            origin,
            dimensions: dims,
            samples,
            error,
        })
    }

    /// 共轭页生成：U 节点映射到 B 坐标双线性采样冻结 B 场，位移向量经逆旋转回 U 基。
    /// U 画布边界节点锁零；B 节点域外（旋转矩形落在同尺寸 B 画布外的角落）恒零。
    /// 连续性与量化误差仍由 validate_grid_window 统一验证，不接受未验证的共轭页。
    fn generate_conjugate_page(
        &self,
        view: &ConjugateView,
        origin: [u32; 2],
        dims: [u32; 2],
        current: &dyn Fn() -> bool,
    ) -> Result<FieldPage, String> {
        let base = view.base.as_ref();
        let [bw, bh] = base.dimensions;
        let mut error = 0.0f64;
        let mut samples = Vec::with_capacity(((dims[0] + 1) * (dims[1] + 1)) as usize);
        for y in 0..=dims[1] {
            check_current(current)?;
            for x in 0..=dims[0] {
                let gx = origin[0] + x;
                let gy = origin[1] + y;
                let boundary =
                    gx == 0 || gy == 0 || gx == self.dimensions[0] || gy == self.dimensions[1];
                let (d, locked) = if boundary {
                    ([0.0f64; 2], true)
                } else {
                    // b = R(u)：绕边界坐标半像素中心，与 apply_rotation/CoordinateMap 同向。
                    let ux = gx as f64 - view.center[0];
                    let uy = gy as f64 - view.center[1];
                    let bx = view.cos * ux - view.sin * uy + view.center[0];
                    let by = view.sin * ux + view.cos * uy + view.center[1];
                    if (0.0..=bw as f64).contains(&bx) && (0.0..=bh as f64).contains(&by) {
                        let x0 = bx.floor().clamp(0.0, (bw - 1) as f64) as u32;
                        let y0 = by.floor().clamp(0.0, (bh - 1) as f64) as u32;
                        let (tx, ty) = (bx - x0 as f64, by - y0 as f64);
                        let (n00, n10) =
                            (base.node(x0, y0, current)?, base.node(x0 + 1, y0, current)?);
                        let (n01, n11) = (
                            base.node(x0, y0 + 1, current)?,
                            base.node(x0 + 1, y0 + 1, current)?,
                        );
                        let blend = |i: usize| {
                            let top = n00[i] as f64 * (1.0 - tx) + n10[i] as f64 * tx;
                            let bottom = n01[i] as f64 * (1.0 - tx) + n11[i] as f64 * tx;
                            top * (1.0 - ty) + bottom * ty
                        };
                        // 基底锁定标记随位移重采样：锁定节点位移恒零，与基底锁定语义一致；
                        // 零旋转整数采样时逐位取基底节点，保证共轭与基底位级恒等。
                        if blend(2) > 0.5 {
                            ([0.0; 2], true)
                        } else {
                            let d_b = [blend(0), blend(1)];
                            // d_U = R⁻¹ 线性部分作用位移向量：[c·dx + s·dy, -s·dx + c·dy]。
                            (
                                [
                                    view.cos * d_b[0] + view.sin * d_b[1],
                                    -view.sin * d_b[0] + view.cos * d_b[1],
                                ],
                                false,
                            )
                        }
                    } else {
                        ([0.0; 2], false)
                    }
                };
                let q = d.map(|v| v as f32);
                error =
                    error.max(((q[0] as f64 - d[0]).powi(2) + (q[1] as f64 - d[1]).powi(2)).sqrt());
                samples.push([q[0], q[1], if locked { 1.0 } else { 0.0 }, 0.0]);
            }
        }
        if !error.is_finite() || error > MAX_QUANTIZATION_ERROR {
            return Err("共轭场 FP32 位移量化误差超限".into());
        }
        validate_grid_window(self.dimensions, origin, dims, &samples, current)?;
        Ok(FieldPage {
            origin,
            dimensions: dims,
            samples,
            error,
        })
    }

    fn generate_inverse_chain_page(
        &self,
        view: &InverseChainView,
        origin: [u32; 2],
        dims: [u32; 2],
        current: &dyn Fn() -> bool,
    ) -> Result<FieldPage, String> {
        let mut error = 0.0f64;
        let mut samples = Vec::with_capacity(((dims[0] + 1) * (dims[1] + 1)) as usize);
        for y in 0..=dims[1] {
            check_current(current)?;
            for x in 0..=dims[0] {
                let gx = origin[0] + x;
                let gy = origin[1] + y;
                let boundary =
                    gx == 0 || gy == 0 || gx == self.dimensions[0] || gy == self.dimensions[1];
                // inv_total = inv_primary ∘ inv_secondary：
                // 子场锁定只表示该子场位移为 0，不能把整链清零（否则大眼边缘硬切断 → Jacobian 爆）。
                let (d, locked) = if boundary {
                    ([0.0f64; 2], true)
                } else {
                    let (mut d_sec, lock_sec) = view
                        .secondary
                        .sample_bilinear(gx as f64, gy as f64, current)?;
                    if lock_sec {
                        d_sec = [0.0; 2];
                    }
                    let sx = gx as f64 + d_sec[0];
                    let sy = gy as f64 + d_sec[1];
                    let (mut d_pri, lock_pri) = view.primary.sample_bilinear(sx, sy, current)?;
                    if lock_pri {
                        d_pri = [0.0; 2];
                    }
                    ([d_sec[0] + d_pri[0], d_sec[1] + d_pri[1]], false)
                };
                let q = d.map(|v| v as f32);
                error =
                    error.max(((q[0] as f64 - d[0]).powi(2) + (q[1] as f64 - d[1]).powi(2)).sqrt());
                samples.push([q[0], q[1], if locked { 1.0 } else { 0.0 }, 0.0]);
            }
        }
        if !error.is_finite() || error > MAX_QUANTIZATION_ERROR {
            return Err("FP32 位移量化误差超限".into());
        }
        validate_grid_window(self.dimensions, origin, dims, &samples, current)?;
        Ok(FieldPage {
            origin,
            dimensions: dims,
            samples,
            error,
        })
    }

    pub fn page(
        &self,
        page: [u32; 2],
        current: &dyn Fn() -> bool,
    ) -> Result<Arc<FieldPage>, String> {
        check_current(current)?;
        let cache = PAGES.get_or_init(|| Mutex::new(PageCache::default()));
        {
            let mut cache = cache.lock().unwrap_or_else(|e| e.into_inner());
            if let Some(i) = cache
                .entries
                .iter()
                .position(|e| e.0 == self.key && e.1 == page)
            {
                let entry = cache.entries.remove(i).ok_or("场页缓存索引错误")?;
                let result = Arc::clone(&entry.2);
                cache.entries.push_back(entry);
                return Ok(result);
            }
        }
        let result = Arc::new(self.generate_page(page, current)?);
        let bytes = result.bytes();
        let mut cache = cache.lock().unwrap_or_else(|e| e.into_inner());
        // 同页并发生成可以丢弃重复副本，不重复记账。
        if let Some(entry) = cache
            .entries
            .iter()
            .find(|e| e.0 == self.key && e.1 == page)
        {
            return Ok(Arc::clone(&entry.2));
        }
        while cache.bytes > PAGE_BUDGET - bytes {
            let i = cache
                .entries
                .iter()
                .position(|e| Arc::strong_count(&e.2) == 1)
                .ok_or("场页预算已被活跃消费者占用")?;
            let (_, _, removed) = cache.entries.remove(i).ok_or("场页缓存索引错误")?;
            cache.bytes -= removed.bytes();
        }
        cache.bytes += bytes;
        cache
            .entries
            .push_back((self.key.clone(), page, Arc::clone(&result)));
        Ok(result)
    }

    pub fn key(&self) -> &str {
        &self.key
    }
    pub fn dimensions(&self) -> [u32; 2] {
        self.dimensions
    }
    pub fn is_identity(&self) -> bool {
        self.identity
    }
    pub fn index_byte_size(&self) -> usize {
        self.index_bytes
    }
    pub fn quantization_error(&self) -> f64 {
        self.quantization_error
    }

    pub fn window_samples(
        &self,
        window: PixelRect,
        current: &dyn Fn() -> bool,
    ) -> Result<Vec<[f32; 4]>, String> {
        if window.width == 0 || window.height == 0 || window.width > 1024 || window.height > 1024 {
            return Err("位移上传窗口超过核心块限制".into());
        }
        let stride = (window.width + 1) as usize;
        let mut samples = vec![[0.0, 0.0, 1.0, 0.0]; stride * (window.height + 1) as usize];
        for y in 0..=window.height {
            check_current(current)?;
            let gy = i64::from(window.y) + i64::from(y);
            if gy < 0 || gy > i64::from(self.dimensions[1]) {
                continue;
            }
            let mut x = 0;
            while x <= window.width {
                let gx = i64::from(window.x) + i64::from(x);
                if gx < 0 || gx > i64::from(self.dimensions[0]) {
                    x += 1;
                    continue;
                }
                let page = self.page(
                    [
                        (gx as u32).min(self.dimensions[0] - 1) / PAGE_SIDE,
                        (gy as u32).min(self.dimensions[1] - 1) / PAGE_SIDE,
                    ],
                    current,
                )?;
                let end = (page.origin[0] + page.dimensions[0]) as i64;
                while x <= window.width && i64::from(window.x) + i64::from(x) <= end {
                    samples[y as usize * stride + x as usize] =
                        page.node((i64::from(window.x) + i64::from(x)) as u32, gy as u32)?;
                    x += 1;
                }
            }
        }
        Ok(samples)
    }

    pub fn node(&self, x: u32, y: u32, current: &dyn Fn() -> bool) -> Result<[f32; 4], String> {
        if x > self.dimensions[0] || y > self.dimensions[1] {
            return Err("位移节点越界".into());
        }
        let page = [
            x.min(self.dimensions[0] - 1) / PAGE_SIDE,
            y.min(self.dimensions[1] - 1) / PAGE_SIDE,
        ];
        self.page(page, current)?.node(x, y)
    }

    /// 双线性单元分量极值位于四角；必须遍历所有相交单元，不能只映射大块四角。
    pub fn inverse_footprint(
        &self,
        target: PixelRect,
        radius: u32,
        current: &dyn Fn() -> bool,
    ) -> Result<PixelRect, String> {
        let mut footprint = [
            target.x as f64,
            target.y as f64,
            target.right() as f64,
            target.bottom() as f64,
        ];
        if let Some(area) = target.intersect(PixelRect::full(self.dimensions)?) {
            let x0 = area.x as u32;
            let y0 = area.y as u32;
            let x1 = area.right() as u32;
            let y1 = area.bottom() as u32;
            if self.derived.is_some() {
                // 派生场没有基元索引；位移可能落在任何节点，必须遍历全部相交单元。
                for y in y0..y1 {
                    check_current(current)?;
                    for x in x0..x1 {
                        for (nx, ny) in [(x, y), (x + 1, y), (x, y + 1), (x + 1, y + 1)] {
                            let d = self.node(nx, ny, current)?;
                            let (u, v) = (nx as f64 + d[0] as f64, ny as f64 + d[1] as f64);
                            footprint[0] = footprint[0].min(u);
                            footprint[1] = footprint[1].min(v);
                            footprint[2] = footprint[2].max(u);
                            footprint[3] = footprint[3].max(v);
                        }
                    }
                }
                return PixelRect::from_bounds(footprint)?
                    .expand(radius.checked_add(1).ok_or("滤波半径溢出")?);
            }
            for py in y0 / PAGE_SIDE..=(y1 - 1) / PAGE_SIDE {
                for px in x0 / PAGE_SIDE..=(x1 - 1) / PAGE_SIDE {
                    check_current(current)?;
                    if !self.index.contains_key(&[px, py]) {
                        continue;
                    }
                    let page = self.page([px, py], current)?;
                    for y in y0.max(page.origin[1])..y1.min(page.origin[1] + page.dimensions[1]) {
                        check_current(current)?;
                        for x in x0.max(page.origin[0])..x1.min(page.origin[0] + page.dimensions[0])
                        {
                            for (nx, ny) in [(x, y), (x + 1, y), (x, y + 1), (x + 1, y + 1)] {
                                let d = page.node(nx, ny)?;
                                let (u, v) = (nx as f64 + d[0] as f64, ny as f64 + d[1] as f64);
                                footprint[0] = footprint[0].min(u);
                                footprint[1] = footprint[1].min(v);
                                footprint[2] = footprint[2].max(u);
                                footprint[3] = footprint[3].max(v);
                            }
                        }
                    }
                }
            }
        }
        PixelRect::from_bounds(footprint)?.expand(radius.checked_add(1).ok_or("滤波半径溢出")?)
    }
}

fn visit_bounds(
    b: [f64; 4],
    origin: [u32; 2],
    dims: [u32; 2],
    full: [u32; 2],
    cells: bool,
    current: &dyn Fn() -> bool,
    mut visit: impl FnMut(u32, u32) -> Result<(), String>,
) -> Result<(), String> {
    let start = if cells {
        origin.map(|v| v.saturating_sub(1))
    } else {
        origin
    };
    let end = [
        (origin[0] + dims[0]).min(full[0] - u32::from(cells)),
        (origin[1] + dims[1]).min(full[1] - u32::from(cells)),
    ];
    let lo = [
        (b[0].floor() - 1.0).max(start[0] as f64),
        (b[1].floor() - 1.0).max(start[1] as f64),
    ];
    let hi = [
        b[2].ceil().min(end[0] as f64),
        b[3].ceil().min(end[1] as f64),
    ];
    if lo[0] > hi[0] || lo[1] > hi[1] {
        return Ok(());
    }
    for y in lo[1] as u32..=hi[1] as u32 {
        check_current(current)?;
        for x in lo[0] as u32..=hi[0] as u32 {
            visit(x, y)?;
        }
    }
    Ok(())
}

/// 调色读源分支取并集，串行 blur 的横纵支持分别扩张；半径只依赖规范尺寸。
#[derive(Clone, Debug)]
pub struct CompositionDependencies {
    pub blur_radii: [u32; 4],
    pub denoise_radius: u32,
    pub sharpen_radius: u32,
    pub ca_red: f64,
    pub ca_blue: f64,
}
impl CompositionDependencies {
    pub fn native(
        full: [u32; 2],
        denoise: bool,
        sharpen: bool,
        ca: [f64; 2],
    ) -> Result<Self, String> {
        check_dimensions(full)?;
        if ca.iter().any(|v| !v.is_finite()) {
            return Err("色差映射非法".into());
        }
        // 与既有 GPU blur 的 f32 计算和最小 1px 规则一致。
        let scale = full[0].min(full[1]) as f32 / 1080.0;
        Ok(Self {
            blur_radii: [1.0, 3.5, 8.0, 40.0].map(|r| (r * scale).ceil().max(1.0) as u32),
            denoise_radius: if denoise { 16 } else { 0 },
            sharpen_radius: if sharpen { 2 } else { 0 },
            ca_red: ca[0],
            ca_blue: ca[1],
        })
    }
    pub fn input(&self, area: PixelRect, full: [u32; 2]) -> Result<PixelRect, String> {
        let radius = self
            .blur_radii
            .into_iter()
            .max()
            .unwrap_or(0)
            .max(self.denoise_radius)
            .max(self.sharpen_radius);
        let expanded = area.expand(radius)?;
        let mut b = [
            expanded.x as f64,
            expanded.y as f64,
            expanded.right() as f64,
            expanded.bottom() as f64,
        ];
        let center = [full[0] as f64 * 0.5, full[1] as f64 * 0.5];
        for ca in [self.ca_red, self.ca_blue] {
            for x in [area.x as f64, area.right() as f64] {
                for y in [area.y as f64, area.bottom() as f64] {
                    let (u, v) = (x - (x - center[0]) * ca, y - (y - center[1]) * ca);
                    b[0] = b[0].min(u - 1.0);
                    b[1] = b[1].min(v - 1.0);
                    b[2] = b[2].max(u + 1.0);
                    b[3] = b[3].max(v + 1.0);
                }
            }
        }
        PixelRect::from_bounds(b)
    }
}

/// 一张纹理与单次上传的保守字节数；上传完成后不继续占用 staging 额度。
/// bpp 为字节每像素：R8Unorm=1、Rgba8Unorm=4、Rgba16Float=8、Rgba32Float=16。
pub fn texture_bytes(dims: [u32; 3], bpp: u32) -> Result<(u64, u64), String> {
    if dims.contains(&0) || !matches!(bpp, 1 | 4 | 8 | 16) {
        return Err("纹理尺寸或像素格式非法".into());
    }
    let row = u64::from(dims[0])
        .checked_mul(u64::from(bpp))
        .ok_or("纹理行溢出")?;
    let rows = u64::from(dims[1])
        .checked_mul(u64::from(dims[2]))
        .ok_or("纹理层溢出")?;
    let resident = row.checked_mul(rows).ok_or("纹理长度溢出")?;
    let staging = row
        .div_ceil(256)
        .checked_mul(256)
        .and_then(|v| v.checked_mul(rows))
        .ok_or("上传长度溢出")?;
    Ok((resident, staging))
}

#[derive(Clone, Copy, Debug)]
pub struct DeviceBudget {
    pub bytes: u64,
    pub texture_side: u32,
    pub buffer_bytes: u64,
    /// 包含已保留的显示、LUT、flare、美肤、mask 及其他工作资源。
    pub resident_bytes: u64,
}
impl DeviceBudget {
    pub fn admits_upload(&self, dims: [u32; 3], bpp: u32) -> Result<bool, String> {
        let (texture, staging) = texture_bytes(dims, bpp)?;
        let peak = self
            .resident_bytes
            .checked_add(texture)
            .and_then(|v| v.checked_add(staging))
            .and_then(|v| v.checked_add(1024 * 1024))
            .ok_or("上传峰值溢出")?;
        Ok(dims[0] <= self.texture_side
            && dims[1] <= self.texture_side
            && staging <= self.buffer_bytes
            && peak <= self.bytes)
    }

    pub fn fits(&self, input: PixelRect, output: PixelRect, fp32: bool) -> Result<bool, String> {
        let node_width = output.width.checked_add(1).ok_or("场纹理宽度溢出")?;
        let node_height = output.height.checked_add(1).ok_or("场纹理高度溢出")?;
        if input.width == 0 || input.height == 0 || output.width == 0 || output.height == 0 {
            return Err("GPU 窗口不能为空".into());
        }
        if [input.width, input.height, node_width, node_height]
            .into_iter()
            .any(|s| s > self.texture_side)
        {
            return Ok(false);
        }
        let bytes_per_pixel = if fp32 { 16u64 } else { 8u64 };
        let pixels = u64::from(input.width)
            .checked_mul(u64::from(input.height))
            .ok_or("纹理尺寸溢出")?;
        let row = u64::from(output.width)
            .checked_mul(bytes_per_pixel)
            .ok_or("读回行宽溢出")?;
        let staging = row
            .div_ceil(256)
            .checked_mul(256)
            .and_then(|v| v.checked_mul(u64::from(output.height)))
            .ok_or("读回长度溢出")?;
        // 输入及上传、四级双通 blur、三张主图、节点及上传、双读回和 uniform 余量。
        let total = pixels
            .checked_mul(bytes_per_pixel * 5 + 8 * 8)
            .and_then(|v| {
                v.checked_add(
                    u64::from(node_width)
                        .checked_mul(u64::from(node_height))?
                        .checked_mul(32)?,
                )
            })
            .and_then(|v| v.checked_add(staging.checked_mul(2)?))
            .and_then(|v| v.checked_add(self.resident_bytes))
            .and_then(|v| v.checked_add(1024 * 1024))
            .ok_or("GPU 预算溢出")?;
        Ok(staging <= self.buffer_bytes
            && pixels * bytes_per_pixel <= self.buffer_bytes
            && total <= self.bytes)
    }
}

/// 仅资源不足允许拆分；足迹、安全或坐标错误直接传播。
pub fn plan_tiles(
    window: PixelRect,
    budget: DeviceBudget,
    fp32: bool,
    footprint: impl Fn(PixelRect) -> Result<PixelRect, String>,
) -> Result<Vec<(PixelRect, PixelRect)>, String> {
    plan_tiles_with_resources(window, budget, fp32, |tile| {
        Ok((footprint(tile)?, tile, 0, 0))
    })
}

/// 预算规划和实际 OOM 使用同一拆分规则；保留有符号绝对原点及原图余块。
pub fn split_core(tile: PixelRect) -> Result<[PixelRect; 2], String> {
    PixelRect::new(tile.x, tile.y, tile.width, tile.height)?;
    let horizontal = tile.width >= tile.height;
    let length = if horizontal { tile.width } else { tile.height };
    if length < 128 {
        return Err("最小 64 像素核心块仍超过 GPU 预算或设备限制".into());
    }
    let half = length / 2;
    let (mut first, mut second) = (tile, tile);
    if horizontal {
        first.width = half;
        second.x = second.x.checked_add(half as i32).ok_or("块原点溢出")?;
        second.width -= half;
    } else {
        first.height = half;
        second.y = second.y.checked_add(half as i32).ok_or("块原点溢出")?;
        second.height -= half;
    }
    Ok([first, second])
}

/// 蒙版 atlas 等局部辅助资源的大小随逆足迹变化，必须逐候选块重新准入。
pub fn plan_tiles_with_resources(
    window: PixelRect,
    budget: DeviceBudget,
    fp32: bool,
    footprint: impl Fn(PixelRect) -> Result<(PixelRect, PixelRect, u64, u64), String>,
) -> Result<Vec<(PixelRect, PixelRect)>, String> {
    PixelRect::new(window.x, window.y, window.width, window.height)?;
    let mut pending = Vec::new();
    for y in (0..window.height).step_by(1024) {
        for x in (0..window.width).step_by(1024) {
            pending.push(PixelRect::new(
                window.x.checked_add(x as i32).ok_or("块原点溢出")?,
                window.y.checked_add(y as i32).ok_or("块原点溢出")?,
                (window.width - x).min(1024),
                (window.height - y).min(1024),
            )?);
        }
    }
    pending.reverse();
    let mut result = Vec::new();
    while let Some(tile) = pending.pop() {
        // 调色 compose 以合成窗为输出，重塑以核心块为输出；只按核心块准入会低估节点/读回。
        let (input, mid, extra_bytes, extra_buffer_bytes) = footprint(tile)?;
        let candidate = DeviceBudget {
            resident_bytes: budget
                .resident_bytes
                .checked_add(extra_bytes)
                .ok_or("局部辅助资源预算溢出")?,
            ..budget
        };
        if extra_buffer_bytes <= budget.buffer_bytes
            && candidate.fits(input, tile, fp32)?
            && candidate.fits(input, mid, fp32)?
            && candidate.fits(mid, tile, fp32)?
        {
            result.push((tile, input));
            continue;
        }
        let [first, second] = split_core(tile)?;
        pending.push(second);
        pending.push(first);
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::lower_strength_retryable;

    #[test]
    fn classifies_gate_and_contract_field_errors() {
        for retryable in [
            "双线性场连续 Jacobian 界超限",
            "位移非有限、保护移动或越界",
            "FP32 位移量化误差超限",
            "共轭场 FP32 位移量化误差超限",
            "规范场重叠逆位移不一致",
        ] {
            assert!(lower_strength_retryable(retryable), "{retryable}");
        }
        for fatal in [
            "位移栅格窗口或长度不符",
            "场索引坐标非法",
            "退化的冻结目标三角形",
            "场页超出规范基底",
            "重塑基底必须为非空图像，单轴不超过 32768、总像素不超过 6400 万",
        ] {
            assert!(!lower_strength_retryable(fatal), "{fatal}");
        }
    }
}
