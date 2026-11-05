//! 重塑坐标使用像素边界坐标，像素中心为 (x + 0.5, y + 0.5)。
//! 仿射只描述 ROI／定向／裁剪；镜头与透视映射复用图像管线，不伪装成仿射。
use crate::portrait_models::FaceRoi;
use nalgebra::{Matrix3, Vector3};
use serde_json::Value;

pub const ROUNDTRIP_TOLERANCE: f64 = 0.05;

#[derive(Clone, Debug)]
pub struct Affine(pub Matrix3<f64>);

impl Affine {
    pub fn identity() -> Self {
        Self(Matrix3::identity())
    }

    pub fn new(a: [f64; 6]) -> Self {
        Self(Matrix3::new(
            a[0], a[1], a[2], a[3], a[4], a[5], 0.0, 0.0, 1.0,
        ))
    }

    pub fn map(&self, p: [f64; 2]) -> Option<[f64; 2]> {
        let q = self.0 * Vector3::new(p[0], p[1], 1.0);
        let result = [q.x, q.y];
        result.iter().all(|v| v.is_finite()).then_some(result)
    }

    pub fn inverse(&self) -> Option<Self> {
        self.0
            .try_inverse()
            .filter(|m| m.iter().all(|v| v.is_finite()))
            .map(Self)
    }

    /// 先执行 self，再执行 next。
    pub fn then(&self, next: &Self) -> Self {
        Self(next.0 * self.0)
    }

    pub fn roi(roi: FaceRoi) -> Self {
        let (s, c) = (roi.angle as f64).sin_cos();
        let side = roi.side as f64;
        Self::new([
            c * side,
            -s * side,
            roi.center[0] as f64 + (s - c) * side * 0.5,
            s * side,
            c * side,
            roi.center[1] as f64 - (s + c) * side * 0.5,
        ])
    }

    /// 未裁剪照片像素 → 裁剪／预览／tile 像素。tile_offset 是预览像素。
    #[allow(dead_code)] // 步骤 08 使用相同契约连接输出窗口。
    pub fn output(crop_origin: [f64; 2], scale: [f64; 2], tile_offset: [f64; 2]) -> Option<Self> {
        if scale.iter().any(|v| !v.is_finite() || *v <= 0.0)
            || crop_origin
                .iter()
                .chain(tile_offset.iter())
                .any(|v| !v.is_finite())
        {
            return None;
        }
        Some(Self::new([
            scale[0],
            0.0,
            -crop_origin[0] * scale[0] - tile_offset[0],
            0.0,
            scale[1],
            -crop_origin[1] * scale[1] - tile_offset[1],
        ]))
    }
}

#[derive(Clone, Debug)]
pub struct CoordinateMap {
    pub source_dimensions: [u32; 2],
    pub base_dimensions: [u32; 2],
    pub spatial: Affine,
    // 只保留几何字段，不保存像素补丁或调色参数的副本。
    geometry: Value,
    inverse_matrix: Matrix3<f32>,
}

impl CoordinateMap {
    pub fn new(source_dimensions: [u32; 2], adjustments: &Value) -> Result<Self, String> {
        if source_dimensions.contains(&0) {
            return Err("坐标基底尺寸不能为空".into());
        }
        let mut geometry = serde_json::Map::new();
        for key in crate::cache_utils::GEOMETRY_KEYS {
            if let Some(value) = adjustments.get(key) {
                geometry.insert((*key).into(), value.clone());
            }
        }
        let geometry = Value::Object(geometry);
        let params = crate::image_processing::get_geometry_params_from_json(&geometry);
        let (matrix, _, _, _) = crate::image_processing::build_transform_matrices(
            &params,
            source_dimensions[0] as f32,
            source_dimensions[1] as f32,
        );
        let inverse_matrix = matrix
            .try_inverse()
            .filter(|m| m.iter().all(|v| v.is_finite()))
            .ok_or("几何矩阵不可逆")?;
        if !matrix.iter().all(|v| v.is_finite()) {
            return Err("几何矩阵包含非有限值".into());
        }
        let steps = adjustments["orientationSteps"].as_u64().unwrap_or(0);
        if steps > 3 {
            return Err("不支持的照片定向步数".into());
        }
        let mut spatial = Affine::identity();
        let [mut w, mut h] = source_dimensions;
        for _ in 0..steps {
            spatial = spatial.then(&Affine::new([0.0, -1.0, h as f64, 1.0, 0.0, 0.0]));
            std::mem::swap(&mut w, &mut h);
        }
        if adjustments["flipHorizontal"] == true {
            spatial = spatial.then(&Affine::new([-1.0, 0.0, w as f64, 0.0, 1.0, 0.0]));
        }
        if adjustments["flipVertical"] == true {
            spatial = spatial.then(&Affine::new([1.0, 0.0, 0.0, 0.0, -1.0, h as f64]));
        }
        let angle = adjustments["rotation"].as_f64().unwrap_or(0.0).to_radians();
        if !angle.is_finite() {
            return Err("旋转角度非法".into());
        }
        let (s, c) = angle.sin_cos();
        // imageproc 在整数像素中心坐标中绕 (w/2,h/2) 旋转，转为边界坐标需加半像素。
        let (cx, cy) = (w as f64 / 2.0 + 0.5, h as f64 / 2.0 + 0.5);
        spatial = spatial.then(&Affine::new([
            c,
            -s,
            cx - c * cx + s * cy,
            s,
            c,
            cy - s * cx - c * cy,
        ]));
        Ok(Self {
            source_dimensions,
            base_dimensions: [w, h],
            spatial,
            geometry,
            inverse_matrix,
        })
    }

    fn inverse_geometry(&self, p: [f64; 2]) -> Option<[f64; 2]> {
        // 既有 CPU 几何以整数为像素中心；只在该接口边界转换，ROI 始终使用半像素中心。
        let center = [p[0] - 0.5, p[1] - 0.5];
        let projected = self.inverse_matrix * Vector3::new(center[0] as f32, center[1] as f32, 1.0);
        if !projected.iter().all(|v| v.is_finite()) || projected.z.abs() <= 1e-6 {
            return None;
        }
        let (x, y) = crate::image_processing::inverse_transform_point(
            center[0],
            center[1],
            self.source_dimensions[0] as f64,
            self.source_dimensions[1] as f64,
            &self.geometry,
        );
        [x, y]
            .iter()
            .all(|v| v.is_finite())
            .then_some([x + 0.5, y + 0.5])
    }

    fn jacobian(&self, p: [f64; 2]) -> Option<[f64; 4]> {
        let left = self.inverse_geometry([p[0] - 0.5, p[1]])?;
        let right = self.inverse_geometry([p[0] + 0.5, p[1]])?;
        let top = self.inverse_geometry([p[0], p[1] - 0.5])?;
        let bottom = self.inverse_geometry([p[0], p[1] + 0.5])?;
        let j = [
            right[0] - left[0],
            bottom[0] - top[0],
            right[1] - left[1],
            bottom[1] - top[1],
        ];
        let det = j[0] * j[3] - j[1] * j[2];
        // 镜像在 spatial 中；几何场的负 Jacobian 表示折叠，不能作为可处理点。
        (det.is_finite() && det > 1e-6).then_some(j)
    }

    /// 通过既有逆采样函数求正向映射；迭代不收敛／局部折叠时拒绝，不回退 identity。
    pub fn source_to_base(&self, source: [f64; 2]) -> Option<[f64; 2]> {
        let mut p = source;
        for _ in 0..32 {
            let q = self.inverse_geometry(p)?;
            let error = [q[0] - source[0], q[1] - source[1]];
            let j = self.jacobian(p)?;
            if error[0].hypot(error[1]) <= ROUNDTRIP_TOLERANCE * 0.25 {
                return self.spatial.map(p);
            }
            let det = j[0] * j[3] - j[1] * j[2];
            let dx = (j[3] * error[0] - j[1] * error[1]) / det;
            let dy = (-j[2] * error[0] + j[0] * error[1]) / det;
            let limit = self.source_dimensions[0].max(self.source_dimensions[1]) as f64 * 0.25;
            let damping = (limit / dx.hypot(dy).max(limit)).min(1.0);
            p = [p[0] - dx * damping, p[1] - dy * damping];
        }
        None
    }

    /// 未裁剪分析基底像素 → 原始定向照片归一化坐标，并验证完整往返。
    pub fn base_to_source(&self, point: [f64; 2]) -> Option<[f64; 2]> {
        let p = self.spatial.inverse()?.map(point)?;
        self.jacobian(p)?;
        let source = self.inverse_geometry(p)?;
        let restored = self.source_to_base(source)?;
        if (restored[0] - point[0]).hypot(restored[1] - point[1]) > ROUNDTRIP_TOLERANCE {
            return None;
        }
        let uv = [
            source[0] / self.source_dimensions[0] as f64,
            source[1] / self.source_dimensions[1] as f64,
        ];
        uv.iter().all(|v| (0.0..=1.0).contains(v)).then_some(uv)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn roi_crop_mirror_rotation_tile_roundtrip() {
        let roi = FaceRoi {
            center: [740.0, 460.0],
            side: 350.0,
            angle: 0.28,
        };
        let roi_map = Affine::roi(roi);
        for steps in 0..4 {
            for flip in [false, true] {
                let map = CoordinateMap::new(
                    [1600, 1000],
                    &json!({"orientationSteps":steps,
                    "flipHorizontal":flip,"flipVertical":!flip,"rotation":13.0}),
                )
                .unwrap();
                let output = Affine::output([32.0, 20.0], [0.4, 0.35], [70.0, 50.0]).unwrap();
                let p = roi_map.map([0.35, 0.6]).unwrap();
                let base = map.source_to_base(p).unwrap();
                let tile = output.map(base).unwrap();
                let uv = map
                    .base_to_source(output.inverse().unwrap().map(tile).unwrap())
                    .unwrap();
                assert!((uv[0] * 1600.0 - p[0]).hypot(uv[1] * 1000.0 - p[1]) < ROUNDTRIP_TOLERANCE);
                let projected = roi.project(0.35, 0.6);
                assert!((projected[0] as f64 - p[0]).abs() < 0.001);
            }
        }
    }

    #[test]
    fn pixel_centers_agree_with_cpu_geometry_and_spatial_sampling() {
        // 坐标坡度图让输出像素直接给出实际采样源坐标，避免只验证自洽的矩阵往返。
        let source =
            image::DynamicImage::ImageRgb32F(image::Rgb32FImage::from_fn(80, 60, |x, y| {
                image::Rgb([(x as f32 + 0.5) / 80.0, (y as f32 + 0.5) / 60.0, 0.5])
            }));
        for settings in [
            json!({"transformScale":120.0,"transformRotate":7.0,"transformXOffset":3.0}),
            json!({"orientationSteps":1,"flipHorizontal":true,"rotation":17.0}),
            json!({"orientationSteps":3,"flipVertical":true,"rotation":-11.0}),
        ] {
            let map = CoordinateMap::new([80, 60], &settings).unwrap();
            let warped =
                crate::adjustment_utils::prepare_patched_warped(&source, &settings).unwrap();
            let (transformed, _) =
                crate::adjustment_utils::apply_spatial_transformations(warped, &settings);
            let pixels = transformed.to_rgb32f();
            for [x, y] in [[24, 25], [32, 31], [38, 39]] {
                let uv = map
                    .base_to_source([x as f64 + 0.5, y as f64 + 0.5])
                    .unwrap();
                let actual = pixels.get_pixel(x, y);
                assert!((actual[0] as f64 - uv[0]).abs() * 80.0 < ROUNDTRIP_TOLERANCE);
                assert!((actual[1] as f64 - uv[1]).abs() * 60.0 < ROUNDTRIP_TOLERANCE);
            }
        }
        let mut horizon = CoordinateMap::new([80, 60], &json!({})).unwrap();
        horizon.inverse_matrix[(2, 2)] = 0.0;
        assert!(horizon.base_to_source([30.5, 20.5]).is_none());
    }

    #[test]
    fn nonlinear_lens_roundtrip_and_singular_rejection() {
        let map = CoordinateMap::new(
            [2000, 1200],
            &json!({"transformDistortion":4.0,
            "lensDistortionParams":{"k1":0.015,"k2":0.003,"k3":0.0},"lensDistortionAmount":30.0,
            "transformVertical":2.0,"transformScale":104.0}),
        )
        .unwrap();
        for p in [[400.0, 300.0], [1000.0, 600.0], [1400.0, 850.0]] {
            let base = map.source_to_base(p).unwrap();
            let uv = map.base_to_source(base).unwrap();
            assert!((uv[0] * 2000.0 - p[0]).hypot(uv[1] * 1200.0 - p[1]) < ROUNDTRIP_TOLERANCE);
        }
        assert!(CoordinateMap::new([100, 100], &json!({"transformScale":0})).is_err());
        assert!(map.base_to_source([f64::NAN, 0.0]).is_none());
        assert!(Affine::output([0.0, 0.0], [0.0, 1.0], [0.0, 0.0]).is_none());
    }
}
