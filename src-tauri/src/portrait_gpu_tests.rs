//! 07：CPU 参考采样与可选实机数值对照。GPU 测试默认忽略，交由用户运行。
use super::*;
use crate::gpu_processing::GpuProcessor;
use std::sync::Arc;

pub(crate) fn context() -> GpuContext {
    let instance = wgpu::Instance::default();
    let adapter = pollster::block_on(instance.request_adapter(&Default::default()))
        .expect("测试需要本地 GPU");
    eprintln!("07 GPU：{:?}", adapter.get_info());
    let limits = adapter.limits();
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        required_limits: limits.clone(),
        ..Default::default()
    }))
    .unwrap();
    GpuContext {
        device: Arc::new(device),
        queue: Arc::new(queue),
        limits,
        display: Arc::new(std::sync::Mutex::new(None)),
    }
}
fn weight(t: f64) -> f64 {
    let t = t.abs();
    if t <= 1.0 {
        1.5 * t.powi(3) - 2.5 * t * t + 1.0
    } else if t < 2.0 {
        -0.5 * t.powi(3) + 2.5 * t * t - 4.0 * t + 2.0
    } else {
        0.0
    }
}
fn reference(base: &[[f32; 4]], field: &ValidatedField, cubic: bool) -> Vec<[f32; 4]> {
    let [w, h] = field.dimensions();
    let load = |x: i32, y: i32| {
        let p = base
            [(y.clamp(0, h as i32 - 1) as u32 * w + x.clamp(0, w as i32 - 1) as u32) as usize]
            .map(f64::from);
        if p[3] == 0.0 { [0.0; 4] } else { p }
    };
    let mut out = Vec::new();
    for y in 0..h {
        for x in 0..w {
            let i = (y * (w + 1) + x) as usize;
            let offset: [f64; 2] = std::array::from_fn(|c| {
                [i, i + 1, i + w as usize + 1, i + w as usize + 2]
                    .iter()
                    .map(|&j| field.samples()[j][c] as f64 * 0.25)
                    .sum()
            });
            let q = [x as f64 + offset[0], y as f64 + offset[1]];
            let p = q.map(|v| v.floor() as i32);
            let t = [q[0] - p[0] as f64, q[1] - p[1] as f64];
            let mut value = [0.0; 4];
            let mut low = [f64::INFINITY; 4];
            let mut high = [f64::NEG_INFINITY; 4];
            let mut color_low = [f64::INFINITY; 3];
            let mut color_high = [f64::NEG_INFINITY; 3];
            let range = if cubic { -1..=2 } else { 0..=1 };
            for dy in range.clone() {
                for dx in range.clone() {
                    let v = load(p[0] + dx, p[1] + dy);
                    let factor = if cubic {
                        weight(dx as f64 - t[0]) * weight(dy as f64 - t[1])
                    } else {
                        (if dx == 0 { 1.0 - t[0] } else { t[0] })
                            * (if dy == 0 { 1.0 - t[1] } else { t[1] })
                    };
                    for c in 0..4 {
                        value[c] += v[c] * factor;
                        if (0..=1).contains(&dx) && (0..=1).contains(&dy) {
                            low[c] = low[c].min(v[c]);
                            high[c] = high[c].max(v[c]);
                            if c < 3 && v[3] > 0.0 {
                                color_low[c] = color_low[c].min(v[c] / v[3]);
                                color_high[c] = color_high[c].max(v[c] / v[3]);
                            }
                        }
                    }
                }
            }
            if cubic {
                for c in 0..4 {
                    value[c] = value[c].clamp(low[c], high[c]);
                }
            }
            if value[3] <= 0.0 {
                value = [0.0; 4];
            } else if cubic {
                for c in 0..3 {
                    value[c] = value[c].clamp(color_low[c] * value[3], color_high[c] * value[3]);
                }
            }
            out.push(value.map(|v| v as f32));
        }
    }
    out
}
fn translation(dims: [u32; 2], half: bool) -> ValidatedField {
    ValidatedField::fixture(dims, half, |x, y| {
        let fade = x.min(dims[0] - x).min(y.min(dims[1] - y)).min(8) as f64 / 8.0;
        [0.3 * fade, -0.2 * fade]
    })
}
fn image(dims: [u32; 2], kind: u32) -> Vec<[f32; 4]> {
    let [w, h] = dims;
    (0..h)
        .flat_map(|y| {
            (0..w).map(move |x| {
                let alpha = if kind == 3 {
                    if x < w / 2 { 0.0 } else { 0.75 }
                } else if kind == 6 {
                    if x % 2 == 0 { 1e-8 } else { 0.5 }
                } else if kind == 5 {
                    (x % 7) as f32 / 8.0
                } else {
                    0.75
                };
                let rgb = match kind {
                    0 => [0.2, 0.4, 0.6],
                    1 => [x as f32 / w as f32, y as f32 / h as f32, 0.3],
                    2 => [if (x / 2 + y / 2) % 2 == 0 { 0.0 } else { 1.0 }; 3],
                    3 => [1.0, 0.2, 0.1],
                    6 => [4.0, 0.2, 0.1],
                    5 => [x as f32 / w as f32, if x % 2 == 0 { 1.0 } else { 0.0 }, 0.3],
                    _ => [-0.2, 1.5, 0.7],
                };
                [rgb[0] * alpha, rgb[1] * alpha, rgb[2] * alpha, alpha]
            })
        })
        .collect()
}
fn bytes(p: &[[f32; 4]], profile: Profile) -> Vec<u8> {
    if profile == Profile::Preview {
        p.iter()
            .flatten()
            .flat_map(|&v| f16::from_f32(v).to_bits().to_le_bytes())
            .collect()
    } else {
        p.iter().flatten().flat_map(|v| v.to_le_bytes()).collect()
    }
}
fn read_warp(node: &ReshapeGpu, context: &GpuContext) -> Vec<[f32; 4]> {
    let bpp = if node.profile == Profile::Preview {
        8
    } else {
        16
    };
    let bytes = crate::gpu_processing::read_texture_data_roi(
        &context.device,
        &context.queue,
        &node.warped.texture,
        wgpu::Origin3d::ZERO,
        extent(node.dimensions),
        bpp,
    )
    .unwrap();
    let floats: Vec<f32> = if bpp == 8 {
        bytes
            .chunks_exact(2)
            .map(|b| f16::from_bits(u16::from_le_bytes(b.try_into().unwrap())).to_f32())
            .collect()
    } else {
        bytes
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect()
    };
    floats
        .chunks_exact(4)
        .map(|p| p.try_into().unwrap())
        .collect()
}
#[test]
fn layout_budget_and_shader_contract() {
    assert_eq!(std::mem::size_of::<Params>(), 16);
    assert_eq!(std::mem::offset_of!(Params, cubic), 8);
    assert!(estimated_bytes([1024, 1024]).unwrap() <= RESOURCE_BUDGET);
    assert!(estimated_bytes([1025, 1]).is_err());
    let shader = include_str!("shaders/portrait_reshape.wgsl");
    assert!(!shader.contains("textureSample"));
    assert!(shader.contains("var<storage, read_write> failure"));
}
#[test]
fn cpu_constant_and_transparency_preserve_premultiplied_color() {
    let dims = [32, 24];
    let field = translation(dims, false);
    for cubic in [false, true] {
        let base = image(dims, 0);
        let out = reference(&base, &field, cubic);
        assert!(
            out.iter()
                .zip(&base)
                .all(|(a, b)| a.iter().zip(b).all(|(a, b)| (a - b).abs() < 1e-6))
        );
        let out = reference(&image(dims, 3), &field, cubic);
        for p in out {
            if p[3] > 1e-5 {
                assert!((p[0] / p[3] - 1.0).abs() < 1e-5);
            } else {
                assert_eq!(p, [0.0; 4]);
            }
        }
    }
}
#[test]
#[ignore = "需要本地 GPU；记录 FP16／FP32 的 mean、p99、max，不加载模型"]
fn gpu_cpu_sampling_and_cache_generation() {
    let context = context();
    let dims = [64, 48];
    let processor = GpuProcessor::new(context.clone(), dims[0], dims[1]).unwrap();
    for profile in [Profile::Preview, Profile::HighQuality] {
        let field = translation(dims, profile == Profile::Preview);
        let mut node = ReshapeGpu::new(
            &context,
            dims,
            profile,
            field.precision(),
            processor.reshape_layout_entries(),
        )
        .unwrap();
        for kind in 0..7 {
            let mut base = image(dims, kind);
            if profile == Profile::Preview {
                for v in base.iter_mut().flatten() {
                    *v = f16::from_f32(*v).to_f32();
                }
            }
            let flat: Vec<_> = base.iter().flatten().copied().collect();
            node.upload(&context, &flat, &field, &format!("case-{kind}"))
                .unwrap();
            write_texture(
                &context,
                &node.base.texture,
                &bytes(&base, profile),
                if profile == Profile::Preview { 8 } else { 16 },
            );
            node.warp(&context).unwrap();
            let first = read_warp(&node, &context);
            let expected = reference(&base, &field, profile == Profile::HighQuality);
            let mut errors: Vec<_> = first
                .iter()
                .flatten()
                .zip(expected.iter().flatten())
                .map(|(a, b)| (a - b).abs())
                .collect();
            assert!(errors.iter().all(|e| e.is_finite()));
            errors.sort_by(f32::total_cmp);
            eprintln!(
                "07 {profile:?} case={kind} 位移量化={} mean={} p99={} max={}",
                field.quantization_error(),
                errors.iter().sum::<f32>() / errors.len() as f32,
                errors[errors.len() * 99 / 100],
                errors.last().unwrap()
            );
            assert!(
                *errors.last().unwrap()
                    < if profile == Profile::Preview {
                        0.002
                    } else {
                        0.00002
                    }
            );
            node.warp(&context).unwrap();
            assert_eq!(
                first,
                read_warp(&node, &context),
                "重复渲染必须从同一基底采样"
            );
            assert_eq!(node.field_key.as_ref().unwrap().0, format!("case-{kind}"));
        }
        assert!(!node.compatible([32, 48], profile, field.precision()));
        assert!(
            node.upload(&context, &[f32::NAN; 4], &field, "bad")
                .is_err()
        );
    }
}
#[test]
#[ignore = "需要本地 GPU；篡改上传场后必须被 shader 拒绝"]
fn gpu_rejects_post_upload_unsafe_field() {
    let context = context();
    let dims = [16, 16];
    let processor = GpuProcessor::new(context.clone(), 16, 16).unwrap();
    let field = translation(dims, false);
    let mut node = ReshapeGpu::new(
        &context,
        dims,
        Profile::HighQuality,
        field.precision(),
        processor.reshape_layout_entries(),
    )
    .unwrap();
    let base = image(dims, 0);
    let flat: Vec<_> = base.iter().flatten().copied().collect();
    node.upload(&context, &flat, &field, "corrupt").unwrap();
    write_texture(
        &context,
        &node.base.texture,
        &bytes(&base, Profile::HighQuality),
        16,
    );
    let mut corrupt = field.samples().to_vec();
    corrupt[8 * 17 + 8][0] = 4.0;
    write_texture(
        &context,
        &node.field.texture,
        &bytes(&corrupt, Profile::HighQuality),
        16,
    );
    assert!(
        node.warp(&context)
            .unwrap_err()
            .contains("GPU 上传／插值后验证失败")
    );
}
