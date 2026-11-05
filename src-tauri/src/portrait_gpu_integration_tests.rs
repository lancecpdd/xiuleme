//! 07 完整调色／基底／重塑／末尾链路：全部需用户显式运行 GPU 测试。
use super::*;
use crate::portrait_gpu::{Profile, ReshapeGpu, TextureSlot, test_context, write_texture};
use crate::portrait_gpu_field::ValidatedField;
use crate::portrait_tiles::CanonicalField;

fn float_texture(
    context: &GpuContext,
    image: &image::Rgba32FImage,
    profile: Profile,
) -> TextureSlot {
    let slot = TextureSlot::new(
        &context.device,
        [image.width(), image.height()],
        profile.format(),
        false,
    );
    let bytes: Vec<u8> = if profile == Profile::Preview {
        image
            .as_raw()
            .iter()
            .flat_map(|&v| f16::from_f32(v).to_bits().to_le_bytes())
            .collect()
    } else {
        image
            .as_raw()
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect()
    };
    write_texture(
        context,
        &slot.texture,
        &bytes,
        if profile == Profile::Preview { 8 } else { 16 },
    );
    slot
}
fn float_pixels(context: &GpuContext, slot: &TextureSlot, profile: Profile) -> image::Rgba32FImage {
    let bpp = if profile == Profile::Preview { 8 } else { 16 };
    let bytes = read_texture_data_roi(
        &context.device,
        &context.queue,
        &slot.texture,
        wgpu::Origin3d::ZERO,
        slot.texture.size(),
        bpp,
    )
    .unwrap();
    let pixels: Vec<f32> = if bpp == 8 {
        bytes
            .chunks_exact(2)
            .map(|v| f16::from_bits(u16::from_le_bytes(v.try_into().unwrap())).to_f32())
            .collect()
    } else {
        bytes
            .chunks_exact(4)
            .map(|v| f32::from_le_bytes(v.try_into().unwrap()))
            .collect()
    };
    assert!(pixels.iter().all(|v| v.is_finite()));
    ImageBuffer::from_raw(slot.texture.width(), slot.texture.height(), pixels).unwrap()
}
fn local_image(image: &image::Rgba32FImage, rect: PixelRect) -> image::Rgba32FImage {
    image::imageops::crop_imm(image, rect.x as u32, rect.y as u32, rect.width, rect.height)
        .to_image()
}
fn compare_float(a: &image::Rgba32FImage, b: &image::Rgba32FImage, profile: Profile) {
    assert_eq!(a.dimensions(), b.dimensions());
    let error = a
        .as_raw()
        .iter()
        .zip(b.as_raw())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f32::max);
    eprintln!("08 {profile:?} 单块／多块最大误差={error}");
    assert!(
        error
            <= if profile == Profile::Preview {
                2e-3
            } else {
                1e-5
            }
    );
    for pixel in b.pixels() {
        if pixel[3] == 0.0 {
            assert_eq!(pixel.0, [0.0; 4]);
        }
    }
}

#[test]
fn portrait_gpu_window_layout_and_shader_contract() {
    use crate::portrait_gpu::TileParams;
    assert_eq!(std::mem::size_of::<TileParams>(), 64);
    assert_eq!(std::mem::offset_of!(TileParams, source_min), 40);
    assert_eq!(std::mem::offset_of!(TileParams, cubic), 56);
    assert_eq!(std::mem::size_of::<CompositionWindow>(), 64);
    assert_eq!(std::mem::offset_of!(CompositionWindow, mask_origin), 40);
    assert_eq!(std::mem::offset_of!(CompositionWindow, radius), 48);
    let photo_shader = include_str!("shaders/shader.wgsl");
    for source in [
        photo_shader,
        include_str!("shaders/blur.wgsl"),
    ] {
        assert!(source.contains("struct CompositionWindow"));
        assert!(source.contains("source_min: vec2<i32>"));
        assert!(source.contains("reserved: vec2<u32>"));
    }
    let sharpen = photo_shader
        .split_once("fn apply_sharpen(")
        .and_then(|(_, tail)| tail.split_once("fn apply_centre_tonal_and_color("))
        .map(|(body, _)| body)
        .expect("锐化 shader 函数边界应保持可识别");
    assert!(sharpen.contains("photo_load("));
    assert!(
        !sharpen.contains("textureLoad(input_texture"),
        "锐化邻域必须通过 photo_load 将全图坐标换算为分块纹理坐标"
    );
    let full = PixelRect::full([17, 19]).unwrap();
    assert!(
        CompositionWindow::new([17, 19], full, PixelRect::new(-1, 0, 17, 19).unwrap(), full)
            .is_err()
    );
}

#[test]
#[ignore = "需要本地 GPU；08 FP32 规范场跨页、分块与缺失足迹，禁止自动运行"]
fn portrait_gpu_native_window_sampling_is_partition_invariant() {
    let context = test_context();
    let dims = [513, 387];
    let full = PixelRect::full(dims).unwrap();
    let scene = crate::portrait_safety::gpu_fixture_dimensions(10.0, false, dims);
    let field = CanonicalField::from_scene(&scene, &|| true).unwrap();
    let original_key = field.key().to_owned();
    let source = ImageBuffer::from_fn(dims[0], dims[1], |x, y| {
        let alpha = if x < 8 || y < 8 {
            0.0
        } else if (x + y) % 7 == 0 {
            0.25
        } else {
            1.0
        };
        Rgba([
            alpha * x as f32 / dims[0] as f32,
            alpha * y as f32 / dims[1] as f32,
            alpha * if (x / 3 + y / 3) % 2 == 0 { 0.9 } else { 0.1 },
            alpha,
        ])
    });
    for profile in [Profile::Preview, Profile::HighQuality] {
        let node = pg::WindowedReshapeGpu::new(&context, profile).unwrap();
        let all = float_texture(&context, &source, profile);
        let baseline = node
            .render_tile(&context, &all, full, full, full, &field, 0, &|| true)
            .unwrap();
        let baseline = float_pixels(&context, &baseline, profile);
        drop(all);
        for size in [64u32, 127, 256] {
            let mut tiled = image::Rgba32FImage::new(dims[0], dims[1]);
            for y in (0..dims[1]).step_by(size as usize).rev() {
                for x in (0..dims[0]).step_by(size as usize) {
                    let tile = PixelRect::new(
                        x as i32,
                        y as i32,
                        size.min(dims[0] - x),
                        size.min(dims[1] - y),
                    )
                    .unwrap();
                    let input = field
                        .inverse_footprint(
                            tile,
                            if profile == Profile::Preview { 1 } else { 2 },
                            &|| true,
                        )
                        .unwrap()
                        .intersect(full)
                        .unwrap();
                    let pixels = float_texture(&context, &local_image(&source, input), profile);
                    let result = node
                        .render_tile(&context, &pixels, input, tile, full, &field, 0, &|| true)
                        .unwrap();
                    image::imageops::replace(
                        &mut tiled,
                        &float_pixels(&context, &result, profile),
                        x as i64,
                        y as i64,
                    );
                }
            }
            compare_float(&baseline, &tiled, profile);
        }
        let tile = PixelRect::new(200, 140, 64, 64).unwrap();
        let missing = float_texture(&context, &local_image(&source, tile), profile);
        assert!(
            node.render_tile(&context, &missing, tile, tile, full, &field, 0, &|| true)
                .is_err()
        );
        assert!(
            node.render_tile(&context, &missing, tile, tile, full, &field, 0, &|| false)
                .is_err()
        );
        assert_eq!(original_key, field.key());
    }
}

#[test]
#[ignore = "需要本地 GPU；08 原生调色、动态 blur、CA、降噪、蒙版与尾部的分块一致性"]
fn portrait_gpu_native_color_and_tail_are_partition_invariant() {
    let context = test_context();
    let dims = [257, 193];
    let full = PixelRect::full(dims).unwrap();
    let source = ImageBuffer::from_fn(dims[0], dims[1], |x, y| {
        Rgba([
            0.1 + x as f32 / 400.0,
            0.1 + y as f32 / 300.0,
            if (x / 3 + y / 3) % 2 == 0 { 0.1 } else { 0.9 },
            if x < 3 { 0.0 } else { 0.75 },
        ])
    });
    let mut a = crate::image_processing::get_all_adjustments_from_json(
        &serde_json::json!({}),
        false,
        crate::white_balance::WhiteBalance::reference(),
        None,
    );
    a.global.chromatic_aberration_red_cyan = 0.03;
    a.global.chromatic_aberration_blue_yellow = -0.02;
    a.global.luma_noise_reduction = 1.0;
    a.global.color_noise_reduction = 1.0;
    a.global.sharpness = 1.0;
    a.global.clarity = 0.3;
    a.global.structure = 0.2;
    a.global.centré = 0.2;
    a.global.vignette_amount = -0.2;
    a.global.grain_amount = 0.2;
    a.global.grain_size = 0.7;
    a.global.grain_roughness = 0.3;
    a.mask_count = 1;
    a.mask_adjustments[0].exposure = 0.25;
    let deps = CompositionDependencies::native(dims, true, true, [0.03, -0.02]).unwrap();
    let dummy = TextureSlot::new(
        &context.device,
        [1, 1],
        wgpu::TextureFormat::Rgba16Float,
        false,
    );
    write_texture(&context, &dummy.texture, &[0u8; 8], 8);
    let portrait_texture = context.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("08 人像 atlas 占位"),
        size: wgpu::Extent3d {
            width: 1,
            height: 1,
            depth_or_array_layers: PORTRAIT_ATLAS_LAYERS,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba16Float,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    write_texture(
        &context,
        &portrait_texture,
        &[0u8; 8 * PORTRAIT_ATLAS_LAYERS as usize],
        8,
    );
    let portrait = TextureSlot {
        view: portrait_texture.create_view(&wgpu::TextureViewDescriptor {
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            ..Default::default()
        }),
        texture: portrait_texture,
    };
    let lut = context.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("08 合成 LUT 占位"),
        size: wgpu::Extent3d {
            width: 2,
            height: 2,
            depth_or_array_layers: 2,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D3,
        format: wgpu::TextureFormat::Rgba16Float,
        usage: wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let nearest = context.device.create_sampler(&Default::default());
    let linear = context.device.create_sampler(&wgpu::SamplerDescriptor {
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        ..Default::default()
    });
    for profile in [Profile::Preview, Profile::HighQuality] {
        let node = WindowedColorGpu::new(&context, profile).unwrap();
        let render = |tile: PixelRect| {
            let input = deps.input(tile, dims).unwrap().intersect(full).unwrap();
            let pixels = float_texture(&context, &local_image(&source, input), profile);
            let mask: Vec<u8> = (tile.y..tile.bottom() as i32)
                .flat_map(|y| (tile.x..tile.right() as i32).map(move |x| ((x + y) % 256) as u8))
                .collect();
            let masks = context.device.create_texture_with_data(
                &context.queue,
                &wgpu::TextureDescriptor {
                    label: Some("08 全局坐标蒙版切片"),
                    size: pg::extent([tile.width, tile.height]),
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: wgpu::TextureFormat::R8Unorm,
                    usage: wgpu::TextureUsages::TEXTURE_BINDING,
                    view_formats: &[],
                },
                TextureDataOrder::MipMajor,
                &mask,
            );
            let resources = ColorTileResources {
                masks: &masks,
                lut: &lut,
                lut_sampler: &nearest,
                flare: &dummy,
                flare_sampler: &linear,
                portrait: &portrait,
                gf_coeffs: &dummy,
                gf_dehaze: &dummy,
                other_resident_bytes: 0,
            };
            let base = node
                .compose(
                    &context,
                    &pixels,
                    input,
                    tile,
                    full,
                    dims,
                    &a,
                    &resources,
                    &|| true,
                )
                .unwrap();
            let finished = node
                .finish(
                    &context,
                    &base,
                    tile,
                    dims,
                    &a,
                    resources.bytes().unwrap(),
                    &|| true,
                )
                .unwrap();
            float_pixels(&context, &finished, profile)
        };
        let baseline = render(full);
        for size in [64u32, 127] {
            let mut tiled = image::Rgba32FImage::new(dims[0], dims[1]);
            for y in (0..dims[1]).step_by(size as usize) {
                for x in (0..dims[0]).step_by(size as usize) {
                    let tile = PixelRect::new(
                        x as i32,
                        y as i32,
                        size.min(dims[0] - x),
                        size.min(dims[1] - y),
                    )
                    .unwrap();
                    image::imageops::replace(&mut tiled, &render(tile), x as i64, y as i64);
                }
            }
            compare_float(&baseline, &tiled, profile);
        }
    }
}

#[test]
#[ignore = "需要本地 GPU；显式验证人像交互缓存，使用 --test-threads=1"]
fn portrait_preview_gpu_resources_follow_pixel_and_lut_dependencies() {
    let context = test_context();
    let base = ImageBuffer::from_pixel(64, 48, Rgba([0.2, 0.3, 0.4, 1.0]));
    let mut adjustments = crate::image_processing::get_all_adjustments_from_json(
        &serde_json::json!({}),
        false,
        crate::white_balance::WhiteBalance::reference(),
        None,
    );
    let cold = NativeColorAssets::prepare_cached(
        &context,
        &base,
        "pixels-a",
        &mut adjustments,
        None,
        None,
        0,
        true,
        &|| true,
    )
    .unwrap();
    let warm = NativeColorAssets::prepare_cached(
        &context,
        &base,
        "pixels-a",
        &mut adjustments,
        None,
        None,
        0,
        true,
        &|| true,
    )
    .unwrap();
    assert_eq!(cold.key, warm.key);
    assert_eq!(
        cold.raw.as_ref().unwrap().texture,
        warm.raw.as_ref().unwrap().texture
    );
    assert_eq!(cold.gf_coeffs.texture, warm.gf_coeffs.texture);
    assert_eq!(cold.lut.texture, warm.lut.texture);
    adjustments.global.exposure = 0.5;
    let tone = NativeColorAssets::prepare_cached(
        &context,
        &base,
        "pixels-a",
        &mut adjustments,
        None,
        None,
        0,
        true,
        &|| true,
    )
    .unwrap();
    assert_eq!(
        warm.raw.as_ref().unwrap().texture,
        tone.raw.as_ref().unwrap().texture
    );
    assert_eq!(warm.gf_coeffs.texture, tone.gf_coeffs.texture);
    assert_eq!(warm.lut.texture, tone.lut.texture);
    assert_ne!(warm.key, tone.key);
    let lut = Lut {
        size: 2,
        data: vec![0.5; 24],
    };
    adjustments.global.has_lut = 1;
    let colored = NativeColorAssets::prepare_cached(
        &context,
        &base,
        "pixels-a",
        &mut adjustments,
        Some(&lut),
        None,
        0,
        true,
        &|| true,
    )
    .unwrap();
    assert_eq!(tone.gf_coeffs.texture, colored.gf_coeffs.texture);
    assert_ne!(tone.lut.texture, colored.lut.texture);
    let changed = NativeColorAssets::prepare_cached(
        &context,
        &base,
        "pixels-b",
        &mut adjustments,
        Some(&lut),
        None,
        0,
        true,
        &|| true,
    )
    .unwrap();
    assert_ne!(
        colored.raw.as_ref().unwrap().texture,
        changed.raw.as_ref().unwrap().texture
    );
    assert_ne!(colored.gf_coeffs.texture, changed.gf_coeffs.texture);
    let uncached = NativeColorAssets::prepare_cached(
        &context,
        &base,
        "pixels-b",
        &mut adjustments,
        Some(&lut),
        None,
        0,
        false,
        &|| true,
    )
    .unwrap();
    assert!(uncached.raw.is_none());
    assert_ne!(changed.gf_coeffs.texture, uncached.gf_coeffs.texture);
    assert!(
        NativeColorAssets::prepare_cached(
            &context,
            &base,
            "pixels-b",
            &mut adjustments,
            Some(&lut),
            None,
            0,
            true,
            &|| false,
        )
        .is_err()
    );
}

#[test]
fn portrait_gpu_flare_preparation_quantizes_before_interpolation() {
    let source = ImageBuffer::from_fn(3, 5, |x, y| {
        Rgba([
            0.123456 + x as f32 * 0.111111,
            0.234567 + y as f32 * 0.111111,
            0.345678,
            1.0,
        ])
    });
    assert_eq!(
        flare_sample(&source, 0, 0)[0],
        f16::from_f32(source.get_pixel(0, 0)[0]).to_f32()
    );
    for (x, y) in [(17, 90), (255, 257), (511, 511)] {
        let got = flare_sample(&source, x, y);
        let u = (x as f32 + 0.5) / 512.0 * 3.0 - 0.5;
        let v = (y as f32 + 0.5) / 512.0 * 5.0 - 0.5;
        let mut expected = [0.0f64; 4];
        for (dx, wx) in [(0, 1.0 - (u - u.floor())), (1, u - u.floor())] {
            for (dy, wy) in [(0, 1.0 - (v - v.floor())), (1, v - v.floor())] {
                let p = source.get_pixel(
                    (u.floor() as i32 + dx).clamp(0, 2) as u32,
                    (v.floor() as i32 + dy).clamp(0, 4) as u32,
                );
                for c in 0..4 {
                    expected[c] += f16::from_f32(p[c]).to_f32() as f64 * wx as f64 * wy as f64;
                }
            }
        }
        for c in 0..4 {
            assert!((got[c] as f64 - expected[c]).abs() < 2e-7);
        }
    }
}

#[test]
#[ignore = "需要本地 GPU；08 全局 LUT／flare／美肤准备、原生执行器与预算拆块，禁止自动运行"]
fn portrait_gpu_native_photo_auxiliary_assets_and_budget_partition_match() {
    use crate::portrait_processing::{PortraitAnalysis, PortraitRenderMaps, PortraitSummary};
    let context = test_context();
    let dims = [513, 387];
    let scene = crate::portrait_safety::gpu_fixture_dimensions(10.0, false, dims);
    let field = CanonicalField::from_scene(&scene, &|| true).unwrap();
    let base = ImageBuffer::from_fn(dims[0], dims[1], |x, y| {
        Rgba([
            0.2 + x as f32 / 640.0,
            0.2 + y as f32 / 480.0,
            0.8,
            if x < 4 { 0.0 } else { 0.75 },
        ])
    });
    let fields = |color| Arc::<[[f32; 4]]>::from(vec![color; 64]);
    let maps = PortraitRenderMaps {
        uv_rect: [0.0, 0.0, 1.0, 1.0],
        analysis: Arc::new(PortraitAnalysis {
            summary: PortraitSummary {
                key: "test-fields".into(),
                face_count: 1,
                skipped_faces: 0,
                has_skin: true,
            },
            width: 8,
            height: 8,
            full_dimensions: (dims[0], dims[1]),
            pixels: fields([0.05, 0.03, 0.9, 0.5]),
            frequency_pixels: fields([0.01, 0.005, 0.003, 0.7]),
            frequency_detail_pixels: fields([0.004, -0.002, 0.001, 0.7]),
            hydrated_pixels: fields([0.008, 0.006, 0.004, 0.7]),
            even_pixels: fields([0.0, 0.0, 0.0, 0.0]),
            whitening_pixels: fields([0.0, 0.0, 0.0, 0.0]),
            evenness_chroma_pixels: fields([0.0, 0.0, 0.0, 0.0]),
            blemish_pixels: fields([0.003, 0.002, 0.001, 0.6]),
            texture_pixels: fields([0.002, 0.003, 0.002, 0.8]),
            shine_pixels: fields([-0.012, -0.008, -0.006, 0.7]),
            eye_white_pixels: fields([0.0, 0.0, 0.0, 0.0]),
        }),
    };
    let lut = Lut {
        size: 2,
        data: vec![
            0.0, 0.0, 0.0, 0.9, 0.0, 0.0, 0.0, 0.9, 0.0, 0.9, 0.9, 0.0, 0.0, 0.0, 0.9, 0.9, 0.0,
            0.9, 0.0, 0.9, 0.9, 0.9, 0.9, 0.9,
        ],
    };
    let masks: Vec<crate::mask_generation::MaskDefinition> = serde_json::from_value(serde_json::json!([
        {"id":"empty","name":"empty","visible":true,"invert":false,"adjustments":{},"subMasks":[]},
        {"id":"all","name":"all","visible":true,"invert":false,"opacity":70,
         "adjustments":{},"subMasks":[{"id":"sub","type":"all","visible":true,"mode":"additive","parameters":{}}]}
    ])).unwrap();
    let cache = crate::portrait_render::NativeMaskCache::default();
    let mut a = crate::image_processing::get_all_adjustments_from_json(
        &serde_json::json!({}),
        false,
        crate::white_balance::WhiteBalance::reference(),
        None,
    );
    a.global.has_lut = 1;
    a.global.lut_intensity = 0.8;
    a.global.flare_amount = 0.5;
    a.global.clarity = 0.2;
    a.global.grain_amount = 0.1;
    a.portrait.strengths = [0.2; 4];
    a.portrait.blemishes = [0.3, 0.2, 0.2, 0.0];
    a.mask_count = 2;
    a.mask_adjustments[0].exposure = 2.0;
    a.mask_adjustments[1].exposure = 0.1;
    let render_adjustments = a;
    let make = |resident| NativePhotoInput {
        base: &base,
        base_key: "fixture",
        field: &field,
        adjustments: render_adjustments,
        masks: &masks,
        mask_cache: &cache,
        mask_source: None,
        cache_preview_resources: false,
        warped: None,
        lut: Some(&lut),
        portrait: Some(&maps),
        other_resident_bytes: resident,
    };
    for profile in [Profile::Preview, Profile::HighQuality] {
        let single = render_native_photo(&context, make(0), profile, &|| true).unwrap();
        // 模拟已有资源占用，只改变准入拆块，不改变图像、辅助资产或冻结场。
        let tiled =
            render_native_photo(&context, make(240 * 1024 * 1024), profile, &|| true).unwrap();
        compare_float(&single, &tiled, profile);
    }
    let original = NativeColorAssets::prepare(
        &context,
        &base,
        "fixture",
        &mut a,
        Some(&lut),
        Some(&maps),
        0,
        &|| true,
    )
    .unwrap();
    // 同一 analysis.key 二次准备应命中美肤五通道缓存，键保持稳定（仅 tone 拖动场景）。
    let cached = NativeColorAssets::prepare(
        &context,
        &base,
        "fixture",
        &mut a,
        Some(&lut),
        Some(&maps),
        0,
        &|| true,
    )
    .unwrap();
    assert_eq!(original.key, cached.key);
    let mut changed = Lut {
        size: lut.size,
        data: lut.data.clone(),
    };
    changed.data[0] = 0.1;
    let changed = NativeColorAssets::prepare(
        &context,
        &base,
        "fixture",
        &mut a,
        Some(&changed),
        Some(&maps),
        original.resident_bytes,
        &|| true,
    )
    .unwrap();
    assert_ne!(original.key, changed.key);
    assert!(
        NativeColorAssets::prepare(
            &context,
            &base,
            "fixture",
            &mut a,
            None,
            Some(&maps),
            0,
            &|| true
        )
        .is_err()
    );
    assert!(
        NativeColorAssets::prepare(
            &context,
            &base,
            "fixture",
            &mut a,
            Some(&lut),
            None,
            0,
            &|| true
        )
        .is_err()
    );
    assert!(render_native_photo(&context, make(0), Profile::HighQuality, &|| false).is_err());
}

fn request(adjustments: AllAdjustments) -> RenderRequest<'static> {
    RenderRequest {
        adjustments,
        mask_bitmaps: &[],
        lut: None,
        roi: None,
        portrait_maps: None,
    }
}
fn original(
    processor: &GpuProcessor,
    input: &DynamicImage,
    adjustments: AllAdjustments,
    profile: Profile,
) -> DynamicImage {
    let texture = TextureSlot::new(
        &processor.context.device,
        [input.width(), input.height()],
        wgpu::TextureFormat::Rgba16Float,
        false,
    );
    write_texture(
        &processor.context,
        &texture.texture,
        bytemuck::cast_slice(&to_rgba_f16(input)),
        8,
    );
    let dummy = TextureSlot::new(
        &processor.context.device,
        [1, 1],
        wgpu::TextureFormat::Rgba16Float,
        false,
    );
    let (pixels, w, h, ..) = processor
        .run(
            &texture.view,
            &dummy.view,
            &dummy.view,
            input.width(),
            input.height(),
            request(adjustments),
            false,
            if profile == Profile::Preview {
                RenderOutputPrecision::EightBit
            } else {
                RenderOutputPrecision::SixteenBit
            },
        )
        .unwrap();
    match pixels {
        RenderedPixels::U8(v) => DynamicImage::ImageRgba8(ImageBuffer::from_raw(w, h, v).unwrap()),
        RenderedPixels::U16(v) => {
            DynamicImage::ImageRgba16(ImageBuffer::from_raw(w, h, v).unwrap())
        }
    }
}
#[test]
#[ignore = "需要本地 GPU；关闭、零值、无脸、重复调参和缓存失效"]
fn bypass_repeated_adjustment_and_no_face_use_original_path() {
    let context = test_context();
    let mut processor = GpuProcessor::new(context.clone(), 128, 128).unwrap();
    let input = DynamicImage::ImageRgba32F(ImageBuffer::from_fn(128, 128, |x, y| {
        Rgba([
            x as f32 / 128.0,
            y as f32 / 128.0,
            if (x / 3 + y / 3) % 2 == 0 { 0.1 } else { 0.9 },
            0.75,
        ])
    }));
    let adjustments = crate::image_processing::get_all_adjustments_from_json(
        &serde_json::json!({}),
        false,
        crate::white_balance::WhiteBalance::reference(),
        None,
    );
    let zero = crate::portrait_safety::gpu_fixture(0.0, false);
    let moving = crate::portrait_safety::gpu_fixture(10.0, false);
    let empty = crate::portrait_safety::gpu_fixture(0.0, true);
    for profile in [Profile::Preview, Profile::HighQuality] {
        let baseline = original(&processor, &input, adjustments, profile);
        for scene in [None, Some(&zero), Some(&empty)] {
            let out = processor
                .render_reshape_fixed(
                    &input,
                    request(adjustments),
                    FixedReshapeRequest {
                        scene,
                        expected_analysis_key: zero.analysis_key(),
                        generation: "first",
                        profile,
                    },
                    &|| true,
                )
                .unwrap();
            assert_eq!(out.image.as_bytes(), baseline.as_bytes());
            assert!(out.field_precision.is_none());
            assert_eq!(out.resource_budget, 0);
            // 旁路无场无量化：量化误差必须为零，与 07 的安全界一致。
            assert_eq!(out.quantization_error, 0.0);
            assert!(processor.reshape_gpu.is_none());
        }
        let render = |processor: &mut GpuProcessor| {
            processor
                .render_reshape_fixed(
                    &input,
                    request(adjustments),
                    FixedReshapeRequest {
                        scene: Some(&moving),
                        expected_analysis_key: moving.analysis_key(),
                        generation: "moving",
                        profile,
                    },
                    &|| true,
                )
                .unwrap()
        };
        let first = render(&mut processor);
        assert!(first.field_precision.is_some());
        // 激活路径的相对位移量化误差必须有限非负（07 连续场安全界）。
        assert!(first.quantization_error.is_finite() && first.quantization_error >= 0.0);
        assert_ne!(first.image.as_bytes(), baseline.as_bytes());
        assert_eq!(
            first.image.as_bytes(),
            render(&mut processor).image.as_bytes()
        );
        let out = processor
            .render_reshape_fixed(
                &input,
                request(adjustments),
                FixedReshapeRequest {
                    scene: Some(&zero),
                    expected_analysis_key: zero.analysis_key(),
                    generation: "zero",
                    profile,
                },
                &|| true,
            )
            .unwrap();
        assert_eq!(out.image.as_bytes(), baseline.as_bytes());
        assert!(processor.reshape_gpu.is_none());
        let other = DynamicImage::ImageRgba32F(ImageBuffer::from_pixel(
            128,
            128,
            Rgba([0.7, 0.2, 0.1, 1.0]),
        ));
        let out = processor
            .render_reshape_fixed(
                &other,
                request(adjustments),
                FixedReshapeRequest {
                    scene: Some(&empty),
                    expected_analysis_key: empty.analysis_key(),
                    generation: "next-photo",
                    profile,
                },
                &|| true,
            )
            .unwrap();
        assert_eq!(
            out.image.as_bytes(),
            original(&processor, &other, adjustments, profile).as_bytes()
        );
        assert!(
            processor
                .render_reshape_fixed(
                    &input,
                    request(adjustments),
                    FixedReshapeRequest {
                        scene: Some(&moving),
                        expected_analysis_key: "stale",
                        generation: "bad",
                        profile
                    },
                    &|| true
                )
                .is_err()
        );
    }
}
#[test]
#[ignore = "需要本地 GPU；恒等测试场验证浮点基底仅拆分一次末尾颗粒／警告／抖动"]
fn split_color_and_tail_preserve_operation_order() {
    let context = test_context();
    let dims = [64, 48];
    let processor = GpuProcessor::new(context.clone(), 64, 48).unwrap();
    // test-only identity 场强制经过节点，生产 identity 必须旁路。
    let field = ValidatedField::fixture(dims, false, |_, _| [0.0; 2]);
    let input = DynamicImage::ImageRgba32F(ImageBuffer::from_fn(64, 48, |x, y| {
        Rgba([0.2 + x as f32 / 160.0, 0.3 + y as f32 / 160.0, 0.4, 0.75])
    }));
    for profile in [Profile::Preview, Profile::HighQuality] {
        let mut node = ReshapeGpu::new(
            &context,
            dims,
            profile,
            field.precision(),
            processor.reshape_layout_entries(),
        )
        .unwrap();
        for clipping in [0, 1] {
            let mut a = crate::image_processing::get_all_adjustments_from_json(
                &serde_json::json!({}),
                false,
                crate::white_balance::WhiteBalance::reference(),
                None,
            );
            a.global.grain_amount = 0.2;
            a.global.grain_size = 0.7;
            a.global.grain_roughness = 0.3;
            a.global.show_clipping = clipping;
            let baseline = original(&processor, &input, a, profile).to_rgba32f();
            node.upload(&context, input.to_rgba32f().as_raw(), &field, "split")
                .unwrap();
            let dummy = TextureSlot::new(
                &context.device,
                [1, 1],
                wgpu::TextureFormat::Rgba16Float,
                false,
            );
            processor
                .run_internal(
                    &node.input.view,
                    &dummy.view,
                    &dummy.view,
                    64,
                    48,
                    request(a),
                    false,
                    RenderOutputPrecision::EightBit,
                    Some(FloatBaseTarget {
                        stage: &node.base_stage,
                        slot: &node.base,
                    }),
                )
                .unwrap();
            node.warp(&context).unwrap();
            let output = node.finish(&context, &a).unwrap().to_rgba32f();
            let maximum = baseline
                .as_raw()
                .iter()
                .zip(output.as_raw())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0, f32::max);
            eprintln!("07 拆分链路 {profile:?} clipping={clipping} max={maximum}");
            assert!(
                maximum
                    <= if profile == Profile::Preview {
                        2.0 / 255.0
                    } else {
                        0.002
                    }
            );
        }
    }
}
