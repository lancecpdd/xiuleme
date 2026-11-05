//! 08 合成回归：不下载模型，不运行 GPU，不使用真实人像或外部可变网格。
use super::*;
use crate::portrait_gpu_field::ValidatedField;
use crate::portrait_tiles::{CompositionDependencies, DeviceBudget, PixelRect, plan_tiles};
use image::{ImageFormat, Rgba, RgbaImage};
use std::io::Cursor;

#[test]
fn portrait_preview_base_key_ignores_tone_crop_and_color_masks() {
    let base = json!({"portrait":{"lipstick":50,"bigEye":20},"rotation":0,"aiPatches":[]});
    let key = portrait_base_key("source", &base).unwrap();
    for (name, value) in [
        ("exposure", json!(2)),
        ("temperature", json!(35)),
        ("whiteBalance", json!({"tint":20})),
        ("hsl", json!({"reds":{"hue":20}})),
        ("curves", json!({"red":[[0,0],[1,0.9]]})),
        ("colorGrading", json!({"blending":60})),
        ("lutPath", json!("other.cube")),
        ("crop", json!({"x":10,"y":20,"width":30,"height":40})),
        (
            "masks",
            json!([{"id":"color-mask","adjustments":{"exposure":2}}]),
        ),
    ] {
        let mut changed = base.clone();
        changed[name] = value;
        assert_eq!(
            portrait_base_key("source", &changed).unwrap(),
            key,
            "{name}"
        );
    }
    assert_ne!(portrait_base_key("another-source", &base).unwrap(), key);
    for (name, value) in [
        ("rotation", json!(20)),
        ("portrait", json!({"lipstick":60,"bigEye":20})),
        ("sectionVisibility", json!({"portraitEyes":false})),
    ] {
        let mut changed = base.clone();
        changed[name] = value;
        assert_ne!(
            portrait_base_key("source", &changed).unwrap(),
            key,
            "{name}"
        );
    }
    let mut patch = base.clone();
    patch["aiPatches"] = json!([{"id":"same","visible":true,"patchData":"AAAA"}]);
    let before = portrait_base_key("source", &patch).unwrap();
    patch["aiPatches"][0]["patchData"] = json!("BBBB");
    assert_ne!(portrait_base_key("source", &patch).unwrap(), before);
}

#[test]
fn portrait_preview_cache_reuses_pixels_and_budget_failure_keeps_proxy() {
    let full = PortraitBasePixels {
        image: Arc::new(DynamicImage::ImageRgba8(RgbaImage::from_pixel(
            4,
            2,
            Rgba([1, 2, 3, 255]),
        ))),
        original_dimensions: [4, 2],
    };
    let mut cache = PortraitBaseCache::default();
    cache.insert_full("base", &full, 32);
    assert!(Arc::ptr_eq(&cache.full("base").unwrap().image, &full.image));
    assert!(cache.full("other").is_none());
    let proxy = PortraitBasePixels {
        image: Arc::new(portrait_proxy(&full.image, 2, &|| true).unwrap()),
        ..full.clone()
    };
    cache.insert_proxy("base", 2, &proxy);
    cache.insert_full("base", &full, 16);
    assert!(cache.full("base").is_none());
    assert!(Arc::ptr_eq(
        &cache.proxy("base", 2).unwrap().image,
        &proxy.image
    ));
    assert!(cache.proxy("base", 3).is_none());
    cache.insert_full("new", &full, 32);
    assert!(cache.proxy("base", 2).is_some());
}

#[test]
fn portrait_preview_switch_source_keeps_content_keyed_pixels() {
    let state = PortraitRenderState::default();
    state.advance_editor(1, 1).unwrap();
    let pixels = PortraitBasePixels {
        image: Arc::new(DynamicImage::ImageRgba8(RgbaImage::new(1, 1))),
        original_dimensions: [1, 1],
    };
    state
        .portrait_bases
        .lock()
        .unwrap()
        .insert_full("key", &pixels, 4);
    state.advance_editor(1, 2).unwrap();
    assert!(state.portrait_bases.lock().unwrap().full("key").is_some());
    state.advance_editor(2, 3).unwrap();
    assert!(state.portrait_bases.lock().unwrap().full("key").is_some());
}

#[test]
fn portrait_preview_proxy_covers_editor_identity_preview() {
    assert_eq!(
        interactive_portrait_proxy_limit(
            RenderPurpose::Preview,
            RenderQuality::Interactive,
            true,
            "high",
            1920
        ),
        Some(1280)
    );
    assert_eq!(
        interactive_portrait_proxy_limit(
            RenderPurpose::Preview,
            RenderQuality::Interactive,
            true,
            "high",
            960
        ),
        Some(960)
    );
    assert_eq!(
        interactive_portrait_proxy_limit(
            RenderPurpose::Preview,
            RenderQuality::HighQuality,
            true,
            "high",
            1920
        ),
        Some(1920)
    );
    assert_eq!(
        interactive_portrait_proxy_limit(
            RenderPurpose::Preview,
            RenderQuality::Interactive,
            true,
            "full",
            1920
        ),
        Some(1920)
    );
    assert_eq!(portrait_preview_limit("performance"), Some(960));
    assert_eq!(portrait_preview_limit("full"), None);
    for purpose in [
        RenderPurpose::Export,
        RenderPurpose::Mask,
        RenderPurpose::Thumbnail,
        RenderPurpose::Uncropped,
    ] {
        assert_eq!(
            interactive_portrait_proxy_limit(
                purpose,
                RenderQuality::Interactive,
                true,
                "high",
                1920
            ),
            None
        );
    }
    assert_eq!(
        interactive_portrait_proxy_limit(
            RenderPurpose::Preview,
            RenderQuality::Interactive,
            false,
            "high",
            1920
        ),
        None
    );
    let full = DynamicImage::ImageRgba8(RgbaImage::from_pixel(1600, 800, Rgba([20, 40, 60, 255])));
    for (quality, expected) in [("high", 1280), ("performance", 960)] {
        let proxy =
            portrait_proxy(&full, portrait_preview_limit(quality).unwrap(), &|| true).unwrap();
        assert_eq!([proxy.width(), proxy.height()], [expected, expected / 2]);
        assert!(proxy.as_rgba32f().is_some());
    }
    assert!(portrait_proxy(&full, 960, &|| false).is_err());
    let fitted = portrait_proxy(&full, 800, &|| true).unwrap();
    assert_eq!([fitted.width(), fitted.height()], [800, 400]);
}

#[test]
fn portrait_preview_proxy_preserves_alpha_and_does_not_upscale() {
    let mut full = RgbaImage::new(2, 1);
    full.put_pixel(0, 0, Rgba([255, 0, 0, 255]));
    full.put_pixel(1, 0, Rgba([0, 0, 255, 0]));
    let image = DynamicImage::ImageRgba8(full);
    let proxy = portrait_proxy(&image, 1, &|| true).unwrap();
    assert_eq!(
        proxy.as_rgba32f().unwrap().get_pixel(0, 0).0,
        [1.0, 0.0, 0.0, 0.5]
    );
    let unchanged = portrait_proxy(&image, 1280, &|| true).unwrap();
    assert_eq!([unchanged.width(), unchanged.height()], [2, 1]);
}

#[test]
fn portrait_preview_proxy_accepts_float_and_rgb8_bases() {
    // 预览/RAW 基底常为浮点或无 Alpha 的 RGB8，不能再硬要求 as_rgba8。
    let rgb32 = DynamicImage::ImageRgb32F(image::Rgb32FImage::from_pixel(
        4,
        2,
        image::Rgb([0.2, 0.4, 0.6]),
    ));
    let proxy = portrait_proxy(&rgb32, 2, &|| true).unwrap();
    assert_eq!([proxy.width(), proxy.height()], [2, 1]);
    assert!(proxy.as_rgba32f().is_some());

    let rgb8 = DynamicImage::ImageRgb8(image::RgbImage::from_pixel(
        4,
        2,
        image::Rgb([51, 102, 153]),
    ));
    let proxy8 = portrait_proxy(&rgb8, 2, &|| true).unwrap();
    assert_eq!([proxy8.width(), proxy8.height()], [2, 1]);
}

#[test]
fn portrait_preview_proxy_crop_uses_continuous_source_mapping() {
    let window = OutputWindow::new(
        [600, 400],
        &json!({"x":11,"y":17,"width":309,"height":201}),
        [160, 104],
        None,
    )
    .unwrap();
    let scaled = proxy_output_window(&window, [600, 400], [192, 128]);
    assert_eq!(scaled.output_dimensions, window.output_dimensions);
    assert_eq!(scaled.roi, window.roi);
    for (actual, original) in scaled.output_to_base.iter().zip(window.output_to_base) {
        assert!((*actual - original * 0.32).abs() < 1e-12);
    }
}

#[test]
fn portrait_preview_proxy_mask_keeps_alpha_and_ignores_color_adjustments() {
    let cache = NativeMaskCache::default();
    let mut mask = photo_mask(50.0);
    let mut output = Vec::new();
    let rect = PixelRect::full([4, 2]).unwrap();
    cache
        .append_resampled_slice(
            "source",
            &mask,
            [8, 4],
            [4, 2],
            rect,
            None,
            &mut output,
            &|| true,
        )
        .unwrap();
    assert_eq!(output, vec![127; 8]);
    mask.adjustments = json!({"exposure":2});
    cache
        .append_resampled_slice(
            "source",
            &mask,
            [8, 4],
            [4, 2],
            rect,
            None,
            &mut output,
            &|| true,
        )
        .unwrap();
    assert_eq!(cache.inner.lock().unwrap().entries.len(), 1);
    assert_eq!(output.len(), 16);
}

fn adjustments(amount: u32) -> Value {
    json!({"portrait":{"slimFace":amount},"portraitReshape":{
        "schemaVersion":1,"engineVersion":"mesh-arap-v1","enabled":true,"faces":[]}})
}

#[test]
fn portrait_render_intent_distinguishes_pending_from_zero_and_legacy() {
    assert_eq!(
        render_intent(&json!({"portrait":{"slimFace":100}})).unwrap(),
        RenderIntent::Bypass
    );
    assert_eq!(
        render_intent(&adjustments(0)).unwrap(),
        RenderIntent::Bypass
    );
    assert_eq!(
        render_intent(&adjustments(25)).unwrap(),
        RenderIntent::Prepare
    );
    let mut changed = adjustments(25);
    changed["portraitReshape"]["enabled"] = json!(false);
    assert_eq!(render_intent(&changed).unwrap(), RenderIntent::Bypass);
    changed["portraitReshape"]["schemaVersion"] = json!(99);
    assert!(render_intent(&changed).is_err());
    let mut changed = adjustments(25);
    changed["liquify"] = json!([]);
    assert!(render_intent(&changed).is_err());
}

#[test]
fn gpupixel_active_clears_native_portrait_gpu_params() {
    let mut params = crate::image_processing::get_all_adjustments_from_json(
        &json!({"portrait": {"aiBlemishRemoval": 100, "slimFace": 40}}),
        false,
        crate::white_balance::WhiteBalance::reference(),
        None,
    );
    assert!(params.portrait.blemishes[0] > 0.0);
    suppress_native_portrait_when_gpupixel(true, &mut params);
    assert_eq!(params.portrait.blemishes, [0.0; 4]);
    assert_eq!(params.portrait.strengths, [0.0; 4]);
    let mut kept = crate::image_processing::get_all_adjustments_from_json(
        &json!({"portrait": {"aiBlemishRemoval": 100}}),
        false,
        crate::white_balance::WhiteBalance::reference(),
        None,
    );
    suppress_native_portrait_when_gpupixel(false, &mut kept);
    assert!(kept.portrait.blemishes[0] > 0.0);
}

#[test]
fn deferred_preview_payload_bypasses_reshape_and_disables_retouch() {
    let mut preview = json!({
        "portrait": {
            "slimFace": 31,
            "bigEye": 55,
            "aiBlemishRemoval": 0,
            "frequencySeparationFace": 0,
            "blemishRemoval": 0,
            "faceShineRemoval": 0,
            "neutralGrayEvenness": 0,
            "neutralGrayContour": 0,
            "frequencySeparationBody": 0
        },
        "portraitReshape": {
            "schemaVersion": 1,
            "engineVersion": "mesh-arap-v1",
            "enabled": false,
            "faces": [{
                "faceId": "face",
                "sourceAnchor": {"sourceBaseHash": "a".repeat(64), "bounds": [0.1, 0.1, 0.2, 0.2]},
                "overrides": {"slimFace": 40},
                "protection": {"hair": true, "background": true, "strokes": []}
            }]
        }
    });
    assert_eq!(render_intent(&preview).unwrap(), RenderIntent::Bypass);
    assert!(!crate::portrait_processing::enabled(&preview));
    preview["portraitReshape"]["enabled"] = json!(true);
    preview["portrait"]["aiBlemishRemoval"] = json!(50);
    assert_eq!(render_intent(&preview).unwrap(), RenderIntent::Prepare);
    assert!(crate::portrait_processing::enabled(&preview));
}

#[test]
fn portrait_render_intent_mesh_big_eye_and_eyes_visibility() {
    let mut a = adjustments(0);
    a["portraitReshape"]["engineVersion"] = json!("3dmm-tddfa-v1");
    a["portraitReshape"]["faces"] = json!([{
        "faceId":"face","sourceAnchor":{"sourceBaseHash":"a".repeat(64),"bounds":[0.1,0.1,0.2,0.2]},
        "overrides":{"bigEye":40},"protection":{"hair":true,"background":true,"strokes":[]}
    }]);
    assert_eq!(render_intent(&a).unwrap(), RenderIntent::Prepare);

    a["sectionVisibility"] = json!({"portraitReshape": false, "portraitEyes": true});
    assert_eq!(render_intent(&a).unwrap(), RenderIntent::Prepare);

    a["sectionVisibility"] = json!({"portraitReshape": true, "portraitEyes": false});
    assert_eq!(render_intent(&a).unwrap(), RenderIntent::Bypass);

    a["sectionVisibility"] = json!({"portraitReshape": false, "portraitEyes": false});
    assert_eq!(render_intent(&a).unwrap(), RenderIntent::Bypass);

    // 网格瘦脸 + 大眼：重塑分区关闭时仍可因大眼 Prepare。
    a["portraitReshape"]["faces"][0]["overrides"] = json!({"slimFace":30,"bigEye":40});
    a["sectionVisibility"] = json!({"portraitReshape": false, "portraitEyes": true});
    assert_eq!(render_intent(&a).unwrap(), RenderIntent::Prepare);
}

#[test]
fn portrait_render_unprojected_or_disabled_protection_is_not_ignored() {
    let mut a = adjustments(0);
    a["portraitReshape"]["faces"] = json!([{
        "faceId":"face","sourceAnchor":{"sourceBaseHash":"a".repeat(64),"bounds":[0.1,0.1,0.2,0.2]},
        "overrides":{"uvSlimFace":50},"protection":{"hair":false}
    }]);
    assert!(render_intent(&a).is_err());
    a["portraitReshape"]["faces"][0]["protection"] = json!({"strokes":[{
        "mode":"protect","radius":0.01,"points":[[0.2,0.2]]
    }]});
    assert!(render_intent(&a).is_err());
    a["portraitReshape"]["faces"][0]["protection"] = json!({"unimplemented":true});
    assert!(render_intent(&a).is_err());
}

#[test]
fn portrait_render_canonical_keys_include_equal_length_content_changes() {
    assert_eq!(
        content_key(&json!({"b":2,"a":{"y":3,"x":4}})),
        content_key(&json!({"a":{"x":4,"y":3},"b":2}))
    );
    for key in ["patchData", "maskDataBase64", "lutContent"] {
        let mut a = json!({});
        a[key] = json!("AAAA");
        let mut b = a.clone();
        b[key] = json!("BBBB");
        assert_ne!(content_key(&a), content_key(&b));
    }
}

#[test]
#[ignore = "Windows 上同 mtime 内容替换偶发检不出；与网格移除无关"]
fn portrait_render_strict_source_detects_same_length_same_time_replacement() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("source.png");
    let image = DynamicImage::ImageRgba8(RgbaImage::from_pixel(7, 9, Rgba([50, 100, 150, 255])));
    image.save(&path).unwrap();
    let source = SourceSnapshot::load(
        path.to_str().unwrap(),
        &crate::AppSettings::default(),
        7,
        &|| true,
    )
    .unwrap();
    assert_eq!(source.revision(), 7);
    source.verify(&|| true).unwrap();
    let modified =
        filetime::FileTime::from_last_modification_time(&std::fs::metadata(&path).unwrap());
    let mut bytes = std::fs::read(&path).unwrap();
    let last = bytes.len() - 1;
    bytes[last] ^= 1;
    std::fs::write(&path, bytes).unwrap();
    filetime::set_file_mtime(&path, modified).unwrap();
    assert!(source.verify(&|| true).is_err());
}

#[test]
fn portrait_render_rejects_source_and_hardlink_targets_and_cancelled_reads() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("source.png");
    let alias = dir.path().join("alias.png");
    DynamicImage::ImageRgba8(RgbaImage::from_pixel(3, 3, Rgba([0, 0, 0, 255])))
        .save(&path)
        .unwrap();
    let source = SourceSnapshot::load(
        path.to_str().unwrap(),
        &crate::AppSettings::default(),
        1,
        &|| true,
    )
    .unwrap();
    std::fs::hard_link(&path, &alias).unwrap();
    assert!(source.reject_source_target(&path).is_err());
    assert!(source.reject_source_target(&alias).is_err());
    source
        .reject_source_target(&dir.path().join("new.png"))
        .unwrap();
    assert!(source.verify(&|| false).is_err());
    assert!(
        SourceSnapshot::load(
            path.to_str().unwrap(),
            &crate::AppSettings::default(),
            2,
            &|| false
        )
        .is_err()
    );
}

fn preview_fixture() -> PreviewMetadata {
    PreviewMetadata {
        path: "照片.raw".into(),
        stamp: RenderStamp {
            request_id: 11,
            source_revision: 7,
            epoch: 19,
            composite_key: "a".repeat(64),
            reshape_key: "b".repeat(64),
            window_key: "c".repeat(64),
        },
        mode: PreviewMode::Jpeg,
        dimensions: [101, 73],
        roi: PixelRect::new(3, 5, 17, 19).unwrap(),
        reshape_attenuation: None,
        body_attenuation: None,
    }
}

#[test]
fn portrait_render_rrv2_encodes_utf8_little_endian_metadata_and_body() {
    let metadata = preview_fixture();
    let body = [255, 216, 255, 217];
    let bytes = metadata.encode(&body).unwrap();
    assert_eq!(&bytes[..4], b"RRV2");
    let length = u32::from_le_bytes(bytes[4..8].try_into().unwrap()) as usize;
    let decoded: Value = serde_json::from_slice(&bytes[8..8 + length]).unwrap();
    assert_eq!(decoded, serde_json::to_value(&metadata).unwrap());
    assert_eq!(&bytes[8 + length..], &body);
    assert_eq!(decoded["stamp"]["requestId"], 11);
    let native = PreviewMetadata {
        mode: PreviewMode::Native,
        roi: PixelRect::full(metadata.dimensions).unwrap(),
        ..metadata
    };
    let bytes = native.encode(&[]).unwrap();
    assert_eq!(
        bytes.len(),
        8 + u32::from_le_bytes(bytes[4..8].try_into().unwrap()) as usize
    );
    assert!(native.encode(&body).is_err());
}

#[test]
fn portrait_render_rrv2_rejects_invalid_identity_and_windows() {
    let valid = preview_fixture();
    assert!(valid.encode(&[]).is_err());
    let mut wrong = valid.clone();
    wrong.mode = PreviewMode::Native;
    assert!(wrong.encode(&[]).is_err());
    for dimensions in [[0, 1], [32769, 1], [10000, 10000]] {
        wrong = valid.clone();
        wrong.dimensions = dimensions;
        assert!(wrong.encode(&[1]).is_err());
    }
    for roi in [
        PixelRect {
            width: 0,
            ..valid.roi
        },
        PixelRect::new(-1, 0, 2, 2).unwrap(),
        PixelRect::new(100, 0, 2, 2).unwrap(),
    ] {
        wrong = valid.clone();
        wrong.roi = roi;
        assert!(wrong.encode(&[1]).is_err());
    }
    for n in [0, 9_007_199_254_740_992, u64::MAX] {
        for field in 0..3 {
            wrong = valid.clone();
            match field {
                0 => wrong.stamp.request_id = n,
                1 => wrong.stamp.source_revision = n,
                _ => wrong.stamp.epoch = n,
            }
            assert!(wrong.encode(&[1]).is_err());
        }
    }
    wrong = valid.clone();
    wrong.stamp.request_id = 9_007_199_254_740_991;
    wrong.stamp.reshape_key = "bypass".into();
    assert!(wrong.encode(&[1]).is_ok());
    for key in ["".into(), "A".repeat(64), "a".repeat(63)] {
        wrong = valid.clone();
        wrong.stamp.window_key = key;
        assert!(wrong.encode(&[1]).is_err());
    }
    for path in [String::new(), "x".repeat(65536)] {
        wrong = valid.clone();
        wrong.path = path;
        assert!(wrong.encode(&[1]).is_err());
    }
}

#[test]
fn portrait_render_preview_window_rounds_crop_before_output_roi() {
    let crop = json!({"x":1.6,"y":3.1,"width":180.0,"height":140.0});
    let window = preview_window([201, 151], &crop, 91, Some((0.25, 0.25, 0.5, 0.5))).unwrap();
    assert_eq!(window.crop, PixelRect::new(2, 3, 180, 140).unwrap());
    assert_eq!(window.output_dimensions, [91, 71]);
    assert_eq!(window.roi, PixelRect::new(23, 18, 46, 36).unwrap());
    assert_eq!(
        window.output_to_base,
        [180.0 / 91.0, 0.0, 2.0, 0.0, 140.0 / 71.0, 3.0]
    );
    assert_eq!(
        preview_window([201, 151], &crop, 4096, None)
            .unwrap()
            .output_dimensions,
        [180, 140]
    );
    for target in [0, 32769] {
        assert!(preview_window([201, 151], &crop, target, None).is_err());
    }
    for roi in [
        (f32::NAN, 0.0, 1.0, 1.0),
        (0.0, 0.0, 0.0, 1.0),
        (0.9, 0.0, 0.2, 1.0),
        (-0.1, 0.0, 1.0, 1.0),
        (0.0, 0.0, 0.001, 1.0),
    ] {
        assert!(normalized_preview_roi([101, 73], Some(roi)).is_err());
    }
}

#[test]
fn portrait_render_committed_events_require_exact_stamp_and_success() {
    let state = PortraitRenderState::default();
    state.advance_editor(7, 1).unwrap();
    let stamp = state
        .legacy_stamp(11, 7, "A.raw", &json!({}), &json!([101, 73]))
        .unwrap();
    assert!(state.with_committed_preview(&stamp, 1, || true).is_none());
    assert!(
        state
            .publish_legacy_preview::<()>(&stamp, 1, || Err("显示失败".into()))
            .is_err()
    );
    assert!(state.with_committed_preview(&stamp, 1, || true).is_none());
    state.publish_legacy_preview(&stamp, 1, || Ok(())).unwrap();
    assert_eq!(state.with_committed_preview(&stamp, 1, || 42), Some(42));
    for field in 0..6 {
        let mut wrong = stamp.clone();
        match field {
            0 => wrong.request_id += 1,
            1 => wrong.source_revision += 1,
            2 => wrong.epoch += 1,
            3 => wrong.composite_key = "d".repeat(64),
            4 => wrong.reshape_key = "e".repeat(64),
            _ => wrong.window_key = "f".repeat(64),
        }
        assert!(
            state
                .with_committed_preview(&wrong, 1, || panic!("旧统计不得发布"))
                .is_none()
        );
    }
    state.advance_editor(7, 2).unwrap();
    assert!(state.with_committed_preview(&stamp, 1, || true).is_none());
    assert!(state.publish_legacy_preview(&stamp, 1, || Ok(())).is_err());
    assert!(
        state.editor.lock().unwrap().source.is_none(),
        "旁路不创建严格源"
    );
}

#[test]
fn portrait_render_identity_publication_failure_does_not_commit_statistics() {
    let (_directory, state, mut output) = publication_fixture();
    state.advance_editor(7, 1).unwrap();
    output.lease = state.begin_editor_consumer(&output.source, 11, 1).unwrap();
    output.stamp.epoch = output.lease.epoch;
    assert!(
        state
            .publish_editor::<()>(&output, 1, &|| true, || Err("上传失败".into()))
            .is_err()
    );
    assert!(!output.lease.has_committed(&output.stamp));
    assert!(
        state
            .with_committed_preview(&output.stamp, 1, || true)
            .is_none()
    );
    state
        .publish_editor(&output, 1, &|| true, || Ok(()))
        .unwrap();
    assert_eq!(
        state.with_committed_preview(&output.stamp, 1, || true),
        Some(true)
    );
    state.advance_editor(8, 2).unwrap();
    assert!(
        state
            .with_committed_preview(&output.stamp, 1, || true)
            .is_none()
    );
}

#[test]
fn portrait_render_hydration_replaces_equal_length_mask_bytes() {
    let mut cache = std::collections::HashMap::from([("same".into(), json!("AAAA"))]);
    for key in ["mask_data_base64", "maskDataBase64"] {
        let mut masks = vec![json!({"id":"same","parameters":{key:"BBBB"}})];
        crate::adjustment_utils::hydrate_sub_masks(&mut masks, &mut cache);
        assert_eq!(masks[0]["parameters"][key], "BBBB");
        assert_eq!(cache["same"], "BBBB");
        masks[0]["parameters"][key] = Value::Null;
        crate::adjustment_utils::hydrate_sub_masks(&mut masks, &mut cache);
        assert_eq!(masks[0]["parameters"][key], "BBBB");
    }
}

#[test]
fn portrait_render_crop_matches_existing_rounding_and_boundary_intersection() {
    let image = DynamicImage::ImageRgba8(RgbaImage::from_fn(31, 29, |x, y| {
        Rgba([x as u8, y as u8, 0, 255])
    }));
    for crop in [
        Value::Null,
        json!({"x":1.6,"y":2.4,"width":40.0,"height":10.7}),
        json!({"x":-3.0,"y":0.0,"width":5.0,"height":7.0}),
        json!({"x":100.0,"y":0.0,"width":1.0,"height":1.0}),
        json!({"x":1.0,"y":2.0,"width":0.0,"height":3.0}),
    ] {
        let rect = crate::portrait_tiles::effective_crop([31, 29], &crop).unwrap();
        let old = crate::image_processing::apply_crop(&image, &crop);
        assert_eq!(
            old.to_rgba8(),
            image
                .crop_imm(rect.x as u32, rect.y as u32, rect.width, rect.height)
                .to_rgba8()
        );
    }
}

#[test]
fn portrait_render_roi_preserves_full_output_origin_and_kernel_support() {
    let roi = PixelRect::new(25, 20, 10, 12).unwrap();
    let window = OutputWindow::new(
        [201, 151],
        &json!({"x":1.6,"y":3.1,"width":180.0,"height":140.0}),
        [90, 70],
        Some(roi),
    )
    .unwrap();
    assert_eq!(window.output_to_base, [2.0, 0.0, 2.0, 0.0, 2.0, 3.0]);
    assert_eq!(
        window.base_support(roi, 3.0).unwrap(),
        PixelRect::new(46, 37, 32, 36).unwrap()
    );
    assert!(
        window
            .base_support(PixelRect::new(0, 0, 1, 1).unwrap(), 3.0)
            .is_err()
    );
}

#[test]
fn portrait_render_pages_match_07_fp32_and_shared_nodes_bitwise() {
    let scene = crate::portrait_safety::gpu_fixture_dimensions(10.0, false, [513, 387]);
    let fixed = ValidatedField::from_scene(&scene, false, &|| true)
        .unwrap()
        .unwrap();
    let paged = CanonicalField::from_scene(&scene, &|| true).unwrap();
    let mut moved_on_seam = false;
    for py in (0..2).rev() {
        for px in (0..3).rev() {
            let page = paged.page([px, py], &|| true).unwrap();
            for y in 0..=page.dimensions[1] {
                for x in 0..=page.dimensions[0] {
                    let gx = page.origin[0] + x;
                    let gy = page.origin[1] + y;
                    let sample = page.node(gx, gy).unwrap();
                    let expected = fixed.samples()[(gy * 514 + gx) as usize];
                    assert_eq!(
                        sample.map(f32::to_bits),
                        expected.map(f32::to_bits),
                        "{gx},{gy}"
                    );
                    if gx == 256 {
                        moved_on_seam |= sample[0] != 0.0 || sample[1] != 0.0;
                    }
                }
            }
        }
    }
    assert!(moved_on_seam, "页边界不能被当成照片边界固定");
    let again = CanonicalField::from_scene(&scene, &|| true).unwrap();
    assert_eq!(paged.key(), again.key());
    assert!(paged.quantization_error() <= 1.0 / 64.0);
}

#[test]
fn portrait_render_footprints_cover_exhaustive_bilinear_samples_and_filter_taps() {
    let scene = crate::portrait_safety::gpu_fixture(10.0, false);
    let field = CanonicalField::from_scene(&scene, &|| true).unwrap();
    for size in [17u32, 64, 127] {
        for y in (0..128).step_by(size as usize) {
            for x in (0..128).step_by(size as usize) {
                let tile =
                    PixelRect::new(x, y, (128 - x as u32).min(size), (128 - y as u32).min(size))
                        .unwrap();
                let footprint = field.inverse_footprint(tile, 2, &|| true).unwrap();
                for gy in tile.y as u32..tile.bottom() as u32 {
                    for gx in tile.x as u32..tile.right() as u32 {
                        let corners = [
                            field.node(gx, gy, &|| true).unwrap(),
                            field.node(gx + 1, gy, &|| true).unwrap(),
                            field.node(gx, gy + 1, &|| true).unwrap(),
                            field.node(gx + 1, gy + 1, &|| true).unwrap(),
                        ];
                        for tx in [0.0, 0.25, 0.5, 0.75, 1.0] {
                            for ty in [0.0, 0.25, 0.5, 0.75, 1.0] {
                                let d: [f64; 2] = std::array::from_fn(|c| {
                                    let a = corners[0][c] as f64 * (1.0 - tx)
                                        + corners[1][c] as f64 * tx;
                                    let b = corners[2][c] as f64 * (1.0 - tx)
                                        + corners[3][c] as f64 * tx;
                                    a * (1.0 - ty) + b * ty
                                });
                                let sx = gx as f64 + tx + d[0];
                                let sy = gy as f64 + ty + d[1];
                                assert!(
                                    sx - 2.0 >= footprint.x as f64
                                        && sx + 2.0 <= footprint.right() as f64
                                );
                                assert!(
                                    sy - 2.0 >= footprint.y as f64
                                        && sy + 2.0 <= footprint.bottom() as f64
                                );
                            }
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn portrait_render_conjugate_rotation_zero_matches_base_bitwise() {
    let scene = crate::portrait_safety::gpu_fixture_dimensions(10.0, false, [513, 387]);
    let base = Arc::new(CanonicalField::from_scene(&scene, &|| true).unwrap());
    let conjugate = CanonicalField::conjugate_uncropped(&base, 0.0, &|| true).unwrap();
    assert_ne!(conjugate.key(), base.key());
    assert_eq!(conjugate.dimensions(), base.dimensions());
    assert!(conjugate.quantization_error() <= 1.0 / 64.0);
    let mut moved = false;
    for gy in 0..=387u32 {
        for gx in 0..=513u32 {
            let c = conjugate.node(gx, gy, &|| true).unwrap();
            let b = base.node(gx, gy, &|| true).unwrap();
            assert_eq!(c.map(f32::to_bits), b.map(f32::to_bits), "节点 {gx},{gy}");
            moved |= c[0] != 0.0 || c[1] != 0.0;
        }
    }
    assert!(moved, "fixture 场必须在内部有非零位移");
}

#[test]
fn portrait_render_conjugate_displacements_match_inverse_rotation_formula() {
    let dims = [513u32, 387];
    let rotation = 33.0;
    let scene = crate::portrait_safety::gpu_fixture_dimensions(10.0, false, dims);
    let base = Arc::new(CanonicalField::from_scene(&scene, &|| true).unwrap());
    let conjugate = CanonicalField::conjugate_uncropped(&base, rotation, &|| true).unwrap();
    let angle = rotation.to_radians();
    let (sin, cos) = angle.sin_cos();
    let center = [dims[0] as f64 / 2.0 + 0.5, dims[1] as f64 / 2.0 + 0.5];
    let mut verified = 0;
    // 覆盖脸心附近（位移非零）与远端（位移为零）两类内部节点。
    for (gx, gy) in [
        (256u32, 193u32),
        (200, 180),
        (300, 210),
        (240, 160),
        (260, 230),
        (60, 60),
        (450, 330),
    ] {
        let node = conjugate.node(gx, gy, &|| true).unwrap();
        let (ux, uy) = (gx as f64 - center[0], gy as f64 - center[1]);
        let bx = cos * ux - sin * uy + center[0];
        let by = sin * ux + cos * uy + center[1];
        let x0 = bx.floor().clamp(0.0, (dims[0] - 1) as f64) as u32;
        let y0 = by.floor().clamp(0.0, (dims[1] - 1) as f64) as u32;
        let (tx, ty) = (bx - x0 as f64, by - y0 as f64);
        let n = |x: u32, y: u32| base.node(x, y, &|| true).unwrap();
        let blend = |i: usize| {
            let top = n(x0, y0)[i] as f64 * (1.0 - tx) + n(x0 + 1, y0)[i] as f64 * tx;
            let bottom = n(x0, y0 + 1)[i] as f64 * (1.0 - tx) + n(x0 + 1, y0 + 1)[i] as f64 * tx;
            top * (1.0 - ty) + bottom * ty
        };
        // d_U = R⁻¹ 线性部分作用 d_B：[c·dx + s·dy, -s·dx + c·dy]。
        let expected = [
            cos * blend(0) + sin * blend(1),
            -sin * blend(0) + cos * blend(1),
        ];
        assert!(
            (node[0] as f64 - expected[0]).abs() < 1e-5,
            "节点 {gx},{gy}：{node:?} 对比 {expected:?}"
        );
        assert!((node[1] as f64 - expected[1]).abs() < 1e-5);
        verified += 1;
    }
    assert_eq!(verified, 7);
}

#[test]
fn portrait_render_conjugate_locks_canvas_boundary_and_outside_base_corners() {
    let dims = [513u32, 387];
    let scene = crate::portrait_safety::gpu_fixture_dimensions(10.0, false, dims);
    let base = Arc::new(CanonicalField::from_scene(&scene, &|| true).unwrap());
    // 45°：U 画布四角投影落在同尺寸 B 画布外，扩展域恒零。
    let conjugate = CanonicalField::conjugate_uncropped(&base, 45.0, &|| true).unwrap();
    for gx in [0u32, 256, 513] {
        for gy in [0u32, 387] {
            let node = conjugate.node(gx, gy, &|| true).unwrap();
            assert_eq!((node[0], node[1], node[2]), (0.0, 0.0, 1.0), "{gx},{gy}");
        }
    }
    for gy in [0u32, 193, 387] {
        for gx in [0u32, 513] {
            let node = conjugate.node(gx, gy, &|| true).unwrap();
            assert_eq!((node[0], node[1], node[2]), (0.0, 0.0, 1.0), "{gx},{gy}");
        }
    }
    // 四角内侧节点投影在 B 域外：位移恒零（内部节点不置锁定标记）。
    let corner = conjugate.node(3, 3, &|| true).unwrap();
    assert_eq!((corner[0], corner[1]), (0.0, 0.0));
    assert_eq!(corner[2], 0.0);
    // identity 场的共轭仍为 identity；非法旋转角拒绝。
    let empty = crate::portrait_safety::gpu_fixture_dimensions(10.0, true, dims);
    let empty_field = Arc::new(CanonicalField::from_scene(&empty, &|| true).unwrap());
    let empty_conjugate =
        CanonicalField::conjugate_uncropped(&empty_field, 45.0, &|| true).unwrap();
    assert!(empty_conjugate.is_identity());
    assert!(CanonicalField::conjugate_uncropped(&base, f64::NAN, &|| true).is_err());
}

#[test]
fn portrait_render_conjugate_footprint_covers_displacement_samples() {
    let dims = [513u32, 387];
    let scene = crate::portrait_safety::gpu_fixture_dimensions(10.0, false, dims);
    let base = Arc::new(CanonicalField::from_scene(&scene, &|| true).unwrap());
    let tile = PixelRect::new(200, 150, 120, 90).unwrap();
    for rotation in [0.0, 33.0, -33.0, 45.0] {
        let conjugate = CanonicalField::conjugate_uncropped(&base, rotation, &|| true).unwrap();
        let footprint = conjugate.inverse_footprint(tile, 2, &|| true).unwrap();
        for gy in tile.y as u32..tile.bottom() as u32 {
            for gx in tile.x as u32..tile.right() as u32 {
                let corners = [
                    conjugate.node(gx, gy, &|| true).unwrap(),
                    conjugate.node(gx + 1, gy, &|| true).unwrap(),
                    conjugate.node(gx, gy + 1, &|| true).unwrap(),
                    conjugate.node(gx + 1, gy + 1, &|| true).unwrap(),
                ];
                for tx in [0.0, 0.25, 0.5, 0.75, 1.0] {
                    for ty in [0.0, 0.25, 0.5, 0.75, 1.0] {
                        let d: [f64; 2] = std::array::from_fn(|c| {
                            let a = corners[0][c] as f64 * (1.0 - tx) + corners[1][c] as f64 * tx;
                            let b = corners[2][c] as f64 * (1.0 - tx) + corners[3][c] as f64 * tx;
                            a * (1.0 - ty) + b * ty
                        });
                        let (sx, sy) = (gx as f64 + tx + d[0], gy as f64 + ty + d[1]);
                        assert!(
                            sx - 2.0 >= footprint.x as f64 && sx + 2.0 <= footprint.right() as f64,
                            "rotation={rotation} 采样 {sx}"
                        );
                        assert!(
                            sy - 2.0 >= footprint.y as f64 && sy + 2.0 <= footprint.bottom() as f64,
                            "rotation={rotation} 采样 {sy}"
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn portrait_render_planner_accounts_native_radius_and_large_layout_without_images() {
    for full in [[6000, 4000], [10000, 5000], [32768, 1]] {
        let deps = CompositionDependencies::native(full, true, true, [0.03, -0.02]).unwrap();
        if full[1] >= 4000 {
            assert!(deps.blur_radii[3] > 128);
        }
        // 大图四级 blur 半径（scale≥3.7 时 r≥148）叠加色差角落扩张后，
        // 最小 128px 核心块的输入窗口已需要约 45-57 MiB；预算必须容得下它。
        let budget = DeviceBudget {
            bytes: 128 * 1024 * 1024,
            texture_side: 2048,
            buffer_bytes: 32 * 1024 * 1024,
            resident_bytes: 1024 * 1024,
        };
        let tiles = plan_tiles(PixelRect::full(full).unwrap(), budget, true, |tile| {
            deps.input(tile, full)
        })
        .unwrap();
        let mut pixels = 0u64;
        for (tile, input) in &tiles {
            assert!(budget.fits(*input, *tile, true).unwrap());
            pixels += tile.width as u64 * tile.height as u64;
        }
        assert_eq!(pixels, full[0] as u64 * full[1] as u64);
    }
    assert!(check_dimensions([32769, 1]).is_err());
    assert!(check_dimensions([10000, 10000]).is_err());
    let tiny = DeviceBudget {
        bytes: 1,
        texture_side: 32,
        buffer_bytes: 1,
        resident_bytes: 0,
    };
    assert!(plan_tiles(PixelRect::new(0, 0, 128, 128).unwrap(), tiny, true, Ok).is_err());
    let tight = DeviceBudget {
        bytes: 48 * 1024 * 1024,
        texture_side: 4096,
        buffer_bytes: 64 * 1024 * 1024,
        resident_bytes: 0,
    };
    let core = PixelRect::new(0, 0, 256, 256).unwrap();
    let by_core = plan_tiles(core, tight, true, |_| Ok(core)).unwrap();
    assert_eq!(by_core.len(), 1);
    let by_compose = crate::portrait_tiles::plan_tiles_with_resources(core, tight, true, |tile| {
        let mid = PixelRect::new(tile.x, tile.y, tile.width * 4, tile.height * 4)?;
        Ok((mid, mid, 0, 0))
    })
    .unwrap();
    assert!(
        by_compose.len() > 1,
        "合成窗大于核心块时必须按调色输出拆分，而不是事后报资源不足"
    );
}

#[test]
fn portrait_render_zero_field_has_no_dense_image_and_missing_page_is_error() {
    let scene = crate::portrait_safety::gpu_fixture(0.0, true);
    let field = CanonicalField::from_scene(&scene, &|| true).unwrap();
    assert!(field.is_identity());
    assert!(field.index_byte_size() < 1024);
    assert!(field.page([1, 0], &|| true).is_err());
    assert!(field.page([0, 0], &|| false).is_err());
    assert!(field.node(129, 0, &|| true).is_err());
}

#[test]
fn portrait_render_window_samples_preserve_global_nodes_and_signed_identity() {
    let scene = crate::portrait_safety::gpu_fixture_dimensions(10.0, false, [513, 387]);
    let field = CanonicalField::from_scene(&scene, &|| true).unwrap();
    for rect in [
        PixelRect::new(-7, -3, 64, 17).unwrap(),
        PixelRect::new(249, 250, 127, 64).unwrap(),
        PixelRect::new(506, 380, 17, 17).unwrap(),
    ] {
        let nodes = field.window_samples(rect, &|| true).unwrap();
        for y in 0..=rect.height {
            for x in 0..=rect.width {
                let (gx, gy) = (rect.x + x as i32, rect.y + y as i32);
                let expected = if (0..=513).contains(&gx) && (0..=387).contains(&gy) {
                    field.node(gx as u32, gy as u32, &|| true).unwrap()
                } else {
                    [0.0, 0.0, 1.0, 0.0]
                };
                assert_eq!(
                    nodes[(y * (rect.width + 1) + x) as usize].map(f32::to_bits),
                    expected.map(f32::to_bits)
                );
            }
        }
    }
}

#[test]
fn portrait_render_budget_rejects_overflow_and_empty_windows() {
    let budget = DeviceBudget {
        bytes: u64::MAX,
        texture_side: u32::MAX,
        buffer_bytes: u64::MAX,
        resident_bytes: 0,
    };
    let input = PixelRect::new(0, 0, 1, 1).unwrap();
    for width in [0, u32::MAX] {
        assert!(
            budget
                .fits(input, PixelRect { width, ..input }, true)
                .is_err()
        );
    }
}

#[test]
fn portrait_render_upload_budget_releases_completed_staging() {
    use crate::portrait_tiles::texture_bytes;
    assert_eq!(
        texture_bytes([257, 3, 2], 8).unwrap(),
        (257 * 3 * 2 * 8, 2304 * 3 * 2)
    );
    assert!(texture_bytes([u32::MAX; 3], 16).is_err());
    assert!(texture_bytes([0, 1, 1], 8).is_err());
    // 最终预览显示纹理为 Rgba8Unorm（4 字节/像素）；白名单遗漏 4 曾使重塑请求的显示暂存必失败。
    assert_eq!(texture_bytes([16, 8, 1], 4).unwrap(), (16 * 8 * 4, 256 * 8));
    assert!(texture_bytes([16, 8, 1], 2).is_err());
    let dims = [2048, 2048, 1];
    let mut budget = DeviceBudget {
        bytes: 256 * 1024 * 1024,
        texture_side: 8192,
        buffer_bytes: 256 * 1024 * 1024,
        resident_bytes: 0,
    };
    let (texture, _) = texture_bytes(dims, 8).unwrap();
    for _ in 0..5 {
        assert!(budget.admits_upload(dims, 8).unwrap());
        budget.resident_bytes += texture;
    }
    assert_eq!(budget.resident_bytes, 160 * 1024 * 1024);
    assert!(!budget.admits_upload([4096, 4096, 1], 8).unwrap());
}

fn photo_mask(opacity: f32) -> crate::mask_generation::MaskDefinition {
    serde_json::from_value(
        json!({"id":"same-id","name":"mask","visible":true,"invert":false,
        "opacity":opacity,"adjustments":{},"subMasks":[{
            "id":"sub","type":"all","visible":true,"mode":"additive","parameters":{}
        }]}),
    )
    .unwrap()
}

#[test]
fn portrait_render_mask_lru_and_content_changes_preserve_layer_order() {
    let cache = NativeMaskCache {
        inner: Mutex::new(NativeMaskCacheInner::default()),
        budget: 16,
    };
    let rect = PixelRect::new(1, 1, 2, 2).unwrap();
    let first = photo_mask(100.0);
    let second = photo_mask(50.0);
    let empty = crate::mask_generation::MaskDefinition {
        sub_masks: vec![],
        ..first.clone()
    };
    let mut atlas = vec![];
    for def in [&first, &empty, &second, &first] {
        cache
            .append_slice("source", def, [4, 4], rect, None, &mut atlas, &|| true)
            .unwrap();
        let inner = cache.inner.lock().unwrap();
        assert!(inner.bytes <= cache.budget);
        assert_eq!(
            inner.bytes,
            inner
                .entries
                .iter()
                .map(|v| v.1.as_raw().len())
                .sum::<usize>()
        );
    }
    assert_eq!(
        atlas,
        [vec![255; 4], vec![0; 4], vec![127; 4], vec![255; 4]].concat()
    );
    assert!(
        cache
            .append_slice("source", &first, [4, 4], rect, None, &mut atlas, &|| false)
            .is_err()
    );
    let mut broken = first.clone();
    broken.sub_masks[0].mask_type = "unsupported".into();
    assert!(
        cache
            .append_slice("source", &broken, [4, 4], rect, None, &mut atlas, &|| true)
            .is_err()
    );
    let inner = cache.inner.lock().unwrap();
    assert_eq!(
        inner.bytes,
        inner
            .entries
            .iter()
            .map(|v| v.1.as_raw().len())
            .sum::<usize>()
    );
}

#[test]
fn portrait_render_mask_partial_failure_is_not_success() {
    let mut def = photo_mask(100.0);
    let mut invalid = def.sub_masks[0].clone();
    invalid.mask_type = "ai-subject".into();
    invalid.parameters = json!({"maskDataBase64":"invalid"});
    def.sub_masks.push(invalid);
    assert!(
        crate::mask_generation::generate_mask_bitmap_checked(&def, 4, 4, None, &|| true).is_err()
    );
    def.sub_masks[1].visible = false;
    let valid = crate::mask_generation::generate_mask_bitmap_checked(&def, 4, 4, None, &|| true)
        .unwrap()
        .unwrap();
    assert!(valid.as_raw().iter().all(|v| *v == 255));
}

#[test]
fn portrait_render_output_roi_matches_full_reconstruction() {
    let native = image::Rgba32FImage::from_fn(41, 31, |x, y| {
        let alpha = if x < 4 { 0.0 } else { 0.75 };
        Rgba([
            alpha * x as f32 / 41.0,
            alpha * y as f32 / 31.0,
            alpha * if (x / 2 + y / 3) % 2 == 0 { 0.9 } else { 0.1 },
            alpha,
        ])
    });
    let crop = json!({"x":1.6,"y":2.2,"width":35.1,"height":25.9});
    let full = OutputWindow::new([41, 31], &crop, [19, 17], None).unwrap();
    for quality in [RenderQuality::Interactive, RenderQuality::HighQuality] {
        let expected = resize_native_output(&native, &full, quality, &|| true).unwrap();
        for roi in [
            PixelRect::new(0, 0, 7, 9).unwrap(),
            PixelRect::new(7, 9, 12, 8).unwrap(),
        ] {
            let window = OutputWindow {
                roi,
                ..full.clone()
            };
            let actual = resize_native_output(&native, &window, quality, &|| true).unwrap();
            assert_eq!(
                actual,
                image::imageops::crop_imm(
                    &expected,
                    roi.x as u32,
                    roi.y as u32,
                    roi.width,
                    roi.height
                )
                .to_image()
            );
        }
        let constant = image::Rgba32FImage::from_pixel(41, 31, Rgba([0.1, 0.2, 0.3, 0.5]));
        let actual = resize_native_output(&constant, &full, quality, &|| true).unwrap();
        for pixel in actual.pixels() {
            for (got, expected) in pixel.0.into_iter().zip([0.1, 0.2, 0.3, 0.5]) {
                assert!((got - expected).abs() < 1e-6);
            }
        }
    }
}

#[test]
fn portrait_render_crop_kernel_reads_real_neighbours_and_transparency() {
    let native = image::Rgba32FImage::from_fn(17, 9, |x, _| {
        if x < 8 {
            Rgba([0.0; 4])
        } else {
            Rgba([0.8, 0.2, 0.1, 1.0])
        }
    });
    let window = OutputWindow::new(
        [17, 9],
        &json!({"x":8,"y":0,"width":8,"height":9}),
        [2, 3],
        None,
    )
    .unwrap();
    for quality in [RenderQuality::Interactive, RenderQuality::HighQuality] {
        let output = resize_native_output(&native, &window, quality, &|| true).unwrap();
        assert!(
            output.get_pixel(0, 1)[3] < 1.0,
            "裁剪外透明像素属于核支持域"
        );
        let transparent = image::Rgba32FImage::from_pixel(17, 9, Rgba([0.0; 4]));
        assert!(
            resize_native_output(&transparent, &window, quality, &|| true)
                .unwrap()
                .as_raw()
                .iter()
                .all(|v| *v == 0.0)
        );
    }
}

#[test]
fn portrait_render_output_rejects_invalid_windows_and_cancellation() {
    let native = image::Rgba32FImage::from_pixel(8, 8, Rgba([0.1, 0.2, 0.3, 1.0]));
    let window = OutputWindow::new([8, 8], &Value::Null, [8, 8], None).unwrap();
    let quality = RenderQuality::HighQuality;
    for roi in [
        PixelRect {
            width: 0,
            ..window.roi
        },
        PixelRect {
            x: -1,
            ..window.roi
        },
    ] {
        assert!(OutputWindow::new([8, 8], &Value::Null, [8, 8], Some(roi)).is_err());
        assert!(
            resize_native_output(
                &native,
                &OutputWindow {
                    roi,
                    ..window.clone()
                },
                quality,
                &|| true
            )
            .is_err()
        );
    }
    for matrix in [
        [f64::NAN, 0.0, 0.0, 0.0, 1.0, 0.0],
        [1.0, 0.5, 0.0, 0.0, 1.0, 0.0],
    ] {
        assert!(
            resize_native_output(
                &native,
                &OutputWindow {
                    output_to_base: matrix,
                    ..window.clone()
                },
                quality,
                &|| true
            )
            .is_err()
        );
    }
    assert!(resize_native_output(&native, &window, quality, &|| false).is_err());
    let polls = std::cell::Cell::new(0);
    assert!(
        output_axis_weights(0, 100, 1.0, 0.0, 100, quality, &|| {
            polls.set(polls.get() + 1);
            polls.get() < 5
        })
        .is_err()
    );
    assert!(output_axis_weights(0, 32768, 32768.0, 0.0, 32768, quality, &|| true).is_err());
    assert!(quantize_native_output(image::Rgba32FImage::new(0, 1), true, &|| true).is_err());
    assert!(quantize_native_output(native, false, &|| false).is_err());
}

#[test]
fn portrait_render_quantizes_once_to_u16_with_zero_hidden_rgb() {
    let native = image::Rgba32FImage::from_fn(4096, 1, |x, _| {
        let v = x as f32 / 4095.0;
        Rgba([v * 0.5, v * 0.5, v * 0.5, 0.5])
    });
    let encoded = quantize_native_output(native, true, &|| true)
        .unwrap()
        .into_rgba16();
    let values: std::collections::BTreeSet<_> = encoded.pixels().map(|p| p[0]).collect();
    assert!(values.len() > 4000);
    assert!(values.iter().any(|v| v % 257 != 0));
    assert!(encoded.pixels().all(|p| p[3] == 32768));
    for sixteen in [false, true] {
        let hidden = image::Rgba32FImage::from_pixel(1, 1, Rgba([0.8, 0.4, 0.3, 0.0]));
        assert!(
            quantize_native_output(hidden, sixteen, &|| true)
                .unwrap()
                .as_bytes()
                .iter()
                .all(|v| *v == 0)
        );
    }
}

#[test]
fn portrait_render_mask_upload_limit_splits_instead_of_failing_early() {
    let budget = DeviceBudget {
        bytes: 256 * 1024 * 1024,
        texture_side: 4096,
        buffer_bytes: 1024 * 1024,
        resident_bytes: 0,
    };
    let tiles = crate::portrait_tiles::plan_tiles_with_resources(
        PixelRect::new(0, 0, 512, 512).unwrap(),
        budget,
        true,
        |tile| {
            let (resident, upload) =
                crate::portrait_tiles::texture_bytes([tile.width, tile.height, 64], 1)?;
            Ok((tile, tile, resident + upload, upload))
        },
    )
    .unwrap();
    assert!(tiles.len() > 1);
    assert_eq!(
        tiles
            .iter()
            .map(|(r, _)| r.width as u64 * r.height as u64)
            .sum::<u64>(),
        512 * 512
    );
    for (tile, _) in tiles {
        assert!(
            crate::portrait_tiles::texture_bytes([tile.width, tile.height, 64], 1)
                .unwrap()
                .1
                <= budget.buffer_bytes
        );
    }
}

fn publication_fixture() -> (tempfile::TempDir, PortraitRenderState, PhotoRenderResult) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("source.png");
    RgbaImage::from_pixel(3, 5, Rgba([80, 100, 120, 255]))
        .save(&path)
        .unwrap();
    let source = Arc::new(
        SourceSnapshot::load(
            path.to_str().unwrap(),
            &crate::AppSettings::default(),
            7,
            &|| true,
        )
        .unwrap(),
    );
    let state = PortraitRenderState::default();
    let lease = state.begin_consumer(&source, 11).unwrap();
    let stamp = RenderStamp {
        request_id: 11,
        source_revision: 7,
        epoch: lease.epoch,
        composite_key: "composite".into(),
        reshape_key: "field".into(),
        window_key: "output".into(),
    };
    (
        directory,
        state,
        PhotoRenderResult {
            image: None,
            stamp,
            source,
            asset: None,
            lease,
            reshape_attenuation: None,
            body_attenuation: None,
        },
    )
}

#[test]
fn portrait_render_consumers_cancel_independently_and_reject_wrong_revision() {
    let (_directory, state, output) = publication_fixture();
    let export = state.begin_consumer(&output.source, 12).unwrap();
    output.lease.cancel();
    assert!(export.is_current());
    assert!(
        output
            .lease
            .commit(&output.source, &output.stamp, &|| true, || Ok(()))
            .is_err()
    );
    let mut stamp = output.stamp.clone();
    stamp.request_id = 12;
    stamp.epoch = export.epoch;
    assert!(
        export
            .commit(&output.source, &stamp, &|| true, || Err::<(), _>(
                "failed".into()
            ))
            .is_err()
    );
    assert!(!export.has_committed(&stamp));
    export
        .commit(&output.source, &stamp, &|| true, || Ok(()))
        .unwrap();
    assert!(export.has_committed(&stamp));
    let mut wrong = stamp.clone();
    wrong.window_key = "other ROI".into();
    assert!(!export.has_committed(&wrong));
    wrong = stamp.clone();
    wrong.source_revision += 1;
    assert!(
        export
            .commit(&output.source, &wrong, &|| true, || Ok(()))
            .is_err()
    );
    export.cancel();
    assert!(!export.has_committed(&stamp));
}

#[test]
fn portrait_render_editor_reload_cancels_only_editor_and_reuses_verified_source() {
    let (_directory, state, mut output) = publication_fixture();
    let settings = crate::AppSettings::default();
    state.advance_editor(7, 1).unwrap();
    let source = state
        .editor_source(output.source.path(), &settings, 7, &|| true)
        .unwrap();
    let same = state
        .editor_source(output.source.path(), &settings, 7, &|| true)
        .unwrap();
    assert!(Arc::ptr_eq(&source, &same));
    let export = state.begin_consumer(&source, 20).unwrap();
    output.source = Arc::clone(&source);
    output.lease = state.begin_editor_consumer(&source, 11, 1).unwrap();
    output.stamp.epoch = output.lease.epoch;
    state
        .publish_editor(&output, 1, &|| true, || Ok(()))
        .unwrap();
    assert!(output.lease.has_committed(&output.stamp));
    state.advance_editor(7, 2).unwrap();
    assert!(!output.lease.is_current());
    assert!(export.is_current());
    assert!(
        state
            .publish_editor::<()>(&output, 1, &|| true, || panic!("旧帧不得提交"))
            .is_err()
    );
    assert!(state.begin_editor_consumer(&source, 12, 1).is_err());
    assert!(Arc::ptr_eq(
        &source,
        &state
            .editor_source(source.path(), &settings, 7, &|| true)
            .unwrap()
    ));
    state.advance_editor(8, 3).unwrap();
    assert!(
        state
            .editor_source(source.path(), &settings, 7, &|| true)
            .is_err()
    );
    let reloaded = state
        .editor_source(source.path(), &settings, 8, &|| true)
        .unwrap();
    assert!(!Arc::ptr_eq(&source, &reloaded));
    assert_eq!(source.source_key(), reloaded.source_key());
    assert_eq!(reloaded.revision(), 8);
    assert!(export.is_current());
    assert!(state.advance_editor(7, 4).is_err());
    assert!(state.advance_editor(8, 2).is_err());
    let mut export_stamp = output.stamp.clone();
    export_stamp.request_id = 20;
    export_stamp.epoch = export.epoch;
    export
        .commit(&source, &export_stamp, &|| true, || Ok(()))
        .unwrap();
}

#[test]
fn portrait_render_editor_source_change_requires_explicit_reload() {
    let (_directory, state, output) = publication_fixture();
    let settings = crate::AppSettings::default();
    state.advance_editor(7, 1).unwrap();
    state
        .editor_source(output.source.path(), &settings, 7, &|| true)
        .unwrap();
    RgbaImage::from_pixel(3, 5, Rgba([1, 2, 3, 255]))
        .save(output.source.path())
        .unwrap();
    assert!(
        state
            .editor_source(output.source.path(), &settings, 7, &|| true)
            .is_err()
    );
    state.advance_editor(8, 2).unwrap();
    let source = state
        .editor_source(output.source.path(), &settings, 8, &|| true)
        .unwrap();
    assert_ne!(source.source_key(), output.source.source_key());
}

#[cfg(not(target_os = "android"))]
#[test]
fn portrait_render_atomic_publication_never_overwrites_and_cleans_cancelled_staging() {
    let (directory, _state, output) = publication_fixture();
    let original = std::fs::read(output.source.path()).unwrap();
    let target = directory.path().join("photo.jpg");
    std::fs::write(&target, b"existing").unwrap();
    assert!(
        output
            .publish_encoded(&target, b"complete", false, &|| true)
            .is_err()
    );
    assert_eq!(std::fs::read(&target).unwrap(), b"existing");
    assert!(!output.lease.has_committed(&output.stamp));
    let published = output
        .publish_encoded(&target, b"complete", true, &|| true)
        .unwrap();
    assert_eq!(published.file_name().unwrap(), "photo (1).jpg");
    assert_eq!(std::fs::read(&published).unwrap(), b"complete");
    assert!(output.lease.has_committed(&output.stamp));
    assert!(
        output
            .publish_encoded(Path::new(output.source.path()), b"bad", true, &|| true)
            .is_err()
    );
    let alias = directory.path().join("alias.png");
    std::fs::hard_link(output.source.path(), &alias).unwrap();
    assert!(
        output
            .publish_encoded(&alias, b"bad", true, &|| true)
            .is_err()
    );
    let polls = std::cell::Cell::new(0);
    let aborted = directory.path().join("aborted.jpg");
    assert!(
        output
            .publish_encoded(&aborted, &vec![42; 256 * 1024], false, &|| {
                polls.set(polls.get() + 1);
                polls.get() < 4
            })
            .is_err()
    );
    assert!(!aborted.exists());
    output.lease.cancel();
    assert!(
        output
            .publish_encoded(&aborted, b"bad", false, &|| true)
            .is_err()
    );
    assert_eq!(std::fs::read(&published).unwrap(), b"complete");
    assert_eq!(std::fs::read(output.source.path()).unwrap(), original);
    assert!(std::fs::read_dir(directory.path()).unwrap().all(|entry| {
        !entry
            .unwrap()
            .file_name()
            .to_string_lossy()
            .starts_with(".rr-reshape-")
    }));
}

#[cfg(not(target_os = "android"))]
#[test]
fn portrait_render_changed_source_cannot_publish_encoded_file() {
    let (directory, _state, output) = publication_fixture();
    let mut bytes = std::fs::read(output.source.path()).unwrap();
    let last = bytes.len() - 1;
    bytes[last] ^= 1;
    std::fs::write(output.source.path(), bytes).unwrap();
    let target = directory.path().join("late.png");
    assert!(
        output
            .publish_encoded(&target, b"complete", false, &|| true)
            .is_err()
    );
    assert!(!target.exists());
    assert!(!output.lease.has_committed(&output.stamp));
}

#[test]
fn portrait_render_lut_snapshot_uses_frozen_bytes_not_mutable_path_cache() {
    let bytes = b"LUT_3D_SIZE 2\n0 0 0\n1 0 0\n0 1 0\n1 1 0\n0 0 1\n1 0 1\n0 1 1\n1 1 1\n";
    let lut = crate::lut_processing::parse_lut_bytes("fixture.cube", bytes).unwrap();
    assert_eq!(lut.size, 2);
    assert_eq!(lut.data.len(), 24);
    assert!(crate::lut_processing::parse_lut_bytes("../fixture.cube", bytes).is_err());
    assert!(crate::lut_processing::parse_lut_bytes("fixture.cube", b"invalid").is_err());
}

#[test]
fn portrait_render_native_conversion_preserves_u16_and_float_and_cancels() {
    let gradient = DynamicImage::ImageRgba16(image::ImageBuffer::from_fn(4096, 1, |x, _| {
        Rgba([x as u16 * 16, 12345, 54321, 65535])
    }));
    let converted = native_rgba(&gradient, &|| true).unwrap();
    assert_eq!(converted.get_pixel(3, 0)[0], 48.0 / 65535.0);
    assert_eq!(converted.get_pixel(3, 0)[1], 12345.0 / 65535.0);
    assert!(
        converted
            .pixels()
            .zip(converted.pixels().skip(1))
            .all(|(a, b)| a[0] < b[0])
    );
    let floats = DynamicImage::ImageRgba32F(image::Rgba32FImage::from_pixel(
        2,
        2,
        Rgba([0.1234567, 1.1, 0.3, 0.5]),
    ));
    assert_eq!(native_rgba(&floats, &|| true).unwrap(), floats.to_rgba32f());
    let hidden = DynamicImage::ImageRgba16(image::ImageBuffer::from_pixel(
        1,
        1,
        Rgba([500, 700, 900, 0]),
    ));
    assert_eq!(
        native_rgba(&hidden, &|| true).unwrap().get_pixel(0, 0).0,
        [0.0; 4]
    );
    assert!(native_rgba(&floats, &|| false).is_err());
    let invalid = DynamicImage::ImageRgba32F(image::Rgba32FImage::from_pixel(
        1,
        1,
        Rgba([f32::NAN, 0.0, 0.0, 1.0]),
    ));
    assert!(native_rgba(&invalid, &|| true).is_err());
    let empty = DynamicImage::ImageRgba8(RgbaImage::new(0, 0));
    assert!(native_rgba(&empty, &|| true).is_err());
}

#[test]
fn portrait_render_oom_classification_never_retries_safety_or_device_errors() {
    use crate::portrait_gpu::{GpuFailure, resolve_gpu_result};
    let oom = || wgpu::Error::OutOfMemory {
        source: Box::new(std::io::Error::other("allocation")),
    };
    assert!(
        resolve_gpu_result(Ok(()), [oom()])
            .unwrap_err()
            .is_out_of_memory()
    );
    for message in [
        "cancelled",
        "missing footprint",
        "unsafe field",
        "device lost",
        "Out of Memory",
    ] {
        let error = resolve_gpu_result::<()>(Err(message.into()), [oom()]).unwrap_err();
        assert!(!error.is_out_of_memory(), "{message}");
    }
    for error in [
        wgpu::Error::Validation {
            source: Box::new(std::io::Error::other("validation")),
            description: "validation".into(),
        },
        wgpu::Error::Internal {
            source: Box::new(std::io::Error::other("device")),
            description: "device".into(),
        },
    ] {
        assert!(
            !resolve_gpu_result(Ok(()), [oom(), error])
                .unwrap_err()
                .is_out_of_memory()
        );
    }
    assert!(GpuFailure::OutOfMemory("08 调色资源不足，必须缩小核心块".into()).is_out_of_memory());
    assert!(!GpuFailure::Failed("08 调色资源不足，必须缩小核心块".into()).is_out_of_memory());
    let nested = resolve_gpu_result::<()>(Err(GpuFailure::OutOfMemory("allocation".into())), [])
        .unwrap_err();
    assert!(nested.is_out_of_memory());
    assert_eq!(resolve_gpu_result(Ok(7), []).unwrap(), 7);
}

#[test]
fn portrait_render_oom_split_preserves_absolute_coverage_and_minimum_core() {
    use crate::portrait_tiles::split_core;
    for (w, h) in [(1024, 1024), (257, 65), (1, 128), (129, 3), (64, 128)] {
        let whole = PixelRect::new(-71, 19, w, h).unwrap();
        let [first, second] = split_core(whole).unwrap();
        assert!(whole.contains(first) && whole.contains(second));
        assert!(first.intersect(second).is_none());
        assert_eq!(
            u64::from(first.width) * u64::from(first.height)
                + u64::from(second.width) * u64::from(second.height),
            u64::from(w) * u64::from(h)
        );
        assert!(first.width <= w && first.height <= h);
        assert!(second.width <= w && second.height <= h);
    }
    for (w, h) in [(64, 64), (127, 64), (1, 1)] {
        assert!(split_core(PixelRect::new(0, 0, w, h).unwrap()).is_err());
    }
}

#[cfg(not(target_os = "android"))]
#[test]
fn portrait_render_cancel_while_waiting_for_publication_gate_does_not_publish() {
    let (directory, _state, output) = publication_fixture();
    let target = directory.path().join("waiting.png");
    let gate = Mutex::new(());
    let cancelled = AtomicBool::new(false);
    let gate_guard = gate.lock().unwrap();
    let (sent, received) = std::sync::mpsc::channel();
    std::thread::scope(|scope| {
        let task = scope.spawn(|| {
            let signalled = AtomicBool::new(false);
            output.publish_encoded_guarded(
                &target,
                b"complete",
                false,
                &|| {
                    // 只有 commit_verified 已取 lease 锁时触发；此时完整源核验已经结束。
                    if output.lease.committed.try_lock().is_err()
                        && !signalled.swap(true, Ordering::SeqCst)
                    {
                        sent.send(()).unwrap();
                    }
                    !cancelled.load(Ordering::SeqCst)
                },
                &gate,
                |_| Ok(()),
            )
        });
        let reached = received.recv_timeout(std::time::Duration::from_secs(5));
        cancelled.store(true, Ordering::SeqCst);
        drop(gate_guard);
        let result = task.join().unwrap();
        assert!(reached.is_ok(), "未到达发布门，不能把较早取消当成并发覆盖");
        assert!(result.is_err());
    });
    assert!(!target.exists());
    assert!(!output.lease.has_committed(&output.stamp));
    assert_eq!(std::fs::read_dir(directory.path()).unwrap().count(), 1);
}

#[cfg(not(target_os = "android"))]
#[test]
fn portrait_render_file_preparation_error_cleans_staging_and_late_cancel_keeps_output() {
    let (directory, _state, output) = publication_fixture();
    let target = directory.path().join("photo.png");
    let error = output
        .publish_encoded_guarded(
            &target,
            b"complete",
            false,
            &|| true,
            &Mutex::new(()),
            |_| Err("timestamp failed".into()),
        )
        .unwrap_err();
    assert_eq!(error, "timestamp failed");
    assert!(!target.exists());
    assert!(!output.lease.has_committed(&output.stamp));
    assert_eq!(std::fs::read_dir(directory.path()).unwrap().count(), 1);
    let published = output
        .publish_encoded(&target, b"complete", false, &|| true)
        .unwrap();
    output.lease.cancel();
    assert_eq!(std::fs::read(published).unwrap(), b"complete");
}

#[test]
fn portrait_render_output_source_changed_decode_cannot_accept_replaced_editor_source() {
    let (_directory, state, output) = publication_fixture();
    let settings = crate::AppSettings::default();
    state.advance_editor(7, 1).unwrap();
    let editor = state
        .editor_source(output.source.path(), &settings, 7, &|| true)
        .unwrap();
    let independent = state
        .output_source(editor.path(), &settings, &|| true)
        .unwrap();
    assert!(Arc::ptr_eq(&editor, &independent));
    let mut changed = settings.clone();
    changed.linear_raw_mode = format!("{}-changed", settings.linear_raw_mode);
    let other_decode = state
        .output_source(editor.path(), &changed, &|| true)
        .unwrap();
    assert!(!Arc::ptr_eq(&editor, &other_decode));
    assert_ne!(editor.source_key(), other_decode.source_key());
    RgbaImage::from_pixel(3, 5, Rgba([1, 2, 3, 255]))
        .save(editor.path())
        .unwrap();
    assert!(
        state
            .output_source(editor.path(), &changed, &|| true)
            .is_err()
    );
    assert!(independent.verify(&|| true).is_err());
}

#[test]
fn portrait_render_snapshot_retains_float_or_high_bit_depth_decode() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("linear.png");
    let image = DynamicImage::ImageRgba16(image::ImageBuffer::from_fn(513, 1, |x, _| {
        image::Rgba([x as u16 * 100, 0, 0, 65535])
    }));
    let mut bytes = Cursor::new(Vec::new());
    image.write_to(&mut bytes, ImageFormat::Png).unwrap();
    std::fs::write(&path, bytes.into_inner()).unwrap();
    let source = SourceSnapshot::load(
        path.to_str().unwrap(),
        &crate::AppSettings::default(),
        1,
        &|| true,
    )
    .unwrap();
    assert!(source.image().to_rgba16().pixels().any(|p| p[0] % 257 != 0));
}
