//! GPU／UV 场测试夹具；网格 prepare_scene 单测已移除。

use super::*;
use crate::face_reshape::{GeometrySolution, Strengths};
use crate::portrait_adapters::ProtectionMaps;
use crate::portrait_analysis::AnalyzedFace;
use crate::portrait_coordinates::CoordinateMap;
use crate::portrait_reshape::{ReshapeFaceSummary, ReshapeSourceAnchor};
use crate::portrait_uv_warp::{UvFaceWarp, UvWarpStrengths};
use image::Rgb;
use serde_json::json;
use std::sync::Arc;

pub(crate) fn map(p: [f32; 4]) -> ProtectionMaps {
    ProtectionMaps {
        width: 256,
        height: 256,
        probabilities: vec![p; 65536],
    }
}

pub(crate) fn face(id: &str, center: [f64; 2], width: f64) -> AnalyzedFace {
    let mut roi_points = vec![[0.0_f32, 0.0, 0.0]; 478];
    for (i, p) in roi_points.iter_mut().enumerate() {
        let t = i as f32 / 478.0;
        p[0] = (t - 0.5) * 0.8;
        p[1] = (t - 0.5) * 0.8;
    }
    AnalyzedFace {
        summary: ReshapeFaceSummary {
            face_id: id.into(),
            source_anchor: ReshapeSourceAnchor {
                source_base_hash: "a".repeat(64),
                bounds: [0.1, 0.1, 0.4, 0.5],
            },
            processable: true,
            reason: None,
        },
        roi_to_base: Affine::new([width, 0.0, center[0], 0.0, width, center[1]]),
        roi_points,
        source_points: vec![[0.5, 0.5]; 478],
        detection_confidence: 0.99,
        landmark_confidence: 0.99,
        refined: true,
        protection: None,
    }
}

pub(crate) fn analysis(faces: Vec<AnalyzedFace>) -> ReshapeAnalysis {
    ReshapeAnalysis {
        key: "analysis-synthetic-06".into(),
        source_base_hash: "a".repeat(64),
        model_hashes: BTreeMap::new(),
        coordinates: CoordinateMap::new([1024, 1024], &json!({})).unwrap(),
        coarse_dimensions: [1024, 1024],
        faces,
        protection: Some(map([0.0, 0.0, 1.0, 0.0])),
        rejected_faces: 0,
    }
}

pub(crate) fn request(a: &ReshapeAnalysis) -> SceneRequest {
    SceneRequest {
        analysis_key: a.key.clone(),
        faces: a
            .faces
            .iter()
            .map(|f| FaceRequest {
                face_id: f.summary.face_id.clone(),
                strengths: Strengths([100.0; 4]),
                extended: crate::face_reshape::ExtendedStrengths::default(),
                occlusion: OcclusionEvidence::Unknown,
            })
            .collect(),
        strokes: Vec::new(),
    }
}

pub(crate) fn image() -> Rgb32FImage {
    Rgb32FImage::from_pixel(1024, 1024, Rgb([0.5; 3]))
}

/// GPU fixture for continuous-field tests（UV 稠密网格，不经 prepare_scene）。
pub(crate) fn gpu_fixture(strength: f64, no_face: bool) -> SafeScene {
    gpu_fixture_dimensions(strength, no_face, [128, 128])
}

pub(crate) fn gpu_fixture_dimensions(strength: f64, no_face: bool, dims: [u32; 2]) -> SafeScene {
    if no_face {
        return SafeScene::from_uv_warps(
            "analysis-synthetic-06".into(),
            dims,
            vec![],
            &|| true,
        )
        .unwrap();
    }
    let w = dims[0] as f64;
    let h = dims[1] as f64;
    let cx = w * 0.5;
    let cy = h * 0.4;
    let source = vec![
        [cx - 20.0, cy - 20.0],
        [cx + 20.0, cy - 20.0],
        [cx + 20.0, cy + 20.0],
        [cx - 20.0, cy + 20.0],
    ];
    let shrink = (strength / 100.0) * 4.0;
    let target = vec![
        [cx - 20.0 + shrink, cy - 20.0 + shrink],
        [cx + 20.0 - shrink, cy - 20.0 + shrink],
        [cx + 20.0 - shrink, cy + 20.0 - shrink],
        [cx - 20.0 + shrink, cy + 20.0 - shrink],
    ];
    let tris = Arc::new(vec![[0usize, 1, 2], [0, 2, 3]]);
    let identity = strength <= 0.0;
    let geometry = GeometrySolution::from_dense_pixel_mesh(
        source,
        target,
        tris,
        40.0,
        Strengths([strength, 0.0, 0.0, 0.0]),
        [strength / 100.0, 0.0, 0.0, 0.0],
        Strengths([strength, 0.0, 0.0, 0.0]),
        identity,
    )
    .unwrap();
    let strengths = UvWarpStrengths::from_ui(strength.clamp(0.0, 100.0), 0.0).unwrap();
    SafeScene::from_uv_warps(
        "analysis-synthetic-06".into(),
        dims,
        vec![UvFaceWarp {
            face_id: "gpu-face".into(),
            strengths,
            geometry,
        }],
        &|| true,
    )
    .unwrap()
}

#[test]
fn prepare_scene_rejects_mesh_requests() {
    let a = analysis(vec![face("f", [512.0, 384.0], 80.0)]);
    let err = prepare_scene(&a, &image(), &request(&a), &|| true);
    assert!(err.is_err());
}
