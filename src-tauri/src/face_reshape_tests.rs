//! 网格 ARAP／3DMM 单测已随网格重塑移除；UV 几何见 portrait_uv_warp。

#[test]
fn mesh_reshape_removed_keeps_geometry_solution_api() {
    use super::*;
    let source = vec![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];
    let target = source.clone();
    let tris = std::sync::Arc::new(vec![[0usize, 1, 2]]);
    let sol = GeometrySolution::from_dense_pixel_mesh(
        source,
        target,
        tris,
        1.0,
        Strengths::default(),
        [0.0; 4],
        Strengths::default(),
        true,
    )
    .unwrap();
    assert!(sol.is_identity());
}
