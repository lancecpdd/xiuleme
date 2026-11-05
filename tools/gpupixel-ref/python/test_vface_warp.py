#!/usr/bin/env python3
"""vface_warp 几何单测（无需 MediaPipe / GPUPixel）。"""

from __future__ import annotations

import numpy as np

from vface_warp import (
    JAW_SLIM_PAIRS,
    VFaceParams,
    apply_vface,
    curve_warp,
    jaw_lateral_narrow,
    warp_uv,
)


def _synth_landmarks() -> np.ndarray:
    p = np.full((478, 2), 500.0, dtype=np.float64)
    p[234] = [350, 400]
    p[454] = [650, 400]
    p[6] = [500, 380]
    p[1] = [500, 420]
    p[98] = [470, 480]
    p[327] = [530, 480]
    p[13] = [500, 520]
    p[132] = [380, 450]
    p[361] = [620, 450]
    p[58] = [400, 500]
    p[288] = [600, 500]
    p[136] = [410, 530]
    p[365] = [590, 530]
    p[172] = [420, 560]
    p[397] = [580, 560]
    p[150] = [450, 580]
    p[379] = [550, 580]
    p[152] = [500, 600]
    return p


def test_curve_warp_linear():
    ox, oy, tx, ty = 0.2, 0.5, 0.5, 0.5
    cx, cy = curve_warp(0.2, 0.5, ox, oy, tx, ty, 0.05, 1.0)
    assert abs(cx - (0.2 - 0.05 * 0.3)) < 1e-9
    assert abs(cy - 0.5) < 1e-9
    farx, fary = curve_warp(0.9, 0.5, ox, oy, tx, ty, 0.05, 1.0)
    assert abs(farx - 0.9) < 1e-9


def test_zero_strength_identity():
    rgba = np.random.randint(0, 255, (64, 64, 4), dtype=np.uint8)
    rgba[:, :, 3] = 255
    lm = _synth_landmarks() * (64 / 1000.0)
    out = apply_vface(rgba, lm, VFaceParams(strength=0.0), grid_step=1)
    assert np.array_equal(out, rgba)


def test_jaw_moves_more_than_midface_uv():
    lm = _synth_landmarks()
    pts = lm / 1000.0
    params = VFaceParams(strength=1.0, slim_gain=0.055, jaw_lateral_gain=1.2)
    cheek_u, cheek_v = pts[132]
    jaw_u, jaw_v = pts[172]
    c2 = warp_uv(cheek_u, cheek_v, pts, 1.0, params)
    j2 = warp_uv(jaw_u, jaw_v, pts, 1.0, params)
    cheek_d = np.hypot(c2[0] - cheek_u, c2[1] - cheek_v)
    jaw_d = np.hypot(j2[0] - jaw_u, j2[1] - jaw_v)
    assert jaw_d > cheek_d, f"jaw={jaw_d} cheek={cheek_d}"
    assert jaw_d > 1e-4


def test_pairs_cover_jaw():
    origins = {o for o, _, _ in JAW_SLIM_PAIRS}
    assert 172 in origins and 397 in origins and 152 in origins
    assert 234 not in origins  # 上颊留给 GPUPixel 原版九对


def test_lateral_only_horizontal():
    lm = _synth_landmarks() / 1000.0
    u, v = float(lm[172, 0]), float(lm[172, 1])
    u2, v2 = jaw_lateral_narrow(u, v, lm, slim_delta=0.05, jaw_lateral_gain=1.2)
    assert abs(v2 - v) < 1e-12
    assert abs(u2 - u) > 1e-6


if __name__ == "__main__":
    test_curve_warp_linear()
    test_zero_strength_identity()
    test_jaw_moves_more_than_midface_uv()
    test_pairs_cover_jaw()
    test_lateral_only_horizontal()
    print("ok")
