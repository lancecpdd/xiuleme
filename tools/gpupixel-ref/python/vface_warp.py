"""V 脸 / 下颌线强化：纯 NumPy 逆采样 warp（实验用，不接入 Tauri）。

在 GPUPixel 原版 thinFace 九对之外，对下半脸加密锚点并叠加水平侧向收束带，
目标是更流畅清晰的下颌轮廓与 V 线。参数刻意与旧商业 UV 试验对齐，便于目检后再移植。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

# MediaPipe 478：下颌／颊侧强化对（origin → target）。不含上颊 234/454，避免整脸过度收窄。
JAW_SLIM_PAIRS: list[tuple[int, int, float]] = [
    (132, 1, 0.90),
    (361, 1, 0.90),
    (58, 98, 1.25),
    (288, 327, 1.25),
    (172, 13, 1.55),
    (397, 13, 1.55),
    (152, 13, 1.15),
    (136, 98, 1.35),
    (365, 327, 1.35),
    (150, 13, 1.35),
    (379, 13, 1.35),
]

# 满档（strength=1）时 curveWarp delta 基线；侧带峰值 = delta * jaw_lateral_gain。
DEFAULT_SLIM_GAIN = 0.055
DEFAULT_JAW_LATERAL_GAIN = 1.20


@dataclass
class VFaceParams:
    """strength ∈ [0, 1]；其余为算法增益。"""

    strength: float = 0.0
    slim_gain: float = DEFAULT_SLIM_GAIN
    jaw_lateral_gain: float = DEFAULT_JAW_LATERAL_GAIN

    def clamped(self) -> "VFaceParams":
        return VFaceParams(
            strength=float(np.clip(self.strength, 0.0, 1.0)),
            slim_gain=float(max(0.0, self.slim_gain)),
            jaw_lateral_gain=float(max(0.0, self.jaw_lateral_gain)),
        )

    @property
    def slim_delta(self) -> float:
        p = self.clamped()
        # smoothstep 满档映射，与产品滑块习惯接近
        t = p.strength
        s = t * t * (3.0 - 2.0 * t)
        return s * p.slim_gain


def _smooth(t: float) -> float:
    t = float(np.clip(t, 0.0, 1.0))
    return t * t * (3.0 - 2.0 * t)


def _bump(t: float) -> float:
    return _smooth(1.0 - abs(t))


def _dist_aspect(ax: float, ay: float, bx: float, by: float, aspect: float) -> float:
    dx = ax - bx
    dy = (ay - by) / aspect
    return float(np.hypot(dx, dy))


def curve_warp(
    cx: float,
    cy: float,
    ox: float,
    oy: float,
    tx: float,
    ty: float,
    delta: float,
    aspect: float,
) -> tuple[float, float]:
    """原版 GPUPixel 线性 curveWarp（在 UV 空间）。"""
    radius = _dist_aspect(tx, ty, ox, oy, aspect)
    if not np.isfinite(radius) or radius <= 1e-8:
        return cx, cy
    ratio = float(np.clip(1.0 - _dist_aspect(cx, cy, ox, oy, aspect) / radius, 0.0, 1.0))
    return cx - (tx - ox) * delta * ratio, cy - (ty - oy) * delta * ratio


def jaw_lateral_narrow(
    cx: float,
    cy: float,
    points: np.ndarray,
    slim_delta: float,
    jaw_lateral_gain: float,
) -> tuple[float, float]:
    """下颌高度侧带：只水平内收，强化削角／V 线。points 为 Nx2 归一化 UV。"""
    left = points[172]
    right = points[397]
    chin = points[152]
    mid_x = 0.5 * (left[0] + right[0])
    jaw_y = 0.5 * (left[1] + right[1])
    half_w = max(abs(right[0] - left[0]) * 0.5, 1e-8)
    jaw_span = max(abs(chin[1] - jaw_y), half_w * 0.35)
    dx = cx - mid_x
    side = _smooth((abs(dx) / half_w - 0.22) / 0.55)
    vert = _bump((cy - jaw_y) / jaw_span)
    amount = slim_delta * jaw_lateral_gain * side * vert
    if amount <= 1e-12 or not np.isfinite(dx):
        return cx, cy
    return cx + np.sign(dx) * amount, cy


def warp_uv(
    u: float,
    v: float,
    points_uv: np.ndarray,
    aspect: float,
    params: VFaceParams,
) -> tuple[float, float]:
    """对单个归一化 UV 做 V 脸逆采样位移。points_uv: (478, 2) 归一化坐标。"""
    p = params.clamped()
    delta = p.slim_delta
    if delta <= 0.0:
        return u, v
    cu, cv = u, v
    for oi, ti, weight in JAW_SLIM_PAIRS:
        o = points_uv[oi]
        t = points_uv[ti]
        cu, cv = curve_warp(cu, cv, o[0], o[1], t[0], t[1], delta * weight, aspect)
    cu, cv = jaw_lateral_narrow(cu, cv, points_uv, delta, p.jaw_lateral_gain)
    return cu, cv


def _bilinear_rgba(image: np.ndarray, x: float, y: float) -> np.ndarray:
    """image HxWx4 uint8；x/y 像素坐标。"""
    h, w = image.shape[:2]
    x = float(np.clip(x, 0.0, w - 1.001))
    y = float(np.clip(y, 0.0, h - 1.001))
    x0 = int(np.floor(x))
    y0 = int(np.floor(y))
    x1 = min(x0 + 1, w - 1)
    y1 = min(y0 + 1, h - 1)
    fx = x - x0
    fy = y - y0
    c00 = image[y0, x0].astype(np.float32)
    c10 = image[y0, x1].astype(np.float32)
    c01 = image[y1, x0].astype(np.float32)
    c11 = image[y1, x1].astype(np.float32)
    top = c00 * (1 - fx) + c10 * fx
    bot = c01 * (1 - fx) + c11 * fx
    return np.clip(top * (1 - fy) + bot * fy, 0, 255).astype(np.uint8)


def apply_vface(
    rgba: np.ndarray,
    landmarks_px: np.ndarray,
    params: VFaceParams,
    *,
    grid_step: int = 1,
) -> np.ndarray:
    """对整图做 V 脸逆采样。

    landmarks_px: (478, 2) 像素坐标（与图像同空间）。
    grid_step>1 时先在稀疏网格求位移再双线性放大（加速预览）。
    """
    if rgba.dtype != np.uint8 or rgba.ndim != 3 or rgba.shape[2] != 4:
        raise ValueError("rgba must be HxWx4 uint8")
    if landmarks_px.shape != (478, 2):
        raise ValueError("landmarks_px must be (478, 2)")
    p = params.clamped()
    if p.strength <= 0.0:
        return rgba.copy()

    h, w = rgba.shape[:2]
    aspect = w / float(h)
    points_uv = landmarks_px.astype(np.float64).copy()
    points_uv[:, 0] /= w
    points_uv[:, 1] /= h

    if grid_step <= 1:
        out = np.empty_like(rgba)
        for y in range(h):
            v = (y + 0.5) / h
            for x in range(w):
                u = (x + 0.5) / w
                su, sv = warp_uv(u, v, points_uv, aspect, p)
                out[y, x] = _bilinear_rgba(rgba, su * w - 0.5, sv * h - 0.5)
        return out

    # 稀疏网格加速：对控制点求源坐标，再逐像素双线性插值位移场
    ys = np.arange(0, h, grid_step, dtype=np.int32)
    xs = np.arange(0, w, grid_step, dtype=np.int32)
    if ys[-1] != h - 1:
        ys = np.append(ys, h - 1)
    if xs[-1] != w - 1:
        xs = np.append(xs, w - 1)
    src_x = np.zeros((len(ys), len(xs)), dtype=np.float32)
    src_y = np.zeros((len(ys), len(xs)), dtype=np.float32)
    for ji, y in enumerate(ys):
        v = (y + 0.5) / h
        for ii, x in enumerate(xs):
            u = (x + 0.5) / w
            su, sv = warp_uv(u, v, points_uv, aspect, p)
            src_x[ji, ii] = su * w - 0.5
            src_y[ji, ii] = sv * h - 0.5

    out = np.empty_like(rgba)
    # 预计算每行：对每个 x 在 xs 上插值
    for y in range(h):
        # 找 ys 区间
        j = int(np.searchsorted(ys, y, side="right") - 1)
        j = max(0, min(j, len(ys) - 2))
        y0, y1 = int(ys[j]), int(ys[j + 1])
        ty = 0.0 if y1 == y0 else (y - y0) / (y1 - y0)
        row0_x = src_x[j]
        row1_x = src_x[j + 1]
        row0_y = src_y[j]
        row1_y = src_y[j + 1]
        for x in range(w):
            i = int(np.searchsorted(xs, x, side="right") - 1)
            i = max(0, min(i, len(xs) - 2))
            x0, x1 = int(xs[i]), int(xs[i + 1])
            tx = 0.0 if x1 == x0 else (x - x0) / (x1 - x0)
            sx = (
                row0_x[i] * (1 - tx) * (1 - ty)
                + row0_x[i + 1] * tx * (1 - ty)
                + row1_x[i] * (1 - tx) * ty
                + row1_x[i + 1] * tx * ty
            )
            sy = (
                row0_y[i] * (1 - tx) * (1 - ty)
                + row0_y[i + 1] * tx * (1 - ty)
                + row1_y[i] * (1 - tx) * ty
                + row1_y[i + 1] * tx * ty
            )
            out[y, x] = _bilinear_rgba(rgba, float(sx), float(sy))
    return out


def make_compare_strip(images: Sequence[np.ndarray], labels: Sequence[str] | None = None) -> np.ndarray:
    """水平拼接 HxWx4 图；可选底部留黑边写字由调用方处理，此处仅拼图。"""
    if not images:
        raise ValueError("no images")
    h = max(im.shape[0] for im in images)
    parts = []
    for im in images:
        if im.shape[0] != h:
            # pad
            pad = np.zeros((h, im.shape[1], 4), dtype=np.uint8)
            pad[: im.shape[0]] = im
            parts.append(pad)
        else:
            parts.append(im)
    return np.concatenate(parts, axis=1)
