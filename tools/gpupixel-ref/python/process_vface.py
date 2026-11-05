#!/usr/bin/env python3
"""离线 V 脸／下颌线实验 CLI（先测 OK，再移植 Tauri）。

流程：
  1) 可选：GPUPixel 社区 wrapper 原版瘦脸
  2) MediaPipe 478 关键点
  3) vface_warp 下颌加密 + 侧向收束
  4) 输出结果图 + 可选三联对比条
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from landmarks_mp import detect_landmarks_478
from vface_warp import VFaceParams, apply_vface, make_compare_strip


def _mae(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a.astype(np.float32) - b.astype(np.float32))))


def _label_bar(img: np.ndarray, text: str) -> np.ndarray:
    """在顶部加 28px 标签条。"""
    h, w = img.shape[:2]
    bar = np.zeros((28, w, 4), dtype=np.uint8)
    bar[:, :, 3] = 255
    bar[:, :, :3] = 32
    canvas = np.concatenate([bar, img], axis=0)
    pil = Image.fromarray(canvas, mode="RGBA")
    draw = ImageDraw.Draw(pil)
    draw.text((8, 6), text, fill=(240, 240, 240, 255))
    return np.asarray(pil)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="V-face / jawline experiment (offline)")
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path, help="V-face result path")
    parser.add_argument(
        "--compare",
        type=Path,
        default=None,
        help="Optional side-by-side strip (orig | gpupixel? | vface)",
    )
    parser.add_argument(
        "--with-gpupixel",
        action="store_true",
        help="Run community GPUPixel thin-face first, then V-face",
    )
    parser.add_argument("--thin-face", type=float, default=8.0, help="GPUPixel thin-face 0–10")
    parser.add_argument(
        "--vface",
        type=float,
        default=7.0,
        help="V-face strength 0–10 (mapped to 0–1 internally)",
    )
    parser.add_argument(
        "--slim-gain",
        type=float,
        default=0.055,
        help="curveWarp delta at full V-face (default 0.055)",
    )
    parser.add_argument(
        "--jaw-lateral",
        type=float,
        default=1.20,
        help="Jaw lateral band gain vs slim delta (default 1.20)",
    )
    parser.add_argument(
        "--grid-step",
        type=int,
        default=2,
        help="Sparse warp grid step (1=full quality, 2–4=faster preview)",
    )
    parser.add_argument("--scale-100", action="store_true", help="Treat --vface/--thin-face as 0–100")
    parser.add_argument("--stats", action="store_true")
    parser.add_argument("--dll-dir", type=Path, default=None)
    args = parser.parse_args(argv)

    inp = args.input.resolve()
    out = args.output.resolve()
    if not inp.is_file():
        print(f"input not found: {inp}", file=sys.stderr)
        return 1

    def map10(v: float) -> float:
        return float(v) / 10.0 if args.scale_100 else float(v)

    thin = map10(args.thin_face)
    vface10 = map10(args.vface)
    if thin > 10 or vface10 > 10:
        print("warning: values >10 will be clamped", file=sys.stderr)
    thin = min(max(thin, 0.0), 10.0)
    vface10 = min(max(vface10, 0.0), 10.0)

    rgba = np.asarray(Image.open(inp).convert("RGBA"))
    stages: list[tuple[str, np.ndarray]] = [("original", rgba.copy())]

    work = rgba
    if args.with_gpupixel:
        try:
            from gpupixel_ctypes import BeautyParams, GPUPixelSession
        except Exception as e:
            print(f"GPUPixel import failed: {e}", file=sys.stderr)
            return 1
        with GPUPixelSession(dll_dir=args.dll_dir) as session:
            work = session.process_rgba(
                work,
                BeautyParams(thin_face=thin, beauty=0, white=0, big_eye=0, lipstick=0, blusher=0),
            )
        stages.append((f"gpupixel thin={thin:.1f}", work.copy()))

    try:
        landmarks = detect_landmarks_478(work)
    except ImportError as e:
        print(str(e), file=sys.stderr)
        return 1
    if landmarks is None:
        print("no face landmarks (MediaPipe)", file=sys.stderr)
        return 2

    params = VFaceParams(
        strength=vface10 / 10.0,
        slim_gain=args.slim_gain,
        jaw_lateral_gain=args.jaw_lateral,
    )
    result = apply_vface(work, landmarks, params, grid_step=max(1, args.grid_step))
    stages.append((f"vface={vface10:.1f}", result.copy()))

    out.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(result, mode="RGBA").save(out)
    print(f"wrote {out}")

    if args.stats:
        print(f"stats vs input: mae={_mae(rgba, result):.4f}")
        if len(stages) >= 2:
            print(f"stats vs pre-vface: mae={_mae(work, result):.4f}")

    if args.compare:
        labeled = [_label_bar(im, name) for name, im in stages]
        strip = make_compare_strip(labeled)
        args.compare.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(strip, mode="RGBA").save(args.compare)
        print(f"wrote compare {args.compare.resolve()}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
