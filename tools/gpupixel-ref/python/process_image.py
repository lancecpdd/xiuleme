#!/usr/bin/env python3
"""CLI: run GPUPixel community pywrapper on an image (Windows)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from gpupixel_ctypes import BeautyParams, GPUPixelSession, default_dll_dir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Process an image with GPUPixel (community pywrapper). Windows only."
    )
    parser.add_argument("input", type=Path, help="Input image path")
    parser.add_argument("output", type=Path, help="Output image path")
    parser.add_argument(
        "--beauty",
        type=float,
        default=0.0,
        help="Smooth/beauty 0–10 (values >10 clamp to 10; optional 0–100 via --scale-100)",
    )
    parser.add_argument("--white", type=float, default=0.0, help="Whitening 0–10")
    parser.add_argument(
        "--thin-face",
        type=float,
        default=0.0,
        help="Thin face 0–10 (= desktop demo max). Internal slim = value/200. >10 clamps unless --scale-100",
    )
    parser.add_argument("--big-eye", type=float, default=0.0, help="Big eye 0–10 (internal = value/100)")
    parser.add_argument("--lipstick", type=float, default=0.0, help="Lipstick 0–10")
    parser.add_argument("--blusher", type=float, default=0.0, help="Blusher 0–10")
    parser.add_argument(
        "--scale-100",
        action="store_true",
        help="Treat slider args as 0–100 UI units (divide by 10 before the 0–10 wrapper API)",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Print mean abs pixel diff vs input (near 0 ⇒ no face / no effect)",
    )
    parser.add_argument(
        "--dll-dir",
        type=Path,
        default=None,
        help=f"Directory with DLLs (default: {default_dll_dir()})",
    )
    args = parser.parse_args(argv)

    # Resolve before GPUPixelSession: it chdirs into the DLL/resource directory.
    input_path = args.input.resolve()
    output_path = args.output.resolve()
    if not input_path.is_file():
        print(f"input not found: {input_path}", file=sys.stderr)
        return 1

    img = Image.open(input_path).convert("RGBA")
    rgba = np.asarray(img)

    def map_param(v: float) -> float:
        if args.scale_100:
            return float(v) / 10.0
        if v > 10.0:
            print(
                f"warning: value {v} > 10 will clamp to 10 (wrapper max). "
                f"Use --scale-100 if you meant a 0–100 UI slider.",
                file=sys.stderr,
            )
        return float(v)

    params = BeautyParams(
        beauty=map_param(args.beauty),
        white=map_param(args.white),
        thin_face=map_param(args.thin_face),
        big_eye=map_param(args.big_eye),
        lipstick=map_param(args.lipstick),
        blusher=map_param(args.blusher),
    )

    with GPUPixelSession(dll_dir=args.dll_dir) as session:
        out = session.process_rgba(rgba, params)

    if args.stats:
        mae = float(np.mean(np.abs(out.astype(np.float32) - rgba.astype(np.float32))))
        changed = int(np.count_nonzero(out != rgba))
        print(f"stats: mae={mae:.4f} changed_pixels={changed}/{out.size}")
        if mae < 0.05:
            print(
                "stats: almost identical to input — likely no face detected "
                "(need a clear frontal face photo; synthetic ovals will not work).",
                file=sys.stderr,
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(out, mode="RGBA").save(output_path)
    print(f"wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
