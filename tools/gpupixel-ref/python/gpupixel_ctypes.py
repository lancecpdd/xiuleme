"""ctypes loader for community libgpupixel_pywrapper.dll (Windows)."""

from __future__ import annotations

import os
import ctypes
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

_DEFAULT_OUT = Path(__file__).resolve().parents[1] / "output" / "windows"
_DLL_NAMES = ("libgpupixel_pywrapper.dll", "gpupixel_pywrapper.dll")


@dataclass
class BeautyParams:
    beauty: float = 0.0
    white: float = 0.0
    thin_face: float = 0.0
    big_eye: float = 0.0
    lipstick: float = 0.0
    blusher: float = 0.0

    def clamped(self) -> "BeautyParams":
        def c(v: float) -> float:
            return float(max(0.0, min(10.0, v)))

        return BeautyParams(
            beauty=c(self.beauty),
            white=c(self.white),
            thin_face=c(self.thin_face),
            big_eye=c(self.big_eye),
            lipstick=c(self.lipstick),
            blusher=c(self.blusher),
        )


def default_dll_dir() -> Path:
    env = os.environ.get("GPUPIXEL_REF_DLL_DIR")
    if env:
        return Path(env)
    return _DEFAULT_OUT


def _find_dll(dll_dir: Path) -> Path:
    for name in _DLL_NAMES:
        path = dll_dir / name
        if path.is_file():
            return path
    raise FileNotFoundError(
        f"No pywrapper DLL in {dll_dir}. Run scripts\\bootstrap.ps1 then scripts\\build.ps1."
    )


class GPUPixelSession:
    """Own one GPUPixelWrapper instance for one or more process_rgba calls."""

    def __init__(self, dll_dir: Optional[Path] = None) -> None:
        self._dll_dir = Path(dll_dir) if dll_dir else default_dll_dir()
        if not self._dll_dir.is_dir():
            raise FileNotFoundError(f"DLL directory missing: {self._dll_dir}")

        # Dependent DLLs (libgpupixel, vnn, glfw, …) must resolve from this dir.
        os.environ["PATH"] = str(self._dll_dir) + os.pathsep + os.environ.get("PATH", "")
        # On Windows, Util::getResourcePath returns a bare filename — resources
        # (face_pc[1.0.0].vnnmodel, lookup_*.png, …) are resolved from CWD.
        self._prev_cwd = Path.cwd()
        os.chdir(self._dll_dir)

        dll_path = _find_dll(self._dll_dir)
        self._lib = ctypes.CDLL(str(dll_path), winmode=0)
        self._bind()
        self._handle = self._lib.GPUPixelWrapper_create()
        if not self._handle:
            raise RuntimeError("GPUPixelWrapper_create failed")
        if not self._lib.GPUPixelWrapper_initialize(self._handle):
            raise RuntimeError(
                "GPUPixelWrapper_initialize failed (OpenGL/GLFW context?). "
                "Need a working GPU driver and display session."
            )
        self._lib.GPUPixelWrapper_setCallbacks(self._handle)
        self._closed = False

    def _bind(self) -> None:
        lib = self._lib
        lib.GPUPixelWrapper_create.restype = ctypes.c_void_p
        lib.GPUPixelWrapper_initialize.argtypes = [ctypes.c_void_p]
        lib.GPUPixelWrapper_initialize.restype = ctypes.c_bool
        lib.GPUPixelWrapper_setCallbacks.argtypes = [ctypes.c_void_p]
        lib.GPUPixelWrapper_setParameters.argtypes = [
            ctypes.c_void_p,
            ctypes.c_float,
            ctypes.c_float,
            ctypes.c_float,
            ctypes.c_float,
            ctypes.c_float,
            ctypes.c_float,
        ]
        lib.GPUPixelWrapper_run.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
        ]
        lib.GPUPixelWrapper_run.restype = ctypes.POINTER(ctypes.c_uint8)
        lib.GPUPixelWrapper_destroy.argtypes = [ctypes.POINTER(ctypes.c_uint8)]
        lib.GPUPixelWrapper_release.argtypes = [ctypes.c_void_p]

    def set_parameters(self, params: BeautyParams) -> None:
        p = params.clamped()
        self._lib.GPUPixelWrapper_setParameters(
            self._handle,
            ctypes.c_float(p.beauty),
            ctypes.c_float(p.white),
            ctypes.c_float(p.thin_face),
            ctypes.c_float(p.big_eye),
            ctypes.c_float(p.lipstick),
            ctypes.c_float(p.blusher),
        )

    def process_rgba(self, rgba: np.ndarray, params: Optional[BeautyParams] = None) -> np.ndarray:
        if self._closed:
            raise RuntimeError("session already closed")
        if rgba.dtype != np.uint8 or rgba.ndim != 3 or rgba.shape[2] != 4:
            raise ValueError("rgba must be HxWx4 uint8")
        if not rgba.flags["C_CONTIGUOUS"]:
            rgba = np.ascontiguousarray(rgba)

        if params is not None:
            self.set_parameters(params)

        height, width, channels = rgba.shape
        ptr = self._lib.GPUPixelWrapper_run(
            self._handle,
            rgba.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            ctypes.c_int(width),
            ctypes.c_int(height),
            ctypes.c_int(channels),
        )
        if not ptr:
            raise RuntimeError("GPUPixelWrapper_run returned null")

        # Copy immediately; free via release() only (avoid destroy+release double-free).
        out = np.ctypeslib.as_array(ptr, shape=(height, width, channels)).copy()
        self._lib.GPUPixelWrapper_release(self._handle)
        return out

    def close(self) -> None:
        if not self._closed and self._handle:
            try:
                self._lib.GPUPixelWrapper_release(self._handle)
            except OSError:
                pass
            self._closed = True
            self._handle = None
        if getattr(self, "_prev_cwd", None) is not None:
            try:
                os.chdir(self._prev_cwd)
            except OSError:
                pass
            self._prev_cwd = None

    def __enter__(self) -> "GPUPixelSession":
        return self

    def __exit__(self, *args) -> None:
        self.close()


def process_rgba(
    rgba: np.ndarray,
    *,
    beauty: float = 0.0,
    white: float = 0.0,
    thin_face: float = 0.0,
    big_eye: float = 0.0,
    lipstick: float = 0.0,
    blusher: float = 0.0,
    dll_dir: Optional[Path] = None,
) -> np.ndarray:
    """One-shot: open session, apply params, process, close."""
    params = BeautyParams(
        beauty=beauty,
        white=white,
        thin_face=thin_face,
        big_eye=big_eye,
        lipstick=lipstick,
        blusher=blusher,
    )
    with GPUPixelSession(dll_dir=dll_dir) as session:
        return session.process_rgba(rgba, params)
