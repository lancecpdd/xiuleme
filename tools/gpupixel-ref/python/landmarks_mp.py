"""MediaPipe Face Landmarker → 478 像素关键点（V 脸实验用）。

兼容：
- mediapipe 0.10.x：`mp.solutions.face_mesh`
- mediapipe 1.x Tasks：`FaceLandmarker` + `face_landmarker.task`
"""

from __future__ import annotations

import urllib.request
from pathlib import Path
from typing import Optional

import numpy as np

_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
)
_MODEL_NAME = "face_landmarker.task"


def _model_path() -> Path:
    env = Path(__file__).resolve().parents[1] / "output" / "windows" / _MODEL_NAME
    return env


def ensure_face_landmarker_model(path: Optional[Path] = None) -> Path:
    dest = path or _model_path()
    if dest.is_file() and dest.stat().st_size > 1_000_000:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading FaceLandmarker model → {dest}")
    urllib.request.urlretrieve(_MODEL_URL, dest)
    return dest


def _fill_iris_if_needed(pts: np.ndarray, n: int) -> np.ndarray:
    if n >= 478:
        return pts
    pts[468] = 0.5 * (pts[33] + pts[133])
    for i in range(469, 473):
        pts[i] = pts[468]
    pts[473] = 0.5 * (pts[263] + pts[362])
    for i in range(474, 478):
        pts[i] = pts[473]
    return pts


def _detect_solutions(rgba: np.ndarray) -> Optional[np.ndarray]:
    import mediapipe as mp

    h, w = rgba.shape[:2]
    rgb = rgba[:, :, :3]
    with mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
    ) as mesh:
        result = mesh.process(rgb)
    if not result.multi_face_landmarks:
        return None
    lm = result.multi_face_landmarks[0].landmark
    n = len(lm)
    if n < 468:
        return None
    pts = np.zeros((478, 2), dtype=np.float64)
    for i in range(min(n, 478)):
        pts[i, 0] = lm[i].x * w
        pts[i, 1] = lm[i].y * h
    return _fill_iris_if_needed(pts, n)


def _detect_tasks(rgba: np.ndarray, model_path: Path) -> Optional[np.ndarray]:
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision
    import mediapipe as mp

    h, w = rgba.shape[:2]
    rgb = np.ascontiguousarray(rgba[:, :, :3])
    options = vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
        running_mode=vision.RunningMode.IMAGE,
        num_faces=1,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
    )
    with vision.FaceLandmarker.create_from_options(options) as landmarker:
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = landmarker.detect(mp_image)
    if not result.face_landmarks:
        return None
    lm = result.face_landmarks[0]
    n = len(lm)
    if n < 468:
        return None
    pts = np.zeros((478, 2), dtype=np.float64)
    for i in range(min(n, 478)):
        pts[i, 0] = lm[i].x * w
        pts[i, 1] = lm[i].y * h
    return _fill_iris_if_needed(pts, n)


def detect_landmarks_478(rgba: np.ndarray) -> Optional[np.ndarray]:
    """返回 (478, 2) 像素坐标；失败返回 None。"""
    try:
        import mediapipe as mp
    except ImportError as e:
        raise ImportError(
            "V 脸实验需要 mediapipe：pip install -r requirements-vface.txt"
        ) from e

    if rgba.dtype != np.uint8 or rgba.ndim != 3 or rgba.shape[2] < 3:
        raise ValueError("rgba must be HxWx3/4 uint8")

    if hasattr(mp, "solutions"):
        return _detect_solutions(rgba)

    model = ensure_face_landmarker_model()
    return _detect_tasks(rgba, model)
