"""MediaPipe Face Landmarker: head pose (from the facial transformation matrix)
and eye-closure / yawn signals (from blendshapes).

Runs in IMAGE mode so it is safe to call from the pipeline's worker thread.
The ~3 MB `.task` file downloads to CV_MODEL_DIR on first use.
"""
from __future__ import annotations

import logging
import math
import os
import urllib.request
from dataclasses import dataclass

import numpy as np

from .config import PipelineConfig

logger = logging.getLogger("cv_pipeline")

_TASK_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)


@dataclass
class FaceSignal:
    bbox: np.ndarray            # (4,) x1,y1,x2,y2 in pixels
    yaw: float
    pitch: float
    roll: float
    blink: float               # 0..1, minimum of both eyes (compatibility field)
    yawn: float                # 0..1, jawOpen blendshape
    eye_left: float | None = None
    eye_right: float | None = None
    pose_valid: bool = False


def _ensure_task_file(model_dir: str) -> str:
    os.makedirs(model_dir, exist_ok=True)
    path = os.path.join(model_dir, "face_landmarker.task")
    if not os.path.exists(path):
        logger.info("Downloading face_landmarker.task ...")
        urllib.request.urlretrieve(_TASK_URL, path)
    return path


def _euler_from_matrix(m: np.ndarray) -> tuple[float, float, float]:
    """Extract yaw/pitch/roll (degrees) from a 4x4 transform. Sign convention:
    pitch + = looking down, yaw + = head turned to subject's left.
    """
    r = m[:3, :3]
    sy = math.sqrt(r[0, 0] ** 2 + r[1, 0] ** 2)
    if sy > 1e-6:
        pitch = math.atan2(r[2, 1], r[2, 2])
        yaw = math.atan2(-r[2, 0], sy)
        roll = math.atan2(r[1, 0], r[0, 0])
    else:
        pitch = math.atan2(-r[1, 2], r[1, 1])
        yaw = math.atan2(-r[2, 0], sy)
        roll = 0.0
    return math.degrees(pitch), math.degrees(yaw), math.degrees(roll)


class FaceLandmarkAnalyzer:
    def __init__(self, cfg: PipelineConfig, max_faces: int = 12):
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        task_path = _ensure_task_file(cfg.model_dir)
        options = vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=task_path),
            running_mode=vision.RunningMode.IMAGE,
            num_faces=max_faces,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
        )
        self._mp = mp
        self._landmarker = vision.FaceLandmarker.create_from_options(options)
        logger.info("MediaPipe FaceLandmarker ready (num_faces=%d)", max_faces)

    def analyze(self, frame_bgr) -> list[FaceSignal]:
        import cv2

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect(mp_image)

        signals: list[FaceSignal] = []
        matrices = result.facial_transformation_matrixes or []
        blendshapes = result.face_blendshapes or []
        for i, landmarks in enumerate(result.face_landmarks or []):
            xs = [lm.x * w for lm in landmarks]
            ys = [lm.y * h for lm in landmarks]
            bbox = np.array([min(xs), min(ys), max(xs), max(ys)], dtype=np.float32)

            yaw = pitch = roll = 0.0
            pose_valid = False
            if i < len(matrices):
                matrix = np.asarray(matrices[i])
                if matrix.shape == (4, 4) and np.isfinite(matrix).all():
                    pitch, yaw, roll = _euler_from_matrix(matrix)
                    pose_valid = True

            blink = yawn = 0.0
            eye_left = eye_right = None
            if i < len(blendshapes):
                cats = {c.category_name: c.score for c in blendshapes[i]}
                eye_left = cats.get("eyeBlinkLeft")
                eye_right = cats.get("eyeBlinkRight")
                if eye_left is not None and eye_right is not None:
                    blink = min(eye_left, eye_right)
                yawn = cats.get("jawOpen", 0.0)

            signals.append(
                FaceSignal(bbox=bbox, yaw=yaw, pitch=pitch, roll=roll, blink=blink, yawn=yawn,
                           eye_left=eye_left, eye_right=eye_right, pose_valid=pose_valid)
            )
        return signals


def match_signals_to_tracks(
    track_boxes: dict[int, np.ndarray], signals: list[FaceSignal], iou_min: float = 0.3
) -> dict[int, FaceSignal]:
    """Greedy IoU match of face-landmark boxes to tracker boxes."""
    out: dict[int, FaceSignal] = {}
    used = set()
    for tid, tb in track_boxes.items():
        best_j, best_iou = -1, iou_min
        for j, sig in enumerate(signals):
            if j in used:
                continue
            iou = _iou(tb, sig.bbox)
            if iou > best_iou:
                best_j, best_iou = j, iou
        if best_j >= 0:
            out[tid] = signals[best_j]
            used.add(best_j)
    return out


def _iou(a, b) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return inter / (area_a + area_b - inter + 1e-6)
