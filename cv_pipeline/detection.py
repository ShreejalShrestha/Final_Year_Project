"""InsightFace wrapper: SCRFD face detection + ArcFace embeddings (buffalo_l).

`get_face_app()` returns a process-wide singleton so the ~350 MB model set is
loaded only once. Weights download automatically to ~/.insightface on first use.
"""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass

import numpy as np

from .config import PipelineConfig

logger = logging.getLogger("cv_pipeline")

_app = None
_lock = threading.Lock()
_dlls_loaded = False


def _preload_cuda_dlls() -> None:
    """Make ONNX Runtime find the CUDA / cuDNN DLLs shipped as `nvidia-*-cu12`
    pip wheels (no system CUDA toolkit needed). Safe to call more than once.

    cuDNN 9 is split into sub-libraries that it loads itself via LoadLibrary, so
    the wheel `bin/` directories must be on PATH (not only `add_dll_directory`).
    """
    global _dlls_loaded
    if _dlls_loaded:
        return
    _dlls_loaded = True

    import os

    try:
        import nvidia

        base = os.path.dirname(nvidia.__file__)
        bin_dirs = [
            os.path.join(base, pkg, "bin")
            for pkg in os.listdir(base)
            if os.path.isdir(os.path.join(base, pkg, "bin"))
        ]
        for d in bin_dirs:
            os.add_dll_directory(d)
        if bin_dirs:
            os.environ["PATH"] = os.pathsep.join(bin_dirs + [os.environ.get("PATH", "")])
            logger.info("Added %d nvidia bin dirs to PATH for CUDA", len(bin_dirs))
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not add nvidia DLL directories: %s", exc)

    try:
        import onnxruntime as ort

        if hasattr(ort, "preload_dlls"):
            ort.preload_dlls()
            logger.info("Called onnxruntime.preload_dlls() for CUDA")
    except Exception as exc:  # noqa: BLE001
        logger.warning("onnxruntime.preload_dlls() unavailable: %s", exc)


@dataclass
class Detection:
    bbox: np.ndarray            # (4,) x1, y1, x2, y2
    det_score: float
    kps: np.ndarray             # (5, 2) five-point landmarks
    embedding: np.ndarray       # (512,) L2-normalised ArcFace embedding
    yaw: float = 0.0            # degrees, + = turned to subject's left
    pitch: float = 0.0         # degrees, + = looking down
    roll: float = 0.0

    @property
    def area(self) -> float:
        x1, y1, x2, y2 = self.bbox
        return float(max(0.0, x2 - x1) * max(0.0, y2 - y1))


def get_face_app(cfg: PipelineConfig | None = None):
    global _app
    if _app is not None:
        return _app
    with _lock:
        if _app is not None:
            return _app
        from insightface.app import FaceAnalysis

        cfg = cfg or PipelineConfig()
        if cfg.device == "cuda":
            _preload_cuda_dlls()
        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if cfg.device == "cuda"
            else ["CPUExecutionProvider"]
        )
        app = FaceAnalysis(
            name=cfg.insightface_name,
            allowed_modules=list(cfg.insightface_modules),
            providers=providers,
        )
        ctx_id = 0 if cfg.device == "cuda" else -1
        app.prepare(ctx_id=ctx_id, det_size=cfg.det_size)
        logger.info(
            "InsightFace ready (%s, modules=%s, providers=%s)",
            cfg.insightface_name, cfg.insightface_modules, providers,
        )
        _app = app
        return _app


def _coarse_pose_from_kps(kps: np.ndarray, bbox: np.ndarray) -> tuple[float, float, float]:
    """Fallback head-pose estimate from the 5 landmarks when the 3D model
    does not populate `.pose`. Rough but cheap and monotonic.
    """
    left_eye, right_eye, nose, left_mouth, right_mouth = kps
    eye_mid = (left_eye + right_eye) / 2.0
    mouth_mid = (left_mouth + right_mouth) / 2.0
    eye_dist = np.linalg.norm(right_eye - left_eye) + 1e-6

    # Yaw: horizontal offset of the nose from the eye midpoint.
    yaw = float(np.degrees(np.arctan2(nose[0] - eye_mid[0], eye_dist))) * 2.2
    # Pitch: vertical position of the nose between eyes and mouth.
    face_h = np.linalg.norm(mouth_mid - eye_mid) + 1e-6
    rel = (nose[1] - eye_mid[1]) / face_h            # ~0.5 neutral, larger = down
    pitch = float((rel - 0.55) * 90.0)
    # Roll: tilt of the eye line.
    roll = float(np.degrees(np.arctan2(right_eye[1] - left_eye[1], right_eye[0] - left_eye[0])))
    return yaw, pitch, roll


def detect_faces(frame_bgr, cfg: PipelineConfig) -> list[Detection]:
    app = get_face_app(cfg)
    out: list[Detection] = []
    for f in app.get(frame_bgr):
        score = float(getattr(f, "det_score", 0.0))
        if score < cfg.min_det_score:
            continue
        kps = np.asarray(f.kps, dtype=np.float32)
        bbox = np.asarray(f.bbox, dtype=np.float32)
        pose = getattr(f, "pose", None)
        if pose is not None and len(pose) == 3:
            pitch, yaw, roll = (float(pose[0]), float(pose[1]), float(pose[2]))
        else:
            yaw, pitch, roll = _coarse_pose_from_kps(kps, bbox)
        out.append(
            Detection(
                bbox=bbox,
                det_score=score,
                kps=kps,
                embedding=np.asarray(f.normed_embedding, dtype=np.float32),
                yaw=yaw,
                pitch=pitch,
                roll=roll,
            )
        )
    return out
