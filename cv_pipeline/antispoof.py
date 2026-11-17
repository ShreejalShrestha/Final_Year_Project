"""MiniFASNetV2 ONNX inference (BGR, unnormalised float32, 2.7x crop).

Weights/export: https://github.com/yakhyo/face-anti-spoofing
Original model: https://github.com/minivision-ai/Silent-Face-Anti-Spoofing
"""
from pathlib import Path
import os
import tempfile
import urllib.request
import cv2
import numpy as np

MODEL_URL = "https://github.com/yakhyo/face-anti-spoofing/releases/download/weights/MiniFASNetV2.onnx"


class AntiSpoofClassifier:
    def __init__(self, cfg):
        import onnxruntime as ort
        path = Path(cfg.model_dir) / "MiniFASNetV2.onnx"
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            # Publish only a completely downloaded file; interrupted downloads retry.
            fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".download")
            os.close(fd)
            try:
                with urllib.request.urlopen(MODEL_URL, timeout=60) as src, open(tmp, "wb") as dst:
                    import shutil
                    shutil.copyfileobj(src, dst)
                ort.InferenceSession(tmp, providers=["CPUExecutionProvider"])
                os.replace(tmp, path)
            except Exception as exc:
                raise RuntimeError(f"Cannot load anti-spoof model. Download {MODEL_URL} to {path}: {exc}") from exc
            finally:
                if os.path.exists(tmp):
                    os.unlink(tmp)
        # Small model: CPU avoids CUDA transfer overhead and GPU dependency. An 80x80 input
        # is far too little work to benefit from ONNX Runtime's default thread pool (one
        # thread per logical core, busy-waiting between calls), and that pool competes with
        # MediaPipe's for CPU while both run each frame. One quiet thread is faster overall.
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        options.add_session_config_entry("session.intra_op.allow_spinning", "0")
        self.session = ort.InferenceSession(
            str(path), sess_options=options, providers=["CPUExecutionProvider"]
        )
        inp = self.session.get_inputs()[0]
        if inp.shape != [1, 3, 80, 80] and inp.shape[1:] != [3, 80, 80]:
            raise ValueError(f"Unexpected MiniFASNet input shape: {inp.shape}")
        self.input_name = inp.name
        self.min_pixels = cfg.liveness_min_face_pixels

    def score(self, frame, bbox):
        box = np.asarray(bbox, dtype=float)
        if box.shape != (4,) or not np.isfinite(box).all():
            return 0.0
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = box
        bw, bh = int(x2 - x1), int(y2 - y1)
        if min(bw, bh) < self.min_pixels or x1 < 0 or y1 < 0 or x2 > w or y2 > h:
            return 0.0
        scale = min((h - 1) / bh, (w - 1) / bw, 2.7)
        cx, cy = int(x1) + bw / 2, int(y1) + bh / 2
        left, top = max(0, int(cx - bw * scale / 2)), max(0, int(cy - bh * scale / 2))
        right, bottom = min(w - 1, int(cx + bw * scale / 2)), min(h - 1, int(cy + bh * scale / 2))
        face = cv2.resize(frame[top:bottom+1, left:right+1], (80, 80))
        tensor = np.ascontiguousarray(face.transpose(2, 0, 1)[None], dtype=np.float32)
        logits = np.asarray(self.session.run(None, {self.input_name: tensor})[0]).reshape(-1)
        if logits.shape != (3,) or not np.isfinite(logits).all():
            return 0.0
        probs = np.exp(logits - logits.max())
        return float(probs[1] / probs.sum())  # class 1 = real; 0/2 = spoof
