"""Thin wrapper around cv2.VideoCapture with reconnect + source parsing."""
from __future__ import annotations

import logging
import time

logger = logging.getLogger("cv_pipeline")


def parse_source(value: str):
    """"0" -> webcam index 0; anything else -> treated as a path/URL."""
    value = (value or "0").strip()
    return int(value) if value.isdigit() else value


class VideoStream:
    def __init__(self, source: str, target_width: int | None = 640):
        import cv2

        self._cv2 = cv2
        self.source = parse_source(source)
        self.target_width = target_width
        self.cap = None
        self._open()

    def _open(self):
        self.cap = self._cv2.VideoCapture(self.source)
        if not self.cap or not self.cap.isOpened():
            raise RuntimeError(f"Could not open video source: {self.source!r}")
        logger.info("Video source opened: %r", self.source)

    def read(self):
        ok, frame = self.cap.read()
        if not ok:
            return None
        if self.target_width and frame.shape[1] > self.target_width:
            scale = self.target_width / frame.shape[1]
            frame = self._cv2.resize(
                frame, (self.target_width, int(frame.shape[0] * scale))
            )
        return frame

    def reconnect(self, delay: float = 1.0):
        self.release()
        time.sleep(delay)
        self._open()

    def release(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None
