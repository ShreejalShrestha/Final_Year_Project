"""Thin wrapper around cv2.VideoCapture with reconnect + source parsing."""
from __future__ import annotations

import logging
import math
import threading
import time
from pathlib import Path

logger = logging.getLogger("cv_pipeline")


def parse_source(value: str):
    """"0" -> webcam index 0; anything else -> treated as a path/URL."""
    value = (value or "0").strip()
    return int(value) if value.isdigit() else value


class VideoStream:
    """Frame source for a webcam, stream URL or video file.

    Live sources (webcam/URL) are read on a background thread that keeps only the
    *newest* frame. The consumer therefore never waits on camera I/O while it is busy,
    never receives a stale buffered frame, and a slow frame drops one camera frame
    instead of stalling the loop out of step with the camera (which halves the frame
    rate). Video files are read synchronously so that every frame is processed.
    """

    READ_TIMEOUT = 2.0   # seconds without a new live frame before read() gives up
    _latest_only = False   # class-level defaults keep partially-initialised instances safe
    _thread = None

    def __init__(self, source: str, target_width: int | None = 640, latest_only: bool | None = None):
        import cv2

        self._cv2 = cv2
        self.source = parse_source(source)
        self._is_file = isinstance(self.source, str) and Path(self.source).is_file()
        self.timestamp_seconds: float | None = None
        self.target_width = target_width
        self.cap = None
        self._latest_only = (not self._is_file) if latest_only is None else latest_only
        self._cond = threading.Condition()
        self._frame = None
        self._seq = 0
        self._consumed = 0
        self._ended = False
        self._stop = False
        self._thread: threading.Thread | None = None
        self._open()
        if self._latest_only:
            self._start_reader()

    def _open(self):
        self.cap = self._cv2.VideoCapture(self.source)
        if not self.cap or not self.cap.isOpened():
            raise RuntimeError(f"Could not open video source: {self.source!r}")
        logger.info("Video source opened: %r", self.source)

    # --- background reader (live sources) ---------------------------
    def _start_reader(self):
        self._stop = False
        self._ended = False
        self._thread = threading.Thread(target=self._reader, name="video-reader", daemon=True)
        self._thread.start()

    def _stop_reader(self):
        with self._cond:
            self._stop = True
            self._cond.notify_all()
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            self._thread = None

    def _reader(self):
        while not self._stop:
            frame = self._grab()
            with self._cond:
                if frame is None:
                    self._ended = True
                    self._cond.notify_all()
                    return
                self._frame = frame
                self._seq += 1
                self._cond.notify_all()

    def read(self):
        if not self._latest_only:
            return self._grab()
        with self._cond:
            self._cond.wait_for(
                lambda: self._seq != self._consumed or self._ended or self._stop,
                timeout=self.READ_TIMEOUT,
            )
            if self._seq == self._consumed:
                return None            # timed out, ended or stopped
            self._consumed = self._seq
            return self._frame

    def _grab(self):
        cap = self.cap
        if cap is None:
            return None
        ok, frame = cap.read()
        if not ok:
            return None
        if self._is_file:
            seconds = cap.get(self._cv2.CAP_PROP_POS_MSEC) / 1000.0
            fps = cap.get(self._cv2.CAP_PROP_FPS)
            frame_number = cap.get(self._cv2.CAP_PROP_POS_FRAMES)
            if seconds <= 0 and fps > 0:
                seconds = max(0.0, frame_number - 1) / fps
            self.timestamp_seconds = seconds if math.isfinite(seconds) and seconds >= 0 else None
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
        if self._latest_only:
            self._start_reader()

    def release(self):
        self._stop_reader()
        if self.cap is not None:
            self.cap.release()
            self.cap = None
