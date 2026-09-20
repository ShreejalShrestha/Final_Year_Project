"""Background monitoring worker: one thread per active session.

The Django dev server is threaded, so a daemon thread can own the camera loop,
run the CV engine, and publish the latest annotated JPEG + track state for the
dashboard to read. DB writes go through the service layer with connection
hygiene (`close_old_connections`).
"""
from __future__ import annotations

import logging
import threading
import time

from django.conf import settings
from django.db import close_old_connections

from cv_pipeline.capture import VideoStream
from cv_pipeline.config import PipelineConfig
from cv_pipeline.engine import MonitoringEngine

logger = logging.getLogger("apps.dashboard")


def _make_config() -> PipelineConfig:
    conf = settings.CV_PIPELINE
    kwargs = dict(
        device=conf["DEVICE"],
        recognition_threshold=conf["RECOGNITION_THRESHOLD"],
        model_dir=conf["MODEL_DIR"],
        # The dashboard draws boxes on a clickable canvas; don't burn them into the stream too.
        burn_in_overlay=False,
    )
    if conf.get("MODEL_NAME"):
        kwargs["insightface_name"] = conf["MODEL_NAME"]
    if conf.get("PROCESS_EVERY_N"):
        kwargs["process_every_n_frames"] = int(conf["PROCESS_EVERY_N"])
    return PipelineConfig(**kwargs)


class DjangoCallbacks:
    """Bridges cv_pipeline events to the database."""

    def __init__(self, session_id: int):
        self.session_id = session_id

    def mark_attendance(self, student_id: int, confidence: float) -> None:
        from apps.attendance.services import mark_present_from_recognition

        close_old_connections()
        try:
            mark_present_from_recognition(
                student_id=student_id, session_id=self.session_id, confidence=confidence
            )
        except Exception:  # noqa: BLE001
            logger.exception("mark_attendance failed")

    def open_event(self, track_id, student_id, condition, quality) -> int:
        from apps.monitoring.services import open_event

        close_old_connections()
        return open_event(
            session_id=self.session_id,
            student_id=student_id,
            track_ref=track_id,
            event_type=condition,
            quality_score=quality,
        )

    def close_event(self, event_pk: int) -> None:
        from apps.monitoring.services import close_event

        close_old_connections()
        close_event(event_pk=event_pk)

    def identity_locked(self, track_id: int, student_id: int) -> None:
        from apps.monitoring.models import MonitoringEvent

        close_old_connections()
        MonitoringEvent.objects.filter(
            session_id=self.session_id, track_ref=str(track_id), student__isnull=True
        ).update(student_id=student_id)


class PipelineWorker:
    def __init__(self, session_id: int, source: str | None = None):
        self.session_id = session_id
        self.source = source or settings.CV_PIPELINE["VIDEO_SOURCE"]
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self.latest_jpeg: bytes | None = None
        self.latest_views: list[dict] = []
        self.fps: float = 0.0
        self.error: str | None = None
        self.started_at: float | None = None

    # --- lifecycle ---------------------------------------------------
    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self.error = None
        self._thread = threading.Thread(
            target=self._run, name=f"pipeline-{self.session_id}", daemon=True
        )
        self._thread.start()
        self.started_at = time.time()

    def stop(self, timeout: float = 5.0):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=timeout)

    @property
    def is_running(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    # --- worker loop -----------------------------------------------
    def _run(self):
        import cv2

        from apps.classes.models import ClassSession

        from .gallery import build_gallery

        cfg = _make_config()
        try:
            close_old_connections()
            session = ClassSession.objects.get(pk=self.session_id)
            gallery, names = build_gallery(cfg)
            logger.info("Session %s: gallery has %d templates", self.session_id, len(gallery))
            engine = MonitoringEngine(
                cfg, gallery, names, DjangoCallbacks(self.session_id),
                session_active=session.is_active,
            )
            stream = VideoStream(self.source, target_width=cfg.capture_width)
        except Exception as exc:  # noqa: BLE001
            self.error = str(exc)
            logger.exception("Pipeline %s failed to start", self.session_id)
            return

        frame_i = 0
        try:
            while not self._stop.is_set():
                frame = stream.read()
                if frame is None:
                    time.sleep(0.1)
                    try:
                        stream.reconnect()
                    except Exception:  # noqa: BLE001
                        self.error = "Video source lost."
                        break
                    continue

                frame_i += 1
                if frame_i % cfg.process_every_n_frames != 0:
                    continue

                try:
                    result = engine.process(frame)
                except Exception:  # noqa: BLE001
                    logger.exception("frame processing error")
                    continue

                ok, buf = cv2.imencode(".jpg", result.frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
                if ok:
                    with self._lock:
                        self.latest_jpeg = buf.tobytes()
                        self.latest_views = result.track_views
                        self.fps = result.fps

                if frame_i % 150 == 0:
                    close_old_connections()
        finally:
            engine.shutdown()
            stream.release()
            close_old_connections()
            logger.info("Pipeline %s stopped", self.session_id)

    # --- readers for the views -----------------------------------
    def snapshot(self):
        with self._lock:
            return self.latest_jpeg, list(self.latest_views), self.fps


class PipelineRegistry:
    def __init__(self):
        self._workers: dict[int, PipelineWorker] = {}
        self._lock = threading.Lock()

    def start(self, session_id: int, source: str | None = None) -> PipelineWorker:
        with self._lock:
            worker = self._workers.get(session_id)
            if worker is None or not worker.is_running:
                worker = PipelineWorker(session_id, source)
                self._workers[session_id] = worker
                worker.start()
            return worker

    def get(self, session_id: int) -> PipelineWorker | None:
        return self._workers.get(session_id)

    def stop(self, session_id: int):
        with self._lock:
            worker = self._workers.pop(session_id, None)
        if worker:
            worker.stop()

    def stop_all(self):
        for sid in list(self._workers):
            self.stop(sid)


pipeline_registry = PipelineRegistry()
