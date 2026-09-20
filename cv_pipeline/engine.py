"""Per-frame orchestration: detect -> track -> recognise -> landmarks ->
indicators. Persistence is delegated to a callbacks object so this module has
no Django import.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from .annotate import draw_overlay
from .config import PipelineConfig
from .detection import detect_faces
from .indicators import FrameSignals, IndicatorEngine
from .landmarks import FaceLandmarkAnalyzer, match_signals_to_tracks
from .recognition import Gallery, TrackIdentity
from .tracking import ByteTracker

logger = logging.getLogger("cv_pipeline")


class PipelineCallbacks(Protocol):
    def mark_attendance(self, student_id: int, confidence: float) -> None: ...
    def open_event(self, track_id: int, student_id: int | None, condition: str,
                   quality: float) -> int: ...
    def close_event(self, event_pk: int) -> None: ...
    def identity_locked(self, track_id: int, student_id: int) -> None: ...


@dataclass
class FrameResult:
    frame: np.ndarray
    track_views: list[dict] = field(default_factory=list)
    fps: float = 0.0
    num_tracks: int = 0


class MonitoringEngine:
    def __init__(
        self,
        cfg: PipelineConfig,
        gallery: Gallery,
        names: dict[int, str],
        callbacks: PipelineCallbacks,
        *,
        session_active: bool = True,
        enable_landmarks: bool = True,
    ):
        self.cfg = cfg
        self.gallery = gallery
        self.names = names
        self.cb = callbacks
        self.session_active = session_active

        self.tracker = ByteTracker(cfg)
        self.indicators = IndicatorEngine(cfg)
        self._identities: dict[int, TrackIdentity] = {}
        self._centroids: dict[int, tuple[float, float]] = {}
        self._open_events: dict[tuple[int, str], int] = {}
        self._marked_students: set[int] = set()
        self._seen_tracks: set[int] = set()

        self._landmarker = None
        if enable_landmarks:
            try:
                self._landmarker = FaceLandmarkAnalyzer(cfg)
            except Exception as exc:  # noqa: BLE001 - degrade gracefully
                logger.warning("Face landmarks disabled: %s", exc)

        self._last_t = time.time()
        self._fps = 0.0
        self._frame_no = 0
        self._sig_cache: dict[int, object] = {}

    # ------------------------------------------------------------------
    def process(self, frame: np.ndarray) -> FrameResult:
        h, w = frame.shape[:2]
        diag = float(np.hypot(w, h))
        self._frame_no += 1

        detections = detect_faces(frame, self.cfg)
        tracks = self.tracker.update(detections)

        track_boxes = {t["track_id"]: np.array(t["tlbr"], dtype=np.float32) for t in tracks}

        # MediaPipe is the second-heaviest step — run it every Nth processed
        # frame and reuse the last result for the frames in between.
        run_lm = (
            self._landmarker is not None
            and track_boxes
            and self._frame_no % max(1, self.cfg.landmark_every_n) == 0
        )
        if run_lm:
            try:
                face_signals = self._landmarker.analyze(frame)
                self._sig_cache = match_signals_to_tracks(track_boxes, face_signals)
            except Exception as exc:  # noqa: BLE001
                logger.debug("landmark analyze failed: %s", exc)
        signals_by_track = {
            tid: s for tid, s in self._sig_cache.items() if tid in track_boxes
        }

        now = time.time()
        views: list[dict] = []
        live_ids = set(track_boxes)

        for t in tracks:
            tid = t["track_id"]
            self._seen_tracks.add(tid)
            box = t["tlbr"]
            cx, cy = (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0

            # --- recognition -----------------------------------------
            ident = self._identities.setdefault(tid, TrackIdentity(self.cfg))
            det_idx = t.get("det_index")
            if det_idx is not None and det_idx < len(detections):
                sid, sim = self.gallery.query(detections[det_idx].embedding)
                was_locked = ident.locked
                ident.add(sid, sim)
                if ident.locked and not was_locked:
                    self.cb.identity_locked(tid, ident.student_id)

            student_id = ident.student_id
            if (
                student_id is not None
                and self.session_active
                and student_id not in self._marked_students
            ):
                self.cb.mark_attendance(student_id, ident.confidence)
                self._marked_students.add(student_id)

            # --- movement -------------------------------------------
            prev = self._centroids.get(tid)
            movement_norm = 0.0
            if prev is not None:
                movement_norm = float(np.hypot(cx - prev[0], cy - prev[1]) / diag)
            self._centroids[tid] = (cx, cy)

            # --- indicator signals ---------------------------------
            sig = signals_by_track.get(tid)
            face_visible = sig is not None or det_idx is not None
            fs = FrameSignals(
                face_visible=face_visible,
                yaw=getattr(sig, "yaw", detections[det_idx].yaw if det_idx is not None and det_idx < len(detections) else 0.0),
                pitch=getattr(sig, "pitch", detections[det_idx].pitch if det_idx is not None and det_idx < len(detections) else 0.0),
                blink=getattr(sig, "blink", 0.0),
                movement_norm=movement_norm,
                quality=t.get("score", 0.0),
            )
            state, transitions = self.indicators.update(tid, now, fs)
            self._apply_transitions(tid, student_id, transitions)

            views.append(
                {
                    "track_id": tid,
                    "bbox": [round(v, 1) for v in box],
                    "student_id": student_id,
                    "student_name": self.names.get(student_id) if student_id else None,
                    "attendance_status": "present" if student_id in self._marked_students else None,
                    "similarity": round(ident.confidence, 3),
                    "indicator": state.current_label,
                    "indicator_is_event": state.current_is_event,
                }
            )

        # Close events for tracks that vanished.
        for gone in list(self._seen_tracks - live_ids):
            for tr in self.indicators.drop_track(gone):
                key = (gone, tr.condition)
                if key in self._open_events:
                    self.cb.close_event(self._open_events.pop(key))
            self._seen_tracks.discard(gone)
            self._identities.pop(gone, None)
            self._centroids.pop(gone, None)

        annotated = draw_overlay(frame.copy(), views) if self.cfg.burn_in_overlay else frame

        dt = now - self._last_t
        self._last_t = now
        if dt > 0:
            self._fps = 0.9 * self._fps + 0.1 * (1.0 / dt)
        return FrameResult(frame=annotated, track_views=views, fps=round(self._fps, 1),
                           num_tracks=len(views))

    def _apply_transitions(self, track_id, student_id, transitions):
        for tr in transitions:
            key = (track_id, tr.condition)
            if tr.kind == "open" and key not in self._open_events:
                self._open_events[key] = self.cb.open_event(
                    track_id, student_id, tr.condition, tr.quality
                )
            elif tr.kind == "close" and key in self._open_events:
                self.cb.close_event(self._open_events.pop(key))

    def shutdown(self):
        for pk in self._open_events.values():
            try:
                self.cb.close_event(pk)
            except Exception:  # noqa: BLE001
                pass
        self._open_events.clear()
