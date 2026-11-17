"""Per-frame orchestration: detect -> track -> recognise -> landmarks ->
indicators. Persistence is delegated to a callbacks object so this module has
no Django import.
"""
from __future__ import annotations

import logging
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from typing import Protocol

import numpy as np

from .annotate import draw_overlay
from .config import PipelineConfig
from .detection import detect_faces, embed_faces
from .indicators import FrameSignals, IndicatorEngine
from .landmarks import FaceLandmarkAnalyzer, match_signals_to_tracks
from .liveness import LivenessGate
from .antispoof import AntiSpoofClassifier
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
    signal_rows: list[dict] = field(default_factory=list)


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
        self.liveness = LivenessGate(cfg)
        self._antispoof = AntiSpoofClassifier(cfg) if cfg.liveness_enabled else None
        self._identities: dict[int, TrackIdentity] = {}
        self._centroids: dict[int, tuple[float, float]] = {}
        self._open_events: dict[tuple[int, str], int] = {}
        self._marked_students: set[int] = set()
        self._seen_tracks: set[int] = set()
        self._last_embed: dict[int, int] = {}      # track id -> frame number of last embedding
        self._pool: ThreadPoolExecutor | None = None

        self._landmarker = None
        if enable_landmarks:
            try:
                self._landmarker = FaceLandmarkAnalyzer(cfg)
            except Exception as exc:  # noqa: BLE001 - degrade gracefully
                logger.warning("Face landmarks disabled: %s", exc)

        self._last_t = time.monotonic()
        self._fps = 0.0
        self._frame_times: deque[float] = deque(maxlen=30)   # wall-clock stamps of recent frames
        self._frame_no = 0
        self._sig_cache: dict[int, object] = {}
        self._sig_cache_time = 0.0

    # ------------------------------------------------------------------
    def _landmark_pool(self) -> ThreadPoolExecutor:
        if self._pool is None:
            self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="landmarks")
        return self._pool

    def process(self, frame: np.ndarray, *, timestamp: float | None = None) -> FrameResult:
        # Recorded clips use video time so event durations do not depend on CPU speed.
        now = time.monotonic() if timestamp is None else timestamp
        h, w = frame.shape[:2]
        diag = float(np.hypot(w, h))
        self._frame_no += 1

        # MediaPipe landmarks are CPU-bound while detection and recognition run on the
        # GPU, so start them first and join after tracking: the two overlap instead of
        # their times adding up.
        lm_due = (
            self._landmarker is not None
            and self._frame_no % max(1, self.cfg.landmark_every_n) == 0
        )
        lm_future = self._landmark_pool().submit(self._landmarker.analyze, frame) if lm_due else None

        # Boxes only: embeddings are the costly part and are computed below, in one
        # batch, for just the tracks that need them.
        detections = detect_faces(frame, self.cfg, with_embeddings=False)
        tracks = self.tracker.update(detections)

        track_boxes = {t["track_id"]: np.array(t["tlbr"], dtype=np.float32) for t in tracks}

        run_lm = lm_due and bool(track_boxes)
        if lm_future is not None:
            try:
                face_signals = lm_future.result()
            except Exception as exc:  # noqa: BLE001
                face_signals = None
                if run_lm:
                    logger.debug("landmark analyze failed: %s", exc)
                    self._sig_cache.clear()
            if run_lm and face_signals is not None:
                self._sig_cache = match_signals_to_tracks(track_boxes, face_signals)
                self._sig_cache_time = now
        signals_by_track = {
            tid: s for tid, s in self._sig_cache.items()
            if tid in track_boxes and 0 <= now - self._sig_cache_time <= self.cfg.landmark_max_age_seconds
        }

        views: list[dict] = []
        signal_rows: list[dict] = []
        live_ids = set(track_boxes)

        # Keep liveness through brief detector flicker; forget long-gone tracks.
        self.liveness.prune(max_age=self.cfg.track_buffer + 5)

        # Pass A: liveness. Rejected tracks are dropped here and never embedded.
        accepted: list[tuple[dict, object]] = []
        for t in tracks:
            tid = t["track_id"]

            # Only fresh measurements can provide liveness evidence.
            det_i = t.get("det_index")
            face_det = detections[det_i] if det_i is not None and 0 <= det_i < len(detections) else None
            lm_sig = signals_by_track.get(tid)
            face_visible = face_det is not None
            real_score = None
            if self._antispoof is not None and face_visible and (
                self._frame_no % max(1, self.cfg.landmark_every_n) == 0
            ):
                try:
                    real_score = self._antispoof.score(frame, face_det.bbox)
                except Exception:
                    logger.exception("Anti-spoof inference failed; rejecting face")
                    real_score = 0.0
            if real_score is not None and (self._frame_no == 1 or self._frame_no % 30 == 0):
                box_size = np.asarray(face_det.bbox)[2:] - np.asarray(face_det.bbox)[:2]
                logger.info(
                    "Liveness track=%s real_score=%.3f threshold=%.3f face_pixels=%sx%s min_pixels=%s blink_required=%s",
                    tid, real_score, self.cfg.liveness_real_threshold,
                    int(box_size[0]), int(box_size[1]), self.cfg.liveness_min_face_pixels,
                    self.cfg.liveness_require_blink,
                )
            if not self.liveness.observe(
                tid,
                blink=getattr(lm_sig, "blink", None) if run_lm and face_visible else None,
                real_score=real_score,
                timestamp=now,
                face_visible=face_visible,
            ):
                # A previously accepted track can be revoked by a spoof score.
                for tr in self.indicators.drop_track(tid):
                    key = (tid, tr.condition)
                    if key in self._open_events:
                        self.cb.close_event(self._open_events.pop(key))
                self._identities.pop(tid, None)
                self._centroids.pop(tid, None)
                self._last_embed.pop(tid, None)
                continue
            accepted.append((t, face_det))

        # Batch-embed only tracks that need a recognition vote this frame: unlocked ones
        # every frame (as before), locked ones just often enough to re-verify the label.
        # With nobody enrolled there is nothing to match, so skip embedding altogether.
        if len(self.gallery):
            pending = []
            for t, face_det in accepted:
                if face_det is None or face_det.embedding is not None:
                    continue
                tid = t["track_id"]
                ident = self._identities.get(tid)
                since = self._frame_no - self._last_embed.get(tid, -(10 ** 9))
                if ident is None or not ident.locked or since >= self.cfg.recognition_recheck_frames:
                    pending.append((tid, face_det))
            if pending:
                embed_faces(frame, [d for _, d in pending], self.cfg)
                for tid, _ in pending:
                    self._last_embed[tid] = self._frame_no

        # Pass B: recognition votes, attendance, movement, indicators.
        for t, face_det in accepted:
            tid = t["track_id"]
            self._seen_tracks.add(tid)
            box = t["tlbr"]
            cx, cy = (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0

            # --- recognition -----------------------------------------
            ident = self._identities.setdefault(tid, TrackIdentity(self.cfg))
            if face_det is not None and (face_det.embedding is not None or not len(self.gallery)):
                sid, sim = self.gallery.query(face_det.embedding)
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
            # A cached landmark must not override a current failed face detection.
            face_visible = face_det is not None
            if not face_visible:
                sig = None
            fs = FrameSignals(
                face_visible=face_visible,
                yaw=getattr(sig, "yaw", 0.0),
                pitch=getattr(sig, "pitch", 0.0),
                blink=getattr(sig, "blink", 0.0),
                movement_norm=movement_norm,
                quality=t.get("score", 0.0),
                pose_valid=getattr(sig, "pose_valid", False),
                eye_left=getattr(sig, "eye_left", None),
                eye_right=getattr(sig, "eye_right", None),
            )
            state, transitions = self.indicators.update(tid, now, fs)
            self._apply_transitions(tid, student_id, transitions)
            signal_rows.append({"timestamp": now, "track_id": tid, **asdict(fs),
                                "landmark_fresh": bool(run_lm and sig is not None),
                                "indicator": state.current_label,
                                "indicator_is_event": state.current_is_event})

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
            self._last_embed.pop(gone, None)
            self._sig_cache.pop(gone, None)

        annotated = draw_overlay(frame.copy(), views) if self.cfg.burn_in_overlay else frame

        wall_now = time.monotonic()
        self._last_t = wall_now
        # Frames per second over the last ~1 s. (Averaging per-frame 1/dt overstates the
        # rate when a camera delivers frames in bursts, e.g. showing 35 fps on a 30 fps camera.)
        self._frame_times.append(wall_now)
        if len(self._frame_times) >= 2 and self._frame_times[-1] > self._frame_times[0]:
            self._fps = (len(self._frame_times) - 1) / (self._frame_times[-1] - self._frame_times[0])
        return FrameResult(frame=annotated, track_views=views, fps=round(self._fps, 1),
                           num_tracks=len(views), signal_rows=signal_rows)

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
        if self._pool is not None:
            self._pool.shutdown(wait=False, cancel_futures=True)
            self._pool = None
