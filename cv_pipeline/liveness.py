"""Require fresh classifier evidence; a blink challenge is optional."""
from __future__ import annotations

from dataclasses import dataclass
import time
import numpy as np
from .config import PipelineConfig


def nonrigid_residual(prev_kps: np.ndarray, cur_kps: np.ndarray) -> float:
    """RMS landmark error left after the best affine fit prev -> cur, divided
    by the current inter-eye distance. Diagnostic only: perspective and detector noise can also produce error."""
    prev = np.asarray(prev_kps, dtype=np.float64)
    cur = np.asarray(cur_kps, dtype=np.float64)
    eye = float(np.linalg.norm(cur[1] - cur[0]))
    if eye < 1e-3:
        return 0.0
    design = np.hstack([prev, np.ones((len(prev), 1))])
    coef, *_ = np.linalg.lstsq(design, cur, rcond=None)
    err = design @ coef - cur
    return float(np.sqrt(np.mean(np.sum(err * err, axis=1))) / eye)


@dataclass
class _TrackLiveness:
    phase: str = "waiting"
    blink_started: float = 0.0
    blink_at: float = float("-inf")
    score_at: float = float("-inf")
    hits: int = 0
    live: bool = False
    last_seen: int = 0
    observed_at: float = float("-inf")


class LivenessGate:
    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg
        self._tracks: dict[int, _TrackLiveness] = {}
        self._tick = 0

    def is_live(self, track_id: int) -> bool:
        if not self.cfg.liveness_enabled:
            return True
        st = self._tracks.get(track_id)
        return bool(st and st.live)

    def observe(self, track_id, kps=None, blink=None, *, real_score=None,
                timestamp=None, face_visible=True):
        if not self.cfg.liveness_enabled:
            return True
        now = time.monotonic() if timestamp is None else timestamp
        cfg = self.cfg
        st = self._tracks.setdefault(track_id, _TrackLiveness())
        st.last_seen = self._tick
        if (not face_visible or now < st.observed_at
                or now - st.observed_at > cfg.liveness_score_max_age):
            st.phase = "waiting"
            st.hits = 0
            st.blink_at = float("-inf")
            st.score_at = float("-inf")
        st.observed_at = now
        if real_score is not None:
            st.score_at = now
            if np.isfinite(real_score) and cfg.liveness_real_threshold <= real_score <= 1:
                st.hits += 1
            else:
                st.hits = 0
                st.blink_at = float("-inf")
                st.phase = "waiting"
        if now - st.score_at > cfg.liveness_score_max_age:
            st.hits = 0
        if st.phase != "waiting" and now - st.blink_started > cfg.liveness_blink_max_seconds:
            st.phase = "waiting"
        if face_visible and blink is not None and np.isfinite(blink) and 0 <= blink <= 1:
            if st.phase == "waiting" and blink <= cfg.liveness_blink_open:
                st.phase = "open"
                st.blink_started = now
            elif st.phase == "open":
                if blink >= cfg.liveness_blink_closed:
                    st.phase = "closed"
                elif blink <= cfg.liveness_blink_open:
                    st.blink_started = now
            elif st.phase == "closed" and blink <= cfg.liveness_blink_open:
                st.blink_at = now
                st.phase = "open"
                st.blink_started = now
        st.live = bool(face_visible and st.hits >= cfg.liveness_min_hits
                       and now - st.score_at <= cfg.liveness_score_max_age
                       and (not cfg.liveness_require_blink
                            or now - st.blink_at <= cfg.liveness_recheck_seconds))
        return st.live

    def prune(self, max_age):
        self._tick += 1
        for tid in [t for t, st in self._tracks.items() if self._tick - st.last_seen > max_age]:
            del self._tracks[tid]

    def drop_track(self, track_id):
        self._tracks.pop(track_id, None)
