"""Reject faces that never show signs of being a real, moving 3-D person.

A still photo (printed, on a phone, on a poster) is detected as a face just
like a real one. What it cannot do is *change shape*: a flat picture only ever
undergoes a 2-D (affine/perspective) transformation of its landmarks, and it
never blinks. A real face blinks, talks and rotates in 3-D, so its five facial
landmarks deform in ways no affine map can explain.

Per track we collect two kinds of evidence and latch the track as "live" once
either is seen (a track stays live for its lifetime):

* **Blink swing** - the MediaPipe eyeBlink blendshape moves by a large amount.
* **Non-rigid landmark motion** - after the best affine fit between the
  landmarks now and a few frames ago, the leftover error (as a fraction of
  inter-eye distance) is above `liveness_nonrigid_thresh` several times.

Until a track is live it is hidden: no box, no recognition, no attendance, no
indicators.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from .config import PipelineConfig


def nonrigid_residual(prev_kps: np.ndarray, cur_kps: np.ndarray) -> float:
    """RMS landmark error left after the best affine fit prev -> cur, divided
    by the current inter-eye distance. ~0 for a flat picture however it moves."""
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
    kps_history: deque
    blink_min: float = 1.0
    blink_max: float = 0.0
    nonrigid_hits: int = 0
    live: bool = False
    frames_seen: int = 0
    last_seen: int = 0


class LivenessGate:
    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg
        self._tracks: dict[int, _TrackLiveness] = {}
        self._tick = 0

    def _state(self, track_id: int) -> _TrackLiveness:
        st = self._tracks.get(track_id)
        if st is None:
            st = _TrackLiveness(kps_history=deque(maxlen=max(2, self.cfg.liveness_lag_frames + 1)))
            self._tracks[track_id] = st
        return st

    def is_live(self, track_id: int) -> bool:
        if not self.cfg.liveness_enabled:
            return True
        st = self._tracks.get(track_id)
        return bool(st and st.live)

    def observe(self, track_id: int, kps: np.ndarray | None, blink: float | None = None) -> bool:
        """Feed one frame of evidence; returns whether the track is live."""
        if not self.cfg.liveness_enabled:
            return True
        st = self._state(track_id)
        st.last_seen = self._tick
        if st.live:
            return True
        cfg = self.cfg

        if blink is not None and np.isfinite(blink):
            st.blink_min = min(st.blink_min, float(blink))
            st.blink_max = max(st.blink_max, float(blink))
            if st.blink_max - st.blink_min >= cfg.liveness_blink_swing:
                st.live = True
                return True

        if kps is not None:
            kps = np.asarray(kps, dtype=np.float64)
            st.frames_seen += 1
            if len(st.kps_history) == st.kps_history.maxlen:
                if nonrigid_residual(st.kps_history[0], kps) >= cfg.liveness_nonrigid_thresh:
                    st.nonrigid_hits += 1
                    if st.nonrigid_hits >= cfg.liveness_min_hits:
                        st.live = True
            st.kps_history.append(kps)
        return st.live

    def prune(self, max_age: int) -> None:
        """Advance one frame and forget tracks not observed for `max_age` frames."""
        self._tick += 1
        for tid in [t for t, st in self._tracks.items() if self._tick - st.last_seen > max_age]:
            del self._tracks[tid]

    def drop_track(self, track_id: int) -> None:
        self._tracks.pop(track_id, None)
