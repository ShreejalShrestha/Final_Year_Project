"""Compact ByteTrack-style multi-object tracker.

Self-contained (Kalman filter + greedy Hungarian matching via SciPy) so the
project does not need the `lap` / `cython_bbox` build chain. Follows the
two-stage association from Zhang et al. (2022): match high-confidence
detections first, then recover low-confidence ones against unmatched tracks.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

from .config import PipelineConfig


class KalmanFilter:
    """8-D state (cx, cy, aspect, h, vx, vy, va, vh) — the SORT/ByteTrack model."""

    def __init__(self):
        ndim, dt = 4, 1.0
        self._motion_mat = np.eye(2 * ndim, 2 * ndim)
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt
        self._update_mat = np.eye(ndim, 2 * ndim)
        self._std_weight_position = 1.0 / 20
        self._std_weight_velocity = 1.0 / 160

    def initiate(self, measurement):
        mean_pos = measurement
        mean_vel = np.zeros_like(measurement)
        mean = np.r_[mean_pos, mean_vel]
        std = [
            2 * self._std_weight_position * measurement[3],
            2 * self._std_weight_position * measurement[3],
            1e-2,
            2 * self._std_weight_position * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            1e-5,
            10 * self._std_weight_velocity * measurement[3],
        ]
        covariance = np.diag(np.square(std))
        return mean, covariance

    def predict(self, mean, covariance):
        std_pos = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-2,
            self._std_weight_position * mean[3],
        ]
        std_vel = [
            self._std_weight_velocity * mean[3],
            self._std_weight_velocity * mean[3],
            1e-5,
            self._std_weight_velocity * mean[3],
        ]
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))
        mean = self._motion_mat @ mean
        covariance = self._motion_mat @ covariance @ self._motion_mat.T + motion_cov
        return mean, covariance

    def update(self, mean, covariance, measurement):
        std = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-1,
            self._std_weight_position * mean[3],
        ]
        innovation_cov = np.diag(np.square(std))
        projected_mean = self._update_mat @ mean
        projected_cov = self._update_mat @ covariance @ self._update_mat.T + innovation_cov

        kalman_gain = np.linalg.solve(
            projected_cov.T, (covariance @ self._update_mat.T).T
        ).T
        innovation = measurement - projected_mean
        new_mean = mean + innovation @ kalman_gain.T
        new_covariance = covariance - kalman_gain @ projected_cov @ kalman_gain.T
        return new_mean, new_covariance


def _tlbr_to_xyah(tlbr):
    x1, y1, x2, y2 = tlbr
    w, h = x2 - x1, y2 - y1
    return np.array([x1 + w / 2, y1 + h / 2, w / max(h, 1e-6), h], dtype=np.float64)


def _xyah_to_tlbr(xyah):
    cx, cy, a, h = xyah
    w = a * h
    return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dtype=np.float64)


def _iou_matrix(tracks_tlbr: np.ndarray, dets_tlbr: np.ndarray) -> np.ndarray:
    if len(tracks_tlbr) == 0 or len(dets_tlbr) == 0:
        return np.zeros((len(tracks_tlbr), len(dets_tlbr)))
    tl = np.maximum(tracks_tlbr[:, None, :2], dets_tlbr[None, :, :2])
    br = np.minimum(tracks_tlbr[:, None, 2:], dets_tlbr[None, :, 2:])
    wh = np.clip(br - tl, 0, None)
    inter = wh[..., 0] * wh[..., 1]
    area_t = np.prod(tracks_tlbr[:, 2:] - tracks_tlbr[:, :2], axis=1)
    area_d = np.prod(dets_tlbr[:, 2:] - dets_tlbr[:, :2], axis=1)
    union = area_t[:, None] + area_d[None, :] - inter + 1e-6
    return inter / union


class Track:
    _next_id = 1

    def __init__(self, tlbr, score, embedding, kf: KalmanFilter):
        self.kf = kf
        self.mean, self.cov = kf.initiate(_tlbr_to_xyah(tlbr))
        self.track_id = Track._next_id
        Track._next_id += 1
        self.score = score
        self.embedding = embedding
        self.hits = 1
        self.age = 0
        self.time_since_update = 0
        self.state = "tentative"

    @staticmethod
    def reset_ids():
        Track._next_id = 1

    @property
    def tlbr(self):
        return _xyah_to_tlbr(self.mean[:4])

    def predict(self):
        self.mean, self.cov = self.kf.predict(self.mean, self.cov)
        self.age += 1
        self.time_since_update += 1

    def update(self, tlbr, score, embedding):
        self.mean, self.cov = self.kf.update(self.mean, self.cov, _tlbr_to_xyah(tlbr))
        self.score = score
        if embedding is not None:
            self.embedding = embedding
        self.hits += 1
        self.time_since_update = 0
        if self.state == "tentative" and self.hits >= 3:
            self.state = "confirmed"


class ByteTracker:
    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg
        self.kf = KalmanFilter()
        self.tracks: list[Track] = []
        Track.reset_ids()

    def _match(self, tracks, dets_tlbr, thresh):
        if not tracks or len(dets_tlbr) == 0:
            return [], list(range(len(tracks))), list(range(len(dets_tlbr)))
        iou = _iou_matrix(np.array([t.tlbr for t in tracks]), dets_tlbr)
        cost = 1.0 - iou
        rows, cols = linear_sum_assignment(cost)
        matches, un_t, un_d = [], [], []
        matched_t, matched_d = set(), set()
        for r, c in zip(rows, cols):
            if iou[r, c] >= (1.0 - thresh):
                matches.append((r, c))
                matched_t.add(r)
                matched_d.add(c)
        un_t = [i for i in range(len(tracks)) if i not in matched_t]
        un_d = [j for j in range(len(dets_tlbr)) if j not in matched_d]
        return matches, un_t, un_d

    def update(self, detections) -> list[dict]:
        """`detections`: list of objects with .bbox, .det_score, .embedding.

        Returns a list of dicts: {track_id, tlbr, score, det_index or None}.
        """
        cfg = self.cfg
        for t in self.tracks:
            t.predict()

        dets = list(detections)
        boxes = np.array([d.bbox for d in dets], dtype=np.float64) if dets else np.zeros((0, 4))
        scores = np.array([d.det_score for d in dets]) if dets else np.zeros((0,))

        high = [i for i, s in enumerate(scores) if s >= cfg.track_high_thresh]
        low = [i for i, s in enumerate(scores) if cfg.track_low_thresh <= s < cfg.track_high_thresh]

        # Stage 1: confirmed + tentative tracks vs high-score detections.
        stage1_tracks = list(range(len(self.tracks)))
        m1, un_t1, un_d1 = self._match(
            [self.tracks[i] for i in stage1_tracks],
            boxes[high] if high else np.zeros((0, 4)),
            cfg.track_match_thresh,
        )
        det_assignment: dict[int, int] = {}          # track_id -> detection index
        for tr, dc in m1:
            di = high[dc]
            track = self.tracks[stage1_tracks[tr]]
            track.update(boxes[di], scores[di], dets[di].embedding)
            det_assignment[track.track_id] = di

        # Stage 2: remaining tracks vs low-score detections (occlusion recovery).
        remaining_tracks = [stage1_tracks[i] for i in un_t1]
        m2, _un_t2, _un_d2 = self._match(
            [self.tracks[i] for i in remaining_tracks],
            boxes[low] if low else np.zeros((0, 4)),
            0.5,
        )
        for tr, dc in m2:
            di = low[dc]
            track = self.tracks[remaining_tracks[tr]]
            track.update(boxes[di], scores[di], None)
            det_assignment[track.track_id] = di

        # New tracks from unmatched high-score detections.
        for dc in un_d1:
            di = high[dc]
            new_track = Track(boxes[di], scores[di], dets[di].embedding, self.kf)
            self.tracks.append(new_track)
            det_assignment[new_track.track_id] = di

        # Drop stale tracks.
        self.tracks = [t for t in self.tracks if t.time_since_update <= cfg.track_buffer]

        results = []
        for t in self.tracks:
            if t.time_since_update > 0:
                continue
            results.append(
                {
                    "track_id": t.track_id,
                    "tlbr": t.tlbr.tolist(),
                    "score": float(t.score),
                    "det_index": det_assignment.get(t.track_id),
                }
            )
        return results
