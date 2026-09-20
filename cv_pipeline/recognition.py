"""Match track embeddings against the enrolled gallery, with per-track voting.

An unknown / low-confidence state is always retained: a face that does not
clear the threshold is reported as ``None`` rather than forced onto the nearest
enrolled student (proposal: Literature Review, Grother et al. 2019).
"""
from __future__ import annotations

from collections import Counter, deque

import numpy as np

from .config import PipelineConfig


class Gallery:
    def __init__(self, entries: dict[int, np.ndarray], labels: dict[int, str] | None = None):
        """`entries`: student_pk -> (N, 512) matrix of L2-normalised embeddings."""
        self.labels = labels or {}
        self._ids: list[int] = []
        mats = []
        for pk, mat in entries.items():
            mat = np.atleast_2d(np.asarray(mat, dtype=np.float32))
            norms = np.linalg.norm(mat, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            mats.append(mat / norms)
            self._ids.extend([pk] * len(mat))
        self.matrix = np.vstack(mats) if mats else np.zeros((0, 512), dtype=np.float32)
        self.ids = np.array(self._ids, dtype=np.int64)

    def __len__(self):
        return len(self.ids)

    def query(self, embedding: np.ndarray) -> tuple[int | None, float]:
        if len(self) == 0:
            return None, 0.0
        v = np.asarray(embedding, dtype=np.float32)
        n = np.linalg.norm(v)
        if n > 0:
            v = v / n
        sims = self.matrix @ v
        best = int(np.argmax(sims))
        return int(self.ids[best]), float(sims[best])


class TrackIdentity:
    """Accumulates recognition votes for one track and exposes a stable label."""

    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg
        self._votes: deque[tuple[int | None, float]] = deque(maxlen=cfg.vote_window)
        self.student_id: int | None = None
        self.confidence: float = 0.0
        self.locked = False

    def add(self, student_id: int | None, similarity: float) -> None:
        if student_id is not None and similarity >= self.cfg.recognition_threshold:
            self._votes.append((student_id, similarity))
        else:
            self._votes.append((None, similarity))
        self._recompute()

    def _recompute(self) -> None:
        named = [(sid, sim) for sid, sim in self._votes if sid is not None]
        if not named:
            return
        counts = Counter(sid for sid, _ in named)
        top_id, top_count = counts.most_common(1)[0]
        if top_count >= self.cfg.min_votes:
            sims = [sim for sid, sim in named if sid == top_id]
            self.student_id = top_id
            self.confidence = float(np.mean(sims))
            self.locked = True
