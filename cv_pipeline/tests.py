"""Lightweight tests for the CV logic that does not need model weights.

Run with:  python manage.py test cv_pipeline
"""
import time
from types import SimpleNamespace

import numpy as np
from django.test import SimpleTestCase

from cv_pipeline.config import PipelineConfig
from cv_pipeline.indicators import FrameSignals, IndicatorEngine
from cv_pipeline.recognition import Gallery, TrackIdentity
from cv_pipeline.tracking import ByteTracker


def _det(x, y, score=0.9):
    return SimpleNamespace(
        bbox=np.array([x, y, x + 50, y + 80], dtype=float),
        det_score=score,
        embedding=np.random.randn(512).astype("f4"),
    )


class TrackerTests(SimpleTestCase):
    def test_ids_are_stable_across_frames(self):
        cfg = PipelineConfig()
        tracker = ByteTracker(cfg)
        ids = []
        for f in range(6):
            out = tracker.update([_det(100 + f, 100), _det(300 - f, 130)])
            ids.append(sorted(o["track_id"] for o in out))
        self.assertEqual(ids[-1], ids[-2])
        self.assertEqual(len(ids[-1]), 2)


class GalleryTests(SimpleTestCase):
    def test_unknown_stays_unknown_below_threshold(self):
        g = Gallery({1: np.random.randn(2, 512).astype("f4")})
        sid, sim = g.query(np.random.randn(512).astype("f4"))
        # A random probe should rarely clear 0.35 cosine similarity.
        self.assertTrue(sim < 0.35 or sid == 1)

    def test_voting_locks_identity(self):
        cfg = PipelineConfig()
        ident = TrackIdentity(cfg)
        for _ in range(cfg.min_votes + 2):
            ident.add(3, 0.5)
        self.assertTrue(ident.locked)
        self.assertEqual(ident.student_id, 3)


class IndicatorTests(SimpleTestCase):
    def test_condition_must_persist_before_event(self):
        cfg = PipelineConfig()
        engine = IndicatorEngine(cfg)
        t0 = time.time()
        opened = []
        for i in range(40):
            _, trans = engine.update(
                1, t0 + i * 0.25, FrameSignals(face_visible=True, pitch=30.0, pose_valid=True, quality=0.9)
            )
            opened.extend(t for t in trans if t.kind == "open")
        self.assertTrue(opened)
        first_open_frame = None
        engine2 = IndicatorEngine(cfg)
        for i in range(40):
            _, trans = engine2.update(
                1, t0 + i * 0.25, FrameSignals(face_visible=True, pitch=30.0, pose_valid=True, quality=0.9)
            )
            if any(t.kind == "open" for t in trans):
                first_open_frame = i
                break
        # 3.0s / 0.25s per frame => ~frame 12
        self.assertGreaterEqual(first_open_frame, 11)
