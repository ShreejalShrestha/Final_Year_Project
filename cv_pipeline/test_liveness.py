"""Regression coverage for photo motion, blink noise, stale evidence and revocation."""
import numpy as np
from django.test import SimpleTestCase
from cv_pipeline.config import PipelineConfig
from cv_pipeline.liveness import LivenessGate


class LivenessTests(SimpleTestCase):
    def setUp(self):
        self.gate = LivenessGate(PipelineConfig())

    def feed(self, blink, t, score=0.99, **kwargs):
        return self.gate.observe(1, blink=blink, real_score=score, timestamp=t, **kwargs)

    def accept(self):
        self.assertFalse(self.feed(0.05, 0))
        self.assertFalse(self.feed(0.9, 0.2))
        self.assertTrue(self.feed(0.05, 0.4))

    def test_complete_blink_and_classifier_required(self):
        self.accept()

    def test_score_swing_without_reopening_rejected(self):
        for i, blink in enumerate([0.05, 0.06, 0.8, 0.9]):
            self.assertFalse(self.feed(blink, i * 0.2))

    def test_motion_and_classifier_alone_never_pass(self):
        rng = np.random.default_rng(0)
        for i in range(300):
            self.assertFalse(self.gate.observe(1, rng.normal(size=(5, 2)), blink=0.05,
                                               real_score=0.99, timestamp=i * 0.1))

    def test_blink_cannot_override_spoof(self):
        for i, blink in enumerate([0.05, 0.9, 0.05] * 4):
            self.assertFalse(self.feed(blink, i * 0.1, score=0.1))

    def test_blink_without_classifier_rejected(self):
        for i, blink in enumerate([0.05, 0.9, 0.05]):
            self.assertFalse(self.feed(blink, i * 0.2, score=None))

    def test_spoof_revokes_accepted_track_immediately(self):
        self.accept()
        self.assertFalse(self.feed(0.05, 0.5, score=0.1))
        self.assertFalse(self.gate.is_live(1))
        for i in range(5):
            self.assertFalse(self.feed(0.05, 0.6 + i * 0.1))

    def test_stale_score_and_missing_detection_revoke(self):
        self.accept()
        self.assertFalse(self.feed(None, 1.3, score=None))
        self.assertFalse(self.feed(0.05, 1.4, face_visible=False))

    def test_blink_must_finish_in_time(self):
        self.feed(0.05, 0)
        for i in range(1, 11):
            self.assertFalse(self.feed(0.9, i * 0.2))
        self.assertFalse(self.feed(0.05, 2.2))

    def test_periodic_blink_required(self):
        self.accept()
        for i in range(1, 150):
            self.feed(0.05, 0.4 + i * 0.1)
        self.assertFalse(self.feed(0.05, 15.5))

    def test_invalid_scores_and_pruning(self):
        self.accept()
        self.assertFalse(self.feed(0.05, 0.5, score=float("nan")))
        for _ in range(40):
            self.gate.prune(35)
        self.assertFalse(self.gate.is_live(1))

    def test_disabled_gate(self):
        self.assertTrue(LivenessGate(PipelineConfig(liveness_enabled=False)).observe(1))
