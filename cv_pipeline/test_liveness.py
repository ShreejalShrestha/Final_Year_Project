"""A still photo must never become a live person; a moving/blinking face must."""
import numpy as np
from django.test import SimpleTestCase

from cv_pipeline.config import PipelineConfig
from cv_pipeline.liveness import LivenessGate, nonrigid_residual

# left eye, right eye, nose, left mouth, right mouth (inter-eye distance 60 px)
FACE = np.array([[100, 100], [160, 100], [130, 130], [110, 160], [150, 160]], dtype=float)


def jitter(rng, scale=0.6):
    return FACE + rng.normal(0, scale, FACE.shape)


class LivenessTests(SimpleTestCase):
    def setUp(self):
        self.gate = LivenessGate(PipelineConfig())
        self.rng = np.random.default_rng(0)

    def test_affine_motion_is_not_nonrigid(self):
        moved = FACE * 1.3 @ np.array([[0.98, 0.1], [-0.1, 0.98]]) + [40, -25]
        self.assertLess(nonrigid_residual(FACE, moved), 0.005)

    def test_static_photo_with_sensor_noise_never_goes_live(self):
        for _ in range(300):
            self.assertFalse(self.gate.observe(1, jitter(self.rng), blink=0.05))

    def test_photo_moved_by_hand_never_goes_live(self):
        for i in range(300):
            shift = [8 * np.sin(i / 7), 5 * np.cos(i / 5)]
            kps = FACE * (1 + 0.1 * np.sin(i / 9)) + shift + self.rng.normal(0, 0.6, FACE.shape)
            self.assertFalse(self.gate.observe(1, kps, blink=0.05))

    def test_blink_makes_track_live(self):
        for blink in (0.05, 0.06, 0.8):
            live = self.gate.observe(1, FACE, blink=blink)
        self.assertTrue(live)

    def test_head_turn_makes_track_live_and_stays_live(self):
        live = False
        for i in range(30):
            kps = FACE.copy()
            kps[2, 0] += 9 * np.sin(i / 2)       # 3-D turn: nose shifts against eyes/mouth
            live = self.gate.observe(1, kps)
        self.assertTrue(live)
        self.assertTrue(self.gate.observe(1, FACE))   # latched

    def test_each_track_is_judged_separately(self):
        for _ in range(50):
            self.gate.observe(1, jitter(self.rng))
        self.gate.observe(2, FACE, blink=0.0)
        self.gate.observe(2, FACE, blink=0.9)
        self.assertFalse(self.gate.is_live(1))
        self.assertTrue(self.gate.is_live(2))

    def test_disabled_gate_lets_everything_through(self):
        gate = LivenessGate(PipelineConfig(liveness_enabled=False))
        self.assertTrue(gate.observe(1, FACE))

    def test_prune_forgets_long_gone_tracks_only(self):
        self.gate.observe(1, FACE, blink=0.0)
        self.gate.observe(1, FACE, blink=0.9)
        for _ in range(10):
            self.gate.prune(max_age=35)
        self.assertTrue(self.gate.is_live(1))
        for _ in range(40):
            self.gate.prune(max_age=35)
        self.assertFalse(self.gate.is_live(1))
