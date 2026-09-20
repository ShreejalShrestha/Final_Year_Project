"""Regression scenarios for uncertainty, eye closure, calibration and missing data."""
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
from django.test import SimpleTestCase, override_settings

from cv_pipeline.config import PipelineConfig, load_calibration
from cv_pipeline.indicators import FrameSignals, IndicatorEngine
from cv_pipeline.engine import MonitoringEngine
from cv_pipeline.landmarks import FaceSignal, _euler_from_matrix
from cv_pipeline.recognition import Gallery


def signal(**overrides):
    return FrameSignals(**dict(dict(face_visible=True, pose_valid=True, quality=0.9,
                                   eye_left=0.1, eye_right=0.1), **overrides))


class RobustIndicatorTests(SimpleTestCase):
    def setUp(self):
        self.engine = IndicatorEngine(PipelineConfig())

    def feed(self, measurements, start=0, count=40, step=0.1):
        transitions = []
        for i in range(count):
            state, changes = self.engine.update(1, start + i * step, measurements)
            transitions.extend(changes)
        return state, transitions

    def test_missing_low_quality_and_nonfinite_pose_never_mean_forward(self):
        for measurement in (FrameSignals(True), signal(pose_valid=False), signal(quality=0.2),
                            signal(yaw=float("nan")), signal(eye_left=None), signal(pitch=-40)):
            with self.subTest(measurement=measurement):
                state, _ = self.engine.update(1, 0, measurement)
                self.assertEqual(state.current_label, "uncertain")
        state, _ = self.engine.update(2, 0, signal())
        self.assertEqual(state.current_label, "looking_forward")

    def test_wink_and_normal_blink_do_not_open_eye_closure_event(self):
        _, changes = self.feed(signal(eye_left=0.9))
        self.assertFalse(changes)
        self.engine = IndicatorEngine(PipelineConfig())
        state, changes = self.feed(signal(eye_left=0.9, eye_right=0.9), count=3)
        self.assertEqual(state.current_label, "eyes_closed")
        self.assertFalse(changes)
        _, changes = self.feed(signal(), start=0.3, count=10)
        self.assertFalse(changes)

    def test_both_eyes_must_remain_closed_to_open_event(self):
        state, changes = self.feed(signal(eye_left=0.9, eye_right=0.9))
        self.assertEqual(state.current_label, "possible_drowsiness")
        self.assertTrue(state.current_is_event)
        self.assertEqual([(c.kind, c.condition) for c in changes], [("open", "possible_drowsiness")])

    def test_hysteresis_and_release_tolerance(self):
        self.feed(signal(yaw=30), count=35)
        state, changes = self.engine.update(1, 3.5, signal(yaw=25))
        self.assertEqual(state.current_label, "looking_away")
        self.assertFalse(changes)
        self.engine.update(1, 3.6, signal(yaw=0))
        state, changes = self.engine.update(1, 3.7, signal(yaw=30))
        self.assertTrue(state.current_is_event)
        self.assertFalse(changes)
        self.engine.update(1, 3.8, signal())
        state, changes = self.engine.update(1, 4.2, signal())
        self.assertEqual(state.current_label, "looking_forward")
        self.assertEqual([c.kind for c in changes], ["close"])

    def test_missing_samples_do_not_count_toward_duration(self):
        self.feed(signal(pitch=30), count=29)  # 2.8 seconds of evidence
        state, changes = self.engine.update(1, 2.9, signal(pose_valid=False))
        self.assertEqual(state.current_label, "uncertain")
        self.assertFalse(changes)
        _, changes = self.engine.update(1, 3.1, signal(pitch=30))
        self.assertFalse(changes)  # the missing interval cannot complete 3 seconds
        _, changes = self.engine.update(1, 3.4, signal(pitch=30))
        self.assertEqual([c.kind for c in changes], ["open"])

    def test_long_missing_interval_or_processing_pause_resets_evidence(self):
        for missing in (True, False):
            with self.subTest(missing=missing):
                self.engine = IndicatorEngine(PipelineConfig())
                self.feed(signal(pitch=30), count=29)
                if missing:
                    self.engine.update(1, 2.9, signal(pose_valid=False))
                _, changes = self.engine.update(1, 5, signal(pitch=30))
                self.assertFalse(changes)

    def test_expired_dropout_resets_even_when_next_sample_recovers(self):
        self.feed(signal(pitch=30), count=29)
        self.engine.update(1, 2.9, signal(pose_valid=False))
        _, changes = self.engine.update(1, 3.5, signal(pitch=30))
        self.assertFalse(changes)

    def test_camera_offsets_and_pitch_direction(self):
        self.engine = IndicatorEngine(PipelineConfig(neutral_yaw_deg=35, neutral_pitch_deg=10,
                                                    pitch_direction=-1))
        state, _ = self.engine.update(1, 0, signal(yaw=35, pitch=10))
        self.assertEqual(state.current_label, "looking_forward")
        state, changes = self.feed(signal(yaw=35, pitch=-20), start=0.1)
        self.assertEqual(state.current_label, "head_down")
        self.assertEqual([c.condition for c in changes], ["head_down"])

    def test_track_removal_closes_event(self):
        self.feed(signal(pitch=30))
        self.assertEqual([c.kind for c in self.engine.drop_track(1)], ["close"])
        self.assertFalse(self.engine.drop_track(1))

    def test_face_absence_still_opens_event(self):
        state, changes = self.feed(signal(face_visible=False))
        self.assertEqual(state.current_label, "face_not_visible")
        self.assertEqual([c.condition for c in changes], ["face_not_visible"])

    def test_open_event_closes_after_missing_measurements(self):
        self.feed(signal(pitch=30))
        state, changes = self.engine.update(1, 4.0, signal(pose_valid=False))
        self.assertEqual(state.current_label, "uncertain")
        self.assertFalse(changes)
        _, changes = self.engine.update(1, 4.4, signal(pose_valid=False))
        self.assertEqual([c.kind for c in changes], ["close"])

    def test_cooldown_prevents_duplicate_events(self):
        self.feed(signal(eye_left=0.9, eye_right=0.9))
        self.engine.update(1, 4, signal())
        self.engine.update(1, 4.4, signal())
        _, changes = self.feed(signal(eye_left=0.9, eye_right=0.9), start=4.5, count=35)
        self.assertFalse(changes)
        _, changes = self.feed(signal(eye_left=0.9, eye_right=0.9), start=8, count=10)
        self.assertEqual([c.kind for c in changes], ["open"])


class LandmarkIntegrationTests(SimpleTestCase):
    def make_engine(self):
        engine = MonitoringEngine(PipelineConfig(landmark_every_n=1, burn_in_overlay=False),
                                  Gallery({}), {}, Mock(), enable_landmarks=False)
        engine._landmarker = Mock()
        engine._landmarker.analyze.return_value = [FaceSignal(
            np.array([10, 10, 50, 50]), 0, 0, 0, 0.1, 0, 0.1, 0.1, True)]
        engine.tracker = Mock()
        engine.tracker.update.return_value = [dict(track_id=1, tlbr=[10, 10, 50, 50],
                                                  score=0.9, det_index=0)]
        return engine

    @patch("cv_pipeline.engine.detect_faces")
    def test_stale_cache_exception_and_disabled_landmarks_are_uncertain(self, detect):
        detect.return_value = [Mock(embedding=None)]
        engine = self.make_engine()
        frame = np.zeros((60, 60, 3), dtype=np.uint8)
        self.assertEqual(engine.process(frame, timestamp=0).track_views[0]["indicator"], "looking_forward")
        engine.cfg.landmark_every_n = 100
        self.assertEqual(engine.process(frame, timestamp=0.2).track_views[0]["indicator"], "looking_forward")
        self.assertEqual(engine.process(frame, timestamp=0.8).track_views[0]["indicator"], "uncertain")
        engine.cfg.landmark_every_n = 1
        engine._landmarker.analyze.side_effect = RuntimeError("measurement failed")
        self.assertEqual(engine.process(frame, timestamp=0.9).track_views[0]["indicator"], "uncertain")
        engine._landmarker = None
        self.assertEqual(engine.process(frame, timestamp=1).track_views[0]["indicator"], "uncertain")

    @patch("cv_pipeline.engine.detect_faces")
    def test_cached_face_cannot_hide_current_detection_loss(self, detect):
        detect.return_value = [Mock(embedding=None)]
        engine = self.make_engine()
        frame = np.zeros((60, 60, 3), dtype=np.uint8)
        engine.process(frame, timestamp=0)
        engine.cfg.landmark_every_n = 100
        engine.tracker.update.return_value[0]["det_index"] = None
        result = engine.process(frame, timestamp=0.1)
        self.assertEqual(result.track_views[0]["indicator"], "face_not_visible")
        self.assertFalse(result.signal_rows[0]["pose_valid"])

    def test_pose_rotation_sign_is_explicit(self):
        angle = np.deg2rad(30)
        matrix = np.eye(4)
        matrix[1:3, 1:3] = [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
        pitch, yaw, roll = _euler_from_matrix(matrix)
        self.assertAlmostEqual(pitch, 30)
        self.assertAlmostEqual(yaw, 0)


class CalibrationTests(SimpleTestCase):
    def test_runtime_loads_calibration_for_dashboard_and_command(self):
        from apps.dashboard.runtime import _make_config
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "profile.json"
            path.write_text('{"neutral_yaw_deg": 17}', encoding="utf-8")
            with override_settings(CV_PIPELINE={"DEVICE": "cpu", "RECOGNITION_THRESHOLD": 0.35,
                                               "MODEL_DIR": directory, "CALIBRATION_FILE": str(path)}):
                cfg = _make_config()
                self.assertEqual(cfg.neutral_yaw_deg, 17)
                self.assertFalse(cfg.burn_in_overlay)

    def test_json_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "calibration.json"
            for values in ({"typo": 1}, {"eye_closed_blink_score": 2},
                           {"looking_away_exit_deg": 30}, {"pitch_direction": 0},
                           {"neutral_yaw_deg": float("nan")}, {"head_down_min_seconds": -1},
                           {"neutral_yaw_deg": True}, []):
                path.write_text(json.dumps(values), encoding="utf-8")
                with self.subTest(values=values), self.assertRaises(ValueError):
                    load_calibration(str(path))
            path.write_text('{"neutral_yaw_deg": 12}', encoding="utf-8")
            self.assertEqual(load_calibration(str(path)), {"neutral_yaw_deg": 12})
