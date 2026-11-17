from unittest.mock import Mock, patch
import numpy as np
from django.test import SimpleTestCase
from cv_pipeline.antispoof import AntiSpoofClassifier
from cv_pipeline.config import PipelineConfig
from cv_pipeline.engine import MonitoringEngine
from cv_pipeline.landmarks import FaceSignal
from cv_pipeline.recognition import Gallery


class AntiSpoofTests(SimpleTestCase):
    def classifier(self):
        model = AntiSpoofClassifier.__new__(AntiSpoofClassifier)
        model.session = Mock()
        model.session.run.return_value = [np.array([[0., 8., 0.]])]
        model.input_name = "input"
        model.min_pixels = 60
        return model

    def test_bgr_raw_input_and_real_class(self):
        model = self.classifier()
        image = np.full((240, 320, 3), [10, 80, 200], dtype=np.uint8)
        self.assertGreater(model.score(image, [100, 70, 180, 160]), 0.99)
        tensor = model.session.run.call_args.args[1]["input"]
        self.assertEqual(tensor.shape, (1, 3, 80, 80))
        self.assertEqual(tensor.dtype, np.float32)
        np.testing.assert_array_equal(tensor[0, :, 40, 40], [10, 80, 200])

    def test_small_invalid_and_spoof_inputs_reject(self):
        model = self.classifier()
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        self.assertEqual(model.score(frame, [0, 0, 10, 10]), 0)
        self.assertEqual(model.score(frame, [float("nan"), 0, 100, 100]), 0)
        model.session.run.assert_not_called()
        model.session.run.return_value = [np.array([[8., 0., 0.]])]
        self.assertLess(model.score(frame, [100, 70, 180, 160]), 0.01)

    @patch("cv_pipeline.engine.detect_faces")
    @patch("cv_pipeline.engine.AntiSpoofClassifier")
    def test_spoof_blocks_views_and_attendance_even_with_blinks(self, classifier, detect):
        classifier.return_value.score.return_value = 0.01
        cb = Mock()
        engine = MonitoringEngine(PipelineConfig(burn_in_overlay=False), Gallery({}), {}, cb,
                                  enable_landmarks=False)
        engine.tracker = Mock()
        engine.tracker.update.return_value = [dict(track_id=1, tlbr=[10, 10, 90, 90],
                                                  score=0.99, det_index=0)]
        detect.return_value = [Mock(bbox=np.array([10, 10, 90, 90]), embedding=None)]
        engine._landmarker = Mock()
        for i, blink in enumerate([0.05, 0.9, 0.05] * 4):
            engine._landmarker.analyze.return_value = [FaceSignal(
                np.array([10, 10, 90, 90]), 0, 0, 0, blink, 0)]
            result = engine.process(np.zeros((120, 120, 3), dtype=np.uint8), timestamp=i * 0.1)
            self.assertEqual(result.num_tracks, 0)
        cb.mark_attendance.assert_not_called()
        cb.identity_locked.assert_not_called()
