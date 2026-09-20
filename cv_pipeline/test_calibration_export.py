"""Verify CSV and media timing without models, cameras, or database writes."""
import csv
import io
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
from django.core.management import call_command
from django.test import SimpleTestCase

from cv_pipeline.capture import VideoStream
from cv_pipeline.config import PipelineConfig
from cv_pipeline.engine import FrameResult


class MeasurementExportTests(SimpleTestCase):
    def test_command_exports_separate_eyes_and_blank_manual_label(self):
        frame = np.zeros((10, 10, 3), dtype=np.uint8)
        row = dict(timestamp=12.5, track_id=1, eye_left=0.8, eye_right=0.1,
                   pose_valid=True, landmark_fresh=True, indicator="uncertain")
        stream = Mock(timestamp_seconds=12.5)
        stream.read.side_effect = [frame, None]
        engine = Mock()
        engine.process.return_value = FrameResult(frame, signal_rows=[row])
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "signals.csv"
            with patch("apps.dashboard.management.commands.run_pipeline.ClassSession.objects.get",
                       return_value=SimpleNamespace(pk=1, is_active=False, video_source="test.mp4")), \
                 patch("apps.dashboard.runtime._make_config", return_value=PipelineConfig()), \
                 patch("apps.dashboard.gallery.build_gallery", return_value=([], {})), \
                 patch("cv_pipeline.engine.MonitoringEngine", return_value=engine), \
                 patch("cv_pipeline.capture.VideoStream", return_value=stream):
                call_command("run_pipeline", session=1, signals_csv=str(destination), stdout=io.StringIO())
            with destination.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(rows[0]["eye_left"], "0.8")
            self.assertEqual(rows[0]["eye_right"], "0.1")
            self.assertEqual(rows[0]["label"], "")
            self.assertEqual(rows[0]["video_seconds"], "12.5")
            self.assertEqual(rows[0]["elapsed_seconds"], "0.0")
            self.assertEqual(engine.process.call_args.kwargs, {"timestamp": 12.5})
            engine.shutdown.assert_called_once()
            stream.release.assert_called_once()


class MediaTimeTests(SimpleTestCase):
    def test_file_uses_video_timestamp_and_frame_rate_fallback(self):
        stream = VideoStream.__new__(VideoStream)
        stream._cv2 = SimpleNamespace(CAP_PROP_POS_MSEC=1, CAP_PROP_FPS=2, CAP_PROP_POS_FRAMES=3)
        stream._is_file = True
        stream.target_width = None
        stream.cap = Mock()
        stream.cap.read.return_value = (True, np.zeros((10, 10, 3)))
        stream.cap.get.side_effect = lambda key: {1: 2500, 2: 20, 3: 51}[key]
        stream.read()
        self.assertEqual(stream.timestamp_seconds, 2.5)
        stream.cap.get.side_effect = lambda key: {1: 0, 2: 20, 3: 51}[key]
        stream.read()
        self.assertEqual(stream.timestamp_seconds, 2.5)
