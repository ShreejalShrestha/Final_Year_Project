"""Behavioural tests for the frame-rate optimisations (no model weights needed)."""
import threading
import time
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
from django.test import SimpleTestCase

from cv_pipeline.capture import VideoStream
from cv_pipeline.config import PipelineConfig
from cv_pipeline.engine import MonitoringEngine
from cv_pipeline.landmarks import FaceSignal
from cv_pipeline.recognition import Gallery

FRAME = np.zeros((120, 160, 3), dtype=np.uint8)
VEC = np.ones(512, dtype=np.float32) / np.sqrt(512)


def _det(x=10):
    """A fresh detection per call, as the real detector returns."""
    return SimpleNamespace(bbox=np.array([x, 10, x + 40, 50], dtype=float),
                           det_score=0.9, kps=np.zeros((5, 2)), embedding=None)


def _engine(gallery, tracks, **cfg):
    cfg.setdefault("liveness_enabled", False)
    engine = MonitoringEngine(PipelineConfig(burn_in_overlay=False, **cfg), gallery, {1: "Asha"},
                              Mock(), enable_landmarks=False)
    engine.tracker = Mock()
    engine.tracker.update.return_value = tracks
    return engine


def _track(tid, det_index, x=10):
    return dict(track_id=tid, tlbr=[x, 10, x + 40, 50], score=0.9, det_index=det_index)


def _fake_embed(frame, dets, cfg=None):
    for d in dets:
        d.embedding = VEC.copy()


class LandmarkOverlapTests(SimpleTestCase):
    @patch("cv_pipeline.engine.detect_faces")
    def test_landmarks_run_while_detection_is_still_running(self, detect):
        """MediaPipe (CPU) must start before detection (GPU) finishes, not after it."""
        started = threading.Event()
        overlapped = []

        def analyze(frame):
            started.set()
            return [FaceSignal(np.array([10, 10, 50, 50]), 0, 0, 0, 0.1, 0, 0.1, 0.1, True)]

        def slow_detect(frame, cfg, with_embeddings=True):
            overlapped.append(started.wait(timeout=2.0))   # only true if analyze runs concurrently
            return [_det()]

        detect.side_effect = slow_detect
        engine = _engine(Gallery({}), [_track(1, 0)])
        engine._landmarker = Mock(analyze=Mock(side_effect=analyze))
        result = engine.process(FRAME, timestamp=0)
        engine.shutdown()
        self.assertEqual(overlapped, [True])
        self.assertEqual(result.num_tracks, 1)
        self.assertEqual(result.track_views[0]["indicator"], "looking_forward")


class EmbeddingPolicyTests(SimpleTestCase):
    @patch("cv_pipeline.engine.embed_faces", side_effect=_fake_embed)
    @patch("cv_pipeline.engine.detect_faces")
    def test_locked_tracks_are_only_reverified_periodically(self, detect, embed):
        detect.side_effect = lambda *a, **k: [_det()]
        cfg = PipelineConfig()
        engine = _engine(Gallery({1: VEC[None, :]}), [_track(1, 0)], recognition_recheck_frames=6)
        views = [engine.process(FRAME, timestamp=i * 0.03).track_views[0] for i in range(30)]
        # Unlocked: embedded every frame until min_votes agree; locked: every 6th frame.
        self.assertEqual(embed.call_count, cfg.min_votes + 4)
        self.assertEqual(views[cfg.min_votes - 1]["student_id"], 1)
        self.assertTrue(all(v["student_id"] == 1 for v in views[cfg.min_votes - 1:]))

    @patch("cv_pipeline.engine.embed_faces", side_effect=_fake_embed)
    @patch("cv_pipeline.engine.detect_faces")
    def test_all_tracks_in_a_frame_share_one_batched_call(self, detect, embed):
        detect.side_effect = lambda *a, **k: [_det(10), _det(80)]
        engine = _engine(Gallery({1: VEC[None, :]}), [_track(1, 0, 10), _track(2, 1, 80)])
        engine.process(FRAME, timestamp=0)
        self.assertEqual(embed.call_count, 1)
        self.assertEqual(len(embed.call_args.args[1]), 2)

    @patch("cv_pipeline.engine.embed_faces")
    @patch("cv_pipeline.engine.detect_faces")
    def test_nothing_is_embedded_when_nobody_is_enrolled(self, detect, embed):
        detect.side_effect = lambda *a, **k: [_det()]
        engine = _engine(Gallery({}), [_track(1, 0)])
        result = engine.process(FRAME, timestamp=0)
        embed.assert_not_called()
        self.assertEqual(result.num_tracks, 1)
        self.assertIsNone(result.track_views[0]["student_id"])

    @patch("cv_pipeline.engine.embed_faces", side_effect=_fake_embed)
    @patch("cv_pipeline.engine.detect_faces")
    def test_rejected_faces_are_never_embedded(self, detect, embed):
        detect.side_effect = lambda *a, **k: [_det()]
        engine = _engine(Gallery({1: VEC[None, :]}), [_track(1, 0)], liveness_enabled=True)
        engine._antispoof = Mock(score=Mock(return_value=0.0))   # classified as a spoof
        for i in range(5):
            self.assertEqual(engine.process(FRAME, timestamp=i * 0.03).num_tracks, 0)
        embed.assert_not_called()

    def test_embed_faces_skips_work_for_already_embedded_detections(self):
        from cv_pipeline.detection import embed_faces
        det = _det()
        det.embedding = VEC
        with patch("cv_pipeline.detection.get_face_app") as app:
            embed_faces(FRAME, [det])
            app.assert_not_called()


class FakeCapture:
    """cv2.VideoCapture stand-in: yields numbered frames quickly, then ends."""

    def __init__(self, source, total=400, interval=0.001):
        self.total, self.interval, self.n, self.released = total, interval, 0, False

    def isOpened(self):
        return True

    def read(self):
        if self.n >= self.total or self.released:
            return False, None
        time.sleep(self.interval)
        self.n += 1
        return True, np.full((4, 4, 3), self.n % 250, dtype=np.uint8)

    def get(self, prop):
        return 0

    def release(self):
        self.released = True


class LatestFrameCaptureTests(SimpleTestCase):
    def test_slow_consumer_gets_the_newest_frame_not_a_stale_one(self):
        with patch("cv2.VideoCapture", FakeCapture):
            stream = VideoStream("0", target_width=None)
            first = stream.read()
            time.sleep(0.12)                      # consumer is busy; camera keeps delivering
            later = stream.read()
            stream.release()
        self.assertIsNotNone(first)
        self.assertGreater(int(later[0, 0, 0]), int(first[0, 0, 0]) + 20)   # skipped ahead

    def test_returns_none_when_the_source_ends_then_recovers_on_reconnect(self):
        with patch("cv2.VideoCapture", lambda src: FakeCapture(src, total=3)):
            stream = VideoStream("0", target_width=None)
            seen = 0
            for _ in range(10):
                if stream.read() is None:
                    break
                seen += 1
            self.assertGreaterEqual(seen, 1)
            self.assertIsNone(stream.read())
            stream.reconnect(delay=0)
            self.assertIsNotNone(stream.read())
            stream.release()

    def test_release_stops_the_reader_thread(self):
        with patch("cv2.VideoCapture", FakeCapture):
            stream = VideoStream("0", target_width=None)
            thread = stream._thread
            self.assertTrue(thread.is_alive())
            stream.release()
        self.assertFalse(thread.is_alive())

    def test_files_are_read_synchronously_so_no_frame_is_skipped(self):
        with patch("cv2.VideoCapture", lambda src: FakeCapture(src, total=5, interval=0)):
            stream = VideoStream("0", target_width=None, latest_only=False)
            frames = [stream.read() for _ in range(5)]
            self.assertIsNone(stream._thread)
            self.assertEqual([int(f[0, 0, 0]) for f in frames], [1, 2, 3, 4, 5])
            stream.release()


class FramePublishTests(SimpleTestCase):
    def test_stream_waits_for_a_new_frame_instead_of_resending(self):
        from apps.dashboard.runtime import PipelineWorker
        worker = PipelineWorker(1, "0")

        def publish():
            time.sleep(0.05)
            with worker._frame_ready:
                worker.latest_jpeg = b"jpeg"
                worker._frame_seq += 1
                worker._frame_ready.notify_all()

        threading.Thread(target=publish).start()
        start = time.perf_counter()
        seq, jpeg = worker.wait_for_frame(0, timeout=2.0)
        self.assertEqual((seq, jpeg), (1, b"jpeg"))
        self.assertLess(time.perf_counter() - start, 1.0)            # woken by notify, not by timeout
        self.assertEqual(worker.wait_for_frame(1, timeout=0.05), (1, b"jpeg"))   # nothing newer
