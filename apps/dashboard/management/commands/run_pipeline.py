"""Run the monitoring pipeline for a session headlessly (no web server).

Useful for testing against a recorded classroom video:
    python manage.py run_pipeline --session 1 --source data/clips/lecture.mp4 --show
"""
import time

from django.core.management.base import BaseCommand, CommandError

from apps.classes.models import ClassSession


class Command(BaseCommand):
    help = "Run the CV monitoring pipeline for a session and write results to the DB."

    def add_arguments(self, parser):
        parser.add_argument("--session", type=int, required=True)
        parser.add_argument("--source", default=None)
        parser.add_argument("--show", action="store_true", help="Preview window (needs a display).")
        parser.add_argument("--max-seconds", type=float, default=0.0)

    def handle(self, *args, **opts):
        import cv2

        from apps.dashboard.gallery import build_gallery
        from apps.dashboard.runtime import DjangoCallbacks, _make_config
        from cv_pipeline.capture import VideoStream
        from cv_pipeline.engine import MonitoringEngine

        try:
            session = ClassSession.objects.get(pk=opts["session"])
        except ClassSession.DoesNotExist:
            raise CommandError(f"No session with id {opts['session']}")

        cfg = _make_config()
        gallery, names = build_gallery(cfg)
        self.stdout.write(f"Gallery: {len(gallery)} templates across {len(names)} students")

        engine = MonitoringEngine(
            cfg, gallery, names, DjangoCallbacks(session.pk),
            session_active=session.is_active,
        )
        from django.conf import settings

        source = opts["source"] or session.video_source or settings.CV_PIPELINE["VIDEO_SOURCE"]
        stream = VideoStream(source, target_width=cfg.capture_width)
        start = time.time()
        try:
            while True:
                frame = stream.read()
                if frame is None:
                    break
                result = engine.process(frame)
                self.stdout.write(
                    f"\rfps={result.fps:5.1f} tracks={result.num_tracks} "
                    + " ".join(
                        f"[{v['track_id']}:{v['student_name'] or '?'}/{v['indicator']}]"
                        for v in result.track_views
                    ),
                    ending="",
                )
                if opts["show"]:
                    cv2.imshow("pipeline", result.frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                if opts["max_seconds"] and time.time() - start > opts["max_seconds"]:
                    break
        except KeyboardInterrupt:
            pass
        finally:
            engine.shutdown()
            stream.release()
            if opts["show"]:
                cv2.destroyAllWindows()
        self.stdout.write(self.style.SUCCESS("\nDone."))
