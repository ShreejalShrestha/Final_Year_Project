"""Run the monitoring pipeline for a session headlessly (no web server).

Useful for testing against a recorded classroom video:
    python manage.py run_pipeline --session 1 --source data/clips/lecture.mp4 --show
"""
import time
import csv

from django.core.management.base import BaseCommand, CommandError

from apps.classes.models import ClassSession


class Command(BaseCommand):
    help = "Run the CV monitoring pipeline for a session and write results to the DB."

    def add_arguments(self, parser):
        parser.add_argument("--session", type=int, required=True)
        parser.add_argument("--source", default=None)
        parser.add_argument("--show", action="store_true", help="Preview window (needs a display).")
        parser.add_argument("--max-seconds", type=float, default=0.0)
        parser.add_argument("--signals-csv", help="Export measurements for manual labeling/calibration (new file).")

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

        try:
            cfg = _make_config()
        except (OSError, ValueError) as exc:
            raise CommandError(f"Invalid calibration configuration: {exc}") from exc
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
        export_file = None
        writer = None
        export_start = None
        try:
            if opts["signals_csv"]:
                export_file = open(opts["signals_csv"], "x", newline="", encoding="utf-8")
            while True:
                frame = stream.read()
                if frame is None:
                    break
                result = engine.process(frame, timestamp=stream.timestamp_seconds)
                if export_file:
                    for row in result.signal_rows:
                        if export_start is None:
                            export_start = row["timestamp"]
                        row = {"session_id": session.pk, **row,
                               "elapsed_seconds": row["timestamp"] - export_start,
                               "video_seconds": stream.timestamp_seconds,
                               "label": ""}
                        if writer is None:
                            writer = csv.DictWriter(export_file, fieldnames=list(row))
                            writer.writeheader()
                        writer.writerow(row)
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
            if export_file:
                export_file.close()
            engine.shutdown()
            stream.release()
            if opts["show"]:
                cv2.destroyAllWindows()
        self.stdout.write(self.style.SUCCESS("\nDone."))
