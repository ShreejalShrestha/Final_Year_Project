from __future__ import annotations

import time

from django.contrib.auth.decorators import login_required
from django.http import JsonResponse, StreamingHttpResponse
from django.shortcuts import get_object_or_404, render
from django.utils import timezone
from django.views.decorators.http import require_POST

from apps.attendance.models import Attendance
from apps.attendance.services import attendance_percentage
from apps.classes.models import ClassSession
from apps.monitoring.models import MonitoringEvent
from apps.students.models import Student

from .runtime import pipeline_registry


@login_required
def home(request):
    sessions = ClassSession.objects.select_related("course_class").exclude(
        status=ClassSession.Status.CLOSED
    )
    recent = ClassSession.objects.select_related("course_class").filter(
        status=ClassSession.Status.CLOSED
    )[:10]
    return render(request, "dashboard/home.html", {"sessions": sessions, "recent": recent})


@login_required
def live(request, session_id):
    session = get_object_or_404(
        ClassSession.objects.select_related("course_class"), pk=session_id
    )
    worker = pipeline_registry.get(session_id)
    return render(
        request,
        "dashboard/live.html",
        {"session": session, "worker_running": bool(worker and worker.is_running)},
    )


@login_required
@require_POST
def start_pipeline(request, session_id):
    session = get_object_or_404(ClassSession, pk=session_id)
    source = request.POST.get("source") or session.video_source or None
    worker = pipeline_registry.start(session_id, source)
    time.sleep(0.3)
    return JsonResponse({"running": worker.is_running, "error": worker.error})


@login_required
@require_POST
def stop_pipeline(request, session_id):
    pipeline_registry.stop(session_id)
    return JsonResponse({"running": False})


@login_required
def video_feed(request, session_id):
    worker = pipeline_registry.get(session_id)
    if worker is None:
        return JsonResponse({"error": "pipeline not started"}, status=409)

    boundary = "frame"

    def generate():
        while worker.is_running:
            jpeg, _, _ = worker.snapshot()
            if jpeg is None:
                time.sleep(0.1)
                continue
            yield (
                b"--" + boundary.encode() + b"\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
            )
            time.sleep(1 / 20)

    resp = StreamingHttpResponse(
        generate(), content_type=f"multipart/x-mixed-replace; boundary={boundary}"
    )
    resp["Cache-Control"] = "no-cache, no-store"
    return resp


@login_required
def tracks_json(request, session_id):
    worker = pipeline_registry.get(session_id)
    if worker is None:
        return JsonResponse({"running": False, "tracks": [], "fps": 0})
    _, views, fps = worker.snapshot()
    return JsonResponse(
        {"running": worker.is_running, "error": worker.error, "fps": fps, "tracks": views}
    )


@login_required
def student_profile_json(request, session_id, student_id):
    """Approved information only (proposal: Appendix D)."""
    session = get_object_or_404(ClassSession, pk=session_id)
    student = get_object_or_404(Student, pk=student_id)

    today_record = Attendance.objects.filter(student=student, session=session).first()
    history = (
        Attendance.objects.filter(student=student)
        .select_related("session__course_class")
        .order_by("-marked_at")[:10]
    )
    events = (
        MonitoringEvent.objects.filter(student=student, session=session)
        .order_by("-start_time")[:10]
    )

    return JsonResponse(
        {
            "identity": {
                "name": student.full_name,
                "student_id": student.student_id,
                "photo": student.profile_image.url if student.profile_image else None,
            },
            "academic": {
                "program": student.program,
                "semester_section": student.semester_section,
            },
            "attendance": {
                "today_status": today_record.status if today_record else "not marked",
                "recognition_time": (
                    timezone.localtime(today_record.marked_at).strftime("%H:%M:%S")
                    if today_record else None
                ),
                "percentage": attendance_percentage(student),
                "history": [
                    {
                        "class": r.session.course_class.name,
                        "date": r.session.date.isoformat(),
                        "status": r.status,
                        "source": r.source,
                    }
                    for r in history
                ],
            },
            "monitoring": {
                "recent_events": [
                    {
                        "type": e.get_event_type_display(),
                        "start": timezone.localtime(e.start_time).strftime("%H:%M:%S"),
                        "duration_s": round(e.duration_seconds, 1),
                        "open": e.is_open,
                    }
                    for e in events
                ]
            },
            "excluded_by_default": [
                "home address", "family details", "inferred personality",
                "emotion", "ethnicity", "disability", "academic ability",
            ],
        }
    )
