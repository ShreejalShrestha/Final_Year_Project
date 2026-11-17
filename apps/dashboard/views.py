from __future__ import annotations

import time
from urllib.parse import urlencode

from django.contrib.auth.decorators import login_required
from django.core.paginator import Paginator
from django.db.models import Case, Count, IntegerField, Q, When
from django.http import JsonResponse, StreamingHttpResponse
from django.shortcuts import get_object_or_404, render
from django.utils import timezone
from django.views.decorators.http import require_POST

from apps.attendance.models import Attendance
from apps.attendance.services import attendance_percentage
from apps.classes.models import ClassSession, CourseClass
from apps.monitoring.models import MonitoringEvent
from apps.students.models import Student

from .runtime import pipeline_registry


SESSIONS_PER_PAGE = 15


def _sessions_with_counts():
    """Sessions plus the per-session figures shown in lists (one query)."""
    return ClassSession.objects.select_related(
        "course_class", "teacher", "teacher__profile"
    ).annotate(
        present_count=Count(
            "attendance_records",
            filter=Q(attendance_records__status=Attendance.Status.PRESENT),
            distinct=True,
        ),
        event_count=Count("monitoring_events", distinct=True),
    )


def _session_totals():
    S = ClassSession.Status
    return ClassSession.objects.aggregate(
        active=Count("pk", filter=Q(status=S.ACTIVE)),
        scheduled=Count("pk", filter=Q(status=S.SCHEDULED)),
        closed=Count("pk", filter=Q(status=S.CLOSED)),
        total=Count("pk"),
    )


@login_required
def home(request):
    """Overview: what is happening now, what is coming up, what just finished."""
    S = ClassSession.Status
    base = _sessions_with_counts()
    today = timezone.localdate()
    ready_students = Student.objects.filter(
        is_active=True, consent_given=True, embeddings__isnull=False
    ).distinct()
    context = {
        "totals": _session_totals(),
        "active_sessions": base.filter(status=S.ACTIVE).order_by("-started_at", "-date"),
        "scheduled_sessions": base.filter(status=S.SCHEDULED).order_by("date")[:5],
        "recent_sessions": base.filter(status=S.CLOSED).order_by("-ended_at", "-date")[:5],
        "present_today": Attendance.objects.filter(
            session__date=today, status=Attendance.Status.PRESENT
        ).count(),
        "active_students": Student.objects.filter(is_active=True).count(),
        "ready_students": ready_students.count(),
        "class_count": CourseClass.objects.count(),
        "recent_attendance": Attendance.objects.select_related(
            "student", "session__course_class"
        ).order_by("-marked_at")[:6],
    }
    return render(request, "dashboard/home.html", context)


@login_required
def sessions(request):
    """All sessions, filterable by status and searchable by class or teacher."""
    S = ClassSession.Status
    status = request.GET.get("status", "")
    if status not in {S.ACTIVE, S.SCHEDULED, S.CLOSED}:
        status = ""
    query = request.GET.get("q", "").strip()

    qs = _sessions_with_counts()
    if status:
        qs = qs.filter(status=status)
    if query:
        qs = qs.filter(
            Q(course_class__name__icontains=query)
            | Q(course_class__code__icontains=query)
            | Q(teacher__username__icontains=query)
            | Q(teacher__first_name__icontains=query)
            | Q(teacher__last_name__icontains=query)
            | Q(teacher__profile__display_name__icontains=query)
        )
    # Live sessions first, then upcoming, then history (newest first).
    qs = qs.annotate(
        rank=Case(
            When(status=S.ACTIVE, then=0),
            When(status=S.SCHEDULED, then=1),
            default=2,
            output_field=IntegerField(),
        )
    ).order_by("rank", "-date", "-started_at", "-pk")

    page = Paginator(qs, SESSIONS_PER_PAGE).get_page(request.GET.get("page"))
    carried = {k: v for k, v in (("status", status), ("q", query)) if v}
    return render(
        request,
        "dashboard/sessions.html",
        {
            "page": page,
            "status": status,
            "q": query,
            "totals": _session_totals(),
            "present_today": Attendance.objects.filter(
                session__date=timezone.localdate(), status=Attendance.Status.PRESENT
            ).count(),
            "carried_qs": urlencode(carried),
            "filtered": bool(status or query),
        },
    )


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
