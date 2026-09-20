"""Attendance recording logic: first valid recognition marks a student present
once per session, with duplicate prevention and teacher correction support.
"""
from __future__ import annotations

import logging

from django.db import IntegrityError, transaction
from django.utils import timezone

from apps.classes.models import ClassSession
from apps.students.models import Student

from .models import Attendance

logger = logging.getLogger("apps.attendance")


def mark_present_from_recognition(
    *, student_id: int, session_id: int, confidence: float | None
) -> tuple[Attendance | None, bool]:
    """Idempotent. Returns (record, created).

    Called by the CV pipeline on the first confident recognition of a student
    in an active session. Safe to call on every subsequent frame.
    """
    existing = Attendance.objects.filter(
        student_id=student_id, session_id=session_id
    ).first()
    if existing:
        return existing, False

    try:
        with transaction.atomic():
            record = Attendance.objects.create(
                student_id=student_id,
                session_id=session_id,
                status=Attendance.Status.PRESENT,
                source=Attendance.Source.RECOGNITION,
                marked_at=timezone.now(),
                recognition_confidence=confidence,
            )
        logger.info(
            "Attendance: student=%s session=%s conf=%.3f",
            student_id, session_id, confidence or 0.0,
        )
        return record, True
    except IntegrityError:
        # Lost a race with another frame/thread — fetch the winner.
        return Attendance.objects.get(student_id=student_id, session_id=session_id), False


def set_attendance_manually(
    *, student: Student, session: ClassSession, status: str, user, note: str = ""
) -> Attendance:
    record, _ = Attendance.objects.update_or_create(
        student=student,
        session=session,
        defaults={
            "status": status,
            "source": Attendance.Source.MANUAL,
            "marked_at": timezone.now(),
            "corrected": True,
            "corrected_by": user,
            "audit_note": note or "Manual correction",
        },
    )
    return record


def attendance_percentage(student: Student, course_class=None) -> float:
    qs = Attendance.objects.filter(student=student)
    sessions = ClassSession.objects.filter(status=ClassSession.Status.CLOSED)
    if course_class is not None:
        qs = qs.filter(session__course_class=course_class)
        sessions = sessions.filter(course_class=course_class)
    total = sessions.count()
    if not total:
        return 0.0
    present = qs.filter(status=Attendance.Status.PRESENT).count()
    return round(100.0 * present / total, 1)
