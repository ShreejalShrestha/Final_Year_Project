"""Persistence helpers for monitoring events emitted by the CV pipeline."""
from __future__ import annotations

import logging

from django.utils import timezone

from .models import MonitoringEvent

logger = logging.getLogger("apps.monitoring")


def open_event(
    *, session_id: int, student_id: int | None, track_ref: str, event_type: str,
    quality_score: float = 0.0,
) -> int:
    event = MonitoringEvent.objects.create(
        session_id=session_id,
        student_id=student_id,
        track_ref=str(track_ref),
        event_type=event_type,
        start_time=timezone.now(),
        quality_score=quality_score,
    )
    logger.info("Event opened %s track=%s type=%s", event.pk, track_ref, event_type)
    return event.pk


def close_event(*, event_pk: int) -> None:
    try:
        event = MonitoringEvent.objects.get(pk=event_pk)
    except MonitoringEvent.DoesNotExist:
        return
    if event.end_time is not None:
        return
    event.end_time = timezone.now()
    event.duration_seconds = (event.end_time - event.start_time).total_seconds()
    event.save(update_fields=["end_time", "duration_seconds"])


def attach_identity(*, event_pk: int, student_id: int) -> None:
    """Backfill the student once a track is confidently recognised."""
    MonitoringEvent.objects.filter(pk=event_pk, student__isnull=True).update(
        student_id=student_id
    )
