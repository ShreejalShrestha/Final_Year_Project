from django.db import models


class EventType(models.TextChoices):
    """Observable conditions only. These are NOT emotion or engagement labels
    (proposal: "The system will not output definitive labels such as bored,
    lazy or interested from facial appearance alone.").
    """

    LOOKING_AWAY = "looking_away", "Looking away (prolonged)"
    HEAD_DOWN = "head_down", "Head-down posture"
    HIGH_MOVEMENT = "high_movement", "High upper-body movement"
    FACE_NOT_VISIBLE = "face_not_visible", "Face not visible"
    POSSIBLE_DROWSINESS = "possible_drowsiness", "Possible drowsiness (eyes closed)"


class MonitoringEvent(models.Model):
    """A condition that persisted past its configured duration threshold during
    a session (proposal: Observable Engagement Indicator Analysis, Appendix B).
    """

    student = models.ForeignKey(
        "students.Student",
        on_delete=models.CASCADE,
        related_name="monitoring_events",
        null=True,
        blank=True,
        help_text="Null when the track was never confidently recognised.",
    )
    session = models.ForeignKey(
        "classes.ClassSession",
        on_delete=models.CASCADE,
        related_name="monitoring_events",
    )
    track_ref = models.CharField(max_length=32, blank=True, help_text="Tracker ID within the session.")
    event_type = models.CharField(max_length=32, choices=EventType.choices)
    start_time = models.DateTimeField()
    end_time = models.DateTimeField(null=True, blank=True)
    duration_seconds = models.FloatField(default=0.0)
    quality_score = models.FloatField(
        default=0.0, help_text="Detector/landmark confidence for this event, 0..1."
    )
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-start_time"]
        indexes = [
            models.Index(fields=["session", "student"]),
            models.Index(fields=["session", "event_type"]),
        ]

    def __str__(self) -> str:
        who = self.student.student_id if self.student else f"track {self.track_ref}"
        return f"{who}: {self.get_event_type_display()} ({self.duration_seconds:.1f}s)"

    @property
    def is_open(self) -> bool:
        return self.end_time is None
