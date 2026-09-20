from django.conf import settings
from django.db import models
from django.utils import timezone


class CourseClass(models.Model):
    """A teachable class/course (e.g. "Data Structures - Sem 4 / B")."""

    name = models.CharField(max_length=120)
    code = models.CharField(max_length=32, blank=True)
    program = models.CharField(max_length=120, blank=True)
    semester_section = models.CharField(max_length=60, blank=True)
    teacher = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="classes",
    )
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["name"]
        verbose_name_plural = "course classes"

    def __str__(self) -> str:
        return f"{self.name}" + (f" [{self.code}]" if self.code else "")


class ClassSession(models.Model):
    """One live class meeting. Attendance and monitoring events are tied to a
    session (proposal: Real-Time Identification, Tracking and Attendance).
    """

    class Status(models.TextChoices):
        SCHEDULED = "scheduled", "Scheduled"
        ACTIVE = "active", "Active"
        CLOSED = "closed", "Closed"

    course_class = models.ForeignKey(
        CourseClass, on_delete=models.CASCADE, related_name="sessions"
    )
    teacher = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="sessions",
    )
    date = models.DateField(default=timezone.localdate)
    started_at = models.DateTimeField(null=True, blank=True)
    ended_at = models.DateTimeField(null=True, blank=True)
    status = models.CharField(
        max_length=12, choices=Status.choices, default=Status.SCHEDULED
    )
    video_source = models.CharField(
        max_length=255,
        blank=True,
        help_text="Webcam index or video path. Blank uses the CV_VIDEO_SOURCE default.",
    )
    notes = models.TextField(blank=True)

    class Meta:
        ordering = ["-date", "-started_at"]

    def __str__(self) -> str:
        return f"{self.course_class.name} — {self.date} ({self.get_status_display()})"

    @property
    def is_active(self) -> bool:
        return self.status == self.Status.ACTIVE

    def activate(self):
        self.status = self.Status.ACTIVE
        self.started_at = self.started_at or timezone.now()
        self.ended_at = None
        self.save(update_fields=["status", "started_at", "ended_at"])

    def close(self):
        self.status = self.Status.CLOSED
        self.ended_at = timezone.now()
        self.save(update_fields=["status", "ended_at"])
