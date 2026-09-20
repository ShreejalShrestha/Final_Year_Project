from django.conf import settings
from django.db import models


class Attendance(models.Model):
    """One record per student per session. A DB-level unique constraint plus a
    check-before-insert in the service layer prevent duplicates (proposal:
    "The system will check for an existing student-session record before
    insertion so repeated detections cannot create duplicate attendance.").
    """

    class Status(models.TextChoices):
        PRESENT = "present", "Present"
        ABSENT = "absent", "Absent"
        EXCUSED = "excused", "Excused"

    class Source(models.TextChoices):
        RECOGNITION = "recognition", "Face recognition"
        MANUAL = "manual", "Manual entry / correction"

    student = models.ForeignKey(
        "students.Student", on_delete=models.CASCADE, related_name="attendance_records"
    )
    session = models.ForeignKey(
        "classes.ClassSession", on_delete=models.CASCADE, related_name="attendance_records"
    )
    status = models.CharField(max_length=10, choices=Status.choices, default=Status.PRESENT)
    source = models.CharField(
        max_length=12, choices=Source.choices, default=Source.RECOGNITION
    )
    marked_at = models.DateTimeField()
    recognition_confidence = models.FloatField(null=True, blank=True)

    corrected = models.BooleanField(default=False)
    corrected_by = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="attendance_corrections",
    )
    audit_note = models.CharField(max_length=255, blank=True)

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        constraints = [
            models.UniqueConstraint(
                fields=["student", "session"], name="uniq_attendance_student_session"
            )
        ]
        ordering = ["marked_at"]

    def __str__(self) -> str:
        return f"{self.student.student_id} @ session {self.session_id}: {self.status}"
