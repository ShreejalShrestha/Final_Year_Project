import base64

import numpy as np
from django.db import models
from django.utils import timezone


class Student(models.Model):
    """Enrolled student. Only identity and academic fields the prototype needs
    are stored (proposal: Data Collection and Student Enrollment, Appendix B/D).
    """

    student_id = models.CharField(max_length=32, unique=True, db_index=True)
    full_name = models.CharField(max_length=120)
    program = models.CharField(max_length=120, blank=True)
    semester_section = models.CharField("semester / section", max_length=60, blank=True)
    profile_image = models.ImageField(upload_to="students/profile/", blank=True, null=True)

    consent_given = models.BooleanField(default=False)
    consent_recorded_at = models.DateTimeField(blank=True, null=True)
    is_active = models.BooleanField(default=True)

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["full_name"]

    def __str__(self) -> str:
        return f"{self.full_name} ({self.student_id})"

    def record_consent(self):
        self.consent_given = True
        self.consent_recorded_at = timezone.now()
        self.save(update_fields=["consent_given", "consent_recorded_at", "updated_at"])

    @property
    def embedding_count(self) -> int:
        return self.embeddings.count()

    @property
    def is_enrolled(self) -> bool:
        """Ready to be recognised: consent on file and at least one face template."""
        return self.consent_given and self.is_active and self.embedding_count > 0


class FaceEmbedding(models.Model):
    """A single ArcFace (512-D, L2-normalised) template for a student.

    Stored as raw bytes. Continuous raw video is never stored; only the source
    still used to derive the template is optionally kept for audit.
    """

    EMBEDDING_DIM = 512

    student = models.ForeignKey(
        Student, on_delete=models.CASCADE, related_name="embeddings"
    )
    vector = models.BinaryField()
    source_image = models.ImageField(
        upload_to="students/enrollment/", blank=True, null=True
    )
    det_score = models.FloatField(default=0.0)
    model_name = models.CharField(max_length=40, default="buffalo_l")
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self) -> str:
        return f"embedding<{self.student.student_id}> #{self.pk}"

    # --- numpy <-> bytes helpers -----------------------------------------
    def set_vector(self, arr: np.ndarray) -> None:
        arr = np.asarray(arr, dtype=np.float32).ravel()
        norm = np.linalg.norm(arr)
        if norm > 0:
            arr = arr / norm
        self.vector = arr.tobytes()

    def get_vector(self) -> np.ndarray:
        return np.frombuffer(self.vector, dtype=np.float32)

    @property
    def vector_b64(self) -> str:
        return base64.b64encode(self.vector).decode("ascii")
