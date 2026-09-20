"""Enrollment services: turn uploaded still images into stored face templates."""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from .models import FaceEmbedding, Student

logger = logging.getLogger("apps.students")


@dataclass
class EnrollmentResult:
    created: int
    skipped: list[str]


class NoFaceDetected(Exception):
    pass


def _largest_face(faces):
    def area(f):
        x1, y1, x2, y2 = f.bbox
        return (x2 - x1) * (y2 - y1)

    return max(faces, key=area)


def embed_image_bytes(image_bytes: bytes) -> tuple[np.ndarray, float]:
    """Return (normalised 512-D embedding, detection score) for the primary face."""
    import cv2

    from apps.dashboard.runtime import _make_config
    from cv_pipeline.detection import get_face_app

    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("Could not decode image.")

    # Same settings-driven config as the live pipeline, so enrolment and
    # recognition always use the same model (the app is a process-wide singleton).
    faces = get_face_app(_make_config()).get(frame)
    if not faces:
        raise NoFaceDetected("No face found in the supplied image.")

    face = _largest_face(faces)
    vec = np.asarray(face.normed_embedding, dtype=np.float32)
    return vec, float(getattr(face, "det_score", 0.0))


def _current_model_name() -> str:
    from apps.dashboard.runtime import _make_config

    return _make_config().insightface_name


def enroll_student_images(student: Student, images: list[tuple[str, bytes]]) -> EnrollmentResult:
    """`images` is a list of (filename, raw_bytes). One FaceEmbedding per usable image."""
    created = 0
    skipped: list[str] = []
    for name, data in images:
        try:
            vec, score = embed_image_bytes(data)
        except (NoFaceDetected, ValueError) as exc:
            logger.warning("Skipped %s for %s: %s", name, student.student_id, exc)
            skipped.append(f"{name}: {exc}")
            continue
        emb = FaceEmbedding(
            student=student, det_score=score, model_name=_current_model_name()
        )
        emb.set_vector(vec)
        emb.save()
        created += 1

    if created and not student.consent_given:
        student.record_consent()
    return EnrollmentResult(created=created, skipped=skipped)
