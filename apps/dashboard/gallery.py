"""Build the recognition gallery from enrolled students in the database."""
from __future__ import annotations

import numpy as np

from apps.students.models import FaceEmbedding, Student
from cv_pipeline.config import PipelineConfig
from cv_pipeline.recognition import Gallery


def build_gallery(cfg: PipelineConfig | None = None) -> tuple[Gallery, dict[int, str]]:
    cfg = cfg or PipelineConfig()
    entries: dict[int, list[np.ndarray]] = {}
    names: dict[int, str] = {}

    students = Student.objects.filter(
        is_active=True, consent_given=True
    ).prefetch_related("embeddings")
    for student in students:
        vecs = [e.get_vector() for e in student.embeddings.all()]
        vecs = [v for v in vecs if v.size == FaceEmbedding.EMBEDDING_DIM]
        if not vecs:
            continue
        entries[student.pk] = np.vstack(vecs)
        names[student.pk] = student.full_name

    return Gallery(entries), names
