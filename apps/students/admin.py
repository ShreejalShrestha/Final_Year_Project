from django.contrib import admin

from .models import FaceEmbedding, Student


class FaceEmbeddingInline(admin.TabularInline):
    model = FaceEmbedding
    extra = 0
    readonly_fields = ("det_score", "model_name", "created_at")
    fields = ("source_image", "det_score", "model_name", "created_at")


@admin.register(Student)
class StudentAdmin(admin.ModelAdmin):
    list_display = (
        "student_id",
        "full_name",
        "program",
        "semester_section",
        "consent_given",
        "embedding_count",
        "is_active",
    )
    list_filter = ("program", "semester_section", "consent_given", "is_active")
    search_fields = ("student_id", "full_name")
    inlines = [FaceEmbeddingInline]


@admin.register(FaceEmbedding)
class FaceEmbeddingAdmin(admin.ModelAdmin):
    list_display = ("student", "det_score", "model_name", "created_at")
    search_fields = ("student__student_id", "student__full_name")
