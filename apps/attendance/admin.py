from django.contrib import admin

from .models import Attendance


@admin.register(Attendance)
class AttendanceAdmin(admin.ModelAdmin):
    list_display = (
        "student",
        "session",
        "status",
        "source",
        "marked_at",
        "recognition_confidence",
        "corrected",
    )
    list_filter = ("status", "source", "corrected", "session")
    search_fields = ("student__student_id", "student__full_name")
    autocomplete_fields = ()
