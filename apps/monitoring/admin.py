from django.contrib import admin

from .models import MonitoringEvent


@admin.register(MonitoringEvent)
class MonitoringEventAdmin(admin.ModelAdmin):
    list_display = (
        "session",
        "student",
        "track_ref",
        "event_type",
        "start_time",
        "duration_seconds",
        "quality_score",
    )
    list_filter = ("event_type", "session")
    search_fields = ("student__student_id", "student__full_name", "track_ref")
