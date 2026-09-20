from django.contrib import admin

from .models import ClassSession, CourseClass


@admin.register(CourseClass)
class CourseClassAdmin(admin.ModelAdmin):
    list_display = ("name", "code", "program", "semester_section", "teacher")
    search_fields = ("name", "code")


@admin.register(ClassSession)
class ClassSessionAdmin(admin.ModelAdmin):
    list_display = ("course_class", "date", "status", "teacher", "started_at", "ended_at")
    list_filter = ("status", "date")
    search_fields = ("course_class__name",)
