from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.shortcuts import get_object_or_404, redirect, render
from django.views.decorators.http import require_POST

from apps.classes.models import ClassSession
from apps.students.models import Student

from .models import Attendance
from .services import set_attendance_manually


@login_required
def session_attendance(request, session_id):
    session = get_object_or_404(ClassSession, pk=session_id)
    records = (
        Attendance.objects.filter(session=session)
        .select_related("student")
        .order_by("student__full_name")
    )
    marked_ids = set(records.values_list("student_id", flat=True))
    unmarked = Student.objects.filter(is_active=True).exclude(pk__in=marked_ids)
    return render(
        request,
        "attendance/session.html",
        {"session": session, "records": records, "unmarked": unmarked},
    )


@login_required
@require_POST
def correct_attendance(request, session_id):
    session = get_object_or_404(ClassSession, pk=session_id)
    student = get_object_or_404(Student, pk=request.POST["student_id"])
    status = request.POST.get("status", Attendance.Status.PRESENT)
    note = request.POST.get("note", "")
    set_attendance_manually(
        student=student, session=session, status=status, user=request.user, note=note
    )
    messages.success(request, f"Updated attendance for {student.full_name}.")
    return redirect("attendance:session", session_id=session.pk)
