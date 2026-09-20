from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.shortcuts import get_object_or_404, redirect, render
from django.urls import reverse
from django.views.decorators.http import require_http_methods

from apps.accounts.permissions import user_is_admin

from .forms import EnrollmentUploadForm, StudentForm
from .models import Student
from .services import enroll_student_images


@login_required
def student_list(request):
    students = Student.objects.all()
    return render(request, "students/list.html", {"students": students})


@login_required
def student_detail(request, pk):
    student = get_object_or_404(Student, pk=pk)
    return render(
        request,
        "students/detail.html",
        {"student": student, "embeddings": student.embeddings.all()},
    )


@login_required
@require_http_methods(["GET", "POST"])
def student_create(request):
    if not user_is_admin(request.user):
        messages.error(request, "Only administrators can enrol students.")
        return redirect("students:list")

    form = StudentForm(request.POST or None, request.FILES or None)
    if request.method == "POST" and form.is_valid():
        student = form.save()
        if student.consent_given and not student.consent_recorded_at:
            student.record_consent()
        # The photo uploaded here doubles as the recognition template, so it
        # only has to be uploaded once.
        if student.profile_image:
            with student.profile_image.open("rb") as fh:
                result = enroll_student_images(student, [(student.profile_image.name, fh.read())])
            if result.created:
                messages.success(
                    request,
                    f"Created {student}. The uploaded photo is enrolled for recognition — "
                    "nothing more to upload.",
                )
                return redirect("students:detail", pk=student.pk)
            for problem in result.skipped:
                messages.warning(request, f"Photo not usable for recognition: {problem}")
        messages.success(request, f"Created {student}. Upload a clear face photo to enable recognition.")
        return redirect("students:enroll", pk=student.pk)
    return render(request, "students/form.html", {"form": form, "title": "New student"})


@login_required
@require_http_methods(["GET", "POST"])
def student_enroll(request, pk):
    student = get_object_or_404(Student, pk=pk)
    if not user_is_admin(request.user):
        messages.error(request, "Only administrators can enrol students.")
        return redirect("students:detail", pk=pk)

    form = EnrollmentUploadForm(request.POST or None, request.FILES or None)
    if request.method == "POST" and form.is_valid():
        uploads = [(f.name, f.read()) for f in request.FILES.getlist("images")]
        result = enroll_student_images(student, uploads)
        if result.created:
            messages.success(request, f"Stored {result.created} face template(s).")
        for problem in result.skipped:
            messages.warning(request, f"Skipped {problem}")
        return redirect("students:detail", pk=pk)

    return render(request, "students/enroll.html", {"student": student, "form": form})


@login_required
@require_http_methods(["POST"])
def embedding_delete(request, pk, emb_pk):
    student = get_object_or_404(Student, pk=pk)
    if not user_is_admin(request.user):
        messages.error(request, "Only administrators can modify enrollment data.")
        return redirect("students:detail", pk=pk)
    student.embeddings.filter(pk=emb_pk).delete()
    messages.info(request, "Face template removed.")
    return redirect(reverse("students:detail", args=[pk]))
