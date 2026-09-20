from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.shortcuts import get_object_or_404, redirect, render
from django.views.decorators.http import require_POST

from .forms import ClassSessionForm, CourseClassForm
from .models import ClassSession, CourseClass


@login_required
def class_list(request):
    return render(
        request,
        "classes/list.html",
        {
            "classes": CourseClass.objects.all(),
            "sessions": ClassSession.objects.select_related("course_class")[:25],
        },
    )


@login_required
def class_create(request):
    form = CourseClassForm(request.POST or None)
    if request.method == "POST" and form.is_valid():
        form.save()
        messages.success(request, "Class created.")
        return redirect("classes:list")
    return render(request, "classes/form.html", {"form": form, "title": "New class"})


@login_required
def session_create(request):
    initial = {"teacher": request.user}
    form = ClassSessionForm(request.POST or None, initial=initial)
    if request.method == "POST" and form.is_valid():
        session = form.save()
        messages.success(request, "Session created.")
        return redirect("dashboard:live", session_id=session.pk)
    return render(request, "classes/form.html", {"form": form, "title": "New session"})


@login_required
@require_POST
def session_activate(request, pk):
    session = get_object_or_404(ClassSession, pk=pk)
    session.activate()
    messages.success(request, "Session is now active — attendance will be recorded.")
    return redirect("dashboard:live", session_id=session.pk)


@login_required
@require_POST
def session_close(request, pk):
    session = get_object_or_404(ClassSession, pk=pk)
    session.close()
    from apps.dashboard.runtime import pipeline_registry

    pipeline_registry.stop(session.pk)
    messages.info(request, "Session closed.")
    return redirect("dashboard:home")
