from django.urls import path

from . import views

app_name = "attendance"

urlpatterns = [
    path("session/<int:session_id>/", views.session_attendance, name="session"),
    path("session/<int:session_id>/correct/", views.correct_attendance, name="correct"),
]
