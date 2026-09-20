from django.urls import path

from . import views

app_name = "dashboard"

urlpatterns = [
    path("", views.home, name="home"),
    path("session/<int:session_id>/", views.live, name="live"),
    path("session/<int:session_id>/start/", views.start_pipeline, name="start"),
    path("session/<int:session_id>/stop/", views.stop_pipeline, name="stop"),
    path("session/<int:session_id>/feed/", views.video_feed, name="feed"),
    path("session/<int:session_id>/tracks/", views.tracks_json, name="tracks"),
    path(
        "session/<int:session_id>/student/<int:student_id>/",
        views.student_profile_json,
        name="student_profile",
    ),
]
