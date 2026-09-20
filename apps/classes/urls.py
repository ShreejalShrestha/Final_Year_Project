from django.urls import path

from . import views

app_name = "classes"

urlpatterns = [
    path("", views.class_list, name="list"),
    path("new/", views.class_create, name="create"),
    path("sessions/new/", views.session_create, name="session_create"),
    path("sessions/<int:pk>/activate/", views.session_activate, name="session_activate"),
    path("sessions/<int:pk>/close/", views.session_close, name="session_close"),
]
