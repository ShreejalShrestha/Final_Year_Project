from django.urls import path

from . import views

app_name = "students"

urlpatterns = [
    path("", views.student_list, name="list"),
    path("new/", views.student_create, name="create"),
    path("<int:pk>/", views.student_detail, name="detail"),
    path("<int:pk>/enroll/", views.student_enroll, name="enroll"),
    path("<int:pk>/embeddings/<int:emb_pk>/delete/", views.embedding_delete, name="embedding_delete"),
]
