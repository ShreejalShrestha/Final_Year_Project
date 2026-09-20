from django.conf import settings
from django.conf.urls.static import static
from django.contrib import admin
from django.urls import include, path
from django.views.generic import RedirectView

urlpatterns = [
    path("admin/", admin.site.urls),
    path("accounts/", include("apps.accounts.urls")),
    path("students/", include("apps.students.urls")),
    path("classes/", include("apps.classes.urls")),
    path("attendance/", include("apps.attendance.urls")),
    path("dashboard/", include("apps.dashboard.urls")),
    path("", RedirectView.as_view(pattern_name="dashboard:home", permanent=False)),
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
