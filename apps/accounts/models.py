from django.conf import settings
from django.db import models


class Profile(models.Model):
    """Role information attached to the built-in Django ``User``.

    The proposal (User Management, Appendix A) requires teacher/admin roles and
    role-based access. Admins can enrol students and manage sessions; teachers
    run monitoring sessions and view dashboards.
    """

    class Role(models.TextChoices):
        ADMIN = "admin", "Administrator"
        TEACHER = "teacher", "Teacher"

    user = models.OneToOneField(
        settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name="profile"
    )
    role = models.CharField(max_length=16, choices=Role.choices, default=Role.TEACHER)
    display_name = models.CharField(max_length=120, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self) -> str:
        return f"{self.user.username} ({self.get_role_display()})"

    @property
    def is_admin(self) -> bool:
        return self.role == self.Role.ADMIN or self.user.is_superuser
