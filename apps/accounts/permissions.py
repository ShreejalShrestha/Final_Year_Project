"""Reusable access-control helpers for role-restricted views."""
from django.contrib.auth.mixins import LoginRequiredMixin, UserPassesTestMixin


def user_is_admin(user) -> bool:
    if not user.is_authenticated:
        return False
    if user.is_superuser:
        return True
    profile = getattr(user, "profile", None)
    return bool(profile and profile.role == "admin")


class StaffRequiredMixin(LoginRequiredMixin, UserPassesTestMixin):
    """Any authenticated teacher or admin may access the view."""

    def test_func(self) -> bool:
        return self.request.user.is_authenticated


class AdminRequiredMixin(LoginRequiredMixin, UserPassesTestMixin):
    """Only administrators (or superusers) may access the view."""

    def test_func(self) -> bool:
        return user_is_admin(self.request.user)
