"""Template context shared by every page: role flags and the active nav item."""
from .permissions import user_is_admin


def _nav_section(request) -> str:
    match = getattr(request, "resolver_match", None)
    if match is None:
        return ""
    namespace, name = match.namespace, match.url_name or ""
    if namespace == "dashboard":
        return "overview" if name == "home" else "sessions"
    if namespace == "attendance":
        return "sessions"
    if namespace == "students":
        return "students"
    if namespace == "classes":
        return "sessions" if name.startswith("session") else "classes"
    return ""


def shell(request):
    user = request.user
    app_admin = user_is_admin(user)
    return {
        "is_app_admin": app_admin,
        # Django's admin site additionally requires is_staff.
        "can_open_admin": app_admin and user.is_authenticated and user.is_staff,
        "nav_section": _nav_section(request),
    }
