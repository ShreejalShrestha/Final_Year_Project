from django import template

register = template.Library()


@register.filter
def person(user, fallback="—"):
    """Friendly name for a user: profile display name, else username.

    Safe for ``None`` (e.g. a session whose teacher was deleted).
    """
    if user is None:
        return fallback
    profile = getattr(user, "profile", None)
    display = getattr(profile, "display_name", "") if profile else ""
    return display or user.get_full_name() or user.get_username()
