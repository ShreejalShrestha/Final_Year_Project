import os

from django import template
from django.contrib.staticfiles import finders
from django.templatetags.static import static

register = template.Library()


@register.simple_tag
def static_v(path):
    """``{% static %}`` URL plus the file's modification time.

    A hand-maintained ``?v=N`` goes stale the moment a file changes without the
    number changing, and browsers then keep serving the old copy. Deriving the
    version from the file itself means an edited asset always gets a new URL.
    """
    url = static(path)
    found = finders.find(path)
    if isinstance(found, str) and os.path.exists(found):
        return f"{url}?v={int(os.path.getmtime(found))}"
    return url
