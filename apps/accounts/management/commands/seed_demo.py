"""Create demo users, a class and a session so the dashboard is usable
immediately. Does NOT create face templates (those need real consented photos).

    python manage.py seed_demo
"""
from django.contrib.auth import get_user_model
from django.core.management.base import BaseCommand

from apps.accounts.models import Profile
from apps.classes.models import ClassSession, CourseClass

User = get_user_model()


class Command(BaseCommand):
    help = "Seed demo admin/teacher accounts and a sample class + session."

    def handle(self, *args, **opts):
        admin, created = User.objects.get_or_create(
            username="admin", defaults={"is_staff": True, "is_superuser": True}
        )
        if created:
            admin.set_password("admin12345")
            admin.save()
        Profile.objects.update_or_create(user=admin, defaults={"role": Profile.Role.ADMIN})

        teacher, created = User.objects.get_or_create(
            username="teacher", defaults={"is_staff": True}
        )
        if created:
            teacher.set_password("teacher12345")
            teacher.save()
        Profile.objects.update_or_create(
            user=teacher, defaults={"role": Profile.Role.TEACHER}
        )

        course, _ = CourseClass.objects.get_or_create(
            name="Data Structures",
            defaults={"code": "CSC201", "program": "BSc. IT", "semester_section": "Sem 4 / B",
                      "teacher": teacher},
        )
        session, _ = ClassSession.objects.get_or_create(
            course_class=course, teacher=teacher,
            defaults={"status": ClassSession.Status.SCHEDULED},
        )

        self.stdout.write(self.style.SUCCESS("Seeded demo data."))
        self.stdout.write("  admin / admin12345  (administrator)")
        self.stdout.write("  teacher / teacher12345  (teacher)")
        self.stdout.write(f"  session id = {session.pk} -> /dashboard/session/{session.pk}/")
