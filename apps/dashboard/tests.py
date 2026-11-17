import re
from datetime import timedelta

from django.contrib.auth import get_user_model
from django.test import TestCase
from django.urls import reverse
from django.utils import timezone

from apps.accounts.models import Profile
from apps.attendance.models import Attendance
from apps.classes.models import ClassSession, CourseClass
from apps.students.models import Student

User = get_user_model()


class UiTestBase(TestCase):
    @classmethod
    def setUpTestData(cls):
        cls.admin = User.objects.create_user(
            "boss", password="pw-boss-12345", is_staff=True, is_superuser=True
        )
        cls.teacher = User.objects.create_user("tina", password="pw-tina-12345", is_staff=True)
        Profile.objects.filter(user=cls.teacher).update(display_name="Tina Rai")

        cls.maths = CourseClass.objects.create(name="Maths", code="MTH101", teacher=cls.teacher)
        cls.physics = CourseClass.objects.create(name="Physics", code="PHY201")
        today = timezone.localdate()
        now = timezone.now()
        cls.live = ClassSession.objects.create(
            course_class=cls.maths, teacher=cls.teacher, date=today,
            status=ClassSession.Status.ACTIVE, started_at=now,
        )
        cls.upcoming = ClassSession.objects.create(
            course_class=cls.physics, teacher=cls.teacher, date=today + timedelta(days=1),
        )
        cls.done = ClassSession.objects.create(
            course_class=cls.maths, teacher=cls.teacher, date=today - timedelta(days=1),
            status=ClassSession.Status.CLOSED,
            started_at=now - timedelta(days=1, hours=1), ended_at=now - timedelta(days=1),
        )
        cls.student = Student.objects.create(student_id="S1", full_name="Asha Gurung")
        Attendance.objects.create(
            student=cls.student, session=cls.live, status="present", marked_at=now
        )


class AccessTests(UiTestBase):
    def test_pages_require_login(self):
        for name in ("dashboard:home", "dashboard:sessions", "students:list", "classes:list"):
            resp = self.client.get(reverse(name))
            self.assertEqual(resp.status_code, 302, name)
            self.assertIn(reverse("accounts:login"), resp["Location"])

    def test_login_page_uses_standalone_layout(self):
        resp = self.client.get(reverse("accounts:login"))
        self.assertContains(resp, 'class="auth"')
        self.assertNotContains(resp, 'id="sidebar"')

    def test_every_page_renders_for_teacher(self):
        self.client.force_login(self.teacher)
        urls = [
            reverse("dashboard:home"),
            reverse("dashboard:sessions"),
            reverse("dashboard:live", args=[self.live.pk]),
            reverse("dashboard:live", args=[self.upcoming.pk]),
            reverse("attendance:session", args=[self.live.pk]),
            reverse("students:list"),
            reverse("students:detail", args=[self.student.pk]),
            reverse("classes:list"),
            reverse("classes:create"),
            reverse("classes:session_create"),
            reverse("accounts:password_change"),
        ]
        for url in urls:
            self.assertEqual(self.client.get(url).status_code, 200, url)


class RoleNavigationTests(UiTestBase):
    def test_teacher_does_not_see_admin_entry_points(self):
        self.client.force_login(self.teacher)
        resp = self.client.get(reverse("dashboard:home"))
        self.assertNotContains(resp, reverse("admin:index"))
        self.assertNotContains(resp, reverse("students:create"))
        self.assertContains(resp, "Tina Rai")
        self.assertContains(resp, "Teacher")

    def test_admin_sees_administration_and_enrol(self):
        self.client.force_login(self.admin)
        resp = self.client.get(reverse("dashboard:home"))
        self.assertContains(resp, reverse("admin:index"))
        self.assertContains(resp, reverse("students:create"))

    def test_enrol_actions_hidden_from_teacher_on_student_pages(self):
        self.client.force_login(self.teacher)
        detail = self.client.get(reverse("students:detail", args=[self.student.pk]))
        self.assertNotContains(detail, reverse("students:enroll", args=[self.student.pk]))
        listing = self.client.get(reverse("students:list"))
        self.assertNotContains(listing, reverse("students:create"))

    def test_current_section_is_marked_in_nav(self):
        self.client.force_login(self.teacher)
        resp = self.client.get(reverse("dashboard:sessions"))
        self.assertContains(
            resp, f'href="{reverse("dashboard:sessions")}" aria-current="page"'
        )
        resp = self.client.get(reverse("dashboard:live", args=[self.live.pk]))
        self.assertContains(
            resp, f'href="{reverse("dashboard:sessions")}" aria-current="page"'
        )


class SessionsPageTests(UiTestBase):
    def setUp(self):
        self.client.force_login(self.teacher)
        self.url = reverse("dashboard:sessions")

    def test_summary_counts_come_from_data(self):
        resp = self.client.get(self.url)
        self.assertEqual(resp.context["totals"]["active"], 1)
        self.assertEqual(resp.context["totals"]["scheduled"], 1)
        self.assertEqual(resp.context["totals"]["closed"], 1)
        self.assertEqual(resp.context["present_today"], 1)

    def test_live_sessions_are_listed_first_with_live_action(self):
        resp = self.client.get(self.url)
        rows = list(resp.context["page"].object_list)
        self.assertEqual([s.status for s in rows], ["active", "scheduled", "closed"])
        self.assertContains(resp, "Open live view")
        self.assertEqual(rows[0].present_count, 1)

    def test_status_filter(self):
        resp = self.client.get(self.url, {"status": "closed"})
        self.assertEqual([s.pk for s in resp.context["page"].object_list], [self.done.pk])
        self.assertNotContains(resp, "Open live view")

    def test_invalid_status_is_ignored(self):
        resp = self.client.get(self.url, {"status": "bogus"})
        self.assertEqual(resp.context["page"].paginator.count, 3)

    def test_search_by_class_and_teacher(self):
        resp = self.client.get(self.url, {"q": "phy"})
        self.assertEqual([s.pk for s in resp.context["page"].object_list], [self.upcoming.pk])
        resp = self.client.get(self.url, {"q": "Tina"})
        self.assertEqual(resp.context["page"].paginator.count, 3)

    def test_empty_states_and_escaping_of_search_term(self):
        resp = self.client.get(self.url, {"q": "<script>alert(1)</script>"})
        self.assertContains(resp, "No sessions match")
        self.assertNotContains(resp, "<script>alert(1)</script>")

    def test_completed_empty_state_copy(self):
        ClassSession.objects.filter(status="closed").delete()
        resp = self.client.get(self.url, {"status": "closed"})
        self.assertContains(resp, "No completed sessions yet")
        self.assertContains(resp, "Your completed classroom sessions will appear here.")

    def test_disclaimer_is_present(self):
        resp = self.client.get(self.url)
        self.assertContains(resp, "Observable indicators only")
        self.assertContains(resp, "human review")


class OverviewAndFlowTests(UiTestBase):
    def setUp(self):
        self.client.force_login(self.teacher)

    def test_overview_shows_real_figures(self):
        resp = self.client.get(reverse("dashboard:home"))
        self.assertEqual(resp.context["totals"]["active"], 1)
        self.assertEqual(len(resp.context["active_sessions"]), 1)
        self.assertContains(resp, "Asha Gurung")  # recent attendance feed

    def test_create_session_redirects_to_live_view(self):
        resp = self.client.post(
            reverse("classes:session_create"),
            {"course_class": self.maths.pk, "teacher": self.teacher.pk,
             "date": timezone.localdate().isoformat(), "video_source": "", "notes": ""},
        )
        session = ClassSession.objects.latest("pk")
        self.assertRedirects(resp, reverse("dashboard:live", args=[session.pk]))

    def test_session_form_shows_validation_errors(self):
        resp = self.client.post(reverse("classes:session_create"), {"date": ""})
        self.assertContains(resp, "field__error")

    def test_class_preselected_from_link(self):
        resp = self.client.get(reverse("classes:session_create"), {"course_class": self.physics.pk})
        self.assertEqual(resp.context["form"].initial["course_class"], self.physics.pk)

    def test_closing_a_session_returns_to_sessions_list(self):
        resp = self.client.post(reverse("classes:session_close", args=[self.live.pk]))
        self.assertRedirects(resp, reverse("dashboard:sessions"))
        self.live.refresh_from_db()
        self.assertEqual(self.live.status, "closed")

    def test_live_page_keeps_script_hooks(self):
        resp = self.client.get(reverse("dashboard:live", args=[self.live.pk]))
        for dom_id in ("btn-start", "btn-stop", "feed", "overlay", "fps",
                       "pipeline-state", "pipeline-error", "panel-body"):
            self.assertContains(resp, f'id="{dom_id}"')
        self.assertContains(resp, f"window.SESSION_ID = {self.live.pk}")
        self.assertContains(resp, "Close session")

    def test_student_search_and_filter(self):
        Student.objects.create(student_id="S2", full_name="Bikash Thapa")
        resp = self.client.get(reverse("students:list"), {"q": "bikash"})
        self.assertEqual([s.student_id for s in resp.context["students"]], ["S2"])
        resp = self.client.get(reverse("students:list"), {"recognition": "ready"})
        self.assertEqual(len(resp.context["students"]), 0)
        resp = self.client.get(reverse("students:list"), {"recognition": "pending"})
        self.assertEqual(len(resp.context["students"]), 2)

    def test_user_generated_names_are_escaped(self):
        Student.objects.create(student_id="X9", full_name="<b>Eve</b>")
        resp = self.client.get(reverse("students:list"))
        self.assertNotContains(resp, "<b>Eve</b>")
        self.assertContains(resp, "&lt;b&gt;Eve&lt;/b&gt;")


class AssetTests(UiTestBase):
    def test_stylesheet_url_is_versioned_from_the_file(self):
        """A stale ?v=N must not be possible: the version comes from the file."""
        import os
        import re

        from django.contrib.staticfiles import finders

        resp = self.client.get(reverse("accounts:login"))
        match = re.search(r'href="/static/css/app\.css\?v=(\d+)"', resp.content.decode())
        self.assertIsNotNone(match, "app.css should carry an mtime-derived version")
        self.assertEqual(int(match.group(1)), int(os.path.getmtime(finders.find("css/app.css"))))

    def test_icons_have_intrinsic_size_so_they_never_render_huge_without_css(self):
        self.client.force_login(self.teacher)
        html = self.client.get(reverse("dashboard:home")).content.decode()
        icons = re.findall(r'<svg class="icon[^"]*"[^>]*>', html)
        self.assertGreater(len(icons), 5)
        for tag in icons:
            self.assertIn('width="18"', tag)
            self.assertIn('fill="none"', tag)
