from django.test import TestCase
from django.utils import timezone

from apps.classes.models import ClassSession, CourseClass
from apps.students.models import Student

from .models import Attendance
from .services import attendance_percentage, mark_present_from_recognition, set_attendance_manually


class AttendanceDedupTests(TestCase):
    def setUp(self):
        self.student = Student.objects.create(
            student_id="T001", full_name="Test Student", consent_given=True
        )
        course = CourseClass.objects.create(name="Test Class")
        self.session = ClassSession.objects.create(
            course_class=course, status=ClassSession.Status.ACTIVE, started_at=timezone.now()
        )

    def test_first_recognition_creates_one_record(self):
        rec, created = mark_present_from_recognition(
            student_id=self.student.pk, session_id=self.session.pk, confidence=0.71
        )
        self.assertTrue(created)
        self.assertEqual(rec.status, Attendance.Status.PRESENT)
        self.assertEqual(rec.source, Attendance.Source.RECOGNITION)

    def test_repeated_recognition_does_not_duplicate(self):
        for _ in range(20):
            mark_present_from_recognition(
                student_id=self.student.pk, session_id=self.session.pk, confidence=0.6
            )
        self.assertEqual(
            Attendance.objects.filter(student=self.student, session=self.session).count(), 1
        )

    def test_manual_correction_marks_audit_fields(self):
        mark_present_from_recognition(
            student_id=self.student.pk, session_id=self.session.pk, confidence=0.6
        )
        rec = set_attendance_manually(
            student=self.student, session=self.session,
            status=Attendance.Status.ABSENT, user=None, note="wrong match",
        )
        self.assertTrue(rec.corrected)
        self.assertEqual(rec.status, Attendance.Status.ABSENT)
        self.assertEqual(rec.source, Attendance.Source.MANUAL)

    def test_attendance_percentage_uses_closed_sessions(self):
        self.session.close()
        mark_present_from_recognition(
            student_id=self.student.pk, session_id=self.session.pk, confidence=0.6
        )
        self.assertEqual(attendance_percentage(self.student), 100.0)
