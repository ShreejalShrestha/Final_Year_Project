"""Enrol a student from a folder of face images.

Example:
    python manage.py enroll_student --id 22BSCIT001 --name "Asha Rai" \\
        --program "BSc. IT" --semester "Sem 6 / A" --images data/enroll/asha/*.jpg
"""
import glob
from pathlib import Path

from django.core.management.base import BaseCommand, CommandError

from apps.students.models import Student
from apps.students.services import enroll_student_images


class Command(BaseCommand):
    help = "Create or update a student and add face templates from image files."

    def add_arguments(self, parser):
        parser.add_argument("--id", required=True, dest="student_id")
        parser.add_argument("--name", required=True, dest="full_name")
        parser.add_argument("--program", default="")
        parser.add_argument("--semester", default="", dest="semester_section")
        parser.add_argument(
            "--images",
            nargs="+",
            required=True,
            help="Image paths or globs (e.g. data/enroll/asha/*.jpg).",
        )

    def handle(self, *args, **opts):
        paths: list[str] = []
        for pattern in opts["images"]:
            matched = glob.glob(pattern)
            paths.extend(matched or ([pattern] if Path(pattern).exists() else []))
        if not paths:
            raise CommandError("No image files matched.")

        student, created = Student.objects.update_or_create(
            student_id=opts["student_id"],
            defaults={
                "full_name": opts["full_name"],
                "program": opts["program"],
                "semester_section": opts["semester_section"],
            },
        )
        self.stdout.write(("Created " if created else "Updated ") + str(student))

        uploads = [(Path(p).name, Path(p).read_bytes()) for p in paths]
        result = enroll_student_images(student, uploads)
        self.stdout.write(self.style.SUCCESS(f"Stored {result.created} template(s)."))
        for problem in result.skipped:
            self.stdout.write(self.style.WARNING(f"  skipped {problem}"))
