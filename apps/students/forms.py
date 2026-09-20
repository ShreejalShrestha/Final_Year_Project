from django import forms

from .models import Student


class StudentForm(forms.ModelForm):
    class Meta:
        model = Student
        fields = [
            "student_id",
            "full_name",
            "program",
            "semester_section",
            "profile_image",
            "consent_given",
            "is_active",
        ]
        widgets = {
            "consent_given": forms.CheckboxInput,
        }
        help_texts = {
            "profile_image": "Clear, front-facing photo. Also used for face recognition — no second upload needed.",
        }


class MultiImageInput(forms.ClearableFileInput):
    allow_multiple_selected = True


class EnrollmentUploadForm(forms.Form):
    images = forms.FileField(
        widget=MultiImageInput(attrs={"multiple": True, "accept": "image/*"}),
        help_text="Upload several clear photos with modest pose and lighting variation.",
    )
