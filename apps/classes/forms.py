from django import forms

from .models import ClassSession, CourseClass


class CourseClassForm(forms.ModelForm):
    class Meta:
        model = CourseClass
        fields = ["name", "code", "program", "semester_section", "teacher"]


class ClassSessionForm(forms.ModelForm):
    class Meta:
        model = ClassSession
        fields = ["course_class", "teacher", "date", "video_source", "notes"]
        widgets = {"date": forms.DateInput(attrs={"type": "date"})}
