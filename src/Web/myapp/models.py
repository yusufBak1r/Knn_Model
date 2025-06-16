from django.db import models

class Student(models.Model):
    student_number = models.CharField(max_length=20, unique=True)
    first_name = models.CharField(max_length=50)
    last_name = models.CharField(max_length=50)
    email = models.EmailField(unique=True)
    download_path = models.CharField(max_length=200)
    def __str__(self):
        return f"{self.student_number} - {self.first_name} {self.last_name}"
