from django.utils import timezone
from django.db import models

class TestData(models.Model):
    name = models.CharField(max_length=100, help_text="Test veri setinin adı")
    description = models.TextField(help_text="Test veri seti açıklaması")
    file = models.FileField(upload_to='test_data/', help_text="Test veri seti dosyası (.csv)")
    uploaded_at = models.DateTimeField(auto_now_add=True)
    is_active = models.BooleanField(default=True, help_text="Aktif test veri seti mi?")

    def __str__(self):
        return f"{self.name} - {self.uploaded_at.strftime('%Y-%m-%d')}"

    class Meta:
        verbose_name = "Test Data"
        verbose_name_plural = "Test Data"

class UserModel(models.Model):
    user = models.ForeignKey('auth.User', on_delete=models.CASCADE, null=True)
    email = models.EmailField(max_length=255)
    school_number = models.CharField(max_length=50)
    first_name = models.CharField(max_length=100)
    last_name = models.CharField(max_length=100)
    model_file = models.FileField(upload_to='user_models/', help_text="KNN model dosyası (.pkl)")
    test_data = models.ForeignKey(TestData, on_delete=models.SET_NULL, null=True, help_text="Kullanılan test veri seti")
    accuracy = models.FloatField(null=True, blank=True, help_text="Model doğruluk oranı")
    uploaded_at = models.DateTimeField(auto_now_add=True)
    evaluated_at = models.DateTimeField(null=True, blank=True)

    def __str__(self):
        return f"{self.first_name} {self.last_name} - {self.uploaded_at.strftime('%Y-%m-%d')}"

    class Meta:
        verbose_name = "User Model"
        verbose_name_plural = "User Models"
        ordering = ['-uploaded_at']
