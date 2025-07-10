from django.contrib import admin
from .models import TestData, UserModel
from .utils import update_model_evaluation

@admin.register(TestData)
class TestDataAdmin(admin.ModelAdmin):
    list_display = ('name', 'uploaded_at', 'is_active')
    list_filter = ('is_active', 'uploaded_at')
    search_fields = ('name', 'description')
    ordering = ('-uploaded_at',)
    readonly_fields = ('uploaded_at',)
    
    fieldsets = (
        ('Test Veri Seti Bilgileri', {
            'fields': ('name', 'description', 'file', 'is_active')
        }),
        ('Zaman Bilgisi', {
            'fields': ('uploaded_at',)
        }),
    )

@admin.register(UserModel)
class UserModelAdmin(admin.ModelAdmin):
    list_display = ('first_name', 'last_name', 'email', 'school_number', 'test_data', 'accuracy', 'uploaded_at', 'evaluated_at')
    list_filter = ('uploaded_at', 'evaluated_at', 'test_data')
    search_fields = ('first_name', 'last_name', 'email', 'school_number')
    ordering = ('-uploaded_at',)
    readonly_fields = ('uploaded_at', 'evaluated_at', 'accuracy')
    
    fieldsets = (
        ('Kullanıcı Bilgileri', {
            'fields': ('user', 'first_name', 'last_name', 'email', 'school_number')
        }),
        ('Model Bilgileri', {
            'fields': ('model_file', 'test_data', 'accuracy')
        }),
        ('Zaman Bilgileri', {
            'fields': ('uploaded_at', 'evaluated_at')
        }),
    )

    def save_model(self, request, obj, form, change):
        super().save_model(request, obj, form, change)
        # Model değerlendirmesini güncelle
        update_model_evaluation(obj)
