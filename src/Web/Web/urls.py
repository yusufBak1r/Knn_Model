# Web/urls.py
from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path('', include('myapp.urls')),  # myapp.urls içindeki URL yapılandırması dahil ediliyor
    path('admin/', admin.site.urls),  # Admin paneli yolu
]
