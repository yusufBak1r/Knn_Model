# Web/urls.py
from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static
from myapp import views

urlpatterns = [
    path('', include('myapp.urls')),  # myapp.urls içindeki URL yapılandırması dahil ediliyor
    path('admin/', admin.site.urls),  # Admin paneli yolu
    path('', views.uploadModelView, name='upload_model'),
    path('model-results/', views.modelResultsView, name='model_results'),
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
