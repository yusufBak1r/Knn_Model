from django.urls import path
from . import views

urlpatterns = [
    path('', views.uploadModelView, name='upload_model'),  # Ana sayfa - model yükleme
    path('model-results/', views.modelResultsView, name='model_results'),  # Model sonuçları sayfası
] 