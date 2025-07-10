from django.urls import path
from . import views

urlpatterns = [
    path('', views.uploadModelView, name='upload_model'),  # Ana sayfa - model yükleme
    path('model-results/', views.modelResultsView, name='model_results'),  # Model sonuçları sayfası
    path('confusion-matrix/<int:model_id>/', views.confusionMatrixView, name='confusion_matrix'),  # Confusion matrix sayfası
] 