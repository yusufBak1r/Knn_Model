from django.urls import path
from . import views

urlpatterns = [
    # path('', views.indexView, name="index"),
        path('', views.uploadView, name='upload'),
    path('upload/', views.uploadView, name='upload')
    # path('home/', views.homeView, name="home"),
    # path('about/', views.aboutView, name="about"),
]