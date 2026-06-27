from django.urls import path
from . import views

urlpatterns = [
    # Frontend pages
    path('', views.index, name='index'),
    path('dashboard/', views.dashboard, name='dashboard'),

    # REST API — ML prediction (scikit-learn)
    path('api/predict/', views.predict, name='predict'),
    path('api/model/info/', views.model_info, name='model-info'),

    # REST API — history & stats
    path('api/history/', views.sign_history, name='sign-history'),
    path('api/history/clear/', views.clear_history, name='clear-history'),
    path('api/stats/', views.stats, name='stats'),
    path('api/sentences/', views.sentences, name='sentences'),
]
