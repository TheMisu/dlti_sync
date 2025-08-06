from django.urls import path
from . import views

urlpatterns = [
    path('', views.landing_page, name='landing'),
    path('signup/', views.signup, name='signup'),
    path('upload/', views.upload_audio, name='upload_audio'),
    path('edit/<int:transcription_id>/', views.edit_transcription, name='edit_transcription'),
]
