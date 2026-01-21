from django.urls import path
from . import views

urlpatterns = [
    path('widget/', views.chat_widget, name='chat-widget'),
    path('message/', views.chat_message, name='chat-message'),
]