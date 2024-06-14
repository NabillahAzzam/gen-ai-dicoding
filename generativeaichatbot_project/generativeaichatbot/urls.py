from django.urls import path
from . import views

urlpatterns = [
    # path('chatbot/', ChatbotAPIView.as_view(), name='chatbot')
    path('', views.chatbot, name='chatbot')
]