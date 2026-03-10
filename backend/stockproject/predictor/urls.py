from django.urls import path
from .views import predict_price, health_check

urlpatterns = [
    path('health/', health_check, name='health_check'),
    path('<str:symbol>/', predict_price, name='predict_price'),  # Direct symbol access under /predict/
]
