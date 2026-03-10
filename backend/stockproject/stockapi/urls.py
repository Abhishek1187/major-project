from django.urls import path
from django.http import JsonResponse
from .views import get_ohlcv
from predictor.views import get_stocks, model_comparison

urlpatterns = [
    path('ohlcv/<str:symbol>/', get_ohlcv, name='get_ohlcv'),
    path('health/', lambda request: JsonResponse({"status": "ok", "message": "API is running"}), name='api_health'),
    path('stocks/', get_stocks, name='get_stocks'),
    path('model_comparison/', model_comparison, name='model_comparison'),
]
