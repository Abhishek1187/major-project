import os
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from typing import Dict, Any, Optional
import traceback

# Define base directory and model paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

# Global variables for caching
_base_model = None
_feature_scaler = None
_target_scaler = None
_stock_scalers_cache = {}

# Popular stocks for pre-initialization
POPULAR_STOCKS = [
    "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS",
    "HINDUNILVR.NS", "ITC.NS", "SBIN.NS", "BHARTIARTL.NS", "KOTAKBANK.NS",
    "ASIANPAINT.NS", "MARUTI.NS", "LT.NS", "AXISBANK.NS", "TITAN.NS"
]

def load_base_model():
    """
    Load the base LSTM model for dynamic scaling.
    Returns the loaded model or None if loading fails.
    """
    global _base_model, _feature_scaler, _target_scaler
    
    if _base_model is not None:
        return _base_model
    
    try:
        # Try to load improved models first
        model_path = os.path.join(MODEL_DIR, "lstm_stock_model_improved.keras")
        feature_scaler_path = os.path.join(MODEL_DIR, "feature_scaler_improved.pkl")
        target_scaler_path = os.path.join(MODEL_DIR, "target_scaler_improved.pkl")
        
        if os.path.exists(model_path) and os.path.exists(feature_scaler_path) and os.path.exists(target_scaler_path):
            _base_model = tf.keras.models.load_model(model_path)
            _feature_scaler = joblib.load(feature_scaler_path)
            _target_scaler = joblib.load(target_scaler_path)
            print("[INFO] Loaded improved LSTM model and scalers for dynamic scaling")
        else:
            # Fallback to original models if improved ones don't exist
            model_path = os.path.join(MODEL_DIR, "lstm_stock_model_hybrid.keras")
            feature_scaler_path = os.path.join(BASE_DIR, "feature_scaler.pkl")
            target_scaler_path = os.path.join(BASE_DIR, "target_scaler.pkl")
            
            if os.path.exists(model_path):
                _base_model = tf.keras.models.load_model(model_path)
                if os.path.exists(feature_scaler_path):
                    _feature_scaler = joblib.load(feature_scaler_path)
                if os.path.exists(target_scaler_path):
                    _target_scaler = joblib.load(target_scaler_path)
                print("[INFO] Loaded original LSTM model for dynamic scaling")
            else:
                print("[ERROR] No suitable LSTM model found for dynamic scaling")
                return None
        
        return _base_model
        
    except Exception as e:
        print(f"[ERROR] Failed to load base model: {str(e)}")
        traceback.print_exc()
        return None

def initialize_scalers_for_popular_stocks():
    """
    Initialize scalers for popular stocks to improve prediction performance.
    This creates a cache of stock-specific scaling parameters.
    """
    global _stock_scalers_cache
    
    try:
        # For now, we'll use the global scalers for all stocks
        # In a more advanced implementation, this could load stock-specific scalers
        if _feature_scaler is None or _target_scaler is None:
            print("[WARNING] Global scalers not loaded, cannot initialize stock-specific scalers")
            return
        
        for stock in POPULAR_STOCKS:
            _stock_scalers_cache[stock] = {
                'feature_scaler': _feature_scaler,
                'target_scaler': _target_scaler,
                'initialized': True
            }
        
        print(f"[INFO] Initialized scalers for {len(POPULAR_STOCKS)} popular stocks")
        
    except Exception as e:
        print(f"[ERROR] Failed to initialize scalers for popular stocks: {str(e)}")
        traceback.print_exc()

def get_stock_scalers(symbol: str) -> tuple:
    """
    Get scalers for a specific stock symbol.
    Returns (feature_scaler, target_scaler) or (None, None) if not available.
    """
    global _stock_scalers_cache, _feature_scaler, _target_scaler
    
    # Check if we have stock-specific scalers
    if symbol in _stock_scalers_cache:
        cache_entry = _stock_scalers_cache[symbol]
        return cache_entry['feature_scaler'], cache_entry['target_scaler']
    
    # Fallback to global scalers
    return _feature_scaler, _target_scaler

def predict_with_dynamic_scaling(symbol: str, data: pd.DataFrame, sequence_length: int = 60) -> Optional[Dict[str, Any]]:
    """
    Predict stock price using dynamic scaling based on the stock symbol.
    
    Args:
        symbol: Stock symbol (e.g., "RELIANCE.NS")
        data: Preprocessed DataFrame with technical indicators
        sequence_length: Length of input sequences for the model
    
    Returns:
        Dictionary with prediction results or None if prediction fails
    """
    global _base_model
    
    try:
        if _base_model is None:
            print("[ERROR] Base model not loaded for dynamic scaling")
            return None
        
        # Get appropriate scalers for this stock
        feature_scaler, target_scaler = get_stock_scalers(symbol)
        
        if feature_scaler is None or target_scaler is None:
            print(f"[ERROR] Scalers not available for symbol {symbol}")
            return None
        
        # Define the features used in training
        features = [
            'open', 'high', 'low', 'volume',
            'sma_10', 'ema_9', 'ema_21', 'rsi_14', 'ma_20',
            'bb_upper', 'bb_lower', 'ema_12', 'ema_26',
            'macd', 'signal_line', '%k', '%d', 'atr_14',
            'news_sentiment'
        ]
        
        # Ensure all required features are present
        missing_features = [f for f in features if f not in data.columns]
        if missing_features:
            print(f"[ERROR] Missing features for prediction: {missing_features}")
            return None
        
        # Prepare features and target
        X = data[features].values
        y = data['close'].values.reshape(-1, 1)
        
        # Apply dynamic scaling
        X_scaled = feature_scaler.transform(X)
        y_scaled = target_scaler.transform(y)
        
        # Create sequences for LSTM input
        X_seq = []
        for i in range(len(X_scaled) - sequence_length):
            X_seq.append(X_scaled[i:i+sequence_length])
        X_seq = np.array(X_seq)
        
        if X_seq.size == 0:
            print("[ERROR] Not enough data to create sequences for prediction")
            return None
        
        # Make prediction
        predictions_scaled = _base_model.predict(X_seq, verbose=0)
        predictions = target_scaler.inverse_transform(predictions_scaled)
        
        # Get the latest prediction and actual values
        latest_prediction = float(predictions[-1][0])
        latest_actual = float(y[-1][0])
        
        # Calculate prediction confidence based on recent model performance
        # This is a simplified confidence metric
        recent_predictions = predictions[-min(10, len(predictions)):]
        recent_actuals = y[-min(10, len(y)):]
        
        if len(recent_predictions) > 1:
            mape = np.mean(np.abs((recent_actuals.flatten() - recent_predictions.flatten()) / recent_actuals.flatten())) * 100
            confidence = max(0, min(100, 100 - mape))
        else:
            confidence = 75.0  # Default confidence
        
        result = {
            'symbol': symbol,
            'predicted_close': round(latest_prediction, 2),
            'actual_close': round(latest_actual, 2),
            'confidence': round(confidence, 2),
            'prediction_count': len(predictions),
            'scaling_method': 'dynamic',
            'model_type': 'LSTM'
        }
        
        print(f"[INFO] Dynamic scaling prediction for {symbol}: {latest_prediction:.2f} (confidence: {confidence:.1f}%)")
        return result
        
    except Exception as e:
        print(f"[ERROR] Dynamic scaling prediction failed for {symbol}: {str(e)}")
        traceback.print_exc()
        return None

def get_dynamic_scaling_status() -> Dict[str, Any]:
    """
    Get the current status of the dynamic scaling system.
    
    Returns:
        Dictionary with system status information
    """
    global _base_model, _feature_scaler, _target_scaler, _stock_scalers_cache
    
    return {
        'base_model_loaded': _base_model is not None,
        'global_scalers_loaded': _feature_scaler is not None and _target_scaler is not None,
        'cached_stocks_count': len(_stock_scalers_cache),
        'popular_stocks_initialized': len(_stock_scalers_cache) >= len(POPULAR_STOCKS),
        'system_ready': _base_model is not None and _feature_scaler is not None and _target_scaler is not None
    }

def reset_dynamic_scaling_cache():
    """
    Reset the dynamic scaling cache. Useful for reloading models or clearing memory.
    """
    global _base_model, _feature_scaler, _target_scaler, _stock_scalers_cache
    
    _base_model = None
    _feature_scaler = None
    _target_scaler = None
    _stock_scalers_cache.clear()
    
    print("[INFO] Dynamic scaling cache reset")
