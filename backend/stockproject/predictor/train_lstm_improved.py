import os
import sys
import re
import json
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler

# Local imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import fetch_last_60_minutes, add_technical_indicators
from nse_data_fetcher import NSEDataFetcher

# Directory and file paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

model_save_path = os.path.join(MODEL_DIR, "lstm_stock_model_improved.keras")
feature_scaler_path = os.path.join(MODEL_DIR, "feature_scaler_improved.pkl")
target_scaler_path = os.path.join(MODEL_DIR, "target_scaler_improved.pkl")
history_path = os.path.join(MODEL_DIR, "lstm_training_history_improved.json")

FEATURES = [
    'open', 'high', 'low', 'volume',
    'sma_10', 'ema_9', 'ema_21', 'rsi_14', 'ma_20',
    'bb_upper', 'bb_lower', 'ema_12', 'ema_26',
    'macd', 'signal_line', '%k', '%d', 'atr_14',
    'news_sentiment'
]

# ⏳ Improved Preprocessing with Conservative Scaling
def preprocess_df_improved(df):
    # Convert MultiIndex or ticker-suffixed columns to simple names
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ['_'.join(filter(None, col)).strip() for col in df.columns.values]
    
    # Remove ticker suffix like '_TCS.NS', '_RELIANCE.NS', '_^NSEI', etc.
    df.columns = [re.sub(r'_[A-Z^]+[A-Z0-9]*\.?[A-Z]*$', '', col) for col in df.columns]

    print(f"[DEBUG] Columns after cleaning: {df.columns}")
    print(f"[DEBUG] Column types: {df.dtypes}")

    df = add_technical_indicators(df)
    df = df.ffill().bfill()

    if 'news_sentiment' not in df.columns:
        df['news_sentiment'] = 0.0

    # Conservative volume normalization - use percentage change instead of ratio
    df['volume_pct_change'] = df['volume'].pct_change().fillna(0)
    # Cap extreme volume changes to prevent scaling issues
    df['volume_pct_change'] = df['volume_pct_change'].clip(-1, 1)
    
    # Replace volume with percentage change for better scaling
    df['volume'] = df['volume_pct_change']

    # Add price percentage changes instead of log returns for better stability
    for col in ['open', 'high', 'low', 'close']:
        df[f'pct_change_{col}'] = df[col].pct_change().fillna(0)
        # Cap extreme price changes to prevent unrealistic predictions
        df[f'pct_change_{col}'] = df[f'pct_change_{col}'].clip(-0.2, 0.2)  # Max 20% change

    df.fillna(0, inplace=True)
    return df

# 📈 Sequence Creation
def create_sequences(X, y, seq_length=60):
    X_seq, y_seq = [], []
    if len(X) <= seq_length:
        return np.array([]), np.array([])
    for i in range(len(X) - seq_length):
        X_seq.append(X[i:i + seq_length])
        y_seq.append(y[i + seq_length])
    return np.array(X_seq), np.array(y_seq)

# 🧠 Model Builder with Conservative Architecture
def build_lstm_model_improved(input_shape):
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=input_shape, recurrent_dropout=0.1),
        BatchNormalization(),
        LSTM(32, recurrent_dropout=0.1),
        BatchNormalization(),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dropout(0.1),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

# 🧪 Improved Trainer with Conservative Scaling
def train_improved(symbol):
    print(f"\n[INFO] Training improved LSTM model for symbol: {symbol}")
    nse_fetcher = NSEDataFetcher()

    # Fetch and preprocess training data
    print(f"\n[START] Fetching data for: {symbol}")
    df_train = nse_fetcher.fetch_data(symbol)
    if df_train is None or df_train.empty:
        print(f"[ERROR] No training data fetched for {symbol}")
        return

    df_train = preprocess_df_improved(df_train)

    # Use improved features with percentage changes
    extended_features = FEATURES + [f'pct_change_{col}' for col in ['open', 'high', 'low', 'close']]
    X_train = df_train[extended_features].values
    y_train = df_train['close'].values.reshape(-1, 1)

    # Use MinMaxScaler instead of StandardScaler for more conservative scaling
    feature_scaler = MinMaxScaler(feature_range=(0, 1))
    target_scaler = MinMaxScaler(feature_range=(0, 1))
    
    X_scaled = feature_scaler.fit_transform(X_train)
    y_scaled = target_scaler.fit_transform(y_train)

    print(f"[INFO] Target price range: {y_train.min():.2f} to {y_train.max():.2f}")
    print(f"[INFO] Scaled target range: {y_scaled.min():.4f} to {y_scaled.max():.4f}")

    X_seq, y_seq = create_sequences(X_scaled, y_scaled)
    if X_seq.size == 0 or y_seq.size == 0:
        print("[ERROR] Not enough data to train LSTM model.")
        return

    model = build_lstm_model_improved((X_seq.shape[1], X_seq.shape[2]))

    # More conservative training parameters
    early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.7, patience=5, min_lr=1e-6)

    history = model.fit(
        X_seq, y_seq,
        epochs=50,  # Reduced epochs to prevent overfitting
        batch_size=min(16, len(X_seq)),  # Smaller batch size for better generalization
        validation_split=0.2,
        callbacks=[early_stop, reduce_lr],
        verbose=1
    )

    model.save(model_save_path)
    joblib.dump(feature_scaler, feature_scaler_path)
    joblib.dump(target_scaler, target_scaler_path)

    print("[INFO] ✅ Improved LSTM training completed and model saved.")

    # Save training history
    with open(history_path, "w") as f:
        json.dump(history.history, f)
    print("[INFO] 📈 Training history saved.")

    # Test prediction to verify scaling
    print("\n[INFO] Testing prediction scaling...")
    test_pred_scaled = model.predict(X_seq[-1:])
    test_pred = target_scaler.inverse_transform(test_pred_scaled)
    actual_price = y_train[-1][0]
    predicted_price = test_pred[0][0]
    
    print(f"[INFO] Last actual price: {actual_price:.2f}")
    print(f"[INFO] Test prediction: {predicted_price:.2f}")
    print(f"[INFO] Prediction difference: {abs(predicted_price - actual_price):.2f} ({abs(predicted_price - actual_price)/actual_price*100:.2f}%)")

# 🏁 Entry Point
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", type=str, required=True, help="Stock symbol (e.g., TCS.NS)")
    args = parser.parse_args()
    train_improved(args.symbol)
