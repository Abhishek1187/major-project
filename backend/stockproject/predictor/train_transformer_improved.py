import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LayerNormalization, Dropout, MultiHeadAttention, Flatten
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import joblib
from sklearn.preprocessing import MinMaxScaler

# Local imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import fetch_last_60_minutes, add_technical_indicators
from nse_data_fetcher import NSEDataFetcher

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
transformer_model_save_path = os.path.join(BASE_DIR, "models", "transformer_stock_model_improved.keras")
feature_scaler_path = os.path.join(BASE_DIR, "models", "feature_scaler_improved.pkl")
target_scaler_path = os.path.join(BASE_DIR, "models", "target_scaler_improved.pkl")

FEATURES = [
    'open', 'high', 'low', 'volume',
    'sma_10', 'ema_9', 'ema_21', 'rsi_14', 'ma_20',
    'bb_upper', 'bb_lower', 'ema_12', 'ema_26',
    'macd', 'signal_line', '%k', '%d', 'atr_14',
    'news_sentiment'
]

def preprocess_df_improved(df):
    import re
    
    # Handle MultiIndex columns
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ['_'.join(filter(None, col)).strip() for col in df.columns.values]
    
    # Remove ticker suffix like '_TCS.NS', '_RELIANCE.NS', '_^NSEI', etc.
    df.columns = [re.sub(r'_[A-Z^]+[A-Z0-9]*\.?[A-Z]*$', '', col) for col in df.columns]
    
    print(f"[DEBUG] Columns after cleaning: {df.columns}")
    print(f"[DEBUG] Column types: {df.dtypes}")
    
    df = add_technical_indicators(df)
    if df is None or df.empty:
        print("[ERROR] Dataframe is empty after adding technical indicators.")
        return df
    df = df.ffill().bfill()
    if 'news_sentiment' not in df.columns:
        df['news_sentiment'] = 0.0
    
    # Conservative volume normalization
    df['volume_pct_change'] = df['volume'].pct_change().fillna(0)
    df['volume_pct_change'] = df['volume_pct_change'].clip(-1, 1)
    df['volume'] = df['volume_pct_change']

    # Add price percentage changes instead of log returns
    for col in ['open', 'high', 'low', 'close']:
        df[f'pct_change_{col}'] = df[col].pct_change().fillna(0)
        df[f'pct_change_{col}'] = df[f'pct_change_{col}'].clip(-0.2, 0.2)

    df.fillna(0, inplace=True)
    return df

def create_sequences(X, y, seq_length=60):
    X_seq, y_seq = [], []
    if len(X) <= seq_length:
        print(f"[ERROR] Not enough data to create sequences. Data length: {len(X)}, required sequence length: {seq_length}")
        return np.array([]), np.array([])
    for i in range(len(X) - seq_length):
        X_seq.append(X[i:i+seq_length])
        y_seq.append(y[i+seq_length])
    return np.array(X_seq), np.array(y_seq)

def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    # Multi-head self-attention
    x = MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(inputs, inputs)
    x = Dropout(dropout)(x)
    x = LayerNormalization(epsilon=1e-6)(x)
    res = x + inputs

    # Feed-forward network
    x = Dense(ff_dim, activation="relu")(res)
    x = Dropout(dropout)(x)
    x = Dense(inputs.shape[-1])(x)
    x = LayerNormalization(epsilon=1e-6)(x)
    return x + res

def build_transformer_model_improved(input_shape, head_size=32, num_heads=2, ff_dim=64, num_layers=1, dropout=0.1):
    """Build a more conservative transformer model"""
    inputs = Input(shape=input_shape)
    x = inputs
    for _ in range(num_layers):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)
    x = Flatten()(x)
    x = Dense(32, activation="relu")(x)
    x = Dropout(dropout)(x)
    x = Dense(16, activation="relu")(x)
    x = Dropout(dropout)(x)
    outputs = Dense(1)(x)
    model = Model(inputs, outputs)
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def train_improved(symbol="RELIANCE.NS"):
    print(f"[INFO] Training improved Transformer model for {symbol}...")
    nse_fetcher = NSEDataFetcher()
    
    # Fetch data using NSE fetcher for consistency
    df = nse_fetcher.fetch_data(symbol)
    if df is None or df.empty:
        print(f"[ERROR] No data fetched for {symbol}")
        return

    df = preprocess_df_improved(df)

    print("[INFO] Preprocessing data...")
    extended_features = FEATURES + [f'pct_change_{col}' for col in ['open', 'high', 'low', 'close']]
    X = df[extended_features].values
    y = df['close'].values.reshape(-1, 1)

    # Use MinMaxScaler for conservative scaling
    feature_scaler = MinMaxScaler(feature_range=(0, 1))
    target_scaler = MinMaxScaler(feature_range=(0, 1))

    X_scaled = feature_scaler.fit_transform(X)
    y_scaled = target_scaler.fit_transform(y)

    print(f"[INFO] Target price range: {y.min():.2f} to {y.max():.2f}")
    print(f"[INFO] Scaled target range: {y_scaled.min():.4f} to {y_scaled.max():.4f}")

    print("[INFO] Creating sequences...")
    X_seq, y_seq = create_sequences(X_scaled, y_scaled)

    if X_seq.size == 0 or y_seq.size == 0:
        print("[ERROR] Not enough data to train Transformer model.")
        return

    print(f"[INFO] Training data shape: {X_seq.shape}, {y_seq.shape}")

    print("[INFO] Building improved Transformer model...")
    model = build_transformer_model_improved((X_seq.shape[1], X_seq.shape[2]))

    # Conservative training parameters
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.7, patience=3, min_lr=1e-6)

    print("[INFO] Starting training...")
    history = model.fit(
        X_seq, y_seq,
        epochs=50,  # Reduced epochs
        batch_size=16,  # Smaller batch size
        validation_split=0.2,
        callbacks=[early_stop, reduce_lr],
        verbose=1
    )

    print("[INFO] Training complete. Saving model and scalers...")
    os.makedirs(os.path.dirname(transformer_model_save_path), exist_ok=True)
    model.save(transformer_model_save_path)
    joblib.dump(feature_scaler, feature_scaler_path)
    joblib.dump(target_scaler, target_scaler_path)
    print("[INFO] Improved Transformer model and scalers saved successfully.")

    # Test prediction to verify scaling
    print("\n[INFO] Testing prediction scaling...")
    test_pred_scaled = model.predict(X_seq[-1:])
    test_pred = target_scaler.inverse_transform(test_pred_scaled)
    actual_price = y[-1][0]
    predicted_price = test_pred[0][0]
    
    print(f"[INFO] Last actual price: {actual_price:.2f}")
    print(f"[INFO] Test prediction: {predicted_price:.2f}")
    print(f"[INFO] Prediction difference: {abs(predicted_price - actual_price):.2f} ({abs(predicted_price - actual_price)/actual_price*100:.2f}%)")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", type=str, default="RELIANCE.NS", help="Stock symbol (e.g., TCS.NS)")
    args = parser.parse_args()
    train_improved(args.symbol)
