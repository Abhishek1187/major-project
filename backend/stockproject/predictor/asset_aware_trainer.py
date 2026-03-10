import os
import sys
import re
import json
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Input, LayerNormalization, MultiHeadAttention, Flatten
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Local imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
try:
    from .utils import add_technical_indicators
    from .nse_data_fetcher import NSEDataFetcher
except ImportError:
    # Fallback for standalone execution
    from utils import add_technical_indicators
    from nse_data_fetcher import NSEDataFetcher

# Directory and file paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

FEATURES = [
    'open', 'high', 'low', 'volume',
    'sma_10', 'ema_9', 'ema_21', 'rsi_14', 'ma_20',
    'bb_upper', 'bb_lower', 'ema_12', 'ema_26',
    'macd', 'signal_line', '%k', '%d', 'atr_14',
    'news_sentiment'
]

# Asset type definitions - Updated to match frontend stockSymbols.js
STOCK_SYMBOLS = [
    "RELIANCE.NS", "AXISBANK.NS", "HDFCBANK.NS", "ONGC.NS", "SBIN.NS",
    "INFY.NS", "TCS.NS", "ICICIBANK.NS", "KOTAKBANK.NS", "ADANIPORTS.NS",
    "ADANIENT.NS", "BAJFINANCE.NS", "BHARTIARTL.NS"
]

INDEX_SYMBOLS = [
    "^NSEI", "^NSEBANK", "^NSEMDCP50", "^CNXAUTO"
]

class AssetAwareTrainer:
    def __init__(self):
        self.nse_fetcher = NSEDataFetcher()
        self.asset_configs = {
            'stocks': {
                'price_range': (100, 5000),  # Typical stock price range
                'scaler_range': (0, 1),  # Standard scaling range for features
                'target_scaler_range': (0, 1),  # Standard range for better generalization
                'sequence_length': 60,
                'model_params': {
                    'lstm_units': [128, 64, 32],  # Deeper architecture for better learning
                    'dropout': 0.1,  # Lower dropout for stocks
                    'epochs': 100,  # More epochs for better convergence
                    'batch_size': 32,  # Larger batch for stability
                    'learning_rate': 0.0005,  # Lower learning rate for precision
                    'patience': 20  # More patience for early stopping
                }
            },
            'indices': {
                'price_range': (10000, 60000),  # Extended range to include Bank Nifty (~52k)
                'scaler_range': (0, 1),  # Standard scaling range for features
                'target_scaler_range': (0, 1),  # Standard range for consistency
                'sequence_length': 60,
                'model_params': {
                    'lstm_units': [128, 64, 32],  # Consistent architecture
                    'dropout': 0.15,  # Slightly higher dropout for indices
                    'epochs': 100,  # More epochs for indices
                    'batch_size': 32,  # Larger batch for indices
                    'learning_rate': 0.0005,  # Consistent learning rate
                    'patience': 20
                }
            }
        }

    def detect_asset_type(self, symbol):
        """Detect if symbol is a stock or index"""
        if symbol.startswith("^"):
            return "indices"
        elif symbol in INDEX_SYMBOLS:
            return "indices"
        else:
            return "stocks"

    def get_asset_config(self, symbol):
        """Get configuration for specific asset type"""
        asset_type = self.detect_asset_type(symbol)
        return self.asset_configs[asset_type], asset_type

    def preprocess_data_consistent(self, df, asset_type):
        """Consistent preprocessing for both training and prediction"""
        # Convert MultiIndex or ticker-suffixed columns to simple names
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = ['_'.join(filter(None, col)).strip() for col in df.columns.values]
        
        # Remove ticker suffix
        df.columns = [re.sub(r'_[A-Z^]+[A-Z0-9]*\.?[A-Z]*$', '', col) for col in df.columns]
        
        # Ensure lowercase columns
        df.columns = df.columns.str.lower()
        
        print(f"[DEBUG] Columns after cleaning: {df.columns}")
        
        # Add technical indicators
        df = add_technical_indicators(df)
        if df is None or df.empty:
            print("[ERROR] Failed to add technical indicators")
            return None
            
        df = df.ffill().bfill()
        
        # Add news sentiment if missing
        if 'news_sentiment' not in df.columns:
            df['news_sentiment'] = 0.0

        # Asset-specific volume normalization
        if asset_type == "stocks":
            # More aggressive volume normalization for stocks
            df['volume_pct_change'] = df['volume'].pct_change().fillna(0)
            df['volume_pct_change'] = df['volume_pct_change'].clip(-0.5, 0.5)
        else:
            # Conservative volume normalization for indices
            df['volume_pct_change'] = df['volume'].pct_change().fillna(0)
            df['volume_pct_change'] = df['volume_pct_change'].clip(-0.2, 0.2)
        
        df['volume'] = df['volume_pct_change']

        # Asset-specific price change normalization
        price_change_limit = 0.1 if asset_type == "indices" else 0.2
        for col in ['open', 'high', 'low', 'close']:
            df[f'pct_change_{col}'] = df[col].pct_change().fillna(0)
            df[f'pct_change_{col}'] = df[f'pct_change_{col}'].clip(-price_change_limit, price_change_limit)

        df.fillna(0, inplace=True)
        return df

    def create_asset_specific_scaler(self, data, asset_config, is_target=False):
        """Create scaler with asset-specific range optimized for stocks vs indices"""
        if is_target:
            # Different target scaling strategies for stocks vs indices
            if asset_config == self.asset_configs['stocks']:
                # For stocks: Use tighter range for better precision
                return MinMaxScaler(feature_range=(0.1, 0.9))
            else:
                # For indices: Use full range due to larger price movements
                return MinMaxScaler(feature_range=(0.0, 1.0))
        else:
            # Feature scaling remains consistent
            scaler_min, scaler_max = asset_config['scaler_range']
            return MinMaxScaler(feature_range=(scaler_min, scaler_max))

    def validate_price_range(self, prices, asset_config, symbol):
        """Validate if prices are within expected range for asset type"""
        min_price, max_price = asset_config['price_range']
        price_mean = np.mean(prices)
        
        if price_mean < min_price * 0.5 or price_mean > max_price * 2:
            print(f"[WARNING] {symbol} prices outside expected range: {price_mean:.2f}")
            print(f"[WARNING] Expected range: {min_price} - {max_price}")
            return False
        return True

    def create_sequences(self, X, y, seq_length=60):
        """Create sequences for LSTM/Transformer training"""
        X_seq, y_seq = [], []
        if len(X) <= seq_length:
            return np.array([]), np.array([])
        for i in range(len(X) - seq_length):
            X_seq.append(X[i:i + seq_length])
            y_seq.append(y[i + seq_length])
        return np.array(X_seq), np.array(y_seq)

    def build_lstm_model(self, input_shape, asset_config):
        """Build improved LSTM model with deeper architecture"""
        params = asset_config['model_params']
        
        # Create optimizer with asset-specific learning rate
        optimizer = tf.keras.optimizers.Adam(learning_rate=params['learning_rate'])
        
        model = Sequential([
            LSTM(params['lstm_units'][0], return_sequences=True, input_shape=input_shape, recurrent_dropout=0.1),
            BatchNormalization(),
            LSTM(params['lstm_units'][1], return_sequences=True, recurrent_dropout=0.1),
            BatchNormalization(),
            LSTM(params['lstm_units'][2], recurrent_dropout=0.1),
            BatchNormalization(),
            Dropout(params['dropout']),
            Dense(32, activation='relu'),
            Dropout(params['dropout'] / 2),
            Dense(16, activation='relu'),
            Dropout(params['dropout'] / 2),
            Dense(1)
        ])
        model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
        return model

    def transformer_encoder(self, inputs, head_size=32, num_heads=2, ff_dim=64, dropout=0.1):
        """Transformer encoder block"""
        x = MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(inputs, inputs)
        x = Dropout(dropout)(x)
        x = LayerNormalization(epsilon=1e-6)(x)
        res = x + inputs

        x = Dense(ff_dim, activation="relu")(res)
        x = Dropout(dropout)(x)
        x = Dense(inputs.shape[-1])(x)
        x = LayerNormalization(epsilon=1e-6)(x)
        return x + res

    def build_transformer_model(self, input_shape, asset_config):
        """Build Transformer model with asset-specific parameters"""
        params = asset_config['model_params']
        
        # Create optimizer with asset-specific learning rate
        optimizer = tf.keras.optimizers.Adam(learning_rate=params['learning_rate'])
        
        inputs = Input(shape=input_shape)
        x = inputs
        
        # Single transformer layer for simplicity
        x = self.transformer_encoder(x, dropout=params['dropout'])
        
        x = Flatten()(x)
        x = Dense(32, activation="relu")(x)
        x = Dropout(params['dropout'])(x)
        x = Dense(16, activation="relu")(x)
        x = Dropout(params['dropout'] / 2)(x)
        outputs = Dense(1)(x)
        
        model = Model(inputs, outputs)
        model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
        return model

    def train_asset_specific_model(self, symbol, model_type="lstm"):
        """Train asset-specific model with proper validation"""
        print(f"\n[INFO] Training {model_type.upper()} model for {symbol}")
        
        # Get asset configuration
        asset_config, asset_type = self.get_asset_config(symbol)
        print(f"[INFO] Detected asset type: {asset_type}")
        
        # Fetch training data - use daily data for consistency
        print(f"[INFO] Fetching daily historical data for {symbol}")
        df = self.nse_fetcher.fetch_data(symbol)
        if df is None or df.empty:
            print(f"[ERROR] No training data fetched for {symbol}")
            return False

        # Preprocess data consistently
        df = self.preprocess_data_consistent(df, asset_type)
        if df is None or df.empty:
            print(f"[ERROR] Preprocessing failed for {symbol}")
            return False

        # Validate price range
        if not self.validate_price_range(df['close'].values, asset_config, symbol):
            print(f"[ERROR] Price validation failed for {symbol}")
            return False

        # Prepare features
        extended_features = FEATURES + [f'pct_change_{col}' for col in ['open', 'high', 'low', 'close']]
        X = df[extended_features].values
        y = df['close'].values.reshape(-1, 1)

        print(f"[INFO] Training data shape: X={X.shape}, y={y.shape}")
        print(f"[INFO] Price range: {y.min():.2f} to {y.max():.2f}")

        # Create asset-specific scalers with different strategies for features vs targets
        feature_scaler = self.create_asset_specific_scaler(X, asset_config, is_target=False)
        target_scaler = self.create_asset_specific_scaler(y, asset_config, is_target=True)

        X_scaled = feature_scaler.fit_transform(X)
        y_scaled = target_scaler.fit_transform(y)

        print(f"[INFO] Scaled ranges - Features: {X_scaled.min():.4f} to {X_scaled.max():.4f}")
        print(f"[INFO] Scaled ranges - Target: {y_scaled.min():.4f} to {y_scaled.max():.4f}")

        # Create sequences
        seq_length = asset_config['sequence_length']
        X_seq, y_seq = self.create_sequences(X_scaled, y_scaled, seq_length)
        
        if X_seq.size == 0 or y_seq.size == 0:
            print(f"[ERROR] Not enough data to create sequences for {symbol}")
            return False

        print(f"[INFO] Sequence data shape: X_seq={X_seq.shape}, y_seq={y_seq.shape}")

        # Build model
        if model_type.lower() == "lstm":
            model = self.build_lstm_model((X_seq.shape[1], X_seq.shape[2]), asset_config)
        else:  # transformer
            model = self.build_transformer_model((X_seq.shape[1], X_seq.shape[2]), asset_config)

        # Train model
        params = asset_config['model_params']
        
        # Training callbacks with improved patience
        early_stop = EarlyStopping(monitor='val_loss', patience=params['patience'], restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=8, min_lr=1e-7)
        print(f"[INFO] Starting training with {params['epochs']} epochs...")
        
        history = model.fit(
            X_seq, y_seq,
            epochs=params['epochs'],
            batch_size=params['batch_size'],
            validation_split=0.2,
            callbacks=[early_stop, reduce_lr],
            verbose=1
        )

        # Test prediction to verify scaling
        print("\n[INFO] Testing prediction scaling...")
        test_pred_scaled = model.predict(X_seq[-5:])  # Test on last 5 sequences
        test_pred = target_scaler.inverse_transform(test_pred_scaled)
        actual_prices = y[-5:]
        
        print(f"[INFO] Last 5 actual prices: {actual_prices.flatten()}")
        print(f"[INFO] Last 5 predictions: {test_pred.flatten()}")
        
        # Calculate prediction accuracy
        mae = mean_absolute_error(actual_prices, test_pred)
        mse = mean_squared_error(actual_prices, test_pred)
        rmse = np.sqrt(mse)
        
        print(f"[INFO] Test Metrics - MAE: {mae:.2f}, RMSE: {rmse:.2f}")
        print(f"[INFO] Average prediction error: {mae/np.mean(actual_prices)*100:.2f}%")

        # Save model and scalers with asset-specific naming
        model_filename = f"{model_type}_{asset_type}_{symbol.replace('.', '_').replace('^', 'INDEX_')}_model.keras"
        feature_scaler_filename = f"feature_scaler_{asset_type}_{symbol.replace('.', '_').replace('^', 'INDEX_')}.pkl"
        target_scaler_filename = f"target_scaler_{asset_type}_{symbol.replace('.', '_').replace('^', 'INDEX_')}.pkl"
        
        model_path = os.path.join(MODEL_DIR, model_filename)
        feature_scaler_path = os.path.join(MODEL_DIR, feature_scaler_filename)
        target_scaler_path = os.path.join(MODEL_DIR, target_scaler_filename)
        
        model.save(model_path)
        joblib.dump(feature_scaler, feature_scaler_path)
        joblib.dump(target_scaler, target_scaler_path)
        
        print(f"[INFO] ✅ {model_type.upper()} model saved: {model_filename}")
        print(f"[INFO] ✅ Scalers saved for {symbol}")

        # Save training metadata
        metadata = {
            'symbol': symbol,
            'asset_type': asset_type,
            'model_type': model_type,
            'training_data_points': len(df),
            'sequence_length': seq_length,
            'price_range': [float(y.min()), float(y.max())],
            'scaled_range': [float(y_scaled.min()), float(y_scaled.max())],
            'test_mae': float(mae),
            'test_rmse': float(rmse),
            'training_history': {
                'final_loss': float(history.history['loss'][-1]),
                'final_val_loss': float(history.history['val_loss'][-1])
            }
        }
        
        metadata_path = os.path.join(MODEL_DIR, f"metadata_{asset_type}_{symbol.replace('.', '_').replace('^', 'INDEX_')}.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"[INFO] ✅ Training metadata saved")
        return True

    def train_all_assets(self):
        """Train models for all defined assets"""
        all_symbols = STOCK_SYMBOLS + INDEX_SYMBOLS
        
        for symbol in all_symbols:
            print(f"\n{'='*60}")
            print(f"Training models for {symbol}")
            print(f"{'='*60}")
            
            # Train LSTM
            lstm_success = self.train_asset_specific_model(symbol, "lstm")
            
            # Train Transformer
            transformer_success = self.train_asset_specific_model(symbol, "transformer")
            
            if lstm_success and transformer_success:
                print(f"[SUCCESS] ✅ Both models trained successfully for {symbol}")
            else:
                print(f"[WARNING] ⚠️ Some models failed for {symbol}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", type=str, help="Specific symbol to train (optional)")
    parser.add_argument("--model", type=str, choices=["lstm", "transformer", "both"], default="both", help="Model type to train")
    parser.add_argument("--all", action="store_true", help="Train all predefined assets")
    
    args = parser.parse_args()
    
    trainer = AssetAwareTrainer()
    
    if args.all:
        trainer.train_all_assets()
    elif args.symbol:
        if args.model in ["lstm", "both"]:
            trainer.train_asset_specific_model(args.symbol, "lstm")
        if args.model in ["transformer", "both"]:
            trainer.train_asset_specific_model(args.symbol, "transformer")
    else:
        print("Please specify --symbol or --all flag")
