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

# Asset type definitions for global training
STOCK_SYMBOLS = [
    "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS",
    "BHARTIARTL.NS", "SBIN.NS", "LT.NS", "WIPRO.NS", "MARUTI.NS"
]

INDEX_SYMBOLS = [
    "^NSEI", "^NSEBANK", "^NSEMDCP50", "^NSEFIN", "^CNXAUTO"
]

class GlobalTrainer:
    def __init__(self):
        self.nse_fetcher = NSEDataFetcher()
        
        # Global model configurations - same for all assets
        self.model_configs = {
            'lstm': {
                'units': [128, 64, 32],  # Deeper architecture
                'dropout': 0.1,
                'epochs': 100,
                'batch_size': 32,
                'learning_rate': 0.0005,
                'patience': 20
            },
            'transformer': {
                'head_size': 32,
                'num_heads': 4,
                'ff_dim': 128,
                'dropout': 0.1,
                'epochs': 100,
                'batch_size': 32,
                'learning_rate': 0.0005,
                'patience': 20
            }
        }
        
        self.sequence_length = 60

    def detect_asset_type(self, symbol):
        """Detect if symbol is a stock or index"""
        if symbol.startswith("^"):
            return "indices"
        elif symbol in INDEX_SYMBOLS:
            return "indices"
        else:
            return "stocks"

    def preprocess_data_consistent(self, df, symbol):
        """Consistent preprocessing for all assets"""
        # Convert MultiIndex or ticker-suffixed columns to simple names
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = ['_'.join(filter(None, col)).strip() for col in df.columns.values]
        
        # Remove ticker suffix
        df.columns = [re.sub(r'_[A-Z^]+[A-Z0-9]*\.?[A-Z]*$', '', col) for col in df.columns]
        
        # Ensure lowercase columns
        df.columns = df.columns.str.lower()
        
        # Add technical indicators
        df = add_technical_indicators(df)
        if df is None or df.empty:
            print(f"[ERROR] Failed to add technical indicators for {symbol}")
            return None
            
        df = df.ffill().bfill()
        
        # Add news sentiment if missing
        if 'news_sentiment' not in df.columns:
            df['news_sentiment'] = 0.0

        # Universal volume normalization (same for all assets)
        df['volume_pct_change'] = df['volume'].pct_change().fillna(0)
        df['volume_pct_change'] = df['volume_pct_change'].clip(-0.3, 0.3)  # Conservative clipping
        df['volume'] = df['volume_pct_change']

        # Universal price change normalization (same for all assets)
        for col in ['open', 'high', 'low', 'close']:
            df[f'pct_change_{col}'] = df[col].pct_change().fillna(0)
            df[f'pct_change_{col}'] = df[f'pct_change_{col}'].clip(-0.15, 0.15)  # Conservative clipping

        df.fillna(0, inplace=True)
        return df

    def collect_global_data(self, symbols):
        """Collect data from multiple symbols to create global scalers"""
        print(f"[INFO] Collecting data from {len(symbols)} symbols for global scaling...")
        
        all_features = []
        all_targets = []
        symbol_data = {}
        
        for symbol in symbols:
            print(f"[INFO] Fetching data for {symbol}...")
            
            try:
                # Fetch data
                df = self.nse_fetcher.fetch_data(symbol)
                if df is None or df.empty:
                    print(f"[WARNING] No data for {symbol}, skipping...")
                    continue

                # Preprocess data
                df = self.preprocess_data_consistent(df, symbol)
                if df is None or df.empty:
                    print(f"[WARNING] Preprocessing failed for {symbol}, skipping...")
                    continue

                # Prepare features and targets
                extended_features = FEATURES + [f'pct_change_{col}' for col in ['open', 'high', 'low', 'close']]
                X = df[extended_features].values
                y = df['close'].values.reshape(-1, 1)

                if len(X) < self.sequence_length:
                    print(f"[WARNING] Not enough data for {symbol}, skipping...")
                    continue

                # Store for global scaling
                all_features.append(X)
                all_targets.append(y)
                symbol_data[symbol] = {'X': X, 'y': y, 'df': df}
                
                print(f"[INFO] ✅ {symbol}: {len(X)} data points, price range: {y.min():.2f} - {y.max():.2f}")
                
            except Exception as e:
                print(f"[ERROR] Failed to process {symbol}: {str(e)}")
                continue

        if not all_features:
            print("[ERROR] No valid data collected for global scaling!")
            return None, None, None

        # Combine all features and targets for global scaling
        global_features = np.vstack(all_features)
        global_targets = np.vstack(all_targets)
        
        print(f"[INFO] ✅ Global dataset created:")
        print(f"[INFO] Features shape: {global_features.shape}")
        print(f"[INFO] Targets shape: {global_targets.shape}")
        print(f"[INFO] Target range: {global_targets.min():.2f} - {global_targets.max():.2f}")
        
        return global_features, global_targets, symbol_data

    def create_global_scalers(self, global_features, global_targets):
        """Create universal scalers trained on all symbols"""
        print("[INFO] Creating global scalers...")
        
        # Create universal scalers
        global_feature_scaler = MinMaxScaler(feature_range=(0, 1))
        global_target_scaler = MinMaxScaler(feature_range=(0, 1))
        
        # Fit on combined data from all symbols
        global_feature_scaler.fit(global_features)
        global_target_scaler.fit(global_targets)
        
        print(f"[INFO] ✅ Global feature scaler range: {global_feature_scaler.data_range_}")
        print(f"[INFO] ✅ Global target scaler range: {global_target_scaler.data_range_}")
        
        return global_feature_scaler, global_target_scaler

    def create_sequences(self, X, y, seq_length=60):
        """Create sequences for LSTM/Transformer training"""
        X_seq, y_seq = [], []
        if len(X) <= seq_length:
            return np.array([]), np.array([])
        for i in range(len(X) - seq_length):
            X_seq.append(X[i:i + seq_length])
            y_seq.append(y[i + seq_length])
        return np.array(X_seq), np.array(y_seq)

    def build_lstm_model(self, input_shape):
        """Build improved LSTM model"""
        config = self.model_configs['lstm']
        
        optimizer = tf.keras.optimizers.Adam(learning_rate=config['learning_rate'])
        
        model = Sequential([
            LSTM(config['units'][0], return_sequences=True, input_shape=input_shape, recurrent_dropout=0.1),
            BatchNormalization(),
            LSTM(config['units'][1], return_sequences=True, recurrent_dropout=0.1),
            BatchNormalization(),
            LSTM(config['units'][2], recurrent_dropout=0.1),
            BatchNormalization(),
            Dropout(config['dropout']),
            Dense(64, activation='relu'),
            Dropout(config['dropout'] / 2),
            Dense(32, activation='relu'),
            Dropout(config['dropout'] / 2),
            Dense(1)
        ])
        model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
        return model

    def transformer_encoder(self, inputs, head_size=32, num_heads=4, ff_dim=128, dropout=0.1):
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

    def build_transformer_model(self, input_shape):
        """Build improved Transformer model"""
        config = self.model_configs['transformer']
        
        optimizer = tf.keras.optimizers.Adam(learning_rate=config['learning_rate'])
        
        inputs = Input(shape=input_shape)
        x = inputs
        
        # Multiple transformer layers for better learning
        x = self.transformer_encoder(x, head_size=config['head_size'], 
                                   num_heads=config['num_heads'], 
                                   ff_dim=config['ff_dim'], 
                                   dropout=config['dropout'])
        x = self.transformer_encoder(x, head_size=config['head_size'], 
                                   num_heads=config['num_heads'], 
                                   ff_dim=config['ff_dim'], 
                                   dropout=config['dropout'])
        
        x = Flatten()(x)
        x = Dense(64, activation="relu")(x)
        x = Dropout(config['dropout'])(x)
        x = Dense(32, activation="relu")(x)
        x = Dropout(config['dropout'] / 2)(x)
        outputs = Dense(1)(x)
        
        model = Model(inputs, outputs)
        model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
        return model

    def train_global_model(self, model_type="lstm"):
        """Train global model using data from all symbols"""
        print(f"\n{'='*60}")
        print(f"Training Global {model_type.upper()} Model")
        print(f"{'='*60}")
        
        # Collect data from all symbols
        all_symbols = STOCK_SYMBOLS + INDEX_SYMBOLS
        global_features, global_targets, symbol_data = self.collect_global_data(all_symbols)
        
        if global_features is None:
            print("[ERROR] Failed to collect global data!")
            return False

        # Create global scalers
        global_feature_scaler, global_target_scaler = self.create_global_scalers(global_features, global_targets)

        # Prepare training data by combining all symbols
        all_X_seq = []
        all_y_seq = []
        
        print(f"[INFO] Creating sequences from all symbols...")
        for symbol, data in symbol_data.items():
            X, y = data['X'], data['y']
            
            # Scale using global scalers
            X_scaled = global_feature_scaler.transform(X)
            y_scaled = global_target_scaler.transform(y)
            
            # Create sequences
            X_seq, y_seq = self.create_sequences(X_scaled, y_scaled, self.sequence_length)
            
            if X_seq.size > 0:
                all_X_seq.append(X_seq)
                all_y_seq.append(y_seq)
                print(f"[INFO] {symbol}: {len(X_seq)} sequences created")

        if not all_X_seq:
            print("[ERROR] No sequences created!")
            return False

        # Combine all sequences
        X_train = np.vstack(all_X_seq)
        y_train = np.vstack(all_y_seq)
        
        print(f"[INFO] ✅ Global training data prepared:")
        print(f"[INFO] X_train shape: {X_train.shape}")
        print(f"[INFO] y_train shape: {y_train.shape}")

        # Build model
        if model_type.lower() == "lstm":
            model = self.build_lstm_model((X_train.shape[1], X_train.shape[2]))
        else:  # transformer
            model = self.build_transformer_model((X_train.shape[1], X_train.shape[2]))

        # Training callbacks
        config = self.model_configs[model_type]
        early_stop = EarlyStopping(monitor='val_loss', patience=config['patience'], restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-7)

        # Train model
        print(f"[INFO] Starting global {model_type} training...")
        history = model.fit(
            X_train, y_train,
            epochs=config['epochs'],
            batch_size=config['batch_size'],
            validation_split=0.2,
            callbacks=[early_stop, reduce_lr],
            verbose=1
        )

        # Test on multiple symbols
        print(f"\n[INFO] Testing global {model_type} model...")
        test_results = {}
        
        for symbol in ["RELIANCE.NS", "TCS.NS", "^NSEI"]:  # Test on key symbols
            if symbol in symbol_data:
                X, y = symbol_data[symbol]['X'], symbol_data[symbol]['y']
                X_scaled = global_feature_scaler.transform(X)
                y_scaled = global_target_scaler.transform(y)
                
                X_seq, y_seq = self.create_sequences(X_scaled, y_scaled, self.sequence_length)
                if X_seq.size > 0:
                    pred_scaled = model.predict(X_seq[-5:])  # Last 5 predictions
                    pred = global_target_scaler.inverse_transform(pred_scaled)
                    actual = global_target_scaler.inverse_transform(y_seq[-5:])
                    
                    mae = mean_absolute_error(actual, pred)
                    error_pct = mae / np.mean(actual) * 100
                    
                    test_results[symbol] = {
                        'mae': mae,
                        'error_pct': error_pct,
                        'actual_avg': np.mean(actual),
                        'pred_avg': np.mean(pred)
                    }
                    
                    print(f"[INFO] {symbol}: MAE={mae:.2f}, Error={error_pct:.2f}%, Actual={np.mean(actual):.2f}, Pred={np.mean(pred):.2f}")

        # Save global model and scalers
        model_filename = f"global_{model_type}_model.keras"
        feature_scaler_filename = f"global_feature_scaler.pkl"
        target_scaler_filename = f"global_target_scaler.pkl"
        
        model_path = os.path.join(MODEL_DIR, model_filename)
        feature_scaler_path = os.path.join(MODEL_DIR, feature_scaler_filename)
        target_scaler_path = os.path.join(MODEL_DIR, target_scaler_filename)
        
        model.save(model_path)
        joblib.dump(global_feature_scaler, feature_scaler_path)
        joblib.dump(global_target_scaler, target_scaler_path)
        
        print(f"[INFO] ✅ Global {model_type} model saved: {model_filename}")
        print(f"[INFO] ✅ Global scalers saved")

        # Save metadata
        # Convert any numpy float32 to native float for JSON serialization
        def convert_floats(obj):
            if isinstance(obj, dict):
                return {k: convert_floats(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_floats(i) for i in obj]
            elif isinstance(obj, np.float32):
                return float(obj)
            else:
                return obj

        metadata = {
            'model_type': f'global_{model_type}',
            'training_symbols': list(symbol_data.keys()),
            'total_sequences': len(X_train),
            'sequence_length': self.sequence_length,
            'global_feature_range': [float(global_features.min()), float(global_features.max())],
            'global_target_range': [float(global_targets.min()), float(global_targets.max())],
            'test_results': convert_floats(test_results),
            'training_history': {
                'final_loss': float(history.history['loss'][-1]),
                'final_val_loss': float(history.history['val_loss'][-1])
            }
        }
        
        metadata_path = os.path.join(MODEL_DIR, f"global_{model_type}_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"[INFO] ✅ Global {model_type} metadata saved")
        return True

    def train_all_global_models(self):
        """Train both LSTM and Transformer global models"""
        print(f"\n{'='*80}")
        print("TRAINING GLOBAL MODELS FOR ALL ASSETS")
        print(f"{'='*80}")
        
        # Train LSTM
        lstm_success = self.train_global_model("lstm")
        
        # Train Transformer
        transformer_success = self.train_global_model("transformer")
        
        if lstm_success and transformer_success:
            print(f"\n[SUCCESS] ✅ Both global models trained successfully!")
            print(f"[INFO] Models can now predict any stock or index using universal scalers")
        else:
            print(f"\n[WARNING] ⚠️ Some global models failed")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=["lstm", "transformer", "both"], default="both", help="Model type to train")
    
    args = parser.parse_args()
    
    trainer = GlobalTrainer()
    
    if args.model == "both":
        trainer.train_all_global_models()
    else:
        trainer.train_global_model(args.model)
