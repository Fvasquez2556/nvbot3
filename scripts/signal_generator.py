"""
🚀 NvBot3 Signal Generator - Main Trading Signal Detection
=========================================================

This is the main script that generates live trading signals using trained models.
Integrates with the web dashboard to track signals automatically.

Models:
🔥 Momentum Model (XGBoost): Detectar movimientos alcistas ≥5%
⚡ Rebound Model (Random Forest): Predecir rebotes 1-3%
📊 Regime Model (LSTM): Clasificar tendencia de mercado
🎯 Advanced Momentum (Ensemble): Momentum con filtros de volumen
"""

import sys
import os
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import logging
import time
import pickle
import ccxt
import warnings
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
warnings.filterwarnings('ignore')

# Import the integration bridge for dashboard tracking
from integration.nvbot3_feedback_bridge import track_signal, update_price

# Data processing components
from src.data.feature_calculator import FeatureCalculator
from src.data.feature_selector import FeatureSelector

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

class NvBot3SignalGenerator:
    """Main signal generator that uses trained models to detect trading opportunities"""
    
    def __init__(self, models_path: str = "data/models", config_path: str = "config/training_config.yaml"):
        self.models_path = Path(models_path)
        self.config_path = Path(config_path)
        
        # Load configuration from YAML
        self._load_config()
        
        # Initialize components
        self.feature_calculator = FeatureCalculator()
        self.feature_selector = FeatureSelector()
        self.exchange = None
        self.loaded_models = {}
        
        # Model configurations
        self.model_types = {
            'momentum': {
                'description': '🔥 Momentum Model: Detectar movimientos ≥5%',
                'threshold': 0.75,
                'target_change': 5.0
            },
            'rebound': {
                'description': '⚡ Rebound Model: Predecir rebotes 1-3%',
                'threshold': 0.70,
                'target_change': 2.0
            },
            'regime': {
                'description': '📊 Regime Model: Clasificar tendencia de mercado',
                'threshold': 0.60,
                'target_change': 0.0  # Classification, not price change
            },
            'momentum_advanced': {
                'description': '🎯 Advanced Momentum: Momentum con filtros',
                'threshold': 0.80,
                'target_change': 5.0
            }
        }
        
        # EXACT FEATURE SELECTIONS USED DURING TRAINING
        # These are the precise features each model expects
        self.model_features = {
            'momentum': [
                'candle_range', 'atr_ratio_30', 'atr_ratio_20', 'atr_ratio_14', 'candle_body', 'upper_shadow', 'regime_low_vol', 
                'lower_shadow', 'asian_session', 'volume_sma_20', 'european_session', 'hour', 'vpt', 'true_range_norm', 
                'distance_to_support_10', 'distance_to_support_20', 'minus_di_20', 'distance_to_support_50', 'minus_di_30', 
                'minus_di_14', 'sma_200', 'ema_200', 'bb_width_10', 'ema_100', 'sma_100', 'min_low_20', 'sma_50', 
                'bb_lower_20', 'ema_50', 'min_low_10', 'bb_middle_30', 'bb_lower_30', 'bb_upper_30', 'bb_middle_20', 
                'sma_20', 'ema_20', 'sma_10', 'ema_10', 'bb_upper_20', 'min_low_50'
            ],
            'rebound': [
                'atr_ratio_30', 'atr_ratio_20', 'candle_range', 'atr_ratio_14', 'candle_body', 'lower_shadow', 'upper_shadow', 
                'local_min_10', 'ema_200', 'sma_200', 'min_low_10', 'min_low_20', 'bb_lower_20', 'ema_10', 'sma_10', 
                'bb_middle_20', 'sma_20', 'ema_20', 'bb_upper_20', 'bb_middle_30', 'bb_lower_30', 'bb_upper_30', 
                'sma_100', 'ema_100', 'min_low_50', 'ema_50', 'sma_50', 'day_of_week', 'local_min_5', 'regime_low_vol', 
                'atr_30', 'true_range_norm', 'atr_20', 'atr_14', 'vpt'
            ],
            'regime': [
                'european_session', 'atr_ratio_30', 'price_above_sma_200', 'atr_ratio_14', 'atr_ratio_20', 'local_min_10', 
                'day_of_week', 'ema_20', 'sma_200', 'hour', 'sma_200_slope', 'ema_100', 'sma_10', 'ema_200', 'ema_10', 
                'sma_50', 'bb_lower_20', 'ma_alignment_bear', 'bb_upper_30', 'min_low_50', 'sma_100', 'sma_100_slope', 
                'min_low_10', 'bb_middle_30', 'ema_200_slope'
            ],
            'momentum_advanced': [
                'candle_range', 'atr_ratio_30', 'atr_ratio_20', 'atr_ratio_14', 'candle_body', 'upper_shadow', 'asian_session', 
                'lower_shadow', 'hour', 'minus_di_14', 'volume_sma_20', 'minus_di_20', 'regime_low_vol', 'european_session', 
                'rsi_14', 'cci_20', 'bb_position_30', 'bb_position_20', 'rsi_21', 'minus_di_30', 'rsi_7', 'true_range_norm', 
                'sma_200', 'ema_200', 'cci_14', 'rsi_30', 'mfi_14', 'distance_to_support_20', 'sma_100', 'ema_100', 
                'bb_upper_30', 'sma_50', 'bb_upper_20', 'ema_50', 'bb_middle_30', 'sma_20', 'bb_middle_20', 'ema_20', 
                'min_low_10', 'min_low_20', 'sma_10', 'ema_10', 'bb_lower_20', 'min_low_50', 'bb_lower_30', 
                'price_above_sma_20', 'distance_to_support_10', 'stoch_k_20', 'williams_r_20', 'momentum_score'
            ]
        }
        
        logger.info("🤖 NvBot3 Signal Generator initialized")
    
    def _load_config(self):
        """Load configuration from YAML file"""
        try:
            import yaml
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            # Get all symbols from tiers
            data_config = config.get('data', {})
            symbols_config = data_config.get('symbols', {})
            
            # Combine all tiers
            tier_1 = symbols_config.get('tier_1', [])
            tier_2 = symbols_config.get('tier_2', [])  
            tier_3 = symbols_config.get('tier_3', [])
            
            self.symbols = tier_1 + tier_2 + tier_3
            self.timeframes = data_config.get('timeframes', ['5m', '15m', '1h', '4h', '1d'])
            
            logger.info(f"📊 Loaded {len(self.symbols)} symbols from config: {len(tier_1)} tier1 + {len(tier_2)} tier2 + {len(tier_3)} tier3")
            logger.info(f"🕐 Timeframes: {self.timeframes}")
            
        except Exception as e:
            logger.warning(f"⚠️ Could not load config, using default symbols: {e}")
            # Fallback to default symbols
            self.symbols = [
                'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'ADAUSDT', 'SOLUSDT',
                'XRPUSDT', 'DOTUSDT', 'LINKUSDT', 'MATICUSDT', 'AVAXUSDT'
            ]
            self.timeframes = ['5m', '15m', '1h', '4h', '1d']
    
    def initialize_exchange(self):
        """Initialize Binance exchange connection"""
        try:
            self.exchange = ccxt.binance({
                'apiKey': os.getenv('BINANCE_API_KEY', ''),
                'secret': os.getenv('BINANCE_SECRET_KEY', ''),
                'sandbox': False,
                'rateLimit': 1200,  # Respect API limits
                'enableRateLimit': True,
            })
            
            logger.info("✅ Binance exchange connected")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error connecting to exchange: {e}")
            return False
    
    def load_model(self, symbol: str, timeframe: str, model_type: str) -> Optional[Any]:
        """Load a trained model for specific symbol, timeframe, and type"""
        model_key = f"{symbol}_{timeframe}_{model_type}"
        
        if model_key in self.loaded_models:
            return self.loaded_models[model_key]
        
        # Try to load model file
        model_file = self.models_path / f"{model_key}.pkl"
        
        if not model_file.exists():
            # Try loading from consolidated models (ALL_SYMBOLS)
            consolidated_model_file = self.models_path / f"ALL_SYMBOLS_{timeframe}_{model_type}.pkl"
            if consolidated_model_file.exists():
                model_file = consolidated_model_file
            else:
                logger.warning(f"⚠️ Model not found: {model_key}")
                return None
        
        try:
            with open(model_file, 'rb') as f:
                model = pickle.load(f)
            
            self.loaded_models[model_key] = model
            logger.info(f"✅ Model loaded: {model_key}")
            return model
            
        except Exception as e:
            logger.error(f"❌ Error loading model {model_key}: {e}")
            return None
    
    def get_market_data(self, symbol: str, timeframe: str, limit: int = 200) -> Optional[pd.DataFrame]:
        """Get recent market data for analysis"""
        if not self.exchange:
            logger.error("❌ Exchange not initialized")
            return None
        
        try:
            # Get OHLCV data
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            
            if not ohlcv:
                logger.error(f"❌ No data received for {symbol} {timeframe}")
                return None
            
            # Convert to DataFrame
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # Add symbol and timeframe info
            df['symbol'] = symbol
            df['timeframe'] = timeframe
            
            logger.info(f"📊 Market data loaded: {symbol} {timeframe} - {len(df)} candles")
            return df
            
        except Exception as e:
            logger.error(f"❌ Error getting market data for {symbol}: {e}")
            return None
    
    def calculate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical features using the EXACT same method as training"""
        try:
            logger.info("🧮 Using FeatureCalculator for identical feature calculation")
            
            # Create temporary DataFrame compatible with FeatureCalculator
            temp_df = df.copy()
            
            # Use FeatureCalculator to compute ALL training features
            # This ensures 100% compatibility with trained models
            temp_calculator = FeatureCalculator()
            
            # Calculate momentum features (ROC, ADX, MACD, RSI, OBV, Bollinger Bands)
            temp_df = temp_calculator.calculate_momentum_features(temp_df)
            
            # Calculate rebound features (RSI oversold, MACD histogram, volume patterns, S/R)
            temp_df = temp_calculator.calculate_rebound_features(temp_df)
            
            # Calculate regime features (ATR, BB width, ADX trending, MA slopes)
            temp_df = temp_calculator.calculate_regime_features(temp_df)
            
            # Calculate additional features (candlestick patterns, stochastics, CCI, MFI, time features)
            temp_df = temp_calculator.calculate_additional_features(temp_df)
            
            # Final cleaning - identical to training
            temp_df.replace([np.inf, -np.inf], np.nan, inplace=True)
            temp_df = temp_df.ffill().fillna(0)
            
            total_features = len(temp_df.columns)
            original_features = len(df.columns)
            added_features = total_features - original_features
            
            logger.info(f"✅ Training-compatible features calculated: {added_features} new features ({total_features} total)")
            return temp_df
            
        except Exception as e:
            logger.error(f"❌ Error calculating training-compatible features: {e}")
            logger.warning("⚠️ Falling back to basic feature calculation...")
            
            # Fallback to basic features if FeatureCalculator fails
            return self._calculate_basic_features_fallback(df)
    
    def verify_feature_compatibility(self, features_df: pd.DataFrame, symbol: str, timeframe: str) -> Dict[str, Dict]:
        """Verify which models have compatible features available using FeatureSelector"""
        compatibility = {}
        
        for model_type in self.model_types.keys():
            try:
                # Get required features for this model
                required_features = self.feature_selector.get_required_features(model_type, symbol, timeframe)
                available_features = list(features_df.columns)
                
                # Check compatibility
                found_features, missing_features = self.feature_selector.validate_features(
                    available_features, required_features
                )
                
                compatibility[model_type] = {
                    'available': len(found_features),
                    'expected': len(required_features),
                    'missing': missing_features,
                    'compatibility_rate': len(found_features) / len(required_features) if required_features else 0
                }
                
            except Exception as e:
                logger.warning(f"Could not check compatibility for {model_type}: {e}")
                compatibility[model_type] = {
                    'available': 0,
                    'expected': 0,
                    'missing': [],
                    'compatibility_rate': 0.0
                }
        
        return compatibility
    
    def _calculate_basic_features_fallback(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fallback basic feature calculation method"""
        try:
            df_with_features = df.copy()
            
            # Basic price features
            df_with_features['price_change'] = df_with_features['close'].pct_change()
            df_with_features['high_low_pct'] = (df_with_features['high'] - df_with_features['low']) / df_with_features['close']
            
            # Moving averages
            for period in [5, 10, 20, 50]:
                df_with_features[f'sma_{period}'] = df_with_features['close'].rolling(period).mean()
                df_with_features[f'ema_{period}'] = df_with_features['close'].ewm(span=period).mean()
            
            # RSI
            delta = df_with_features['close'].diff()
            gain = (delta.where(delta.gt(0), 0)).rolling(window=14).mean()
            loss = (-delta.where(delta.lt(0), 0)).rolling(window=14).mean()
            rs = gain / loss
            df_with_features['rsi_14'] = 100 - (100 / (1 + rs))
            
            # Volume indicators
            df_with_features['volume_sma_20'] = df_with_features['volume'].rolling(20).mean()
            df_with_features['volume_ratio'] = df_with_features['volume'] / df_with_features['volume_sma_20']
            
            # Bollinger Bands (basic)
            bb_middle = df_with_features['close'].rolling(20).mean()
            bb_std_dev = df_with_features['close'].rolling(20).std()
            df_with_features['bb_upper_20'] = bb_middle + (bb_std_dev * 2)
            df_with_features['bb_lower_20'] = bb_middle - (bb_std_dev * 2)
            df_with_features['bb_width_20'] = (df_with_features['bb_upper_20'] - df_with_features['bb_lower_20']) / bb_middle
            df_with_features['bb_position_20'] = (df_with_features['close'] - df_with_features['bb_lower_20']) / (df_with_features['bb_upper_20'] - df_with_features['bb_lower_20'])
            
            # Clean data
            df_with_features = df_with_features.ffill().fillna(0)
            df_with_features = df_with_features.replace([np.inf, -np.inf], 0)
            
            logger.info(f"⚠️ Fallback features calculated: {len(df_with_features.columns)} features")
            return df_with_features
            
        except Exception as e:
            logger.error(f"❌ Error in fallback feature calculation: {e}")
            return df
    
    def make_prediction(self, model: Any, features: pd.DataFrame, model_type: str, symbol: str, timeframe: str) -> Optional[Dict]:
        """Make a prediction using the loaded model with FeatureSelector compatibility layer"""
        try:
            # Get the latest feature row for prediction
            if len(features) == 0:
                logger.error("❌ No features available for prediction")
                return None
            
            latest_features = features.iloc[-1:].copy()
            
            # USE FEATURESELECTOR FOR EXACT COMPATIBILITY
            try:
                # Select features using FeatureSelector
                selected_features = self.feature_selector.select_features(
                    latest_features, model_type, symbol, timeframe, strict=True
                )
                
                logger.info(f"✅ FeatureSelector: Selected {len(selected_features.columns)} features for {model_type}")
                logger.debug(f"   Model: {symbol}_{timeframe}_{model_type}")
                logger.debug(f"   Sample features: {list(selected_features.columns)[:5]}...")
                
                X = selected_features.copy()
                
            except Exception as e:
                logger.error(f"❌ FeatureSelector failed for {model_type}: {e}")
                return None
            
            # Handle different model types
            if model_type == 'regime' and isinstance(model, dict):
                # LSTM model with scaler
                scaler = model.get('scaler')
                lstm_model = model.get('model')
                sequence_length = model.get('sequence_length', 20)
                
                if scaler and lstm_model:
                    # Scale features
                    X_scaled = scaler.transform(X)
                    
                    # Create sequence (use last sequence_length rows)
                    if len(features) >= sequence_length:
                        recent_features = features.iloc[-sequence_length:][selected_features.columns]
                        X_sequence = scaler.transform(recent_features)
                        X_sequence = X_sequence.reshape(1, sequence_length, -1)
                        
                        # Make prediction
                        pred_proba = lstm_model.predict(X_sequence, verbose=0)
                        prediction = np.argmax(pred_proba, axis=1)[0]
                        confidence = np.max(pred_proba)
                    else:
                        logger.warning(f"⚠️ Not enough data for LSTM sequence ({len(features)} < {sequence_length})")
                        return None
                else:
                    logger.error("❌ Invalid LSTM model structure")
                    return None
                    
            elif model_type == 'momentum_advanced' and isinstance(model, dict):
                # Ensemble model
                xgb_model = model.get('xgb')
                rf_model = model.get('rf')
                weights = model.get('weights', [0.7, 0.3])
                
                if xgb_model and rf_model:
                    # Get predictions from both models
                    xgb_proba = xgb_model.predict_proba(X)
                    rf_proba = rf_model.predict_proba(X)
                    
                    # Weighted average
                    ensemble_proba = weights[0] * xgb_proba + weights[1] * rf_proba
                    prediction = np.argmax(ensemble_proba, axis=1)[0]
                    confidence = np.max(ensemble_proba)
                else:
                    logger.error("❌ Invalid ensemble model structure")
                    return None
                    
            else:
                # Standard models (XGBoost, Random Forest)
                if hasattr(model, 'predict_proba'):
                    pred_proba = model.predict_proba(X)
                    prediction = np.argmax(pred_proba, axis=1)[0]
                    confidence = np.max(pred_proba)
                else:
                    prediction = model.predict(X)[0]
                    confidence = 0.8  # Default confidence for models without probability
            
            # Prepare prediction result
            result = {
                'type': model_type,
                'prediction': int(prediction),
                'confidence': float(confidence),
                'predicted_change': self.model_types[model_type]['target_change'] if prediction == 1 else 0,
                'timestamp': datetime.now().isoformat(),
                'model_description': self.model_types[model_type]['description']
            }
            
            logger.info(f"🎯 Prediction made: {model_type} = {prediction} (conf: {confidence:.3f})")
            return result
            
        except Exception as e:
            logger.error(f"❌ Error making prediction: {e}")
            return None
    
    def generate_signals_for_symbol(self, symbol: str, timeframe: str = '5m') -> List[Dict]:
        """Generate signals for a specific symbol and timeframe"""
        logger.info(f"🔍 Generating signals for {symbol} {timeframe}")
        
        signals = []
        
        try:
            # Get market data
            market_data = self.get_market_data(symbol, timeframe)
            if market_data is None:
                logger.error(f"❌ Could not get market data for {symbol}")
                return signals
            
            # Get current price for tracking
            current_price = float(market_data['close'].iloc[-1])
            
            # Update price in dashboard
            update_price(symbol, current_price)
            
            # Calculate features
            features_df = self.calculate_features(market_data)
            
            # Verify feature compatibility
            compatibility = self.verify_feature_compatibility(features_df, symbol, timeframe)
            logger.info(f"🔍 Feature compatibility check:")
            for model_type, comp in compatibility.items():
                rate = comp['compatibility_rate']
                status = "✅" if rate == 1.0 else "⚠️" if rate > 0.8 else "❌"
                logger.info(f"   {status} {model_type}: {comp['available']}/{comp['expected']} ({rate:.1%})")
                if rate < 1.0:
                    logger.warning(f"      Missing: {comp['missing'][:3]}...")
            
            # Generate predictions for each model type
            for model_type in self.model_types.keys():
                try:
                    # Load model
                    model = self.load_model(symbol, timeframe, model_type)
                    if model is None:
                        continue
                    
                    # Make prediction
                    prediction = self.make_prediction(model, features_df, model_type, symbol, timeframe)
                    if prediction is None:
                        continue
                    
                    # Check if signal meets threshold
                    threshold = self.model_types[model_type]['threshold']
                    if prediction['confidence'] >= threshold and prediction['prediction'] == 1:
                        
                        # Track signal in dashboard
                        signal_id = track_signal(symbol, prediction, current_price)
                        
                        # Add to results
                        signal_data = {
                            'signal_id': signal_id,
                            'symbol': symbol,
                            'timeframe': timeframe,
                            'current_price': current_price,
                            **prediction
                        }
                        
                        signals.append(signal_data)
                        
                        logger.info(f"🚨 SIGNAL: {symbol} {model_type} - Confidence: {prediction['confidence']:.3f}")
                
                except Exception as e:
                    logger.error(f"❌ Error processing {model_type} for {symbol}: {e}")
                    continue
            
            return signals
            
        except Exception as e:
            logger.error(f"❌ Error generating signals for {symbol}: {e}")
            return signals
    
    def run_signal_scan(self, symbols: Optional[List[str]] = None, timeframes: Optional[List[str]] = None) -> Dict:
        """Run a complete signal scan across symbols and timeframes"""
        if symbols is None:
            symbols = self.symbols[:5]  # Limit to first 5 symbols for testing
        if timeframes is None:
            timeframes = ['5m', '1h']  # Limit to key timeframes
        
        # Type checker assertions
        assert symbols is not None
        assert timeframes is not None
        
        logger.info(f"🚀 Starting signal scan for {len(symbols)} symbols x {len(timeframes)} timeframes")
        
        all_signals = []
        scan_stats = {
            'total_symbols': len(symbols),
            'total_timeframes': len(timeframes),
            'signals_generated': 0,
            'errors': 0,
            'start_time': time.time()
        }
        
        for symbol in symbols:
            for timeframe in timeframes:
                try:
                    signals = self.generate_signals_for_symbol(symbol, timeframe)
                    all_signals.extend(signals)
                    scan_stats['signals_generated'] += len(signals)
                    
                    # Rate limiting
                    time.sleep(0.5)
                    
                except Exception as e:
                    logger.error(f"❌ Error scanning {symbol} {timeframe}: {e}")
                    scan_stats['errors'] += 1
                    continue
        
        scan_stats['total_time'] = time.time() - scan_stats['start_time']
        scan_stats['signals'] = all_signals
        
        logger.info(f"✅ Signal scan completed: {scan_stats['signals_generated']} signals generated")
        return scan_stats
    
    def run_continuous_monitoring(self, interval_minutes: int = 5):
        """Run continuous signal monitoring"""
        logger.info(f"🔄 Starting continuous monitoring (interval: {interval_minutes} minutes)")
        
        while True:
            try:
                logger.info("🔍 Running signal scan...")
                scan_results = self.run_signal_scan()
                
                if scan_results['signals_generated'] > 0:
                    logger.info(f"🚨 {scan_results['signals_generated']} new signals detected!")
                else:
                    logger.info("📊 No signals detected this cycle")
                
                # Wait for next scan
                logger.info(f"⏰ Waiting {interval_minutes} minutes until next scan...")
                time.sleep(interval_minutes * 60)
                
            except KeyboardInterrupt:
                logger.info("⚠️ Monitoring stopped by user")
                break
            except Exception as e:
                logger.error(f"❌ Error in monitoring loop: {e}")
                time.sleep(60)  # Wait 1 minute before retry

def main():
    """Main function for running the signal generator"""
    import argparse
    
    parser = argparse.ArgumentParser(description='NvBot3 Signal Generator')
    parser.add_argument('--mode', choices=['scan', 'monitor'], default='scan',
                       help='Run mode: single scan or continuous monitoring')
    parser.add_argument('--symbol', type=str, help='Specific symbol to analyze')
    parser.add_argument('--timeframe', type=str, default='5m', help='Timeframe to use')
    parser.add_argument('--interval', type=int, default=5, 
                       help='Monitoring interval in minutes')
    
    args = parser.parse_args()
    
    # Initialize signal generator
    generator = NvBot3SignalGenerator()
    
    # Initialize exchange connection
    if not generator.initialize_exchange():
        logger.error("❌ Could not initialize exchange connection")
        return 1
    
    try:
        if args.mode == 'scan':
            if args.symbol:
                # Analyze single symbol
                signals = generator.generate_signals_for_symbol(args.symbol, args.timeframe)
                logger.info(f"📊 Generated {len(signals)} signals for {args.symbol}")
                for signal in signals:
                    print(f"🚨 {signal['symbol']} {signal['type']}: {signal['confidence']:.3f}")
            else:
                # Full scan
                results = generator.run_signal_scan()
                logger.info(f"📈 Scan completed: {results['signals_generated']} total signals")
        
        elif args.mode == 'monitor':
            # Continuous monitoring
            generator.run_continuous_monitoring(args.interval)
        
        return 0
        
    except Exception as e:
        logger.error(f"💥 Critical error: {e}")
        return 1

if __name__ == "__main__":
    print("🤖 === NVBOT3 SIGNAL GENERATOR ===")
    print("🎯 Real-time signal detection with dashboard integration")
    print("="*60)
    
    exit(main())