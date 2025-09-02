"""
🚀 NvBot3 Signal Generator - Real-Time Continuous Market Scanner
==============================================================

Sistema de monitoreo continuo que escanea el mercado en tiempo real 24/7.
Detecta señales de trading en múltiples timeframes simultáneamente.
Se integra con el dashboard web para visualización en tiempo real.

Características nuevas:
🔄 Monitoreo continuo sin intervención manual
📊 Escaneo de 30+ monedas entrenadas + 30 adicionales
⏱️ Múltiples timeframes: 3m, 5m, 15m, 1h, 4h
🎯 Detección automática de tendencias, subidas y rebotes
💰 Precios de referencia: Binance + precio óptimo calculado
📈 Actualización en tiempo real del dashboard

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
import threading
import queue
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
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

class NvBot3RealTimeScanner:
    """Sistema de escaneo en tiempo real mejorado con monitoreo continuo"""
    
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
        
        # Configuración de monitoreo continuo
        self.is_running = False
        self.scan_queue = queue.Queue()
        self.results_queue = queue.Queue()
        
        # Timeframes optimizados para detección
        self.active_timeframes = ['3m', '5m', '15m', '1h', '4h']
        
        # Cache de precios para optimización
        self.price_cache = {}
        self.last_cache_update = {}
        
        # Threading configuration
        self.max_workers = 8  # Número de threads para procesamiento paralelo
        
        # Configurar símbolos expandidos
        self._setup_expanded_symbols()
        
        # Model configurations with enhanced thresholds
        self.model_types = {
            'momentum': {
                'description': '🔥 Momentum Model: Detectar movimientos ≥5%',
                'threshold': 0.75,
                'target_change': 5.0,
                'category': 'momentum'
            },
            'rebound': {
                'description': '⚡ Rebound Model: Predecir rebotes 1-3%',
                'threshold': 0.70,
                'target_change': 2.0,
                'category': 'rebound'
            },
            'regime': {
                'description': '📊 Regime Model: Clasificar tendencia de mercado',
                'threshold': 0.60,
                'target_change': 0.0,
                'category': 'trend'
            },
            'momentum_advanced': {
                'description': '🎯 Advanced Momentum: Momentum con filtros',
                'threshold': 0.80,
                'target_change': 5.0,
                'category': 'momentum'
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
        logger.info("🤖 NvBot3 Real-Time Scanner initialized")
    
    def _setup_expanded_symbols(self):
        """Configurar lista expandida de símbolos (entrenados + adicionales)"""
        # Símbolos adicionales para monitoreo (30 nuevas monedas populares)
        additional_symbols = [
            # DeFi y Layer 1s populares
            'ATOMUSDT', 'ALGOUSDT', 'FTMUSDT', 'NEARUSDT', 'HBARUSDT',
            'FLOWUSDT', 'ICPUSDT', 'FILUSDT', 'VETUSDT', 'EGLDUSDT',
            
            # Gaming y NFTs
            'AXSUSDT', 'GALAUSDT', 'GMTUSDT', 'APECUSDT', 'LDOUSDT',
            
            # Meme coins y trending
            'DOGEUSDT', 'SHIBUSDT', 'PEPEUSDT', 'FLOKIUSDT', 'BONKUSDT',
            
            # Layer 2 y Scaling
            'OPUSDT', 'ARBUSDT', 'STRKUSDT', 'POLSUSDT', 'LRCUSDT',
            
            # Nuevos proyectos populares
            'APTUSDT', 'SUIUSDT', 'LDTUSDT', 'RNDRUSDT', 'INJUSDT'
        ]
        
        # Combinar símbolos entrenados + adicionales
        self.trained_symbols = self.symbols.copy()
        self.all_symbols = self.symbols + additional_symbols
        
        logger.info(f"📊 Símbolos configurados:")
        logger.info(f"   🎯 Entrenados: {len(self.trained_symbols)} símbolos")
        logger.info(f"   🔍 Adicionales: {len(additional_symbols)} símbolos")
        logger.info(f"   📈 Total: {len(self.all_symbols)} símbolos")
    
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
    
    def get_optimal_price_reference(self, symbol: str, market_data: pd.DataFrame) -> Dict[str, Any]:
        """Calcular precios de referencia: Binance actual + precio óptimo"""
        try:
            current_price = float(market_data['close'].iloc[-1])
            
            # Precio de Binance (actual)
            binance_price = current_price
            
            # Calcular precio óptimo basado en análisis técnico
            # Usar múltiples indicadores para determinar precio justo
            
            # 1. Media ponderada de diferentes MAs
            sma_20 = market_data['close'].rolling(20).mean().iloc[-1]
            ema_20 = market_data['close'].ewm(span=20).mean().iloc[-1]
            sma_50 = market_data['close'].rolling(50).mean().iloc[-1]
            
            # 2. Bollinger Bands para rango de precio
            bb_middle = sma_20
            bb_std = market_data['close'].rolling(20).std().iloc[-1]
            bb_upper = bb_middle + (bb_std * 2)
            bb_lower = bb_middle - (bb_std * 2)
            
            # 3. RSI para determinar sobre/subcompra
            delta = market_data['close'].diff()
            gain = (delta.where(delta.gt(0), 0)).rolling(14).mean().iloc[-1]
            loss = (-delta.where(delta.lt(0), 0)).rolling(14).mean().iloc[-1]
            rs = gain / loss if loss != 0 else float('inf')
            rsi = 100 - (100 / (1 + rs))
            
            # 4. Calcular precio óptimo ponderado
            weights = [0.3, 0.3, 0.25, 0.15]  # SMA20, EMA20, SMA50, BB_middle
            optimal_price = (
                sma_20 * weights[0] + 
                ema_20 * weights[1] + 
                sma_50 * weights[2] + 
                bb_middle * weights[3]
            )
            
            # 5. Ajuste por RSI (si está sobrecomprado/sobrevendido)
            if rsi > 70:  # Sobrecomprado
                optimal_price = optimal_price * 0.98  # 2% descuento
            elif rsi < 30:  # Sobrevendido
                optimal_price = optimal_price * 1.02  # 2% premium
            
            # 6. Calcular diferencia porcentual
            price_diff_pct = ((current_price - optimal_price) / optimal_price) * 100
            
            return {
                'binance_price': round(binance_price, 6),
                'optimal_price': round(optimal_price, 6),
                'price_difference_pct': round(price_diff_pct, 2),
                'bb_upper': round(bb_upper, 6),
                'bb_lower': round(bb_lower, 6),
                'rsi': round(rsi, 2),
                'price_analysis': {
                    'is_overvalued': price_diff_pct > 5,
                    'is_undervalued': price_diff_pct < -5,
                    'is_fair': abs(price_diff_pct) <= 5,
                    'recommendation': self._get_price_recommendation(price_diff_pct, rsi)
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Error calculating price reference for {symbol}: {e}")
            current_price = float(market_data['close'].iloc[-1])
            return {
                'binance_price': current_price,
                'optimal_price': current_price,
                'price_difference_pct': 0.0,
                'bb_upper': current_price,
                'bb_lower': current_price,
                'rsi': 50.0,
                'price_analysis': {
                    'is_overvalued': False,
                    'is_undervalued': False,
                    'is_fair': True,
                    'recommendation': 'hold'
                }
            }
    
    def _get_price_recommendation(self, price_diff_pct: float, rsi: float) -> str:
        """Generar recomendación basada en análisis de precio"""
        if price_diff_pct > 10 and rsi > 70:
            return 'strong_sell'
        elif price_diff_pct > 5 and rsi > 60:
            return 'sell'
        elif price_diff_pct < -10 and rsi < 30:
            return 'strong_buy'
        elif price_diff_pct < -5 and rsi < 40:
            return 'buy'
        else:
            return 'hold'
    
    def update_price_cache(self, symbol: str, price_data: Dict):
        """Actualizar cache de precios con timestamp"""
        self.price_cache[symbol] = {
            **price_data,
            'timestamp': datetime.now(),
            'last_update': time.time()
        }
        
        # Limpiar cache antiguo (mayor a 5 minutos)
        current_time = time.time()
        expired_symbols = [
            s for s, data in self.price_cache.items() 
            if current_time - data.get('last_update', 0) > 300
        ]
        
        for symbol in expired_symbols:
            del self.price_cache[symbol]
    
    def get_cached_price(self, symbol: str) -> Optional[Dict]:
        """Obtener precio del cache si está disponible y fresco"""
        if symbol in self.price_cache:
            data = self.price_cache[symbol]
            # Verificar si el cache tiene menos de 1 minuto
            if time.time() - data.get('last_update', 0) < 60:
                return data
        return None
    
    def start_continuous_monitoring(self, update_interval_seconds: int = 180):
        """Iniciar monitoreo continuo del mercado (cada 3 minutos por defecto)"""
        if self.is_running:
            logger.warning("⚠️ El monitoreo ya está ejecutándose")
            return
        
        logger.info("🚀 Iniciando monitoreo continuo del mercado...")
        logger.info(f"⏱️ Intervalo de actualización: {update_interval_seconds} segundos")
        logger.info(f"📊 Monitoreando {len(self.all_symbols)} símbolos en {len(self.active_timeframes)} timeframes")
        
        self.is_running = True
        
        # Inicializar exchange si no está conectado
        if not self.exchange:
            if not self.initialize_exchange():
                logger.error("❌ No se pudo conectar al exchange")
                return
        
        # Thread principal de monitoreo
        monitoring_thread = threading.Thread(
            target=self._continuous_scan_loop,
            args=(update_interval_seconds,),
            daemon=True
        )
        monitoring_thread.start()
        
        # Thread para procesar resultados
        results_thread = threading.Thread(
            target=self._process_results_loop,
            daemon=True
        )
        results_thread.start()
        
        logger.info("✅ Monitoreo continuo iniciado exitosamente")
        
        try:
            # Mantener el programa ejecutándose
            while self.is_running:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("⚠️ Deteniendo monitoreo por solicitud del usuario...")
            self.stop_monitoring()
    
    def stop_monitoring(self):
        """Detener el monitoreo continuo"""
        self.is_running = False
        logger.info("🛑 Monitoreo continuo detenido")
    
    def _continuous_scan_loop(self, interval_seconds: int):
        """Loop principal de escaneo continuo"""
        while self.is_running:
            try:
                start_time = time.time()
                logger.info("🔍 Iniciando ciclo de escaneo...")
                
                # Escanear todos los símbolos en paralelo
                self._parallel_symbol_scan()
                
                # Calcular tiempo transcurrido
                elapsed_time = time.time() - start_time
                logger.info(f"✅ Ciclo completado en {elapsed_time:.2f} segundos")
                
                # Esperar hasta el próximo ciclo
                remaining_time = max(0, interval_seconds - elapsed_time)
                if remaining_time > 0:
                    logger.info(f"⏰ Esperando {remaining_time:.0f}s hasta el próximo ciclo...")
                    time.sleep(remaining_time)
                else:
                    logger.warning("⚠️ El escaneo tomó más tiempo del esperado")
                    time.sleep(10)  # Pausa mínima
                    
            except Exception as e:
                logger.error(f"❌ Error en loop de monitoreo: {e}")
                time.sleep(30)  # Pausa de recuperación
    
    def _parallel_symbol_scan(self):
        """Escanear múltiples símbolos en paralelo para optimizar rendimiento"""
        symbols_to_scan = self.all_symbols.copy()
        
        # Priorizar símbolos entrenados
        trained_first = [s for s in symbols_to_scan if s in self.trained_symbols]
        additional_symbols = [s for s in symbols_to_scan if s not in self.trained_symbols]
        
        # Organizar por prioridad
        priority_symbols = trained_first + additional_symbols
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Crear tareas para cada combinación symbol-timeframe
            futures = {}
            
            for symbol in priority_symbols[:40]:  # Limitar a 40 símbolos para evitar límites de API
                for timeframe in self.active_timeframes:
                    future = executor.submit(self._scan_symbol_timeframe, symbol, timeframe)
                    futures[future] = (symbol, timeframe)
            
            # Procesar resultados conforme van completándose
            for future in as_completed(futures):
                symbol, timeframe = futures[future]
                try:
                    result = future.result(timeout=30)  # Timeout de 30 segundos
                    if result:
                        self.results_queue.put(result)
                except Exception as e:
                    logger.error(f"❌ Error escaneando {symbol} {timeframe}: {e}")
    
    def _scan_symbol_timeframe(self, symbol: str, timeframe: str) -> Optional[Dict]:
        """Escanear un símbolo específico en un timeframe específico"""
        try:
            # Verificar cache de precios primero
            cached_price = self.get_cached_price(symbol)
            if cached_price and timeframe == '5m':  # Usar cache solo para timeframe principal
                logger.debug(f"💾 Usando precio en cache para {symbol}")
                price_data = cached_price
            else:
                # Obtener datos de mercado
                market_data = self.get_market_data(symbol, timeframe, limit=100)
                if market_data is None:
                    return None
                
                # Calcular precios de referencia
                price_data = self.get_optimal_price_reference(symbol, market_data)
                
                # Actualizar cache
                self.update_price_cache(symbol, price_data)
            
            # Generar señales solo si tenemos datos válidos
            signals = self.generate_signals_for_symbol(symbol, timeframe)
            
            if signals:
                result = {
                    'symbol': symbol,
                    'timeframe': timeframe,
                    'signals': signals,
                    'price_data': price_data,
                    'scan_timestamp': datetime.now().isoformat(),
                    'is_trained_symbol': symbol in self.trained_symbols
                }
                
                return result
            
            return None
            
        except Exception as e:
            logger.error(f"❌ Error en _scan_symbol_timeframe {symbol} {timeframe}: {e}")
            return None
    
    def _process_results_loop(self):
        """Loop para procesar resultados de escaneo y enviar al dashboard"""
        while self.is_running:
            try:
                # Obtener resultado de la cola (con timeout)
                result = self.results_queue.get(timeout=1)
                
                # Procesar resultado
                self._process_scan_result(result)
                
                # Marcar tarea como completada
                self.results_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"❌ Error procesando resultados: {e}")
    
    def _process_scan_result(self, result: Dict):
        """Procesar un resultado individual de escaneo"""
        try:
            symbol = result['symbol']
            timeframe = result['timeframe']
            signals = result['signals']
            price_data = result['price_data']
            
            if signals:
                # Agrupar señales por categoría
                categorized_signals = self._categorize_signals(signals)
                
                # Log de señales detectadas
                for category, category_signals in categorized_signals.items():
                    if category_signals:
                        logger.info(f"🚨 {symbol} {timeframe} - {category.upper()}: {len(category_signals)} señales")
                
                # Actualizar precio en dashboard
                update_price(symbol, price_data['binance_price'])
                
                # Enviar señales categorizadas al dashboard
                self._send_categorized_signals_to_dashboard(symbol, categorized_signals, price_data)
            
        except Exception as e:
            logger.error(f"❌ Error procesando resultado: {e}")
    
    def _categorize_signals(self, signals: List[Dict]) -> Dict[str, List[Dict]]:
        """Categorizar señales por tipo: momentum, rebotes, tendencias"""
        categorized = {
            'momentum': [],
            'rebound': [],
            'trend': []
        }
        
        for signal in signals:
            signal_type = signal.get('type', '')
            category = self.model_types.get(signal_type, {}).get('category', 'trend')
            categorized[category].append(signal)
        
        return categorized
    
    def _send_categorized_signals_to_dashboard(self, symbol: str, categorized_signals: Dict, price_data: Dict):
        """Enviar señales categorizadas al dashboard web"""
        try:
            # Preparar datos para el dashboard
            dashboard_data = {
                'symbol': symbol,
                'price_data': price_data,
                'signals_by_category': categorized_signals,
                'timestamp': datetime.now().isoformat(),
                'total_signals': sum(len(signals) for signals in categorized_signals.values())
            }
            
            # Enviar cada señal individual para tracking
            for category, signals in categorized_signals.items():
                for signal in signals:
                    try:
                        # Track individual signal
                        signal_id = track_signal(symbol, signal, price_data['binance_price'])
                        logger.info(f"📊 Dashboard actualizado: {symbol} - {category} signal #{signal_id}")
                    except Exception as e:
                        logger.error(f"❌ Error enviando señal al dashboard: {e}")
            
        except Exception as e:
            logger.error(f"❌ Error enviando datos al dashboard: {e}")
    
    def run_signal_scan_new(self, symbols: Optional[List[str]] = None, timeframes: Optional[List[str]] = None) -> Dict:
        """Ejecutar un escaneo completo de señales (compatibilidad con versión anterior)"""
        if symbols is None:
            symbols = self.trained_symbols[:10]  # Limitar a primeros 10 símbolos entrenados
        if timeframes is None:
            timeframes = ['5m', '1h']  # Timeframes principales
        
        # Type checkers - ensuring symbols and timeframes are not None
        assert symbols is not None
        assert timeframes is not None
        
        logger.info(f"🚀 Iniciando escaneo de señales para {len(symbols)} símbolos x {len(timeframes)} timeframes")
        
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
                    logger.error(f"❌ Error escaneando {symbol} {timeframe}: {e}")
                    scan_stats['errors'] += 1
                    continue
        
        scan_stats['total_time'] = time.time() - scan_stats['start_time']
        scan_stats['signals'] = all_signals
        
        logger.info(f"✅ Escaneo completado: {scan_stats['signals_generated']} señales generadas")
        return scan_stats
    
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
    """Función principal mejorada para ejecutar el scanner en tiempo real"""
    import argparse
    
    parser = argparse.ArgumentParser(description='NvBot3 Real-Time Market Scanner')
    parser.add_argument('--mode', choices=['scan', 'monitor'], default='monitor',
                       help='Modo: escaneo único o monitoreo continuo (por defecto: monitor)')
    parser.add_argument('--symbol', type=str, help='Símbolo específico para analizar')
    parser.add_argument('--timeframe', type=str, default='5m', help='Timeframe a usar')
    parser.add_argument('--interval', type=int, default=180, 
                       help='Intervalo de monitoreo en segundos (por defecto: 180)')
    parser.add_argument('--max-symbols', type=int, default=40,
                       help='Máximo número de símbolos a escanear simultáneamente')
    
    args = parser.parse_args()
    
    # Inicializar scanner en tiempo real
    scanner = NvBot3RealTimeScanner()
    
    # Configurar límite de símbolos
    if args.max_symbols:
        scanner.all_symbols = scanner.all_symbols[:args.max_symbols]
    
    # Inicializar conexión al exchange
    if not scanner.initialize_exchange():
        logger.error("❌ No se pudo conectar al exchange")
        return 1
    
    try:
        if args.mode == 'scan':
            logger.info("🔍 Ejecutando escaneo único...")
            if args.symbol:
                # Analizar símbolo específico
                signals = scanner.generate_signals_for_symbol(args.symbol, args.timeframe)
                logger.info(f"📊 Generadas {len(signals)} señales para {args.symbol}")
                for signal in signals:
                    print(f"🚨 {signal['symbol']} {signal['type']}: {signal['confidence']:.3f}")
            else:
                # Escaneo completo una vez
                results = scanner.run_signal_scan_new()
                logger.info(f"📈 Escaneo completado: {results['signals_generated']} señales totales")
        
        elif args.mode == 'monitor':
            # Monitoreo continuo (modo principal)
            logger.info("🚀 Iniciando monitoreo continuo...")
            logger.info("💡 Presiona Ctrl+C para detener")
            scanner.start_continuous_monitoring(args.interval)
        
        return 0
        
    except Exception as e:
        logger.error(f"💥 Error crítico: {e}")
        return 1
    finally:
        if scanner.is_running:
            scanner.stop_monitoring()

if __name__ == "__main__":
    print("🤖 === NVBOT3 REAL-TIME MARKET SCANNER ===")
    print("🎯 Monitoreo continuo del mercado con detección automática")
    print("📊 Escaneo de 30+ monedas entrenadas + 30 adicionales")
    print("⏱️ Múltiples timeframes: 3m, 5m, 15m, 1h, 4h")
    print("🌐 Integración en tiempo real con dashboard web")
    print("="*70)
    
    exit(main())