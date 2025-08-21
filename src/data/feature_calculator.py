"""
🧮 Feature Calculator - NvBot3
Calculador de ~50 indicadores técnicos específicos para trading

Funcionalidades especializadas:
🔥 Para Momentum Alcista (≥5%): ROC, ADX, MACD, RSI, OBV, Bollinger Bands
⚡ Para Rebotes (1-3%): RSI oversold, MACD Histogram, Volume spikes, S/R levels
📊 Para Régimen de Mercado: ATR, BB Width, ADX, Moving Averages slopes
"""

import pandas as pd
import numpy as np
import talib
import logging
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def to_numpy(series: pd.Series) -> np.ndarray:
    """Convertir pandas Series a numpy array para TA-Lib"""
    return series.values.astype(np.float64)

class FeatureCalculator:
    """Calculador comprehensivo de features técnicas para trading"""
    
    def __init__(self, data_path: str = "data/raw", output_path: str = "data/processed"):
        self.data_path = Path(data_path)
        self.output_path = Path(output_path)
        self.output_path.mkdir(exist_ok=True)
        
        # Configuración de períodos para indicadores
        self.periods = {
            'short': [5, 7, 10, 14],
            'medium': [20, 26, 30],
            'long': [50, 100, 200]
        }
        
    def load_data(self, symbol: str, timeframe: str) -> Optional[pd.DataFrame]:
        """Cargar datos desde archivo CSV"""
        try:
            filename = f"{symbol}_{timeframe}.csv"
            filepath = self.data_path / filename
            
            if not filepath.exists():
                logger.error(f"❌ Archivo no encontrado: {filepath}")
                return None
                
            df = pd.read_csv(filepath)
            
            # Convertir timestamp a datetime si es necesario
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df.set_index('timestamp', inplace=True)
            elif 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)
            else:
                # Si no hay columna de fecha, usar índice como timestamp
                df.index = pd.to_datetime(df.index)
                
            # Verificar que tenemos las columnas OHLCV necesarias
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                logger.error(f"❌ Columnas faltantes: {missing_cols}")
                return None
                
            logger.info(f"✅ Datos cargados: {len(df)} registros de {symbol}_{timeframe}")
            return df
            
        except Exception as e:
            logger.error(f"❌ Error cargando datos: {e}")
            return None
    
    def calculate_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        🔥 Calcular features para Momentum Alcista (≥5%)
        
        Indicadores primarios:
        - ROC (Rate of Change)
        - ADX (Average Directional Index)
        - MACD (Moving Average Convergence Divergence)
        
        Indicadores secundarios:
        - RSI (Relative Strength Index)
        - OBV (On-Balance Volume)
        - Bollinger Bands
        """
        logger.info("🔥 Calculando features de Momentum Alcista")
        
        result_df = df.copy()
        
        try:
            # === ROC (Rate of Change) ===
            for period in [5, 10, 14, 20]:
                result_df[f'roc_{period}'] = talib.ROC(to_numpy(df['close']), timeperiod=period)
            
            # === ADX (Average Directional Index) ===
            for period in [14, 20, 30]:
                result_df[f'adx_{period}'] = talib.ADX(to_numpy(df['high']), to_numpy(df['low']), to_numpy(df['close']), timeperiod=period)
                result_df[f'plus_di_{period}'] = talib.PLUS_DI(to_numpy(df['high']), to_numpy(df['low']), to_numpy(df['close']), timeperiod=period)
                result_df[f'minus_di_{period}'] = talib.MINUS_DI(to_numpy(df['high']), to_numpy(df['low']), to_numpy(df['close']), timeperiod=period)
            
            # === MACD (Moving Average Convergence Divergence) ===
            # MACD estándar (12, 26, 9)
            macd, macdsignal, macdhist = talib.MACD(to_numpy(df['close']), fastperiod=12, slowperiod=26, signalperiod=9)
            result_df['macd'] = macd
            result_df['macd_signal'] = macdsignal
            result_df['macd_histogram'] = macdhist
            
            # MACD variaciones
            macd_fast, macdsignal_fast, macdhist_fast = talib.MACD(to_numpy(df['close']), fastperiod=8, slowperiod=21, signalperiod=5)
            result_df['macd_fast'] = macd_fast
            result_df['macd_fast_signal'] = macdsignal_fast
            result_df['macd_fast_histogram'] = macdhist_fast
            
            # === RSI (Relative Strength Index) ===
            for period in [7, 14, 21, 30]:
                result_df[f'rsi_{period}'] = talib.RSI(to_numpy(df['close']), timeperiod=period)
            
            # === OBV (On-Balance Volume) ===
            result_df['obv'] = talib.OBV(to_numpy(df['close']), to_numpy(df['volume']))
            
            # OBV normalizado
            result_df['obv_sma_20'] = talib.SMA(to_numpy(result_df['obv']), timeperiod=20)
            result_df['obv_ratio'] = result_df['obv'] / result_df['obv_sma_20']
            
            # === Bollinger Bands ===
            for period in [20, 30]:
                upper, middle, lower = talib.BBANDS(to_numpy(df['close']), timeperiod=period, nbdevup=2, nbdevdn=2)
                result_df[f'bb_upper_{period}'] = upper
                result_df[f'bb_middle_{period}'] = middle
                result_df[f'bb_lower_{period}'] = lower
                result_df[f'bb_width_{period}'] = (upper - lower) / middle
                result_df[f'bb_position_{period}'] = (df['close'] - lower) / (upper - lower)
            
            # === Features derivadas para momentum ===
            # Momentum combinado
            result_df['momentum_score'] = (
                (result_df['rsi_14'] > 50).astype(int) +
                (result_df['adx_14'] > 25).astype(int) +
                (result_df['macd'] > result_df['macd_signal']).astype(int) +
                (result_df['roc_14'] > 0).astype(int)
            ) / 4
            
            # Fuerza del momentum
            result_df['momentum_strength'] = (
                result_df['roc_14'] * 0.3 +
                (result_df['rsi_14'] - 50) / 50 * 0.3 +
                result_df['adx_14'] / 100 * 0.4
            )
            
            logger.info("✅ Features de Momentum calculadas")
            
        except Exception as e:
            logger.error(f"❌ Error calculando features de momentum: {e}")
            
        return result_df
    
    def calculate_rebound_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        ⚡ Calcular features para Rebotes (1-3%)
        
        - RSI para condiciones oversold
        - MACD Histogram para divergencias
        - Volume patterns (sudden spikes)
        - Support/Resistance levels
        """
        logger.info("⚡ Calculando features de Rebotes")
        
        result_df = df.copy()
        
        try:
            # === RSI Oversold Conditions ===
            result_df['rsi_oversold'] = (result_df['rsi_14'] < 30).astype(int)
            result_df['rsi_oversold_extreme'] = (result_df['rsi_14'] < 20).astype(int)
            
            # RSI divergencia (precio baja, RSI sube)
            result_df['price_change_5'] = result_df['close'].pct_change(5)
            result_df['rsi_change_5'] = result_df['rsi_14'].diff(5)
            result_df['rsi_divergence'] = (
                (result_df['price_change_5'] < 0) & 
                (result_df['rsi_change_5'] > 0)
            ).astype(int)
            
            # === MACD Histogram Divergencias ===
            result_df['macd_hist_increasing'] = (result_df['macd_histogram'] > result_df['macd_histogram'].shift(1)).astype(int)
            result_df['macd_hist_divergence'] = (
                (result_df['price_change_5'] < 0) & 
                (result_df['macd_histogram'] > result_df['macd_histogram'].shift(5))
            ).astype(int)
            
            # === Volume Patterns ===
            result_df['volume_sma_20'] = talib.SMA(to_numpy(df['volume']), timeperiod=20)
            result_df['volume_ratio'] = df['volume'] / result_df['volume_sma_20']
            result_df['volume_spike'] = (result_df['volume_ratio'] > 2.0).astype(int)
            result_df['volume_spike_extreme'] = (result_df['volume_ratio'] > 3.0).astype(int)
            
            # Volume durante declive (buying the dip)
            result_df['volume_on_decline'] = (
                (result_df['close'] < result_df['close'].shift(1)) & 
                (result_df['volume_ratio'] > 1.5)
            ).astype(int)
            
            # === Support/Resistance Levels ===
            # Soporte dinámico (mínimos locales)
            result_df['local_min_5'] = (
                (df['low'] <= df['low'].rolling(5, center=True).min()) & 
                (df['low'] <= df['low'].rolling(5, center=True).min())
            ).astype(int)
            
            result_df['local_min_10'] = (
                (df['low'] <= df['low'].rolling(10, center=True).min()) & 
                (df['low'] <= df['low'].rolling(10, center=True).min())
            ).astype(int)
            
            # Distancia a soportes recientes
            for period in [10, 20, 50]:
                result_df[f'min_low_{period}'] = df['low'].rolling(period).min()
                result_df[f'distance_to_support_{period}'] = (df['close'] - result_df[f'min_low_{period}']) / result_df[f'min_low_{period}']
            
            # === Williams %R (otra medida de oversold) ===
            for period in [14, 20]:
                result_df[f'williams_r_{period}'] = talib.WILLR(to_numpy(df['high']), to_numpy(df['low']), to_numpy(df['close']), timeperiod=period)
                result_df[f'williams_oversold_{period}'] = (result_df[f'williams_r_{period}'] < -80).astype(int)
            
            # === Features combinadas para rebotes ===
            # Score de oversold
            result_df['oversold_score'] = (
                result_df['rsi_oversold'] +
                result_df['williams_oversold_14'] +
                (result_df['bb_position_20'] < 0.2).astype(int)
            ) / 3
            
            # Score de rebote potencial
            result_df['rebound_potential'] = (
                result_df['oversold_score'] * 0.4 +
                result_df['rsi_divergence'] * 0.2 +
                result_df['macd_hist_divergence'] * 0.2 +
                result_df['volume_on_decline'] * 0.2
            )
            
            logger.info("✅ Features de Rebotes calculadas")
            
        except Exception as e:
            logger.error(f"❌ Error calculando features de rebotes: {e}")
            
        return result_df
    
    def calculate_regime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        📊 Calcular features para Régimen de Mercado
        
        - ATR (Average True Range)
        - Bollinger Bands Width
        - ADX para trending vs sideways
        - Moving Average slopes
        """
        logger.info("📊 Calculando features de Régimen de Mercado")
        
        result_df = df.copy()
        
        try:
            # === ATR (Average True Range) ===
            for period in [14, 20, 30]:
                result_df[f'atr_{period}'] = talib.ATR(to_numpy(df['high']), to_numpy(df['low']), to_numpy(df['close']), timeperiod=period)
                result_df[f'atr_ratio_{period}'] = result_df[f'atr_{period}'] / df['close']
            
            # === Moving Averages y Slopes ===
            for period in [10, 20, 50, 100, 200]:
                result_df[f'sma_{period}'] = talib.SMA(to_numpy(df['close']), timeperiod=period)
                result_df[f'ema_{period}'] = talib.EMA(to_numpy(df['close']), timeperiod=period)
                
                # Slope (pendiente) de las medias móviles
                result_df[f'sma_{period}_slope'] = result_df[f'sma_{period}'].diff(5) / result_df[f'sma_{period}']
                result_df[f'ema_{period}_slope'] = result_df[f'ema_{period}'].diff(5) / result_df[f'ema_{period}']
            
            # === Relaciones entre medias móviles ===
            # Price position relative to MAs
            for period in [20, 50, 200]:
                result_df[f'price_above_sma_{period}'] = (df['close'] > result_df[f'sma_{period}']).astype(int)
                result_df[f'price_distance_sma_{period}'] = (df['close'] - result_df[f'sma_{period}']) / result_df[f'sma_{period}']
            
            # MA alignment (bullish cuando MA corta > MA larga)
            result_df['ma_alignment_bull'] = (
                (result_df['sma_20'] > result_df['sma_50']) & 
                (result_df['sma_50'] > result_df['sma_200'])
            ).astype(int)
            
            result_df['ma_alignment_bear'] = (
                (result_df['sma_20'] < result_df['sma_50']) & 
                (result_df['sma_50'] < result_df['sma_200'])
            ).astype(int)
            
            # === Volatilidad y Rangos ===
            # Bollinger Bands Width (ya calculado en momentum, pero añadimos más períodos)
            for period in [10, 20, 50]:
                upper, middle, lower = talib.BBANDS(to_numpy(df['close']), timeperiod=period, nbdevup=2, nbdevdn=2)
                result_df[f'bb_width_{period}'] = (upper - lower) / middle
            
            # True Range normalizado
            result_df['true_range'] = talib.TRANGE(to_numpy(df['high']), to_numpy(df['low']), to_numpy(df['close']))
            result_df['true_range_norm'] = result_df['true_range'] / df['close']
            
            # === Market Regime Classification ===
            # ADX para trending vs consolidación
            result_df['regime_trending'] = (result_df['adx_14'] > 25).astype(int)
            result_df['regime_strong_trend'] = (result_df['adx_14'] > 40).astype(int)
            result_df['regime_consolidation'] = (result_df['adx_14'] < 20).astype(int)
            
            # Volatilidad regime
            result_df['regime_low_vol'] = (result_df['bb_width_20'] < 0.04).astype(int)  # Consolidación
            result_df['regime_high_vol'] = (result_df['bb_width_20'] > 0.08).astype(int)  # Alta volatilidad
            
            # === Trend Direction ===
            # Trend dirección basada en pendientes de MA
            result_df['trend_bullish'] = (
                (result_df['sma_20_slope'] > 0) & 
                (result_df['sma_50_slope'] > 0) & 
                (result_df['adx_14'] > 20)
            ).astype(int)
            
            result_df['trend_bearish'] = (
                (result_df['sma_20_slope'] < 0) & 
                (result_df['sma_50_slope'] < 0) & 
                (result_df['adx_14'] > 20)
            ).astype(int)
            
            # === Score compuesto de régimen ===
            # Regime Score: -1 (Bear) to +1 (Bull)
            result_df['regime_score'] = (
                result_df['ma_alignment_bull'] * 0.3 +
                result_df['trend_bullish'] * 0.3 +
                result_df['price_above_sma_20'] * 0.2 +
                result_df['price_above_sma_50'] * 0.2 -
                result_df['ma_alignment_bear'] * 0.3 -
                result_df['trend_bearish'] * 0.3
            )
            
            logger.info("✅ Features de Régimen de Mercado calculadas")
            
        except Exception as e:
            logger.error(f"❌ Error calculando features de régimen: {e}")
            
        return result_df
    
    def calculate_additional_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calcular features adicionales y técnicas avanzadas"""
        logger.info("🔧 Calculando features adicionales")
        
        result_df = df.copy()
        
        try:
            # === Patrones de Velas ===
            # Doji
            result_df['doji'] = talib.CDLDOJI(to_numpy(df['open']), to_numpy(df['high']), to_numpy(df['low']), to_numpy(df['close']))
            
            # Hammer
            result_df['hammer'] = talib.CDLHAMMER(to_numpy(df['open']), to_numpy(df['high']), to_numpy(df['low']), to_numpy(df['close']))
            
            # Engulfing
            result_df['engulfing_bull'] = talib.CDLENGULFING(to_numpy(df['open']), to_numpy(df['high']), to_numpy(df['low']), to_numpy(df['close']))
            
            # === Stochastic ===
            for period in [14, 20]:
                slowk, slowd = talib.STOCH(to_numpy(df['high']), to_numpy(df['low']), to_numpy(df['close']), 
                                         fastk_period=period, slowk_period=3, slowd_period=3)
                result_df[f'stoch_k_{period}'] = slowk
                result_df[f'stoch_d_{period}'] = slowd
                result_df[f'stoch_oversold_{period}'] = (slowk < 20).astype(int)
                result_df[f'stoch_overbought_{period}'] = (slowk > 80).astype(int)
            
            # === CCI (Commodity Channel Index) ===
            for period in [14, 20]:
                result_df[f'cci_{period}'] = talib.CCI(to_numpy(df['high']), to_numpy(df['low']), to_numpy(df['close']), timeperiod=period)
                result_df[f'cci_oversold_{period}'] = (result_df[f'cci_{period}'] < -100).astype(int)
                result_df[f'cci_overbought_{period}'] = (result_df[f'cci_{period}'] > 100).astype(int)
            
            # === Money Flow Index ===
            result_df['mfi_14'] = talib.MFI(to_numpy(df['high']), to_numpy(df['low']), to_numpy(df['close']), to_numpy(df['volume']), timeperiod=14)
            result_df['mfi_oversold'] = (result_df['mfi_14'] < 20).astype(int)
            result_df['mfi_overbought'] = (result_df['mfi_14'] > 80).astype(int)
            
            # === Price Action Features ===
            # Rangos y cuerpos de velas
            result_df['candle_range'] = (df['high'] - df['low']) / df['close']
            result_df['candle_body'] = abs(df['close'] - df['open']) / df['close']
            result_df['upper_shadow'] = (df['high'] - np.maximum(df['open'], df['close'])) / df['close']
            result_df['lower_shadow'] = (np.minimum(df['open'], df['close']) - df['low']) / df['close']
            
            # === Volume Analysis ===
            # Volume Price Trend
            result_df['vpt'] = talib.TRIX(to_numpy(df['close'] * df['volume']), timeperiod=14)
            
            # Accumulation/Distribution Line
            result_df['ad'] = talib.AD(to_numpy(df['high']), to_numpy(df['low']), to_numpy(df['close']), to_numpy(df['volume']))
            
            # === Features de tiempo ===
            # Hora del día (importante para crypto)
            try:
                if isinstance(result_df.index, pd.DatetimeIndex):
                    result_df['hour'] = result_df.index.hour
                    result_df['day_of_week'] = result_df.index.dayofweek
                    
                    # Sesiones de trading
                    result_df['asian_session'] = ((result_df['hour'] >= 0) & (result_df['hour'] < 8)).astype(int)
                    result_df['european_session'] = ((result_df['hour'] >= 8) & (result_df['hour'] < 16)).astype(int)
                    result_df['american_session'] = ((result_df['hour'] >= 16) & (result_df['hour'] < 24)).astype(int)
                else:
                    logger.warning("⚠️ Índice no es DatetimeIndex, omitiendo features temporales")
            except Exception as e:
                logger.warning(f"⚠️ Error calculando features temporales: {e}")
            
            logger.info("✅ Features adicionales calculadas")
            
        except Exception as e:
            logger.error(f"❌ Error calculando features adicionales: {e}")
            
        return result_df
    
    def calculate_all_features(self, symbol: str, timeframe: str) -> Optional[pd.DataFrame]:
        """Calcular todos los features para un símbolo y timeframe"""
        logger.info(f"🧮 === CALCULANDO TODAS LAS FEATURES: {symbol}_{timeframe} ===")
        
        # Cargar datos
        df = self.load_data(symbol, timeframe)
        if df is None:
            return None
        
        # Verificar que tenemos suficientes datos (necesitamos al menos 200 para MA200)
        if len(df) < 250:
            logger.warning(f"⚠️ Pocos datos para cálculos robustos: {len(df)} registros")
        
        # Calcular features por categorías
        df_with_features = df.copy()
        
        # 1. Features de Momentum
        df_with_features = self.calculate_momentum_features(df_with_features)
        
        # 2. Features de Rebotes  
        df_with_features = self.calculate_rebound_features(df_with_features)
        
        # 3. Features de Régimen
        df_with_features = self.calculate_regime_features(df_with_features)
        
        # 4. Features adicionales
        df_with_features = self.calculate_additional_features(df_with_features)
        
        # Limpieza final: eliminar inf y NaN
        df_with_features.replace([np.inf, -np.inf], np.nan, inplace=True)
        
        # Contar features añadidas
        original_cols = len(df.columns)
        final_cols = len(df_with_features.columns)
        features_added = final_cols - original_cols
        
        logger.info(f"✅ FEATURES COMPLETADAS: {features_added} indicadores añadidos")
        logger.info(f"📊 Total columnas: {original_cols} → {final_cols}")
        
        return df_with_features
    
    def save_features(self, df: pd.DataFrame, symbol: str, timeframe: str):
        """Guardar features calculadas"""
        try:
            output_file = self.output_path / f"{symbol}_{timeframe}_features.csv"
            df.to_csv(output_file)
            logger.info(f"💾 Features guardadas en: {output_file}")
            
            # Guardar también info de las features
            info_file = self.output_path / f"{symbol}_{timeframe}_features_info.txt"
            with open(info_file, 'w', encoding='utf-8') as f:
                f.write(f"Features calculadas para {symbol}_{timeframe}\n")
                f.write(f"Total registros: {len(df)}\n")
                f.write(f"Total features: {len(df.columns)}\n")
                f.write(f"Rango de fechas: {df.index.min()} - {df.index.max()}\n\n")
                f.write("Columnas:\n")
                for col in df.columns:
                    f.write(f"  - {col}\n")
            
            logger.info(f"📝 Info guardada en: {info_file}")
            
        except Exception as e:
            logger.error(f"❌ Error guardando features: {e}")
    
    def get_feature_summary(self, df: pd.DataFrame) -> Dict:
        """Obtener resumen de features calculadas"""
        try:
            # Categorizar features
            momentum_features = [col for col in df.columns if any(x in col for x in ['roc', 'adx', 'macd', 'momentum'])]
            rebound_features = [col for col in df.columns if any(x in col for x in ['oversold', 'divergence', 'rebound', 'williams'])]
            regime_features = [col for col in df.columns if any(x in col for x in ['regime', 'trend', 'slope', 'alignment'])]
            volume_features = [col for col in df.columns if any(x in col for x in ['volume', 'obv', 'vpt', 'mfi'])]
            price_features = [col for col in df.columns if any(x in col for x in ['bb', 'sma', 'ema', 'atr', 'rsi'])]
            pattern_features = [col for col in df.columns if any(x in col for x in ['doji', 'hammer', 'engulfing', 'stoch', 'cci'])]
            
            summary = {
                'total_features': len(df.columns),
                'categories': {
                    'momentum': len(momentum_features),
                    'rebound': len(rebound_features),
                    'regime': len(regime_features),
                    'volume': len(volume_features),
                    'price': len(price_features),
                    'patterns': len(pattern_features)
                },
                'data_quality': {
                    'total_rows': len(df),
                    'missing_values': df.isnull().sum().sum(),
                    'missing_percentage': (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
                }
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"❌ Error generando resumen: {e}")
            return {}
    
    def print_feature_summary(self, summary: Dict, symbol: str, timeframe: str):
        """Imprimir resumen de features"""
        print(f"\n{'='*60}")
        print(f"🧮 RESUMEN DE FEATURES: {symbol}_{timeframe}")
        print(f"{'='*60}")
        
        print(f"📊 Total features calculadas: {summary['total_features']}")
        print(f"📈 Total registros: {summary['data_quality']['total_rows']:,}")
        print(f"🎯 Valores faltantes: {summary['data_quality']['missing_percentage']:.2f}%")
        
        print(f"\n📋 FEATURES POR CATEGORÍA:")
        categories = summary['categories']
        print(f"  🔥 Momentum:   {categories['momentum']} features")
        print(f"  ⚡ Rebotes:    {categories['rebound']} features")
        print(f"  📊 Régimen:    {categories['regime']} features")
        print(f"  📊 Volumen:    {categories['volume']} features")
        print(f"  💰 Precios:    {categories['price']} features")
        print(f"  🕯️  Patrones:   {categories['patterns']} features")
        
        print(f"{'='*60}")
    
    def process_all_symbols(self) -> Dict:
        """Procesar todos los símbolos disponibles"""
        logger.info("🧮 === PROCESAMIENTO MASIVO DE FEATURES ===")
        
        results = {}
        csv_files = list(self.data_path.glob("*.csv"))
        
        if not csv_files:
            logger.error("❌ No se encontraron archivos CSV en data/raw")
            return {}
        
        for csv_file in csv_files:
            try:
                # Extraer symbol y timeframe del nombre del archivo
                filename = csv_file.stem
                parts = filename.split('_')
                
                if len(parts) >= 2:
                    symbol = parts[0]
                    timeframe = '_'.join(parts[1:])
                    
                    logger.info(f"🧮 Procesando {symbol}_{timeframe}")
                    
                    # Calcular features
                    df_features = self.calculate_all_features(symbol, timeframe)
                    
                    if df_features is not None:
                        # Guardar features
                        self.save_features(df_features, symbol, timeframe)
                        
                        # Generar resumen
                        summary = self.get_feature_summary(df_features)
                        self.print_feature_summary(summary, symbol, timeframe)
                        
                        results[f"{symbol}_{timeframe}"] = {
                            'success': True,
                            'features_count': len(df_features.columns),
                            'summary': summary
                        }
                    else:
                        results[f"{symbol}_{timeframe}"] = {
                            'success': False,
                            'error': 'No se pudieron calcular features'
                        }
                    
            except Exception as e:
                logger.error(f"❌ Error procesando {csv_file}: {e}")
                results[f"{csv_file.stem}"] = {
                    'success': False,
                    'error': str(e)
                }
        
        # Resumen general
        self.print_processing_summary(results)
        
        return results
    
    def print_processing_summary(self, results: Dict):
        """Imprimir resumen del procesamiento"""
        print(f"\n{'='*80}")
        print(f"📋 RESUMEN GENERAL DE PROCESAMIENTO")
        print(f"{'='*80}")
        
        successful = sum(1 for r in results.values() if r.get('success', False))
        total = len(results)
        
        print(f"📊 Símbolos procesados: {successful}/{total}")
        
        if successful > 0:
            avg_features = np.mean([r['features_count'] for r in results.values() if r.get('success', False)])
            print(f"🧮 Promedio features por símbolo: {avg_features:.0f}")
            
            print(f"\n✅ PROCESADOS EXITOSAMENTE:")
            for symbol, result in results.items():
                if result.get('success', False):
                    features_count = result['features_count']
                    print(f"  • {symbol}: {features_count} features")
        
        failed = [symbol for symbol, result in results.items() if not result.get('success', False)]
        if failed:
            print(f"\n❌ ERRORES EN:")
            for symbol in failed:
                error = results[symbol].get('error', 'Error desconocido')
                print(f"  • {symbol}: {error}")
        
        print(f"{'='*80}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Calculador de Features NvBot3')
    parser.add_argument('--symbol', type=str, help='Símbolo específico (ej: BTCUSDT)')
    parser.add_argument('--timeframe', type=str, help='Timeframe específico (ej: 5m)')
    parser.add_argument('--all-symbols', action='store_true', help='Procesar todos los símbolos')
    parser.add_argument('--save', action='store_true', help='Guardar features calculadas')
    
    args = parser.parse_args()
    
    # Crear calculador
    calculator = FeatureCalculator()
    
    if args.all_symbols:
        # Procesar todos los símbolos
        results = calculator.process_all_symbols()
    elif args.symbol and args.timeframe:
        # Procesar símbolo específico
        df_features = calculator.calculate_all_features(args.symbol, args.timeframe)
        
        if df_features is not None:
            # Mostrar resumen
            summary = calculator.get_feature_summary(df_features)
            calculator.print_feature_summary(summary, args.symbol, args.timeframe)
            
            # Guardar si se especifica
            if args.save:
                calculator.save_features(df_features, args.symbol, args.timeframe)
        else:
            print("❌ No se pudieron calcular features")
    else:
        print("❌ Especifica --symbol y --timeframe, o usa --all-symbols")
        print("Ejemplo: python feature_calculator.py --symbol BTCUSDT --timeframe 5m --save")
        print("Ejemplo: python feature_calculator.py --all-symbols")
