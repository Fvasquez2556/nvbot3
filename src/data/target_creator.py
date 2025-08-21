"""
🎯 Target Creator - NvBot3
Definidor de targets específicos para cada modelo de trading

Targets especializados:
🔥 Momentum Target: Detectar movimientos alcistas ≥5% en próximas 4 horas
⚡ Rebound Target: Predecir rebotes de 1-3% en próximas 2 horas  
📊 Regime Target: Clasificar tendencia (Bearish/Sideways/Bullish) en próximas 8 horas
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TargetCreator:
    """Creador de targets específicos para modelos de trading"""
    
    def __init__(self, data_path: str = "data/processed", output_path: str = "data/processed"):
        self.data_path = Path(data_path)
        self.output_path = Path(output_path)
        self.output_path.mkdir(exist_ok=True)
        
    def load_features_data(self, symbol: str, timeframe: str) -> Optional[pd.DataFrame]:
        """Cargar datos con features ya calculadas"""
        try:
            filename = f"{symbol}_{timeframe}_features.csv"
            filepath = self.data_path / filename
            
            if not filepath.exists():
                logger.error(f"❌ Archivo no encontrado: {filepath}")
                logger.info("💡 Asegúrate de haber ejecutado feature_calculator.py primero")
                return None
                
            df = pd.read_csv(filepath, index_col=0, parse_dates=True)
            logger.info(f"✅ Features cargadas: {len(df)} registros de {symbol}_{timeframe}")
            return df
            
        except Exception as e:
            logger.error(f"❌ Error cargando features: {e}")
            return None
    
    def create_momentum_target(self, df: pd.DataFrame, lookforward_periods: int = 48) -> pd.Series:
        """
        🔥 Crear target para Momentum Alcista (≥5%)
        
        Busca máximo precio en próximos períodos
        Target = 1 si (max_future_price / current_price) >= 1.05
        Target = 0 en caso contrario
        
        Args:
            df: DataFrame con datos OHLCV
            lookforward_periods: Períodos a mirar hacia adelante
                                - 48 para 5m = 4 horas
                                - 4 para 1h = 4 horas
        
        Returns:
            Serie con targets binarios (0/1)
        """
        logger.info(f"🔥 Creando Momentum Target (lookforward: {lookforward_periods} períodos)")
        
        close_prices = df['close'].copy()
        targets = pd.Series(0, index=df.index, name='momentum_target')
        
        try:
            for i in range(len(df) - lookforward_periods):
                current_price = close_prices.iloc[i]
                
                # Buscar máximo precio en próximos períodos
                future_prices = close_prices.iloc[i+1:i+1+lookforward_periods]
                max_future_price = future_prices.max()
                
                # Calcular retorno máximo
                max_return = (max_future_price / current_price) - 1
                
                # Target = 1 si retorno >= 5%
                if max_return >= 0.05:
                    targets.iloc[i] = 1
            
            # Estadísticas del target
            positive_ratio = targets.mean()
            total_signals = targets.sum()
            
            logger.info(f"✅ Momentum Target creado:")
            logger.info(f"   📊 Señales positivas: {total_signals:,} ({positive_ratio:.2%})")
            logger.info(f"   📊 Total registros válidos: {len(targets) - lookforward_periods:,}")
            
            return targets
            
        except Exception as e:
            logger.error(f"❌ Error creando momentum target: {e}")
            return pd.Series(0, index=df.index, name='momentum_target')
    
    def create_rebound_target(self, df: pd.DataFrame, lookforward_periods: int = 24) -> pd.Series:
        """
        ⚡ Crear target para Rebotes (1-3%)
        
        Busca rebote específico en rango 1-3% en próximos períodos
        Target = 1 si rebote está en rango 1%-3%
        Target = 0 si rebote < 1% o > 3%
        
        Args:
            df: DataFrame con datos OHLCV
            lookforward_periods: Períodos a mirar hacia adelante
                                - 24 para 5m = 2 horas
                                - 2 para 1h = 2 horas
        
        Returns:
            Serie con targets binarios (0/1)
        """
        logger.info(f"⚡ Creando Rebound Target (lookforward: {lookforward_periods} períodos)")
        
        close_prices = df['close'].copy()
        targets = pd.Series(0, index=df.index, name='rebound_target')
        
        try:
            for i in range(len(df) - lookforward_periods):
                current_price = close_prices.iloc[i]
                
                # Buscar máximo precio en próximos períodos
                future_prices = close_prices.iloc[i+1:i+1+lookforward_periods]
                max_future_price = future_prices.max()
                
                # Calcular retorno máximo
                max_return = (max_future_price / current_price) - 1
                
                # Target = 1 si rebote está en rango 1%-3%
                if 0.01 <= max_return <= 0.03:
                    targets.iloc[i] = 1
            
            # Estadísticas del target
            positive_ratio = targets.mean()
            total_signals = targets.sum()
            
            logger.info(f"✅ Rebound Target creado:")
            logger.info(f"   📊 Señales positivas: {total_signals:,} ({positive_ratio:.2%})")
            logger.info(f"   📊 Total registros válidos: {len(targets) - lookforward_periods:,}")
            
            return targets
            
        except Exception as e:
            logger.error(f"❌ Error creando rebound target: {e}")
            return pd.Series(0, index=df.index, name='rebound_target')
    
    def create_regime_target(self, df: pd.DataFrame, lookforward_periods: int = 96) -> pd.Series:
        """
        📊 Crear target para Régimen de Mercado
        
        Analiza tendencia en próximos períodos
        0 = Bearish (tendencia bajista)
        1 = Sideways (consolidación)  
        2 = Bullish (tendencia alcista)
        
        Args:
            df: DataFrame con datos OHLCV
            lookforward_periods: Períodos a mirar hacia adelante
                                - 96 para 5m = 8 horas
                                - 8 para 1h = 8 horas
        
        Returns:
            Serie con targets multiclass (0/1/2)
        """
        logger.info(f"📊 Creando Regime Target (lookforward: {lookforward_periods} períodos)")
        
        close_prices = df['close'].copy()
        targets = pd.Series(1, index=df.index, name='regime_target')  # Default: Sideways
        
        try:
            for i in range(len(df) - lookforward_periods):
                current_price = close_prices.iloc[i]
                
                # Precio al final del período
                future_price = close_prices.iloc[i + lookforward_periods]
                
                # Calcular retorno total
                total_return = (future_price / current_price) - 1
                
                # Calcular volatilidad del período
                future_prices = close_prices.iloc[i:i+lookforward_periods]
                volatility = future_prices.pct_change().std()
                
                # Clasificar régimen basado en retorno y volatilidad
                if total_return > 0.02:  # >2% alcista
                    targets.iloc[i] = 2  # Bullish
                elif total_return < -0.02:  # <-2% bajista
                    targets.iloc[i] = 0  # Bearish
                else:  # Entre -2% y +2%
                    targets.iloc[i] = 1  # Sideways
            
            # Estadísticas del target
            regime_counts = targets.value_counts().sort_index()
            total_valid = len(targets) - lookforward_periods
            
            logger.info(f"✅ Regime Target creado:")
            logger.info(f"   📊 Bearish (0): {regime_counts.get(0, 0):,} ({regime_counts.get(0, 0)/total_valid:.1%})")
            logger.info(f"   📊 Sideways (1): {regime_counts.get(1, 0):,} ({regime_counts.get(1, 0)/total_valid:.1%})")
            logger.info(f"   📊 Bullish (2): {regime_counts.get(2, 0):,} ({regime_counts.get(2, 0)/total_valid:.1%})")
            logger.info(f"   📊 Total registros válidos: {total_valid:,}")
            
            return targets
            
        except Exception as e:
            logger.error(f"❌ Error creando regime target: {e}")
            return pd.Series(1, index=df.index, name='regime_target')
    
    def create_advanced_momentum_target(self, df: pd.DataFrame, lookforward_periods: int = 48) -> pd.Series:
        """
        🔥 Target de momentum avanzado con filtros adicionales
        
        Requiere:
        1. Retorno >= 5%
        2. Volumen por encima del promedio durante el movimiento
        3. Movimiento sostenido (no solo spike)
        """
        logger.info(f"🔥 Creando Momentum Target Avanzado")
        
        close_prices = df['close'].copy()
        volume = df['volume'].copy()
        volume_ma = volume.rolling(20).mean()
        
        targets = pd.Series(0, index=df.index, name='momentum_advanced_target')
        
        try:
            for i in range(len(df) - lookforward_periods):
                current_price = close_prices.iloc[i]
                
                # Precios y volúmenes futuros
                future_prices = close_prices.iloc[i+1:i+1+lookforward_periods]
                future_volumes = volume.iloc[i+1:i+1+lookforward_periods]
                
                max_future_price = future_prices.max()
                max_return = (max_future_price / current_price) - 1
                
                # Filtro 1: Retorno >= 5%
                if max_return >= 0.05:
                    
                    # Filtro 2: Volumen promedio durante movimiento > promedio histórico
                    avg_volume_during_move = future_volumes.mean()
                    historical_avg_volume = volume_ma.iloc[i]
                    
                    if avg_volume_during_move > historical_avg_volume:
                        
                        # Filtro 3: Movimiento sostenido (al menos 50% del tiempo por encima de 2%)
                        sustained_periods = 0
                        for j in range(len(future_prices)):
                            partial_return = (future_prices.iloc[j] / current_price) - 1
                            if partial_return >= 0.02:
                                sustained_periods += 1
                        
                        if sustained_periods >= (lookforward_periods * 0.3):  # 30% del tiempo
                            targets.iloc[i] = 1
            
            positive_ratio = targets.mean()
            total_signals = targets.sum()
            
            logger.info(f"✅ Momentum Advanced Target creado:")
            logger.info(f"   📊 Señales positivas: {total_signals:,} ({positive_ratio:.2%})")
            
            return targets
            
        except Exception as e:
            logger.error(f"❌ Error creando momentum advanced target: {e}")
            return pd.Series(0, index=df.index, name='momentum_advanced_target')
    
    def create_all_targets(self, symbol: str, timeframe: str) -> Optional[pd.DataFrame]:
        """Crear todos los targets para un símbolo y timeframe"""
        logger.info(f"🎯 === CREANDO TODOS LOS TARGETS: {symbol}_{timeframe} ===")
        
        # Cargar datos con features
        df = self.load_features_data(symbol, timeframe)
        if df is None:
            return None
        
        # Determinar períodos según timeframe
        if '5m' in timeframe.lower():
            momentum_periods = 48  # 4 horas
            rebound_periods = 24   # 2 horas
            regime_periods = 96    # 8 horas
        elif '1h' in timeframe.lower():
            momentum_periods = 4   # 4 horas
            rebound_periods = 2    # 2 horas
            regime_periods = 8     # 8 horas
        elif '4h' in timeframe.lower():
            momentum_periods = 1   # 4 horas
            rebound_periods = 1    # 4 horas
            regime_periods = 2     # 8 horas
        else:
            # Default para otros timeframes
            momentum_periods = 48
            rebound_periods = 24
            regime_periods = 96
            logger.warning(f"⚠️ Timeframe '{timeframe}' no reconocido, usando períodos default")
        
        # Crear targets
        logger.info(f"🎯 Períodos configurados: Momentum={momentum_periods}, Rebound={rebound_periods}, Regime={regime_periods}")
        
        # 1. Target básico de momentum
        momentum_target = self.create_momentum_target(df, momentum_periods)
        df['momentum_target'] = momentum_target
        
        # 2. Target de rebotes
        rebound_target = self.create_rebound_target(df, rebound_periods)
        df['rebound_target'] = rebound_target
        
        # 3. Target de régimen
        regime_target = self.create_regime_target(df, regime_periods)
        df['regime_target'] = regime_target
        
        # 4. Target avanzado de momentum
        momentum_advanced = self.create_advanced_momentum_target(df, momentum_periods)
        df['momentum_advanced_target'] = momentum_advanced
        
        # Estadísticas finales
        self.print_target_summary(df, symbol, timeframe)
        
        return df
    
    def print_target_summary(self, df: pd.DataFrame, symbol: str, timeframe: str):
        """Imprimir resumen de targets creados"""
        print(f"\n{'='*70}")
        print(f"🎯 RESUMEN DE TARGETS: {symbol}_{timeframe}")
        print(f"{'='*70}")
        
        # Target básico de momentum
        momentum_signals = df['momentum_target'].sum()
        momentum_ratio = df['momentum_target'].mean()
        print(f"🔥 Momentum (≥5%):     {momentum_signals:,} señales ({momentum_ratio:.2%})")
        
        # Target de rebotes
        rebound_signals = df['rebound_target'].sum()
        rebound_ratio = df['rebound_target'].mean()
        print(f"⚡ Rebotes (1-3%):     {rebound_signals:,} señales ({rebound_ratio:.2%})")
        
        # Target de régimen
        regime_counts = df['regime_target'].value_counts().sort_index()
        print(f"📊 Régimen:")
        print(f"   • Bearish:          {regime_counts.get(0, 0):,} ({regime_counts.get(0, 0)/len(df):.1%})")
        print(f"   • Sideways:         {regime_counts.get(1, 0):,} ({regime_counts.get(1, 0)/len(df):.1%})")
        print(f"   • Bullish:          {regime_counts.get(2, 0):,} ({regime_counts.get(2, 0)/len(df):.1%})")
        
        # Target avanzado de momentum
        momentum_adv_signals = df['momentum_advanced_target'].sum()
        momentum_adv_ratio = df['momentum_advanced_target'].mean()
        print(f"🔥 Momentum Avanzado:  {momentum_adv_signals:,} señales ({momentum_adv_ratio:.2%})")
        
        print(f"\n📊 Total registros: {len(df):,}")
        print(f"📊 Features + Targets: {len(df.columns)} columnas")
        print(f"{'='*70}")
    
    def save_targets_data(self, df: pd.DataFrame, symbol: str, timeframe: str):
        """Guardar datos con features y targets"""
        try:
            output_file = self.output_path / f"{symbol}_{timeframe}_with_targets.csv"
            df.to_csv(output_file)
            logger.info(f"💾 Datos con targets guardados en: {output_file}")
            
            # Guardar resumen de targets
            summary_file = self.output_path / f"{symbol}_{timeframe}_targets_summary.txt"
            with open(summary_file, 'w', encoding='utf-8') as f:
                f.write(f"Targets creados para {symbol}_{timeframe}\n")
                f.write(f"Total registros: {len(df)}\n")
                f.write(f"Total columnas: {len(df.columns)}\n\n")
                
                # Estadísticas de targets
                f.write("Distribución de Targets:\n")
                f.write(f"Momentum: {df['momentum_target'].sum()} señales ({df['momentum_target'].mean():.2%})\n")
                f.write(f"Rebound: {df['rebound_target'].sum()} señales ({df['rebound_target'].mean():.2%})\n")
                
                regime_counts = df['regime_target'].value_counts().sort_index()
                f.write(f"Regime Bearish: {regime_counts.get(0, 0)} ({regime_counts.get(0, 0)/len(df):.1%})\n")
                f.write(f"Regime Sideways: {regime_counts.get(1, 0)} ({regime_counts.get(1, 0)/len(df):.1%})\n")
                f.write(f"Regime Bullish: {regime_counts.get(2, 0)} ({regime_counts.get(2, 0)/len(df):.1%})\n")
                
                f.write(f"Momentum Advanced: {df['momentum_advanced_target'].sum()} señales ({df['momentum_advanced_target'].mean():.2%})\n")
                
                # Lista de columnas de targets
                target_columns = [col for col in df.columns if 'target' in col]
                f.write(f"\nColumnas de targets ({len(target_columns)}):\n")
                for col in target_columns:
                    f.write(f"  - {col}\n")
            
            logger.info(f"📝 Resumen guardado en: {summary_file}")
            
        except Exception as e:
            logger.error(f"❌ Error guardando targets: {e}")
    
    def analyze_target_quality(self, df: pd.DataFrame, symbol: str, timeframe: str) -> Dict:
        """Analizar calidad y balance de targets"""
        logger.info(f"🔍 Analizando calidad de targets para {symbol}_{timeframe}")
        
        analysis = {
            'symbol': symbol,
            'timeframe': timeframe,
            'total_records': len(df),
            'targets': {}
        }
        
        target_columns = [col for col in df.columns if 'target' in col]
        
        for target_col in target_columns:
            target_data = df[target_col].dropna()
            
            if target_col == 'regime_target':
                # Análisis multiclass
                value_counts = target_data.value_counts().sort_index()
                analysis['targets'][target_col] = {
                    'type': 'multiclass',
                    'classes': len(value_counts),
                    'distribution': value_counts.to_dict(),
                    'balance_score': self._calculate_balance_score(value_counts),
                    'most_common_class': value_counts.idxmax(),
                    'most_common_ratio': value_counts.max() / len(target_data)
                }
            else:
                # Análisis binario
                positive_count = target_data.sum()
                total_count = len(target_data)
                positive_ratio = positive_count / total_count
                
                analysis['targets'][target_col] = {
                    'type': 'binary',
                    'positive_signals': int(positive_count),
                    'total_signals': int(total_count),
                    'positive_ratio': positive_ratio,
                    'balance_score': min(positive_ratio, 1 - positive_ratio) * 2,  # 0 a 1
                    'quality': self._assess_binary_quality(positive_ratio)
                }
        
        return analysis
    
    def _calculate_balance_score(self, value_counts: pd.Series) -> float:
        """Calcular score de balance para targets multiclass"""
        proportions = value_counts / value_counts.sum()
        # Score más alto cuando las clases están más balanceadas
        return 1 - proportions.var()
    
    def _assess_binary_quality(self, positive_ratio: float) -> str:
        """Evaluar calidad de target binario basado en ratio de positivos"""
        if 0.05 <= positive_ratio <= 0.3:
            return "EXCELENTE"
        elif 0.03 <= positive_ratio <= 0.4:
            return "BUENA"
        elif 0.01 <= positive_ratio <= 0.5:
            return "ACEPTABLE"
        else:
            return "PROBLEMÁTICA"
    
    def process_all_symbols(self) -> Dict:
        """Procesar todos los símbolos con features disponibles"""
        logger.info("🎯 === PROCESAMIENTO MASIVO DE TARGETS ===")
        
        results = {}
        feature_files = list(self.data_path.glob("*_features.csv"))
        
        if not feature_files:
            logger.error("❌ No se encontraron archivos de features")
            logger.info("💡 Ejecuta feature_calculator.py primero")
            return {}
        
        for feature_file in feature_files:
            try:
                # Extraer symbol y timeframe del nombre del archivo
                filename = feature_file.stem.replace('_features', '')
                parts = filename.split('_')
                
                if len(parts) >= 2:
                    symbol = parts[0]
                    timeframe = '_'.join(parts[1:])
                    
                    logger.info(f"🎯 Procesando {symbol}_{timeframe}")
                    
                    # Crear targets
                    df_with_targets = self.create_all_targets(symbol, timeframe)
                    
                    if df_with_targets is not None:
                        # Guardar datos con targets
                        self.save_targets_data(df_with_targets, symbol, timeframe)
                        
                        # Analizar calidad
                        quality_analysis = self.analyze_target_quality(df_with_targets, symbol, timeframe)
                        
                        results[f"{symbol}_{timeframe}"] = {
                            'success': True,
                            'total_columns': len(df_with_targets.columns),
                            'quality_analysis': quality_analysis
                        }
                    else:
                        results[f"{symbol}_{timeframe}"] = {
                            'success': False,
                            'error': 'No se pudieron crear targets'
                        }
                    
            except Exception as e:
                logger.error(f"❌ Error procesando {feature_file}: {e}")
                results[f"{feature_file.stem}"] = {
                    'success': False,
                    'error': str(e)
                }
        
        # Resumen general
        self.print_processing_summary(results)
        
        return results
    
    def print_processing_summary(self, results: Dict):
        """Imprimir resumen del procesamiento"""
        print(f"\n{'='*80}")
        print(f"📋 RESUMEN GENERAL DE TARGETS")
        print(f"{'='*80}")
        
        successful = sum(1 for r in results.values() if r.get('success', False))
        total = len(results)
        
        print(f"📊 Símbolos procesados: {successful}/{total}")
        
        if successful > 0:
            print(f"\n✅ PROCESADOS EXITOSAMENTE:")
            for symbol, result in results.items():
                if result.get('success', False):
                    cols = result['total_columns']
                    print(f"  • {symbol}: {cols} columnas totales")
        
        failed = [symbol for symbol, result in results.items() if not result.get('success', False)]
        if failed:
            print(f"\n❌ ERRORES EN:")
            for symbol in failed:
                error = results[symbol].get('error', 'Error desconocido')
                print(f"  • {symbol}: {error}")
        
        print(f"{'='*80}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Creador de Targets NvBot3')
    parser.add_argument('--symbol', type=str, help='Símbolo específico (ej: BTCUSDT)')
    parser.add_argument('--timeframe', type=str, help='Timeframe específico (ej: 5m)')
    parser.add_argument('--all-symbols', action='store_true', help='Procesar todos los símbolos con features')
    parser.add_argument('--save', action='store_true', help='Guardar targets calculados')
    parser.add_argument('--analyze-quality', action='store_true', help='Analizar calidad de targets')
    
    args = parser.parse_args()
    
    # Crear creador de targets
    creator = TargetCreator()
    
    if args.all_symbols:
        # Procesar todos los símbolos
        results = creator.process_all_symbols()
    elif args.symbol and args.timeframe:
        # Procesar símbolo específico
        df_with_targets = creator.create_all_targets(args.symbol, args.timeframe)
        
        if df_with_targets is not None:
            # Guardar si se especifica
            if args.save:
                creator.save_targets_data(df_with_targets, args.symbol, args.timeframe)
            
            # Analizar calidad si se especifica
            if args.analyze_quality:
                quality = creator.analyze_target_quality(df_with_targets, args.symbol, args.timeframe)
                print(f"\n🔍 ANÁLISIS DE CALIDAD:")
                for target, analysis in quality['targets'].items():
                    print(f"  {target}: {analysis}")
        else:
            print("❌ No se pudieron crear targets")
    else:
        print("❌ Especifica --symbol y --timeframe, o usa --all-symbols")
        print("Ejemplo: python target_creator.py --symbol BTCUSDT --timeframe 5m --save")
        print("Ejemplo: python target_creator.py --all-symbols")
