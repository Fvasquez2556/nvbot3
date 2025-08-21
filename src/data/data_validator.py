"""
🔍 Data Validator - NvBot3
Validador exhaustivo de calidad de datos para entrenamiento confiable

Funcionalidades:
- Validar completeness (sin gaps > 2 períodos)
- Verificar price validity (no cambios > 50%)
- Comprobar volume validity (> 0 en 95%+ registros)
- Detectar outliers extremos
- Verificar consistencia temporal
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataValidator:
    """Validador comprehensivo de datos para trading"""
    
    def __init__(self, data_path: str = "data/raw"):
        self.data_path = Path(data_path)
        self.validation_results = {}
        
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
                
            logger.info(f"✅ Datos cargados: {len(df)} registros de {symbol}_{timeframe}")
            return df
            
        except Exception as e:
            logger.error(f"❌ Error cargando datos: {e}")
            return None
    
    def validate_completeness(self, df: pd.DataFrame, symbol: str, timeframe: str) -> Dict:
        """Validar completeness - detectar gaps en datos"""
        logger.info(f"🔍 Validando completeness para {symbol}_{timeframe}")
        
        results = {
            'total_records': len(df),
            'date_range': (df.index.min(), df.index.max()),
            'gaps_detected': [],
            'max_gap_hours': 0,
            'completeness_score': 0.0
        }
        
        try:
            # Detectar gaps en el índice temporal
            if len(df) > 1:
                # Calcular diferencias esperadas según timeframe
                if '5m' in timeframe:
                    expected_freq = pd.Timedelta(minutes=5)
                elif '1h' in timeframe:
                    expected_freq = pd.Timedelta(hours=1)
                elif '4h' in timeframe:
                    expected_freq = pd.Timedelta(hours=4)
                elif '1d' in timeframe:
                    expected_freq = pd.Timedelta(days=1)
                else:
                    expected_freq = pd.Timedelta(minutes=5)  # Default
                
                # Encontrar gaps
                time_diffs = df.index.to_series().diff()[1:]
                gaps = time_diffs[time_diffs > expected_freq * 2]  # Gaps > 2 períodos
                
                if len(gaps) > 0:
                    gap_data = []
                    for gap_time in gaps.index:
                        gap_duration = gaps.loc[gap_time]
                        if pd.notna(gap_duration):
                            duration_hours = gap_duration.total_seconds() / 3600
                            missing_periods = int(gap_duration / expected_freq)
                            gap_data.append({
                                'start': str(gap_time),
                                'duration_hours': duration_hours,
                                'missing_periods': missing_periods
                            })
                    results['gaps_detected'] = gap_data
                    results['max_gap_hours'] = max(gap['duration_hours'] for gap in results['gaps_detected'])
                
                # Calcular score de completeness
                total_expected_periods = (df.index.max() - df.index.min()) / expected_freq
                actual_periods = len(df)
                results['completeness_score'] = min(1.0, actual_periods / total_expected_periods)
                
        except Exception as e:
            logger.error(f"❌ Error validando completeness: {e}")
            
        return results
    
    def validate_price_validity(self, df: pd.DataFrame, symbol: str) -> Dict:
        """Validar validez de precios - detectar cambios extremos"""
        logger.info(f"💰 Validando validez de precios para {symbol}")
        
        results = {
            'extreme_changes': [],
            'negative_prices': 0,
            'zero_prices': 0,
            'max_change_percent': 0.0,
            'price_validity_score': 0.0
        }
        
        try:
            # Verificar precios negativos o cero
            for col in ['open', 'high', 'low', 'close']:
                if col in df.columns:
                    negative_count = (df[col] <= 0).sum()
                    if negative_count > 0:
                        results['negative_prices'] += negative_count
                        logger.warning(f"⚠️ {negative_count} precios negativos/cero en {col}")
            
            # Detectar cambios extremos (>50% entre períodos consecutivos)
            if 'close' in df.columns and len(df) > 1:
                price_changes = df['close'].pct_change().abs()
                extreme_changes = price_changes[price_changes > 0.5]  # >50%
                
                if len(extreme_changes) > 0:
                    extreme_data = []
                    for idx in extreme_changes.index:
                        change = extreme_changes.loc[idx]
                        try:
                            position = df.index.get_loc(idx)
                            if isinstance(position, int) and position > 0:
                                price_before = df.iloc[position-1]['close']
                                price_after = df.loc[idx, 'close']
                                extreme_data.append({
                                    'timestamp': str(idx),
                                    'change_percent': change * 100,
                                    'price_before': price_before,
                                    'price_after': price_after
                                })
                        except (KeyError, IndexError):
                            continue
                    results['extreme_changes'] = extreme_data
                    results['max_change_percent'] = extreme_changes.max() * 100
                
                # Score de validez (penalizar cambios extremos)
                normal_changes = (price_changes <= 0.5).sum()
                results['price_validity_score'] = normal_changes / len(price_changes)
                
        except Exception as e:
            logger.error(f"❌ Error validando precios: {e}")
            
        return results
    
    def validate_volume_validity(self, df: pd.DataFrame, symbol: str) -> Dict:
        """Validar validez de volumen"""
        logger.info(f"📊 Validando validez de volumen para {symbol}")
        
        results = {
            'zero_volume_count': 0,
            'zero_volume_percentage': 0.0,
            'avg_volume': 0.0,
            'volume_validity_score': 0.0
        }
        
        try:
            if 'volume' in df.columns:
                zero_volume = (df['volume'] <= 0).sum()
                total_records = len(df)
                
                results['zero_volume_count'] = zero_volume
                results['zero_volume_percentage'] = (zero_volume / total_records) * 100
                results['avg_volume'] = df['volume'].mean()
                
                # Score: >95% de registros con volumen > 0
                valid_volume_pct = ((df['volume'] > 0).sum() / total_records)
                results['volume_validity_score'] = min(1.0, valid_volume_pct / 0.95)
                
                if zero_volume > 0:
                    logger.warning(f"⚠️ {zero_volume} registros con volumen cero ({results['zero_volume_percentage']:.1f}%)")
                    
        except Exception as e:
            logger.error(f"❌ Error validando volumen: {e}")
            
        return results
    
    def detect_outliers(self, df: pd.DataFrame, symbol: str) -> Dict:
        """Detectar outliers extremos usando IQR method"""
        logger.info(f"🎯 Detectando outliers para {symbol}")
        
        results = {
            'outliers_detected': {},
            'outlier_count': 0,
            'outlier_percentage': 0.0
        }
        
        try:
            numeric_columns = ['open', 'high', 'low', 'close', 'volume']
            total_outliers = 0
            
            for col in numeric_columns:
                if col in df.columns:
                    Q1 = df[col].quantile(0.25)
                    Q3 = df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    
                    # Definir límites para outliers extremos
                    lower_bound = Q1 - 3 * IQR
                    upper_bound = Q3 + 3 * IQR
                    
                    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
                    
                    if len(outliers) > 0:
                        results['outliers_detected'][col] = {
                            'count': len(outliers),
                            'percentage': (len(outliers) / len(df)) * 100,
                            'extreme_values': outliers[col].tolist()[:10]  # Primeros 10
                        }
                        total_outliers += len(outliers)
            
            results['outlier_count'] = total_outliers
            results['outlier_percentage'] = (total_outliers / len(df)) * 100 if len(df) > 0 else 0
            
        except Exception as e:
            logger.error(f"❌ Error detectando outliers: {e}")
            
        return results
    
    def validate_temporal_consistency(self, df: pd.DataFrame, symbol: str) -> Dict:
        """Validar consistencia temporal (OHLC relationships)"""
        logger.info(f"⏰ Validando consistencia temporal para {symbol}")
        
        results = {
            'ohlc_violations': 0,
            'ohlc_violations_percentage': 0.0,
            'temporal_consistency_score': 0.0
        }
        
        try:
            required_cols = ['open', 'high', 'low', 'close']
            if all(col in df.columns for col in required_cols):
                
                # Verificar relaciones OHLC básicas
                violations = 0
                
                # High debe ser >= Open, Low, Close
                high_violations = ((df['high'] < df['open']) | 
                                 (df['high'] < df['low']) | 
                                 (df['high'] < df['close'])).sum()
                
                # Low debe ser <= Open, High, Close  
                low_violations = ((df['low'] > df['open']) | 
                                (df['low'] > df['high']) | 
                                (df['low'] > df['close'])).sum()
                
                violations = high_violations + low_violations
                
                results['ohlc_violations'] = violations
                results['ohlc_violations_percentage'] = (violations / len(df)) * 100
                results['temporal_consistency_score'] = max(0.0, 1.0 - (violations / len(df)))
                
                if violations > 0:
                    logger.warning(f"⚠️ {violations} violaciones OHLC detectadas ({results['ohlc_violations_percentage']:.2f}%)")
                    
        except Exception as e:
            logger.error(f"❌ Error validando consistencia temporal: {e}")
            
        return results
    
    def validate_dataset(self, symbol: str, timeframe: str) -> Dict:
        """Validación completa de un dataset"""
        logger.info(f"🔍 === VALIDACIÓN COMPLETA: {symbol}_{timeframe} ===")
        
        # Cargar datos
        df = self.load_data(symbol, timeframe)
        if df is None:
            return {'error': 'No se pudieron cargar los datos'}
        
        # Ejecutar todas las validaciones
        validation_results = {
            'symbol': symbol,
            'timeframe': timeframe,
            'dataset_info': {
                'total_records': len(df),
                'date_range': {
                    'start': df.index.min().strftime('%Y-%m-%d %H:%M:%S'),
                    'end': df.index.max().strftime('%Y-%m-%d %H:%M:%S')
                },
                'columns': df.columns.tolist()
            }
        }
        
        # Ejecutar validaciones individuales
        validation_results['completeness'] = self.validate_completeness(df, symbol, timeframe)
        validation_results['price_validity'] = self.validate_price_validity(df, symbol)
        validation_results['volume_validity'] = self.validate_volume_validity(df, symbol)
        validation_results['outliers'] = self.detect_outliers(df, symbol)
        validation_results['temporal_consistency'] = self.validate_temporal_consistency(df, symbol)
        
        # Calcular score general de calidad
        scores = [
            validation_results['completeness'].get('completeness_score', 0),
            validation_results['price_validity'].get('price_validity_score', 0),
            validation_results['volume_validity'].get('volume_validity_score', 0),
            validation_results['temporal_consistency'].get('temporal_consistency_score', 0)
        ]
        
        validation_results['overall_quality_score'] = np.mean(scores)
        
        # Determinar estado de calidad
        if validation_results['overall_quality_score'] >= 0.9:
            validation_results['quality_status'] = '✅ EXCELENTE'
        elif validation_results['overall_quality_score'] >= 0.7:
            validation_results['quality_status'] = '⚠️ BUENA (requiere correcciones menores)'
        elif validation_results['overall_quality_score'] >= 0.5:
            validation_results['quality_status'] = '⚠️ REGULAR (requiere correcciones importantes)'
        else:
            validation_results['quality_status'] = '❌ MALA (requiere limpieza exhaustiva)'
        
        # Guardar resultados
        self.validation_results[f"{symbol}_{timeframe}"] = validation_results
        
        return validation_results
    
    def print_validation_report(self, results: Dict):
        """Imprimir reporte de validación formateado"""
        symbol = results['symbol']
        timeframe = results['timeframe']
        
        print(f"\n{'='*60}")
        print(f"📊 REPORTE DE VALIDACIÓN: {symbol}_{timeframe}")
        print(f"{'='*60}")
        
        # Info general
        info = results['dataset_info']
        print(f"📈 Registros totales: {info['total_records']:,}")
        print(f"📅 Rango de fechas: {info['date_range']['start']} → {info['date_range']['end']}")
        print(f"📋 Columnas: {', '.join(info['columns'])}")
        
        # Score general
        score = results['overall_quality_score']
        status = results['quality_status']
        print(f"\n🎯 CALIDAD GENERAL: {score:.2%} - {status}")
        
        # Detalles por categoría
        print(f"\n📊 DETALLES DE VALIDACIÓN:")
        
        # Completeness
        comp = results['completeness']
        print(f"  ✅ Completeness: {comp['completeness_score']:.2%}")
        if comp['gaps_detected']:
            print(f"     ⚠️ {len(comp['gaps_detected'])} gaps detectados (máx: {comp['max_gap_hours']:.1f}h)")
        
        # Price validity
        price = results['price_validity']
        print(f"  💰 Validez precios: {price['price_validity_score']:.2%}")
        if price['extreme_changes']:
            print(f"     ⚠️ {len(price['extreme_changes'])} cambios extremos (máx: {price['max_change_percent']:.1f}%)")
        
        # Volume validity
        vol = results['volume_validity']
        print(f"  📊 Validez volumen: {vol['volume_validity_score']:.2%}")
        if vol['zero_volume_count'] > 0:
            print(f"     ⚠️ {vol['zero_volume_count']} registros sin volumen ({vol['zero_volume_percentage']:.1f}%)")
        
        # Outliers
        outliers = results['outliers']
        print(f"  🎯 Outliers: {outliers['outlier_percentage']:.2f}% del dataset")
        
        # Temporal consistency
        temp = results['temporal_consistency']
        print(f"  ⏰ Consistencia OHLC: {temp['temporal_consistency_score']:.2%}")
        if temp['ohlc_violations'] > 0:
            print(f"     ⚠️ {temp['ohlc_violations']} violaciones OHLC ({temp['ohlc_violations_percentage']:.2f}%)")
        
        print(f"{'='*60}")
    
    def validate_all_datasets(self) -> Dict:
        """Validar todos los datasets disponibles"""
        logger.info("🔍 === VALIDACIÓN MASIVA DE DATASETS ===")
        
        all_results = {}
        csv_files = list(self.data_path.glob("*.csv"))
        
        if not csv_files:
            logger.error("❌ No se encontraron archivos CSV en data/raw")
            return {}
        
        for csv_file in csv_files:
            try:
                # Extraer symbol y timeframe del nombre del archivo
                filename = csv_file.stem  # sin extensión
                parts = filename.split('_')
                
                if len(parts) >= 2:
                    symbol = parts[0]
                    timeframe = '_'.join(parts[1:])  # Por si hay múltiples guiones bajos
                    
                    logger.info(f"🔍 Validando {symbol}_{timeframe}")
                    results = self.validate_dataset(symbol, timeframe)
                    all_results[f"{symbol}_{timeframe}"] = results
                    
                    # Imprimir reporte individual
                    self.print_validation_report(results)
                    
            except Exception as e:
                logger.error(f"❌ Error validando {csv_file}: {e}")
        
        # Resumen general
        self.print_summary_report(all_results)
        
        return all_results
    
    def print_summary_report(self, all_results: Dict):
        """Imprimir resumen general de todas las validaciones"""
        print(f"\n{'='*80}")
        print(f"📋 RESUMEN GENERAL DE VALIDACIÓN")
        print(f"{'='*80}")
        
        if not all_results:
            print("❌ No hay resultados de validación")
            return
        
        # Estadísticas generales
        total_datasets = len(all_results)
        quality_scores = [r['overall_quality_score'] for r in all_results.values() if 'overall_quality_score' in r]
        avg_quality = np.mean(quality_scores) if quality_scores else 0
        
        print(f"📊 Datasets analizados: {total_datasets}")
        print(f"🎯 Calidad promedio: {avg_quality:.2%}")
        
        # Clasificación por calidad
        excellent = sum(1 for score in quality_scores if score >= 0.9)
        good = sum(1 for score in quality_scores if 0.7 <= score < 0.9)
        regular = sum(1 for score in quality_scores if 0.5 <= score < 0.7)
        poor = sum(1 for score in quality_scores if score < 0.5)
        
        print(f"\n📈 DISTRIBUCIÓN DE CALIDAD:")
        print(f"  ✅ Excelente (≥90%): {excellent} datasets")
        print(f"  ⚠️  Buena (70-89%):   {good} datasets")
        print(f"  ⚠️  Regular (50-69%): {regular} datasets")
        print(f"  ❌ Mala (<50%):      {poor} datasets")
        
        # Top y bottom performers
        if quality_scores:
            sorted_results = sorted(all_results.items(), 
                                  key=lambda x: x[1].get('overall_quality_score', 0), 
                                  reverse=True)
            
            print(f"\n🏆 MEJORES DATASETS:")
            for i, (dataset, results) in enumerate(sorted_results[:3]):
                score = results.get('overall_quality_score', 0)
                print(f"  {i+1}. {dataset}: {score:.2%}")
            
            if poor > 0:
                print(f"\n⚠️ DATASETS QUE REQUIEREN ATENCIÓN:")
                for dataset, results in sorted_results[-3:]:
                    score = results.get('overall_quality_score', 0)
                    if score < 0.7:
                        print(f"  • {dataset}: {score:.2%}")
        
        print(f"{'='*80}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Validador de Datos NvBot3')
    parser.add_argument('--symbol', type=str, help='Símbolo específico a validar (ej: BTCUSDT)')
    parser.add_argument('--timeframe', type=str, help='Timeframe específico (ej: 5m)')
    parser.add_argument('--all-symbols', action='store_true', help='Validar todos los símbolos disponibles')
    
    args = parser.parse_args()
    
    # Crear validador
    validator = DataValidator()
    
    if args.all_symbols:
        # Validar todos los datasets
        validator.validate_all_datasets()
    elif args.symbol and args.timeframe:
        # Validar dataset específico
        results = validator.validate_dataset(args.symbol, args.timeframe)
        validator.print_validation_report(results)
    else:
        print("❌ Especifica --symbol y --timeframe, o usa --all-symbols")
        print("Ejemplo: python data_validator.py --symbol BTCUSDT --timeframe 5m")
        print("Ejemplo: python data_validator.py --all-symbols")
