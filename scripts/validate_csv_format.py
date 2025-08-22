#!/usr/bin/env python3
"""
📊 CSV Format Validator - NvBot3
===============================

Validador completo de formato CSV para garantizar compatibilidad con:
- Feature Calculator (indicadores técnicos)
- Target Creator (detección momentum/rebound)
- Model Trainer (pipeline de entrenamiento)

Basado en el formato usado por download_training_data_only.py
Detecta problemas críticos vs warnings para optimizar calidad de datos.
"""

import os
import sys
import pandas as pd
import numpy as np
import json
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import warnings

# Configurar logging
log_file = Path(__file__).parent.parent / 'logs' / 'csv_validation_report.log'
log_file.parent.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_file, encoding='utf-8')
    ]
)

logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

class CSVFormatValidator:
    """
    Validador de formato CSV específico para NvBot3
    Garantiza compatibilidad con Feature Calculator y Target Creator
    """
    
    def __init__(self):
        # Directorio de datos raw
        self.raw_data_dir = Path(__file__).parent.parent / 'data' / 'raw'
        
        # Formato estándar requerido (compatible con HistoricalDataDownloader)
        self.required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        self.optional_columns = ['close_time', 'quote_volume', 'trades', 'taker_buy_base', 'taker_buy_quote']
        
        # Criterios de calidad específicos para entrenamiento
        self.quality_criteria = {
            'min_records_for_features': 200,      # Para SMA 200
            'min_records_for_training': 1000,     # Para training split confiable
            'max_null_percentage': 5.0,           # Max 5% valores nulos
            'max_extreme_change_percentage': 1.0, # Max 1% cambios extremos (>500%)
            'max_gap_percentage': 2.0,            # Max 2% gaps temporales
            'lookforward_periods': {              # Para Target Creator
                '5m': 48,   # 4h ahead
                '15m': 16,  # 4h ahead  
                '1h': 4,    # 4h ahead
                '4h': 1,    # 4h ahead
                '1d': 1     # 1d ahead
            }
        }
        
        # Estadísticas globales
        self.total_files = 0
        self.valid_files = 0
        self.files_with_warnings = 0
        self.files_with_errors = 0
        self.validation_results = {}
        
    def validate_all_csv_files(self) -> bool:
        """
        Validar todos los archivos CSV en data/raw/
        Retorna True si todos son válidos para entrenamiento
        """
        logger.info("🔍 INICIANDO VALIDACIÓN COMPLETA DE ARCHIVOS CSV")
        logger.info("=" * 55)
        logger.info(f"📁 Directorio: {self.raw_data_dir}")
        
        if not self.raw_data_dir.exists():
            logger.error(f"❌ Directorio de datos no existe: {self.raw_data_dir}")
            return False
        
        # Obtener todos los archivos CSV
        csv_files = list(self.raw_data_dir.glob("*.csv"))
        
        if not csv_files:
            logger.warning("⚠️ No se encontraron archivos CSV en el directorio")
            return False
        
        self.total_files = len(csv_files)
        logger.info(f"📊 Total archivos CSV encontrados: {self.total_files}")
        logger.info("")
        
        # Validar cada archivo
        for i, csv_file in enumerate(csv_files, 1):
            logger.info(f"🔍 [{i}/{self.total_files}] Validando: {csv_file.name}")
            
            validation_result = self._validate_single_file(csv_file)
            self.validation_results[csv_file.name] = validation_result
            
            if validation_result['is_valid']:
                self.valid_files += 1
                if validation_result['warnings']:
                    self.files_with_warnings += 1
                    logger.warning(f"⚠️ [{i}/{self.total_files}] {csv_file.name}: Válido con advertencias")
                else:
                    logger.info(f"✅ [{i}/{self.total_files}] {csv_file.name}: Válido")
            else:
                self.files_with_errors += 1
                logger.error(f"❌ [{i}/{self.total_files}] {csv_file.name}: ERRORES CRÍTICOS")
                
                # Mostrar errores críticos
                for error in validation_result['critical_errors']:
                    logger.error(f"   • {error}")
        
        # Generar reporte final
        self._generate_final_report()
        
        # Retornar si todos los archivos están listos para entrenamiento
        return self.files_with_errors == 0
    
    def _validate_single_file(self, csv_file: Path) -> Dict:
        """
        Validar un archivo CSV individual
        Retorna diccionario con resultados de validación
        """
        result = {
            'filename': csv_file.name,
            'is_valid': True,
            'critical_errors': [],
            'warnings': [],
            'stats': {},
            'feature_calculator_ready': False,
            'target_creator_ready': False,
            'training_ready': False
        }
        
        try:
            # Cargar CSV
            df = pd.read_csv(csv_file)
            result['stats']['total_records'] = len(df)
            
            if len(df) == 0:
                result['critical_errors'].append("empty_file: Archivo vacío")
                result['is_valid'] = False
                return result
            
            # 1. Validar estructura de columnas
            self._validate_columns(df, result)
            
            # 2. Validar tipos de datos
            self._validate_data_types(df, result)
            
            # 3. Validar timestamps
            self._validate_timestamps(df, result)
            
            # 4. Validar datos OHLCV
            self._validate_ohlcv_data(df, result)
            
            # 5. Validar suficiencia de datos
            self._validate_data_sufficiency(df, csv_file.name, result)
            
            # 6. Validar continuidad temporal
            self._validate_temporal_continuity(df, csv_file.name, result)
            
            # 7. Validar compatibilidad con componentes
            self._validate_component_compatibility(df, csv_file.name, result)
            
        except Exception as e:
            result['critical_errors'].append(f"file_read_error: {str(e)}")
            result['is_valid'] = False
        
        # Determinar si hay errores críticos
        if result['critical_errors']:
            result['is_valid'] = False
        
        return result
    
    def _validate_columns(self, df: pd.DataFrame, result: Dict):
        """Validar que existan las columnas requeridas"""
        missing_columns = [col for col in self.required_columns if col not in df.columns]
        
        if missing_columns:
            result['critical_errors'].append(f"missing_columns: Columnas faltantes: {missing_columns}")
        
        # Verificar columnas extra (no crítico)
        extra_columns = [col for col in df.columns if col not in self.required_columns + self.optional_columns]
        if extra_columns:
            result['warnings'].append(f"extra_columns: Columnas adicionales: {extra_columns}")
        
        result['stats']['columns'] = list(df.columns)
    
    def _validate_data_types(self, df: pd.DataFrame, result: Dict):
        """Validar tipos de datos de columnas críticas"""
        numeric_columns = ['open', 'high', 'low', 'close', 'volume']
        
        for col in numeric_columns:
            if col in df.columns:
                if not pd.api.types.is_numeric_dtype(df[col]):
                    result['critical_errors'].append(f"invalid_data_type: {col} no es numérico")
                
                # Verificar valores nulos
                null_count = df[col].isnull().sum()
                null_percentage = (null_count / len(df)) * 100
                
                if null_percentage > self.quality_criteria['max_null_percentage']:
                    result['critical_errors'].append(f"excessive_nulls: {col} tiene {null_percentage:.1f}% valores nulos")
                elif null_count > 0:
                    result['warnings'].append(f"some_nulls: {col} tiene {null_count} valores nulos ({null_percentage:.1f}%)")
    
    def _validate_timestamps(self, df: pd.DataFrame, result: Dict):
        """Validar formato y orden de timestamps"""
        if 'timestamp' not in df.columns:
            return
        
        try:
            # Intentar convertir timestamps
            timestamps = pd.to_datetime(df['timestamp'])
            
            # Verificar orden cronológico
            if not timestamps.is_monotonic_increasing:
                result['critical_errors'].append("temporal_order: Timestamps no están ordenados cronológicamente")
            
            # Verificar duplicados
            duplicates = timestamps.duplicated().sum()
            if duplicates > 0:
                result['warnings'].append(f"duplicate_timestamps: {duplicates} timestamps duplicados")
            
            result['stats']['date_range'] = {
                'start': str(timestamps.min()),
                'end': str(timestamps.max()),
                'duration_days': (timestamps.max() - timestamps.min()).days
            }
            
        except Exception as e:
            result['critical_errors'].append(f"timestamp_format: Error parseando timestamps: {str(e)}")
    
    def _validate_ohlcv_data(self, df: pd.DataFrame, result: Dict):
        """Validar consistencia de datos OHLCV"""
        required_ohlc = ['open', 'high', 'low', 'close']
        
        if not all(col in df.columns for col in required_ohlc):
            return
        
        # Verificar relaciones OHLC válidas
        invalid_ohlc = (
            (df['high'] < df['low']) | 
            (df['high'] < df['open']) | 
            (df['high'] < df['close']) |
            (df['low'] > df['open']) | 
            (df['low'] > df['close'])
        )
        
        invalid_count = invalid_ohlc.sum()
        invalid_percentage = (invalid_count / len(df)) * 100
        
        if invalid_percentage > 1.0:  # Más del 1% inválido es crítico
            result['critical_errors'].append(f"invalid_ohlc_relationships: {invalid_count} registros con relaciones OHLC inválidas ({invalid_percentage:.1f}%)")
        elif invalid_count > 0:
            result['warnings'].append(f"some_invalid_ohlc: {invalid_count} registros con relaciones OHLC inválidas ({invalid_percentage:.1f}%)")
        
        # Verificar cambios extremos de precio
        if 'close' in df.columns:
            price_changes = df['close'].pct_change().abs()
            extreme_changes = (price_changes > 5.0).sum()  # >500% cambio
            extreme_percentage = (extreme_changes / len(df)) * 100
            
            if extreme_percentage > self.quality_criteria['max_extreme_change_percentage']:
                result['critical_errors'].append(f"excessive_extreme_changes: {extreme_changes} cambios extremos de precio (>{extreme_percentage:.1f}%)")
            elif extreme_changes > 0:
                result['warnings'].append(f"some_extreme_changes: {extreme_changes} cambios extremos de precio ({extreme_percentage:.1f}%)")
        
        # Verificar volumen
        if 'volume' in df.columns:
            zero_volume = (df['volume'] <= 0).sum()
            if zero_volume > len(df) * 0.1:  # >10% volumen cero es preocupante
                result['warnings'].append(f"high_zero_volume: {zero_volume} registros con volumen cero o negativo")
    
    def _validate_data_sufficiency(self, df: pd.DataFrame, filename: str, result: Dict):
        """Validar suficiencia de datos para Feature Calculator y Target Creator"""
        record_count = len(df)
        
        # Para Feature Calculator (necesita indicadores largos)
        if record_count < self.quality_criteria['min_records_for_features']:
            result['critical_errors'].append(f"insufficient_data_for_features: {record_count} < {self.quality_criteria['min_records_for_features']} períodos")
        else:
            result['feature_calculator_ready'] = True
        
        # Para entrenamiento confiable
        if record_count < self.quality_criteria['min_records_for_training']:
            result['warnings'].append(f"limited_training_data: {record_count} < {self.quality_criteria['min_records_for_training']} períodos recomendados")
        
        # Para Target Creator (necesita períodos futuros)
        timeframe = self._extract_timeframe_from_filename(filename)
        if timeframe and timeframe in self.quality_criteria['lookforward_periods']:
            required_lookforward = self.quality_criteria['lookforward_periods'][timeframe]
            usable_records = record_count - required_lookforward
            
            if usable_records < self.quality_criteria['min_records_for_features']:
                result['critical_errors'].append(f"insufficient_usable_data: Solo {usable_records} registros utilizables después de lookforward")
            else:
                result['target_creator_ready'] = True
    
    def _validate_temporal_continuity(self, df: pd.DataFrame, filename: str, result: Dict):
        """Validar continuidad temporal apropiada para el timeframe"""
        if 'timestamp' not in df.columns:
            return
        
        try:
            timestamps = pd.to_datetime(df['timestamp'])
            
            # Determinar timeframe esperado
            timeframe = self._extract_timeframe_from_filename(filename)
            if timeframe:
                expected_delta = self._get_expected_timedelta(timeframe)
            else:
                expected_delta = None
            
            if expected_delta:
                # Calcular diferencias temporales
                time_diffs = timestamps.diff().dropna()
                
                # Detectar gaps grandes (>2x el intervalo esperado)
                large_gaps = (time_diffs > expected_delta * 2).sum()
                gap_percentage = (large_gaps / len(time_diffs)) * 100
                
                if gap_percentage > self.quality_criteria['max_gap_percentage']:
                    result['critical_errors'].append(f"excessive_temporal_gaps: {large_gaps} gaps grandes ({gap_percentage:.1f}%)")
                elif large_gaps > 0:
                    result['warnings'].append(f"temporal_gaps: {large_gaps} gaps grandes detectados ({gap_percentage:.1f}%)")
                
                result['stats']['temporal_analysis'] = {
                    'expected_interval': str(expected_delta),
                    'large_gaps': int(large_gaps),
                    'gap_percentage': round(gap_percentage, 2)
                }
                
        except Exception as e:
            result['warnings'].append(f"temporal_analysis_failed: {str(e)}")
    
    def _validate_component_compatibility(self, df: pd.DataFrame, filename: str, result: Dict):
        """Validar compatibilidad específica con Feature Calculator y Target Creator"""
        
        # Verificar que esté listo para Feature Calculator
        if result['feature_calculator_ready'] and not result['critical_errors']:
            # Verificar datos para indicadores técnicos comunes
            if 'close' in df.columns and len(df) >= 200:
                result['feature_calculator_ready'] = True
            else:
                result['feature_calculator_ready'] = False
        
        # Verificar que esté listo para Target Creator
        if result['target_creator_ready'] and not result['critical_errors']:
            timeframe = self._extract_timeframe_from_filename(filename)
            if timeframe and 'close' in df.columns:
                lookforward = self.quality_criteria['lookforward_periods'].get(timeframe, 1)
                if len(df) > lookforward + 200:
                    result['target_creator_ready'] = True
                else:
                    result['target_creator_ready'] = False
        
        # Archivo listo para entrenamiento si ambos componentes están listos
        result['training_ready'] = (result['feature_calculator_ready'] and 
                                   result['target_creator_ready'] and 
                                   not result['critical_errors'])
    
    def _extract_timeframe_from_filename(self, filename: str) -> Optional[str]:
        """Extraer timeframe del nombre del archivo (ej: BTCUSDT_5m.csv -> 5m)"""
        try:
            # Formato esperado: SYMBOL_TIMEFRAME.csv
            parts = filename.replace('.csv', '').split('_')
            if len(parts) >= 2:
                return parts[-1]  # Último elemento es el timeframe
        except:
            pass
        return None
    
    def _get_expected_timedelta(self, timeframe: str) -> Optional[timedelta]:
        """Obtener timedelta esperado para un timeframe"""
        timeframe_map = {
            '1m': timedelta(minutes=1),
            '5m': timedelta(minutes=5),
            '15m': timedelta(minutes=15),
            '1h': timedelta(hours=1),
            '4h': timedelta(hours=4),
            '1d': timedelta(days=1)
        }
        return timeframe_map.get(timeframe)
    
    def _generate_final_report(self):
        """Generar reporte final de validación"""
        logger.info("")
        logger.info("=" * 55)
        logger.info("📊 REPORTE FINAL DE VALIDACIÓN CSV")
        logger.info("=" * 55)
        
        # Estadísticas generales
        success_rate = (self.valid_files / self.total_files * 100) if self.total_files > 0 else 0
        
        logger.info(f"📁 Total archivos analizados: {self.total_files}")
        logger.info(f"✅ Archivos válidos: {self.valid_files}")
        logger.info(f"⚠️ Archivos con problemas: {self.files_with_warnings}")
        logger.info(f"🚨 Errores críticos: {self.files_with_errors}")
        logger.info(f"📈 Tasa de éxito: {success_rate:.1f}%")
        logger.info("")
        
        # Compatibilidad por componente
        feature_ready = sum(1 for r in self.validation_results.values() if r.get('feature_calculator_ready', False))
        target_ready = sum(1 for r in self.validation_results.values() if r.get('target_creator_ready', False))
        training_ready = sum(1 for r in self.validation_results.values() if r.get('training_ready', False))
        
        logger.info("🤖 COMPATIBILIDAD PARA ENTRENAMIENTO:")
        logger.info(f"🔧 Feature Calculator listos: {feature_ready}/{self.total_files}")
        logger.info(f"🎯 Target Creator listos: {target_ready}/{self.total_files}")
        logger.info(f"🚀 Listos para entrenamiento: {training_ready}/{self.total_files}")
        logger.info("")
        
        # Problemas más comunes
        self._report_common_issues()
        
        # Recomendaciones finales
        self._provide_recommendations(training_ready)
        
        # Guardar reporte detallado en JSON
        self._save_detailed_report()
    
    def _report_common_issues(self):
        """Reportar los problemas más comunes encontrados"""
        error_counts = {}
        warning_counts = {}
        
        for result in self.validation_results.values():
            for error in result.get('critical_errors', []):
                error_type = error.split(':')[0]
                error_counts[error_type] = error_counts.get(error_type, 0) + 1
            
            for warning in result.get('warnings', []):
                warning_type = warning.split(':')[0]
                warning_counts[warning_type] = warning_counts.get(warning_type, 0) + 1
        
        if error_counts:
            logger.warning("🚨 ERRORES CRÍTICOS MÁS COMUNES:")
            for error_type, count in sorted(error_counts.items(), key=lambda x: x[1], reverse=True):
                logger.warning(f"   • {error_type}: {count} archivos")
        
        if warning_counts:
            logger.info("⚠️ ADVERTENCIAS MÁS COMUNES:")
            for warning_type, count in sorted(warning_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
                logger.info(f"   • {warning_type}: {count} archivos")
    
    def _provide_recommendations(self, training_ready: int):
        """Proporcionar recomendaciones basadas en resultados"""
        logger.info("💡 RECOMENDACIONES:")
        
        if training_ready == self.total_files:
            logger.info("🎉 ¡TODOS LOS ARCHIVOS ESTÁN LISTOS PARA ENTRENAMIENTO!")
            logger.info("   ✅ Puedes proceder con Feature Calculator")
            logger.info("   ✅ Puedes proceder con Target Creator")
            logger.info("   ✅ Puedes proceder con Model Trainer")
        elif training_ready >= self.total_files * 0.9:
            logger.info("🟢 La mayoría de archivos están listos")
            logger.info("   🔧 Corregir archivos con errores críticos")
            logger.info("   ✅ Sistema mayormente funcional")
        elif training_ready >= self.total_files * 0.7:
            logger.warning("🟡 Sistema parcialmente listo")
            logger.warning("   ⚠️ Revisar y corregir archivos problemáticos")
            logger.warning("   🔄 Re-descargar archivos con errores críticos")
        else:
            logger.error("🔴 SISTEMA NO LISTO PARA ENTRENAMIENTO")
            logger.error("   ❌ Demasiados archivos con problemas críticos")
            logger.error("   🔄 Re-ejecutar download_training_data_only.py")
            logger.error("   🔍 Verificar configuración de descarga")
    
    def _save_detailed_report(self):
        """Guardar reporte detallado en formato JSON"""
        report_path = Path(__file__).parent.parent / 'logs' / 'csv_validation_detailed_report.json'
        
        detailed_report = {
            'validation_timestamp': datetime.now().isoformat(),
            'summary': {
                'total_files': self.total_files,
                'valid_files': self.valid_files,
                'files_with_warnings': self.files_with_warnings,
                'files_with_errors': self.files_with_errors,
                'success_rate': (self.valid_files / self.total_files * 100) if self.total_files > 0 else 0
            },
            'quality_criteria': self.quality_criteria,
            'file_results': self.validation_results
        }
        
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(detailed_report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"📄 Reporte detallado guardado: {report_path}")


def main():
    """Función principal"""
    logger.info("🚀 VALIDADOR CSV NVBOT3 - INICIO")
    logger.info("🎯 Objetivo: Verificar compatibilidad para entrenamiento")
    logger.info("")
    
    validator = CSVFormatValidator()
    
    try:
        success = validator.validate_all_csv_files()
        
        if success:
            logger.info("🎉 VALIDACIÓN EXITOSA: Todos los archivos listos para entrenamiento")
            return 0
        else:
            logger.error("❌ VALIDACIÓN FALLIDA: Corregir errores antes del entrenamiento")
            return 1
            
    except KeyboardInterrupt:
        logger.warning("⚠️ Validación interrumpida por usuario")
        return 1
    except Exception as e:
        logger.error(f"❌ Error durante validación: {str(e)}")
        return 1


if __name__ == "__main__":
    exit(main())
