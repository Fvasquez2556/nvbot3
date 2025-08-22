"""
🔧 Verificador Dual Strategy COMBINADO - Criterios Adaptativos Inteligentes
Combina validación estricta para trading + flexibilidad para entrenamiento
Versión mejorada que integra lo mejor de ambos enfoques.

Uso:
  python scripts/verify_dual_strategy_data.py          # Modo flexible (por defecto)
  python scripts/verify_dual_strategy_data.py strict   # Modo estricto para trading
"""

import os
import yaml
import pandas as pd
import requests
from datetime import datetime, timedelta
import logging
from pathlib import Path
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DualStrategyVerifier:
    def __init__(self, validation_mode='flexible'):
        """
        Inicializar verificador con modo de validación configurable.
        
        Args:
            validation_mode: 'strict' para trading en vivo, 'flexible' para entrenamiento
        """
        self.validation_mode = validation_mode
        self.data_dir = Path('data/raw')
        self.config_dir = Path('config')
        
        # Cargar configuraciones
        self.training_config = self.load_yaml_config('training_config.yaml')
        self.monitoring_config = self.load_yaml_config('monitoring_config.yaml')
        
        # Estadísticas de verificación
        self.stats = {
            'training': {'valid': 0, 'missing': 0, 'corrupted': 0, 'total': 0},
            'monitoring': {'available': 0, 'unavailable': 0, 'total': 0}
        }
        
        self.missing_training_files = []
        self.corrupted_training_files = []
        self.unavailable_monitoring_symbols = []
        
        # Configurar criterios según el modo
        self.setup_validation_criteria()
    
    def setup_validation_criteria(self):
        """Configurar criterios de validación según el modo seleccionado."""
        if self.validation_mode == 'strict':
            # CRITERIOS ESTRICTOS para trading en vivo
            self.min_records = {
                '5m': 20000,   # ~69 días - Necesario para análisis profundo
                '15m': 15000,  # ~156 días - Patrones intraday robustos
                '1h': 15000,   # ~625 días - Análisis técnico confiable
                '4h': 4000,    # ~666 días - Tendencias de mediano plazo
                '1d': 1000     # ~2.7 años - Análisis fundamental sólido
            }
            self.max_data_age_days = 30  # Datos muy recientes para trading
            logger.info("🔒 Modo ESTRICTO: Criterios optimizados para trading en vivo")
            
        else:  # flexible
            # CRITERIOS FLEXIBLES para entrenamiento y desarrollo
            self.min_records = {
                '5m': 1000,   # ~3.5 días - Suficiente para patterns básicos
                '15m': 1000,  # ~10 días - Análisis intraday elemental
                '1h': 1000,   # ~42 días - Tendencias de corto plazo
                '4h': 1000,   # ~166 días - Análisis de medio plazo
                '1d': 600     # ~1.6 años - Base sólida para ML (ACEPTA 730 registros)
            }
            self.max_data_age_days = 60  # Más tolerante para entrenamiento
            logger.info("🔓 Modo FLEXIBLE: Criterios optimizados para entrenamiento y desarrollo")
    
    def load_yaml_config(self, filename):
        """Cargar archivo de configuración YAML."""
        config_path = self.config_dir / filename
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"❌ Error cargando {filename}: {e}")
            return {}
    
    def verify_file_exists_and_quality(self, filename):
        """Verificar archivo con criterios adaptativos según el modo de validación."""
        file_path = self.data_dir / filename
        
        if not file_path.exists():
            return False, "Archivo no existe"
        
        try:
            df = pd.read_csv(file_path)
            
            # Extraer timeframe del nombre de archivo
            timeframe = filename.split('_')[1].replace('.csv', '')
            
            # Aplicar criterios según el modo actual
            required_records = self.min_records.get(timeframe, 1000)
            if len(df) < required_records:
                if self.validation_mode == 'strict':
                    return False, f"Insuficientes datos ({len(df)} < {required_records}) - Modo ESTRICTO"
                else:
                    return False, f"Muy pocos datos ({len(df)})"
            
            # Verificar columnas esenciales
            required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            missing_cols = [col for col in required_columns if col not in df.columns]
            if missing_cols:
                return False, f"Columnas faltantes: {missing_cols}"
            
            # Validación de fechas según el modo
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            latest_date = df['timestamp'].max()
            cutoff_date = datetime.now() - timedelta(days=self.max_data_age_days)
            
            if latest_date < cutoff_date:
                days_old = (datetime.now() - latest_date).days
                return False, f"Datos obsoletos ({days_old} días)"
            
            # Validación adicional: verificar que no hay datos futuros
            future_cutoff = datetime.now() + timedelta(days=1)
            if latest_date > future_cutoff:
                return False, f"Datos futuros detectados: {latest_date.date()}"
            
            # Validación de integridad de precios (solo en modo estricto)
            if self.validation_mode == 'strict':
                price_columns = ['open', 'high', 'low', 'close']
                for col in price_columns:
                    if df[col].isna().any():
                        return False, f"Precios nulos en columna {col}"
                    if (df[col] <= 0).any():
                        return False, f"Precios inválidos en columna {col}"
                
                # Verificar que high >= low y close/open están en rango
                invalid_ohlc = df[(df['high'] < df['low']) | 
                                 (df['high'] < df['open']) | 
                                 (df['high'] < df['close']) |
                                 (df['low'] > df['open']) | 
                                 (df['low'] > df['close'])].shape[0]
                
                if invalid_ohlc > 0:
                    return False, f"OHLC inválido en {invalid_ohlc} registros"
            
            return True, f"OK - {len(df)} registros hasta {latest_date.date()}"
            
        except Exception as e:
            return False, f"Error leyendo: {e}"
    
    def verify_training_data_files(self):
        """Verificar datos históricos para entrenamiento con criterios adaptativos."""
        logger.info("🔍 Verificando datos históricos para entrenamiento...")
        
        # Obtener símbolos de entrenamiento
        training_symbols = []
        if 'data' in self.training_config and 'symbols' in self.training_config['data']:
            symbols_config = self.training_config['data']['symbols']
            for tier in ['tier_1', 'tier_2', 'tier_3']:
                if tier in symbols_config:
                    training_symbols.extend(symbols_config[tier])
        
        timeframes = self.training_config.get('data', {}).get('timeframes', [])
        
        if not training_symbols or not timeframes:
            logger.error("❌ Configuración de entrenamiento incompleta")
            return False
        
        total_files = len(training_symbols) * len(timeframes)
        self.stats['training']['total'] = total_files
        
        logger.info(f"📊 Total archivos esperados: {total_files}")
        logger.info(f"   📚 {len(training_symbols)} símbolos × {len(timeframes)} timeframes")
        logger.info(f"   🔧 Modo validación: {self.validation_mode.upper()}")
        
        file_count = 0
        for symbol in training_symbols:
            for timeframe in timeframes:
                file_count += 1
                filename = f"{symbol}_{timeframe}.csv"
                
                is_valid, message = self.verify_file_exists_and_quality(filename)
                
                if is_valid:
                    self.stats['training']['valid'] += 1
                    logger.info(f"✅ [{file_count}/{total_files}] {filename}: {message}")
                else:
                    if "no existe" in message:
                        self.stats['training']['missing'] += 1
                        self.missing_training_files.append(filename)
                        logger.error(f"❌ [{file_count}/{total_files}] {filename}: {message}")
                    else:
                        self.stats['training']['corrupted'] += 1
                        self.corrupted_training_files.append(filename)
                        logger.error(f"❌ [{file_count}/{total_files}] {filename}: {message}")
        
        return True
    
    def verify_monitoring_symbols_availability(self):
        """Verificar disponibilidad de símbolos para monitoreo en Binance."""
        logger.info("🔍 Verificando disponibilidad de símbolos para monitoreo...")
        
        # Obtener símbolos de monitoreo
        monitoring_data = self.monitoring_config.get('live_monitoring', {})
        symbols_config = monitoring_data.get('symbols', {})
        
        # Combinar todos los símbolos de monitoreo
        all_monitoring_symbols = []
        for tier, symbols in symbols_config.items():
            if isinstance(symbols, list):
                all_monitoring_symbols.extend(symbols)
        
        if not all_monitoring_symbols:
            logger.error("❌ No se encontraron símbolos de monitoreo")
            return False
        
        self.stats['monitoring']['total'] = len(all_monitoring_symbols)
        
        # Obtener información de símbolos disponibles en Binance
        try:
            response = requests.get('https://api.binance.com/api/v3/exchangeInfo', timeout=10)
            exchange_info = response.json()
            available_symbols = {s['symbol'] for s in exchange_info['symbols'] if s['status'] == 'TRADING'}
            
            logger.info(f"📊 Verificando {len(all_monitoring_symbols)} símbolos de monitoreo...")
            
        except Exception as e:
            logger.error(f"❌ Error obteniendo información de Binance: {e}")
            return False
        
        # Verificar cada símbolo
        for i, symbol in enumerate(all_monitoring_symbols, 1):
            if symbol in available_symbols:
                self.stats['monitoring']['available'] += 1
                logger.info(f"✅ [{i}/{len(all_monitoring_symbols)}] {symbol}: Disponible en Binance")
            else:
                self.stats['monitoring']['unavailable'] += 1
                self.unavailable_monitoring_symbols.append(symbol)
                logger.error(f"❌ [{i}/{len(all_monitoring_symbols)}] {symbol}: NO disponible en Binance")
        
        return True
    
    def generate_detailed_report(self):
        """Generar reporte detallado con criterios adaptativos."""
        logger.info("\n" + "="*70)
        logger.info(f"📊 REPORTE VERIFICACIÓN DUAL STRATEGY - MODO {self.validation_mode.upper()}")
        logger.info("="*70)
        
        # Configuración dual
        training_symbols = len([s for tier in ['tier_1', 'tier_2', 'tier_3'] 
                               for s in self.training_config.get('data', {}).get('symbols', {}).get(tier, [])])
        training_timeframes = len(self.training_config.get('data', {}).get('timeframes', []))
        
        logger.info("🎯 CONFIGURACIÓN DUAL:")
        logger.info(f"   📚 Entrenamiento: {training_symbols} monedas × {training_timeframes} timeframes")
        logger.info(f"   📡 Monitoreo: {self.stats['monitoring']['total']} monedas (solo WebSocket)")
        logger.info(f"   🔧 Criterios: {self.validation_mode.upper()} ({self.max_data_age_days} días máx, 1d≥{self.min_records['1d']} registros)")
        
        # Resultados entrenamiento
        total_training = self.stats['training']['total']
        valid_training = self.stats['training']['valid']
        missing_training = self.stats['training']['missing']
        corrupted_training = self.stats['training']['corrupted']
        
        logger.info(f"\n📈 RESULTADOS ENTRENAMIENTO:")
        logger.info(f"   ✅ Archivos válidos: {valid_training}/{total_training} ({valid_training/total_training*100:.1f}%)")
        logger.info(f"   ❌ Archivos faltantes: {missing_training}")
        logger.info(f"   ⚠️ Archivos con problemas: {corrupted_training}")
        
        # Resultados monitoreo
        total_monitoring = self.stats['monitoring']['total']
        available_monitoring = self.stats['monitoring']['available']
        unavailable_monitoring = self.stats['monitoring']['unavailable']
        
        logger.info(f"\n📡 RESULTADOS MONITOREO:")
        logger.info(f"   ✅ Símbolos disponibles: {available_monitoring}/{total_monitoring} ({available_monitoring/total_monitoring*100:.1f}%)")
        logger.info(f"   ❌ Símbolos no disponibles: {unavailable_monitoring}")
        
        # Listas detalladas (solo primeros 10)
        if self.missing_training_files:
            logger.info(f"\n📋 Archivos entrenamiento faltantes ({len(self.missing_training_files)}):")
            for i, filename in enumerate(self.missing_training_files[:10], 1):
                logger.info(f"   • {filename}")
            if len(self.missing_training_files) > 10:
                logger.info(f"   • ... y {len(self.missing_training_files) - 10} más")
        
        if self.corrupted_training_files:
            logger.info(f"\n⚠️ Archivos entrenamiento con problemas ({len(self.corrupted_training_files)}):")
            for i, filename in enumerate(self.corrupted_training_files[:10], 1):
                logger.info(f"   • {filename}")
            if len(self.corrupted_training_files) > 10:
                logger.info(f"   • ... y {len(self.corrupted_training_files) - 10} más")
        
        if self.unavailable_monitoring_symbols:
            logger.info(f"\n⚠️ Símbolos monitoreo no disponibles ({len(self.unavailable_monitoring_symbols)}):")
            for symbol in self.unavailable_monitoring_symbols:
                logger.info(f"   • {symbol}")
        
        # Evaluación con criterios adaptativos
        training_completeness = valid_training / total_training if total_training > 0 else 0
        monitoring_availability = available_monitoring / total_monitoring if total_monitoring > 0 else 0
        
        # Umbrales según el modo
        if self.validation_mode == 'strict':
            training_threshold = 0.95  # 95% para trading
            monitoring_threshold = 0.90  # 90% para trading
        else:
            training_threshold = 0.80  # 80% para entrenamiento
            monitoring_threshold = 0.75  # 75% para entrenamiento
        
        logger.info(f"\n🎯 EVALUACIÓN DUAL STRATEGY ({self.validation_mode.upper()}):")
        
        if training_completeness >= training_threshold:
            logger.info(f"✅ ENTRENAMIENTO: {training_completeness*100:.1f}% completitud - APROBADO")
        else:
            logger.warning(f"⚠️ ENTRENAMIENTO: {training_completeness*100:.1f}% completitud - Necesita mejora")
            logger.info(f"   🔧 Faltan {total_training - valid_training} archivos para {training_threshold*100:.0f}% mínimo")
        
        if monitoring_availability >= monitoring_threshold:
            logger.info(f"✅ MONITOREO: {monitoring_availability*100:.1f}% disponibilidad - APROBADO")
        else:
            logger.warning(f"⚠️ MONITOREO: {monitoring_availability*100:.1f}% disponibilidad - Necesita ajuste")
            logger.info(f"   🔧 Remover {unavailable_monitoring} símbolos no disponibles")
        
        # Recomendaciones específicas
        logger.info(f"\n💡 RECOMENDACIONES:")
        
        if missing_training or corrupted_training:
            logger.info("   🔧 Ejecutar: python scripts/fix_training_data.py")
        
        if unavailable_monitoring:
            logger.info("   📝 Actualizar: config/monitoring_config_v2.yaml")
            logger.info("   🗑️ Remover símbolos no disponibles de la lista")
        
        if training_completeness >= training_threshold and monitoring_availability >= monitoring_threshold:
            logger.info("   🎉 DUAL STRATEGY LISTA para Model Trainer!")
            return True
        else:
            logger.warning("   ⚠️ Completar correcciones antes del Model Trainer")
            return False

def main():
    """Función principal con modo de validación configurable."""
    import sys
    
    # Permitir especificar modo por línea de comandos
    validation_mode = 'flexible'
    if len(sys.argv) > 1 and sys.argv[1] in ['strict', 'flexible']:
        validation_mode = sys.argv[1]
    
    try:
        logger.info("🔧 Iniciando Verificación Dual Strategy COMBINADA")
        logger.info(f"📋 Modo seleccionado: {validation_mode.upper()}")
        logger.info("="*70)
        
        verifier = DualStrategyVerifier(validation_mode=validation_mode)
        
        # Verificar datos de entrenamiento
        if not verifier.verify_training_data_files():
            logger.error("❌ Error verificando datos de entrenamiento")
            return 1
        
        # Verificar disponibilidad de monitoreo
        if not verifier.verify_monitoring_symbols_availability():
            logger.error("❌ Error verificando símbolos de monitoreo")
            return 1
        
        # Generar reporte combinado
        success = verifier.generate_detailed_report()
        
        if success:
            logger.info("\n🎯 PRÓXIMO PASO: Ejecutar correcciones si es necesario")
            logger.info("   📝 Comando: python scripts/fix_training_data.py")
        
        return 0 if success else 1
        
    except Exception as e:
        logger.error(f"❌ ERROR CRÍTICO en verificación: {e}")
        return 1

if __name__ == "__main__":
    exit(main())
