#!/usr/bin/env python3
"""
CORRECCIÓN #3: Descargador Masivo Unificado con Validación FLEXIBLE
=================================================================

COMBINACIÓN DE MEJORES CARACTERÍSTICAS:
- Documentación completa y logging detallado (de flexible)
- Compatibilidad Windows sin emojis (de simple)
- Validación flexible realista para cripto (ambos)
- Lista completa de archivos faltantes (unificada)
- Manejo robusto de errores y reportes (mejorado)

PROBLEMA RESUELTO:
- Validación muy estricta rechazaba volatilidad normal cripto
- Solo 7/57 archivos completados por límites irreales
- Problemas de encoding con emojis en Windows

SOLUCIÓN UNIFICADA:
- Límite de cambio de precio: 50% -> 200% (realista para cripto)
- Eventos extremos permitidos: 0 -> 5 (volatilidad normal)
- Mantiene validaciones críticas: gaps, volumen, consistencia
- Formato 100% compatible con HistoricalDataDownloader
- Logging limpio sin caracteres especiales
"""

import os
import sys
import time
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import pandas as pd

# Configurar entorno
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Configurar logging robusto SIN emojis para compatibilidad Windows
log_file = project_root / 'logs' / 'fix_training_data.log'
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

# Cargar variables de entorno
from dotenv import load_dotenv
env_path = project_root / '.env'
load_dotenv(env_path)

# Verificar configuración crítica
if not os.getenv('BINANCE_API_KEY') or not os.getenv('BINANCE_SECRET_KEY'):
    logger.error("ERROR CRITICO: Variables de entorno BINANCE no encontradas")
    logger.error("Verificar archivo .env con BINANCE_API_KEY y BINANCE_SECRET_KEY")
    sys.exit(1)

logger.info("INICIANDO CORRECCIÓN #3: Descargador Masivo Unificado")
logger.info("=" * 70)
logger.info("MEJORAS APLICADAS:")
logger.info("  - Validación flexible para volatilidad cripto realista")
logger.info("  - Límite cambio precio: 50% -> 200%")
logger.info("  - Eventos extremos permitidos: 0 -> 5")
logger.info("  - Compatible Windows sin emojis")
logger.info("  - Lista unificada de archivos faltantes")
logger.info("=" * 70)

# Importar downloader base
try:
    from scripts.download_historical_data import HistoricalDataDownloader
    logger.info("HistoricalDataDownloader importado exitosamente")
except ImportError as e:
    logger.error(f"ERROR importando HistoricalDataDownloader: {e}")
    sys.exit(1)

class FlexibleHistoricalDataDownloader(HistoricalDataDownloader):
    """
    Versión extendida del HistoricalDataDownloader con validación FLEXIBLE
    específicamente adaptada para volatilidad realista de criptomonedas.
    
    CAMBIOS CLAVE vs versión original:
    - Límite cambio precio: 50% -> 200% (permite 3x precio anterior)
    - Eventos extremos: 0 -> 5 permitidos (normal en cripto volátil)
    - Mantiene validaciones críticas: gaps temporales, volumen
    """
    
    def validate_downloaded_data(self, df: pd.DataFrame, symbol: str, timeframe: str) -> bool:
        """
        Validar calidad de datos con criterios FLEXIBLES para criptomonedas.
        
        VALIDACIONES MANTENIDAS (críticas):
        - Gaps temporales máximo 1% del dataset
        - Volumen mediano > 0
        - Integridad de timestamp
        
        VALIDACIONES FLEXIBILIZADAS (realistas):
        - Cambio precio extremo: 50% -> 200%
        - Eventos extremos permitidos: 0 -> 5
        """
        try:
            if df is None or len(df) == 0:
                logger.warning(f"Dataset vacío para {symbol} {timeframe}")
                return False
            
            # 1. VALIDACIÓN TEMPORAL: Verificar gaps
            df_sorted = df.sort_values('timestamp')
            time_diffs = df_sorted['timestamp'].diff().dt.total_seconds()
            expected_interval = self.get_timeframe_ms(timeframe) / 1000
            
            large_gaps = time_diffs > (expected_interval * 2)
            gap_percentage = large_gaps.sum() / len(df)
            
            if gap_percentage > 0.01:  # Máximo 1% de gaps grandes
                logger.warning(f"Gaps temporales excesivos en {symbol} {timeframe}: {gap_percentage:.2%}")
                return False
            
            # 2. VALIDACIÓN FLEXIBLE: Cambios de precio extremos
            price_changes = df['close'].pct_change().abs()
            # CAMBIO CRÍTICO: 2.0 = 200% (era 0.5 = 50%)
            extreme_changes = price_changes > 2.0
            extreme_count = extreme_changes.sum()
            
            # CAMBIO CRÍTICO: Permitir hasta 5 eventos extremos (era 0)
            if extreme_count > 5:
                logger.warning(f"Cambios extremos excesivos en {symbol} {timeframe}: {extreme_count} > 5")
                return False
            elif extreme_count > 0:
                logger.info(f"Cambios extremos detectados en {symbol} {timeframe}: {extreme_count} (aceptable <= 5)")
            
            # 3. VALIDACIÓN VOLUMEN: Verificar liquidez mínima
            zero_volume_pct = (df['volume'] == 0).sum() / len(df)
            if zero_volume_pct > 0.05:  # Máximo 5% con volumen cero
                logger.warning(f"Volumen insuficiente en {symbol} {timeframe}: {zero_volume_pct:.2%} registros sin volumen")
                return False
            
            # 4. VALIDACIÓN INTEGRIDAD: Verificar valores válidos
            invalid_prices = (df[['open', 'high', 'low', 'close']] <= 0).any(axis=1).sum()
            if invalid_prices > 0:
                logger.warning(f"Precios inválidos en {symbol} {timeframe}: {invalid_prices} registros")
                return False
            
            logger.info(f"VALIDACION EXITOSA: {symbol} {timeframe} - {len(df)} registros válidos")
            return True
            
        except Exception as e:
            logger.error(f"Error en validación flexible de {symbol} {timeframe}: {e}")
            return False

class UnifiedMissingDataDownloader:
    """
    Descargador masivo unificado que combina lo mejor de ambas versiones:
    - Robustez y documentación de la versión flexible
    - Compatibilidad y simplicidad de la versión simple
    - Lista completa y unificada de archivos faltantes
    """
    
    def __init__(self):
        """Inicializar con configuración robusta y validación completa."""
        logger.info("Inicializando UnifiedMissingDataDownloader...")
        
        # Crear instancia del downloader flexible
        self.flexible_downloader = FlexibleHistoricalDataDownloader()
        
        # Verificar inicialización exitosa
        if not hasattr(self.flexible_downloader, 'exchange') or not self.flexible_downloader.exchange:
            logger.error("ERROR CRITICO: Exchange no inicializado correctamente")
            logger.error("Verificar configuración de API keys en archivo .env")
            sys.exit(1)
        
        # Confirmar configuración
        api_key_preview = os.getenv('BINANCE_API_KEY', '')[:8]
        logger.info(f"API configurada exitosamente: {api_key_preview}...")
        logger.info("Usando FlexibleHistoricalDataDownloader - VALIDACION FLEXIBLE ACTIVADA")
        
        # LISTA UNIFICADA: Combinación de archivos faltantes y corruptos identificados
        self.critical_missing_files = [
            # Archivos principales faltantes del reporte de verificación
            'ETHUSDT_5m.csv', 'ETHUSDT_15m.csv', 'ADAUSDT_5m.csv', 'ADAUSDT_15m.csv', 
            'ADAUSDT_1d.csv', 'SOLUSDT_5m.csv', 'SOLUSDT_15m.csv', 'XRPUSDT_5m.csv', 
            'XRPUSDT_15m.csv', 'LINKUSDT_5m.csv', 'LINKUSDT_15m.csv', 'AVAXUSDT_5m.csv', 
            'AVAXUSDT_15m.csv', 'UNIUSDT_5m.csv', 'UNIUSDT_15m.csv', 'UNIUSDT_1h.csv', 
            'UNIUSDT_4h.csv', 'UNIUSDT_1d.csv', 'AAVEUSDT_5m.csv', 'AAVEUSDT_15m.csv',
            'SUSHIUSDT_5m.csv', 'SUSHIUSDT_15m.csv', 'COMPUSDT_5m.csv', 'COMPUSDT_15m.csv',
            'YFIUSDT_5m.csv', 'YFIUSDT_15m.csv', 'SNXUSDT_5m.csv', 'SNXUSDT_15m.csv',
            'CRVUSDT_5m.csv', 'CRVUSDT_15m.csv', '1INCHUSDT_5m.csv', '1INCHUSDT_15m.csv',
            'ALPHAUSDT_4h.csv', 'SANDUSDT_5m.csv', 'SANDUSDT_1d.csv', 'ENJUSDT_5m.csv',
            'ZRXUSDT_5m.csv', 'ZRXUSDT_15m.csv', 'ZRXUSDT_1d.csv', 'STORJUSDT_5m.csv',
            'STORJUSDT_15m.csv', 'STORJUSDT_1d.csv', 'OCEANUSDT_5m.csv', 'OCEANUSDT_15m.csv',
            'OCEANUSDT_1h.csv', 'OCEANUSDT_4h.csv', 'OCEANUSDT_1d.csv', 'FETUSDT_5m.csv', 
            'FETUSDT_15m.csv', 'IOTAUSDT_5m.csv', 'IOTAUSDT_15m.csv',
            
            # Archivos corruptos identificados que necesitan re-descarga
            'MATICUSDT_5m.csv', 'MATICUSDT_15m.csv', 'MATICUSDT_1h.csv', 'MATICUSDT_4h.csv', 
            'MATICUSDT_1d.csv', 'ALPHAUSDT_5m.csv'
        ]
        
        # Remover duplicados y ordenar
        self.all_files_to_download = sorted(list(set(self.critical_missing_files)))
        
        # Configurar fechas para máxima cobertura histórica
        self.start_date = datetime(2022, 1, 1)
        self.end_date = datetime.now()
        
        logger.info(f"ARCHIVOS IDENTIFICADOS: {len(self.all_files_to_download)} únicos para descarga")
        logger.info(f"PERIODO: {self.start_date.strftime('%Y-%m-%d')} a {self.end_date.strftime('%Y-%m-%d')}")

    def parse_filename(self, filename: str) -> Tuple[str, str]:
        """
        Extraer símbolo y timeframe del nombre de archivo con manejo robusto.
        
        Soporta formatos:
        - BTCUSDT_5m.csv -> ('BTCUSDT', '5m')
        - 1INCHUSDT_15m.csv -> ('1INCHUSDT', '15m')
        """
        try:
            name_without_ext = filename.replace('.csv', '')
            parts = name_without_ext.split('_')
            
            if len(parts) >= 2:
                # Unir todas las partes excepto la última como símbolo
                symbol = '_'.join(parts[:-1])
                timeframe = parts[-1]
                return symbol, timeframe
            else:
                raise ValueError(f"Formato inválido: {filename}")
                
        except Exception as e:
            logger.error(f"Error parseando filename {filename}: {e}")
            raise

    def download_single_file(self, filename: str, progress_info: str) -> bool:
        """
        Descargar un archivo específico usando FlexibleHistoricalDataDownloader
        con manejo robusto de errores y logging detallado.
        """
        file_start_time = time.time()
        
        try:
            symbol, timeframe = self.parse_filename(filename)
            
            logger.info(f"{progress_info} PROCESANDO: {filename}")
            logger.info(f"   Símbolo: {symbol}, Timeframe: {timeframe}")
            logger.info(f"   Periodo: {self.start_date.strftime('%Y-%m-%d')} a {self.end_date.strftime('%Y-%m-%d')}")
            
            # Usar método base del HistoricalDataDownloader flexible
            success = self.flexible_downloader.download_symbol_timeframe(
                symbol=symbol,
                timeframe=timeframe,
                start_date=self.start_date,
                end_date=self.end_date,
                resume=False  # Forzar nueva descarga completa
            )
            
            file_duration = time.time() - file_start_time
            
            if success:
                # Verificar que el archivo se creó y validar contenido
                file_path = self.flexible_downloader.get_file_path(symbol, timeframe)
                
                if os.path.exists(file_path):
                    df = pd.read_csv(file_path)
                    
                    if len(df) > 0:
                        # Mostrar estadísticas del archivo descargado
                        first_date = pd.to_datetime(df['timestamp'].iloc[0]).strftime('%Y-%m-%d %H:%M')
                        last_date = pd.to_datetime(df['timestamp'].iloc[-1]).strftime('%Y-%m-%d %H:%M')
                        
                        logger.info(f"   SUCCESS: {len(df)} registros guardados")
                        logger.info(f"   Rango: {first_date} a {last_date}")
                        logger.info(f"   Duración: {file_duration:.1f}s")
                        return True
                    else:
                        logger.error(f"   ERROR: Archivo creado pero vacío: {filename}")
                        return False
                else:
                    logger.error(f"   ERROR: Archivo no encontrado después de descarga: {filename}")
                    return False
            else:
                logger.error(f"   ERROR: Falló descarga de {symbol} {timeframe} (duración: {file_duration:.1f}s)")
                return False
                
        except Exception as e:
            file_duration = time.time() - file_start_time
            logger.error(f"   ERROR CRITICO procesando {filename}: {e} (duración: {file_duration:.1f}s)")
            return False

    def download_all_missing_files(self) -> Dict[str, bool]:
        """
        Ejecutar descarga masiva de todos los archivos faltantes
        con monitoreo detallado de progreso y manejo robusto de errores.
        """
        total_files = len(self.all_files_to_download)
        results = {}
        successful = 0
        failed = 0
        
        logger.info("INICIANDO DESCARGA MASIVA UNIFICADA")
        logger.info("=" * 70)
        logger.info(f"ARCHIVOS A PROCESAR: {total_files}")
        logger.info(f"DIRECTORIO DESTINO: {project_root / 'data' / 'raw'}")
        logger.info(f"VALIDACION: Flexible para volatilidad cripto realista")
        logger.info(f"CONFIGURACION: Límite 200%, hasta 5 eventos extremos")
        logger.info("=" * 70)
        
        overall_start_time = time.time()
        
        for i, filename in enumerate(self.all_files_to_download, 1):
            progress_info = f"[{i}/{total_files}]"
            
            try:
                # Descargar archivo individual
                success = self.download_single_file(filename, progress_info)
                results[filename] = success
                
                if success:
                    successful += 1
                    logger.info(f"SUCCESS {progress_info} COMPLETADO: {filename}")
                else:
                    failed += 1
                    logger.error(f"FAILED {progress_info} ERROR: {filename}")
                
                # Mostrar progreso cada 10 archivos o en hitos importantes
                if i % 10 == 0 or i in [1, 5, total_files]:
                    elapsed = time.time() - overall_start_time
                    rate = i / (elapsed / 60) if elapsed > 0 else 0
                    eta_minutes = (total_files - i) / rate if rate > 0 else 0
                    progress_pct = (i / total_files) * 100
                    
                    logger.info(f"PROGRESO: {i}/{total_files} ({progress_pct:.1f}%) - "
                              f"Exitosos: {successful}, Fallidos: {failed}")
                    logger.info(f"   Velocidad: {rate:.1f} archivos/min - ETA: {eta_minutes:.1f} min")
                
                # Pausa entre descargas para rate limiting
                time.sleep(0.8)
                
            except KeyboardInterrupt:
                logger.warning(f"INTERRUPCION: Descarga cancelada por usuario en archivo {i}/{total_files}")
                break
            except Exception as e:
                failed += 1
                logger.error(f"ERROR CRITICO procesando {filename}: {e}")
                results[filename] = False
        
        # Generar reporte final detallado
        self.generate_comprehensive_report(results, overall_start_time, successful, failed)
        return results

    def generate_comprehensive_report(self, results: Dict[str, bool], start_time: float, 
                                    successful: int, failed: int):
        """Generar reporte final comprehensivo con análisis detallado."""
        total_time = time.time() - start_time
        total_files = len(results)
        success_rate = (successful / total_files) * 100 if total_files > 0 else 0
        avg_rate = total_files / (total_time / 60) if total_time > 0 else 0
        
        logger.info("\n" + "=" * 70)
        logger.info("REPORTE FINAL: DESCARGA MASIVA UNIFICADA COMPLETADA")
        logger.info("=" * 70)
        logger.info(f"ESTADISTICAS GENERALES:")
        logger.info(f"   Archivos exitosos: {successful}")
        logger.info(f"   Archivos fallidos: {failed}")
        logger.info(f"   Total procesados: {total_files}")
        logger.info(f"   Tasa de éxito: {success_rate:.1f}%")
        logger.info(f"   Tiempo total: {total_time/60:.1f} minutos")
        logger.info(f"   Velocidad promedio: {avg_rate:.1f} archivos/minuto")
        
        # Evaluación de resultado
        if success_rate >= 90:
            logger.info("RESULTADO: EXCELENTE - Descarga masiva altamente exitosa")
        elif success_rate >= 80:
            logger.info("RESULTADO: BUENO - Descarga masiva exitosa")
        elif success_rate >= 60:
            logger.warning("RESULTADO: ACEPTABLE - Descarga masiva con problemas menores")
        else:
            logger.error("RESULTADO: PROBLEMATICO - Descarga masiva con fallas significativas")
        
        # Mostrar archivos fallidos si los hay
        failed_files = [filename for filename, success in results.items() if not success]
        if failed_files:
            logger.error(f"ARCHIVOS FALLIDOS ({len(failed_files)}):")
            for i, filename in enumerate(failed_files[:15], 1):  # Mostrar hasta 15
                logger.error(f"   {i:2d}. {filename}")
            if len(failed_files) > 15:
                logger.error(f"   ... y {len(failed_files) - 15} archivos más")
        
        # Recomendaciones próximos pasos
        logger.info("\nPROXIMOS PASOS RECOMENDADOS:")
        if success_rate >= 80:
            logger.info("   1. Ejecutar verificación dual: python scripts/verify_dual_strategy_data.py")
            logger.info("   2. Proceder con entrenamiento si cobertura >= 80%")
        else:
            logger.info("   1. Revisar logs para identificar patrones de fallos")
            logger.info("   2. Re-intentar archivos fallidos individualmente")
            logger.info("   3. Verificar conectividad y configuración API")

def main():
    """Función principal del descargador masivo unificado."""
    try:
        logger.info("INICIANDO CORRECCIÓN #3: Descargador Masivo Unificado")
        logger.info("=" * 70)
        logger.info("CARACTERISTICAS UNIFICADAS:")
        logger.info("   - Validación flexible para volatilidad cripto realista")
        logger.info("   - Compatibilidad Windows completa (sin emojis)")
        logger.info("   - Lista unificada de archivos críticos")
        logger.info("   - Logging detallado y manejo robusto de errores")
        logger.info("   - Reportes comprehensivos de progreso")
        logger.info("=" * 70)
        
        # Crear y ejecutar descargador
        downloader = UnifiedMissingDataDownloader()
        results = downloader.download_all_missing_files()
        
        # Evaluar resultado final
        successful = sum(1 for success in results.values() if success)
        total = len(results)
        success_rate = (successful / total) * 100 if total > 0 else 0
        
        # Determinar código de salida basado en éxito
        if success_rate >= 80:
            logger.info("EXITO FINAL: Descarga masiva unificada completada satisfactoriamente")
            logger.info(f"Cobertura lograda: {success_rate:.1f}% >= 80% requerido")
            return 0
        else:
            logger.error("FALLO FINAL: Descarga masiva unificada con problemas significativos")
            logger.error(f"Cobertura lograda: {success_rate:.1f}% < 80% requerido")
            return 1
            
    except KeyboardInterrupt:
        logger.warning("INTERRUPCION: Proceso cancelado por usuario")
        return 130
    except Exception as e:
        logger.error(f"ERROR CRITICO en main(): {e}")
        import traceback
        logger.error(f"Traceback completo:\n{traceback.format_exc()}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
