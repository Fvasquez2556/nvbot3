"""
🔧 Descarga inteligente SOLO para las 30 monedas de entrenamiento.
Las 60 de monitoreo NO requieren datos históricos.
Actualizado para trabajar con el verificador combinado.
"""

import os
import sys
import yaml
from pathlib import Path

# Agregar el directorio scripts al path para importaciones
scripts_path = Path(__file__).parent
sys.path.insert(0, str(scripts_path))

# Importar desde el mismo directorio scripts
from verify_dual_strategy_data import DualStrategyVerifier
from download_historical_data import HistoricalDataDownloader
import logging
from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TrainingDataDownloader:
    def __init__(self):
        # Usar modo flexible para entrenamiento
        self.verifier = DualStrategyVerifier(validation_mode='flexible')
        self.downloader = HistoricalDataDownloader()
        
    def run_verification(self):
        """Ejecutar verificación completa y obtener archivos faltantes."""
        try:
            # Verificar datos de entrenamiento
            if not self.verifier.verify_training_data_files():
                logger.error("❌ Error verificando datos de entrenamiento")
                return False
            
            # Verificar disponibilidad de monitoreo
            if not self.verifier.verify_monitoring_symbols_availability():
                logger.error("❌ Error verificando símbolos de monitoreo")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error en verificación: {e}")
            return False
        
    def download_missing_training_data_only(self):
        """Descargar solo datos faltantes para entrenamiento."""
        logger.info("🎯 INICIANDO DESCARGA PARA ENTRENAMIENTO (30 MONEDAS)")
        logger.info("🔧 Modo: FLEXIBLE - Criterios optimizados para entrenamiento")
        
        # 1. Verificar estado actual
        success = self.run_verification()
        
        if success and not self.verifier.missing_training_files and not self.verifier.corrupted_training_files:
            logger.info("✅ Todos los datos de entrenamiento están completos.")
            return True
        
        # 2. Obtener archivos de entrenamiento faltantes
        training_files_to_download = self.verifier.missing_training_files + self.verifier.corrupted_training_files
        
        if not training_files_to_download:
            logger.info("✅ No hay archivos de entrenamiento para descargar.")
            return True
        
        logger.info(f"📥 Descargando {len(training_files_to_download)} archivos para entrenamiento...")
        logger.info("💡 NOTA: Monedas de monitoreo NO requieren datos históricos")
        
        # 3. Descargar solo archivos de entrenamiento
        successful_downloads = 0
        failed_downloads = 0
        
        for i, filename in enumerate(training_files_to_download, 1):
            try:
                symbol, timeframe_ext = filename.split('_', 1)
                timeframe = timeframe_ext.replace('.csv', '')
                
                logger.info(f"📥 [{i}/{len(training_files_to_download)}] Descargando {symbol} {timeframe}...")
                
                # Configurar fechas para descarga (últimos 2 años)
                end_date = datetime.now()
                start_date = end_date - timedelta(days=730)  # 2 años
                
                # Usar el downloader existente
                success = self.downloader.download_symbol_timeframe(symbol, timeframe, start_date, end_date)
                
                if success:
                    successful_downloads += 1
                    logger.info(f"✅ [{i}/{len(training_files_to_download)}] Completado: {filename}")
                else:
                    failed_downloads += 1
                    logger.error(f"⚠️ [{i}/{len(training_files_to_download)}] Falló: {filename}")
                
            except Exception as e:
                failed_downloads += 1
                logger.error(f"❌ Error descargando {filename}: {e}")
                continue
        
        # Reporte de descarga
        logger.info(f"\n📊 REPORTE DESCARGA:")
        logger.info(f"   ✅ Exitosos: {successful_downloads}")
        logger.info(f"   ❌ Fallidos: {failed_downloads}")
        logger.info(f"   📈 Tasa éxito: {successful_downloads/len(training_files_to_download)*100:.1f}%")
        
        # 4. Verificar nuevamente después de descarga
        logger.info("\n🔍 Verificando datos después de descarga...")
        
        # Crear nuevo verificador para re-verificar
        final_verifier = DualStrategyVerifier(validation_mode='flexible')
        final_success = final_verifier.verify_training_data_files()
        
        if final_success:
            # Generar reporte final
            final_verifier.verify_monitoring_symbols_availability()
            success = final_verifier.generate_detailed_report()
            
            if success:
                logger.info("🎉 ¡DESCARGA COMPLETADA! Dual strategy lista.")
                return True
            else:
                logger.warning("⚠️ Descarga completada pero algunos criterios no se cumplen.")
                return False
        else:
            logger.error("⚠️ Algunos archivos de entrenamiento aún faltan.")
            return False

def main():
    downloader = TrainingDataDownloader()
    success = downloader.download_missing_training_data_only()
    return 0 if success else 1

if __name__ == "__main__":
    exit(main())
