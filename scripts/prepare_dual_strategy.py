"""
Preparación completa para estrategia dual:
- Verificar/descargar 30 monedas entrenamiento
- Verificar disponibilidad 60 monedas monitoreo  
- Confirmar readiness para Model Trainer
"""

import subprocess
import sys
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_command(command, description):
    """Ejecutar comando y manejar errores."""
    logger.info(f"🔄 {description}...")
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True, shell=True)
        logger.info(f"✅ {description} completado exitosamente.")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ {description} falló:")
        logger.error(f"   Error: {e}")
        if e.stdout:
            logger.error(f"   Output: {e.stdout}")
        if e.stderr:
            logger.error(f"   Error Output: {e.stderr}")
        return False

def main():
    logger.info("🚀 PREPARACIÓN DUAL STRATEGY PARA MODEL TRAINER")
    logger.info("="*60)
    logger.info("📚 30 monedas para entrenamiento (datos históricos)")
    logger.info("📡 60 monedas DIFERENTES para monitoreo (WebSocket)")
    logger.info("="*60)
    
    # 1. Verificar entorno virtual
    if 'nvbot3_env' not in sys.executable:
        logger.error("❌ Entorno virtual nvbot3_env no está activo!")
        logger.error("   Ejecutar: nvbot3_env\\Scripts\\activate")
        return 1
    
    logger.info("✅ Entorno virtual nvbot3_env activo")
    
    # 2. Verificar estrategia dual
    step1 = run_command(
        "python scripts/verify_dual_strategy_data.py",
        "Verificando dual strategy (30 entrenamiento + 60 monitoreo)"
    )
    
    # 3. Descargar datos de entrenamiento faltantes (si es necesario)
    if not step1:
        step2 = run_command(
            "python scripts/download_training_data_only.py", 
            "Descargando datos SOLO para entrenamiento (30 monedas)"
        )
        
        if not step2:
            logger.error("💥 No se pudieron descargar datos de entrenamiento")
            return 1
    
    # 4. Verificación final dual
    step3 = run_command(
        "python scripts/verify_dual_strategy_data.py",
        "Verificación final dual strategy"
    )
    
    if not step3:
        logger.error("💥 Verificación dual final fallida")
        return 1
    
    # 5. Confirmar dual strategy readiness
    logger.info("\n" + "="*60)
    logger.info("🎉 ¡DUAL STRATEGY PREPARADA EXITOSAMENTE!")
    logger.info("="*60)
    logger.info("✅ Entorno virtual activo")
    logger.info("✅ 30 monedas entrenamiento: datos históricos completos")
    logger.info("✅ 60 monedas monitoreo: símbolos disponibles en Binance")
    logger.info("✅ Separación completa: 0 overlap entre entrenamiento/monitoreo")
    logger.info("✅ Sistema optimizado para laptop (60 streams)")
    logger.info("✅ Anti-overfitting: prueba real de generalización")
    logger.info("\n🚀 LISTO PARA: python scripts/model_trainer.py")
    
    return 0

if __name__ == "__main__":
    exit(main())
