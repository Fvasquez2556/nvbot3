#!/usr/bin/env python3
"""
🧪 Script de prueba simplificado para verificar el estado del sistema
"""

import os
import sys
import yaml
import logging
from pathlib import Path
from dotenv import load_dotenv

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_environment():
    """Probar el entorno básico."""
    print("🧪 PROBANDO ENTORNO BÁSICO")
    print("=" * 50)
    
    # 1. Verificar Python y entorno virtual
    print(f"🐍 Python: {sys.executable}")
    print(f"📁 Working dir: {os.getcwd()}")
    
    # 2. Verificar archivo .env
    load_dotenv()
    api_key = os.getenv('BINANCE_API_KEY')
    if api_key:
        print(f"🔑 API Key: {api_key[:10]}...")
        logger.info("✅ Archivo .env cargado correctamente")
    else:
        print("❌ No se pudo cargar BINANCE_API_KEY del .env")
        logger.error("❌ Problema con archivo .env")
    
    # 3. Verificar archivos de configuración
    config_dir = Path('config')
    training_config_path = config_dir / 'training_config.yaml'
    monitoring_config_path = config_dir / 'monitoring_config.yaml'
    
    if training_config_path.exists():
        try:
            with open(training_config_path, 'r', encoding='utf-8') as f:
                training_config = yaml.safe_load(f)
            print(f"✅ Training config: {len(training_config.get('symbols', []))} símbolos")
            logger.info("✅ training_config.yaml cargado correctamente")
        except Exception as e:
            print(f"❌ Error cargando training_config.yaml: {e}")
            logger.error(f"❌ Error en training_config.yaml: {e}")
    else:
        print("❌ No existe training_config.yaml")
        logger.error("❌ Archivo training_config.yaml no encontrado")
    
    if monitoring_config_path.exists():
        try:
            with open(monitoring_config_path, 'r', encoding='utf-8') as f:
                monitoring_config = yaml.safe_load(f)
            print(f"✅ Monitoring config: {len(monitoring_config.get('symbols', []))} símbolos")
            logger.info("✅ monitoring_config.yaml cargado correctamente")
        except Exception as e:
            print(f"❌ Error cargando monitoring_config.yaml: {e}")
            logger.error(f"❌ Error en monitoring_config.yaml: {e}")
    else:
        print("❌ No existe monitoring_config.yaml")
        logger.error("❌ Archivo monitoring_config.yaml no encontrado")
    
    # 4. Verificar directorio de datos
    data_dir = Path('data/raw')
    if data_dir.exists():
        csv_files = list(data_dir.glob('*.csv'))
        print(f"📊 Archivos de datos: {len(csv_files)} archivos CSV")
        logger.info(f"✅ Directorio data/raw existe con {len(csv_files)} archivos")
    else:
        print("❌ No existe directorio data/raw")
        logger.error("❌ Directorio data/raw no encontrado")
    
    # 5. Probar importaciones críticas
    try:
        import pandas as pd
        import requests
        print("✅ Importaciones básicas: pandas, requests OK")
        logger.info("✅ Dependencias básicas importadas correctamente")
    except ImportError as e:
        print(f"❌ Error importando dependencias: {e}")
        logger.error(f"❌ Error en importaciones: {e}")
    
    print("\n" + "=" * 50)
    logger.info("🎯 Prueba de entorno completada")

if __name__ == "__main__":
    test_environment()
