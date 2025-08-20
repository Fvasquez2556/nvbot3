#!/usr/bin/env python3
"""
NvBot3 - Validador de Setup
===========================

Script para validar que el setup del proyecto está correcto y todas las
dependencias funcionan apropiadamente.

Uso:
    # Primero activar el entorno virtual
    nvbot3_env\Scripts\activate
    # Luego ejecutar validación
    python scripts/validate_setup.py

Autor: NvBot3 Team
Fecha: Agosto 2025
"""

import sys
import os
import importlib
from pathlib import Path
import yaml


def check_virtual_environment():
    """Verificar que el entorno virtual está activo."""
    print("🔍 Verificando entorno virtual...")
    
    if 'nvbot3_env' in sys.executable:
        print("✅ Entorno virtual nvbot3_env activo")
        print(f"   Python path: {sys.executable}")
        return True
    else:
        print("❌ Entorno virtual nvbot3_env NO está activo")
        print("   Por favor ejecuta: nvbot3_env\\Scripts\\activate")
        return False


def check_dependencies():
    """Verificar que todas las dependencias están instaladas."""
    print("\n📦 Verificando dependencias...")
    
    required_packages = {
        'numpy': 'Computación numérica',
        'pandas': 'Manipulación de datos',
        'sklearn': 'Machine Learning',
        'xgboost': 'Gradient Boosting',
        'tensorflow': 'Deep Learning',
        'ccxt': 'Conectores de exchanges',
        'ta': 'Análisis técnico',
        'tqdm': 'Barras de progreso',
        'yaml': 'Configuración YAML',
        'dotenv': 'Variables de entorno'
    }
    
    missing_packages = []
    
    for package, description in required_packages.items():
        try:
            if package == 'sklearn':
                importlib.import_module('sklearn')
            elif package == 'dotenv':
                importlib.import_module('dotenv')
            else:
                importlib.import_module(package)
            print(f"✅ {package:12} - {description}")
        except ImportError:
            print(f"❌ {package:12} - {description} (NO INSTALADO)")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n❌ Paquetes faltantes: {', '.join(missing_packages)}")
        print("   Ejecuta: pip install -r requirements.txt")
        return False
    else:
        print("\n✅ Todas las dependencias están instaladas")
        return True


def check_directory_structure():
    """Verificar que la estructura de directorios es correcta."""
    print("\n📁 Verificando estructura de directorios...")
    
    required_dirs = [
        'src',
        'src/data',
        'src/models', 
        'src/analysis',
        'src/utils',
        'config',
        'data',
        'data/raw',
        'data/processed',
        'data/models',
        'scripts',
        'tests'
    ]
    
    missing_dirs = []
    
    for directory in required_dirs:
        if Path(directory).exists():
            print(f"✅ {directory}")
        else:
            print(f"❌ {directory} (FALTANTE)")
            missing_dirs.append(directory)
    
    if missing_dirs:
        print(f"\n❌ Directorios faltantes: {', '.join(missing_dirs)}")
        return False
    else:
        print("\n✅ Estructura de directorios correcta")
        return True


def check_config_files():
    """Verificar que los archivos de configuración existen."""
    print("\n⚙️  Verificando archivos de configuración...")
    
    config_files = {
        'requirements.txt': 'Dependencias de Python',
        'config/training_config.yaml': 'Configuración de entrenamiento',
        '.env.example': 'Ejemplo de variables de entorno'
    }
    
    missing_files = []
    
    for file_path, description in config_files.items():
        if Path(file_path).exists():
            print(f"✅ {file_path:30} - {description}")
        else:
            print(f"❌ {file_path:30} - {description} (FALTANTE)")
            missing_files.append(file_path)
    
    # Verificar .env (opcional pero recomendado)
    if Path('.env').exists():
        print("✅ .env                          - Variables de entorno (configurado)")
    else:
        print("⚠️  .env                          - Variables de entorno (no configurado)")
        print("   Recomendación: Copia .env.example a .env y configura tus valores")
    
    if missing_files:
        print(f"\n❌ Archivos faltantes: {', '.join(missing_files)}")
        return False
    else:
        print("\n✅ Archivos de configuración presentes")
        return True


def check_config_validity():
    """Verificar que la configuración YAML es válida."""
    print("\n📋 Verificando validez de configuración...")
    
    try:
        with open('config/training_config.yaml', 'r', encoding='utf-8') as file:
            config = yaml.safe_load(file)
        
        # Verificar secciones principales
        required_sections = ['data', 'models', 'validation', 'api', 'download']
        missing_sections = []
        
        for section in required_sections:
            if section in config:
                print(f"✅ Sección '{section}' presente")
            else:
                print(f"❌ Sección '{section}' faltante")
                missing_sections.append(section)
        
        # Verificar configuración de datos
        if 'data' in config:
            data_config = config['data']
            if 'symbols' in data_config and len(data_config['symbols']) > 0:
                print(f"✅ Símbolos configurados: {', '.join(data_config['symbols'])}")
            else:
                print("❌ No hay símbolos configurados")
                missing_sections.append('symbols')
            
            if 'timeframes' in data_config and len(data_config['timeframes']) > 0:
                print(f"✅ Timeframes configurados: {', '.join(data_config['timeframes'])}")
            else:
                print("❌ No hay timeframes configurados")
                missing_sections.append('timeframes')
        
        if missing_sections:
            print(f"\n❌ Configuración incompleta: {', '.join(missing_sections)}")
            return False
        else:
            print("\n✅ Configuración YAML válida y completa")
            return True
            
    except yaml.YAMLError as e:
        print(f"❌ Error en formato YAML: {e}")
        return False
    except FileNotFoundError:
        print("❌ Archivo config/training_config.yaml no encontrado")
        return False


def test_import_main_module():
    """Probar importar el módulo principal de descarga."""
    print("\n🧪 Probando importación de módulos...")
    
    try:
        # Agregar el directorio actual al path para imports
        sys.path.insert(0, '.')
        
        # Intentar importar el módulo de descarga
        from scripts.download_historical_data import HistoricalDataDownloader
        print("✅ HistoricalDataDownloader importado correctamente")
        
        # Intentar crear instancia (sin ejecutar descarga)
        downloader = HistoricalDataDownloader()
        print("✅ HistoricalDataDownloader instanciado correctamente")
        
        return True
        
    except ImportError as e:
        print(f"❌ Error importando módulo: {e}")
        return False
    except Exception as e:
        print(f"❌ Error creando instancia: {e}")
        return False


def show_next_steps():
    """Mostrar los próximos pasos recomendados."""
    print("\n🎯 Próximos Pasos Recomendados")
    print("=" * 35)
    
    print("1. 📝 Configurar variables de entorno:")
    print("   - Copia .env.example a .env")
    print("   - Configura tus API keys de Binance (opcional para datos históricos)")
    
    print("\n2. 🔄 Probar descarga de datos:")
    print("   - python scripts/download_historical_data.py")
    
    print("\n3. 📊 Verificar datos descargados:")
    print("   - Revisar carpeta data/raw/ para archivos CSV")
    
    print("\n4. 🧪 Ejecutar tests:")
    print("   - python -m pytest tests/ (cuando estén implementados)")


def main():
    """Función principal de validación."""
    print("🤖 NvBot3 - Validador de Setup")
    print("=" * 50)
    
    all_checks_passed = True
    
    # Ejecutar todas las verificaciones
    checks = [
        check_virtual_environment,
        check_dependencies,
        check_directory_structure,
        check_config_files,
        check_config_validity,
        test_import_main_module
    ]
    
    for check_function in checks:
        if not check_function():
            all_checks_passed = False
    
    # Resultado final
    print("\n" + "=" * 50)
    if all_checks_passed:
        print("🎉 ¡Todas las verificaciones pasaron!")
        print("✅ El proyecto está correctamente configurado")
        show_next_steps()
    else:
        print("❌ Algunas verificaciones fallaron")
        print("⚠️  Por favor corrige los problemas indicados arriba")
    
    return all_checks_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
