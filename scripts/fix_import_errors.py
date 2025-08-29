#!/usr/bin/env python3
# scripts/fix_import_errors.py
"""
Script para diagnosticar y solucionar errores de importación
en el sistema de retroalimentación de NvBot3
"""

import os
import sys
from pathlib import Path

def check_file_exists(filepath, description):
    """Verifica si un archivo existe y reporta el estado"""
    path = Path(filepath)
    if path.exists():
        size = path.stat().st_size
        print(f"   ✅ {description}: {filepath} ({size} bytes)")
        return True
    else:
        print(f"   ❌ {description}: {filepath} - NO EXISTE")
        return False

def create_missing_files():
    """Crea archivos faltantes esenciales"""
    
    print("🔧 Creando archivos faltantes...")
    
    files_to_create = {
        "web_dashboard/__init__.py": "# NvBot3 Web Dashboard Package\n",
        "web_dashboard/database/__init__.py": "# NvBot3 Database Package\n",
        "integration/__init__.py": "# NvBot3 Integration Package\n"
    }
    
    for filepath, content in files_to_create.items():
        path = Path(filepath)
        
        # Crear directorio si no existe
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Crear archivo si no existe
        if not path.exists():
            with open(path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"   ✅ Creado: {filepath}")
        else:
            print(f"   ⚠️ Ya existe: {filepath}")

def test_imports():
    """Prueba las importaciones críticas"""
    
    print("🧪 Probando importaciones...")
    
    # Test 1: SignalTracker
    try:
        sys.path.append('web_dashboard/database')
        from web_dashboard.database.signal_tracker import SignalTracker
        tracker = SignalTracker()
        print("   ✅ SignalTracker: Importación exitosa")
        return True
    except ImportError as e:
        print(f"   ❌ SignalTracker: Error de importación - {e}")
        return False
    except Exception as e:
        print(f"   ⚠️ SignalTracker: Importación OK, pero error en inicialización - {e}")
        return True  # Importación funciona, solo hay problema de inicialización

def test_integration_bridge():
    """Prueba el bridge de integración"""
    
    print("🌉 Probando bridge de integración...")
    
    try:
        sys.path.append('integration')
        from web_dashboard.database.signal_tracker import SignalTracker 
        from integration.nvbot3_feedback_bridge import get_tracking_stats

        status = get_tracking_stats()
        print("   ✅ Bridge: Importación exitosa")
        print(f"   📊 Estado: {status}")
        
        return True
        
    except ImportError as e:
        print(f"   ❌ Bridge: Error de importación - {e}")
        return False
    except Exception as e:
        print(f"   ⚠️ Bridge: Error - {e}")
        return False

def fix_permission_issues():
    """Intenta solucionar problemas de permisos"""
    
    print("🔒 Verificando permisos...")
    
    directories = [
        "web_dashboard",
        "web_dashboard/database", 
        "integration"
    ]
    
    for directory in directories:
        path = Path(directory)
        if path.exists():
            if os.access(path, os.R_OK):
                print(f"   ✅ Lectura OK: {directory}")
            else:
                print(f"   ❌ Sin permisos de lectura: {directory}")
            
            if os.access(path, os.W_OK):
                print(f"   ✅ Escritura OK: {directory}")
            else:
                print(f"   ❌ Sin permisos de escritura: {directory}")

def install_missing_dependencies():
    """Verifica e instala dependencias faltantes"""
    
    print("📦 Verificando dependencias...")
    
    required_packages = [
        'sqlite3',  # Parte de Python estándar
        'pandas',
        'flask', 
        'flask_socketio'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'flask_socketio':
                import flask_socketio
            elif package == 'sqlite3':
                import sqlite3
            elif package == 'pandas':
                import pandas
            elif package == 'flask':
                import flask
            
            print(f"   ✅ {package}: Instalado")
            
        except ImportError:
            print(f"   ❌ {package}: Faltante")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n📋 Instalar dependencias faltantes:")
        if 'flask_socketio' in missing_packages:
            print("   pip install flask flask-socketio")
        if 'pandas' in missing_packages:
            print("   pip install pandas")

def comprehensive_diagnosis():
    """Diagnóstico completo del sistema"""
    
    print("🔍 DIAGNÓSTICO COMPLETO DEL SISTEMA")
    print("=" * 50)
    
    # 1. Verificar estructura de archivos
    print("\n1. 📁 Estructura de Archivos:")
    
    critical_files = {
        "web_dashboard/database/signal_tracker.py": "SignalTracker principal",
        "integration/nvbot3_feedback_bridge.py": "Bridge de integración",
        "web_dashboard/app.py": "Aplicación Flask",
        "web_dashboard/templates/dashboard.html": "Template HTML",
        "web_dashboard/__init__.py": "Package init",
        "web_dashboard/database/__init__.py": "Database package init",
        "integration/__init__.py": "Integration package init"
    }
    
    all_files_exist = True
    for filepath, description in critical_files.items():
        exists = check_file_exists(filepath, description)
        if not exists:
            all_files_exist = False
    
    # 2. Crear archivos faltantes
    if not all_files_exist:
        print("\n2. 🔧 Creando archivos faltantes:")
        create_missing_files()
    
    # 3. Verificar permisos
    print("\n3. 🔒 Permisos:")
    fix_permission_issues()
    
    # 4. Verificar dependencias
    print("\n4. 📦 Dependencias:")
    install_missing_dependencies()
    
    # 5. Probar importaciones
    print("\n5. 🧪 Pruebas de Importación:")
    signal_tracker_ok = test_imports()
    bridge_ok = test_integration_bridge()
    
    # 6. Resumen final
    print("\n" + "=" * 50)
    print("📊 RESUMEN DEL DIAGNÓSTICO")
    print("=" * 50)
    
    if all_files_exist and signal_tracker_ok and bridge_ok:
        print("🎉 ¡TODO ESTÁ FUNCIONANDO CORRECTAMENTE!")
        print("✅ Todos los archivos existen")
        print("✅ Todas las importaciones funcionan") 
        print("✅ Sistema listo para usar")
        print("\n🚀 Próximos pasos:")
        print("   1. Ejecutar: python scripts/start_dashboard.py")
        print("   2. Integrar con tu código: from integration.nvbot3_feedback_bridge import track_signal")
    else:
        print("⚠️ SE ENCONTRARON ALGUNOS PROBLEMAS:")
        if not all_files_exist:
            print("❌ Faltan archivos críticos")
        if not signal_tracker_ok:
            print("❌ Error en SignalTracker")
        if not bridge_ok:
            print("❌ Error en Bridge de integración")
        
        print("\n🔧 SOLUCIONES RECOMENDADAS:")
        print("   1. Ejecutar: python scripts/fix_import_errors.py")
        print("   2. Verificar que el entorno virtual esté activo")
        print("   3. Instalar dependencias: pip install flask flask-socketio pandas")
        print("   4. Ejecutar nuevamente este diagnóstico")

def quick_fix():
    """Solución rápida para problemas comunes"""
    
    print("⚡ SOLUCIÓN RÁPIDA DE PROBLEMAS")
    print("=" * 40)
    
    # Crear estructura básica
    create_missing_files()
    
    # Crear SignalTracker básico si no existe
    signal_tracker_path = Path("web_dashboard/database/signal_tracker.py")
    if not signal_tracker_path.exists():
        print("🔧 Creando SignalTracker...")
        # Aquí irían las instrucciones para crear el archivo completo
        print("   ⚠️ Necesitas copiar el SignalTracker completo del artifact")
    
    # Verificar nuevamente
    print("\n🔄 Verificando después de las correcciones...")
    test_imports()

if __name__ == "__main__":
    comprehensive_diagnosis()