#!/usr/bin/env python3
# scripts/start_dashboard.py
"""
Script para iniciar el dashboard web de NvBot3
Ejecutar en una terminal separada mientras tu bot principal está corriendo
"""

import sys
import os
import subprocess
from pathlib import Path
import time

# Ensure the project root is in sys.path
sys.path.append(os.path.abspath('.'))

# Debug block to verify web_dashboard import
try:
    from web_dashboard.database.signal_tracker import SignalTracker
    print("✅ Módulo web_dashboard importado correctamente")
except ModuleNotFoundError as e:
    print(f"❌ Error importando web_dashboard: {e}")

def check_requirements():
    """Verifica que todos los requisitos estén presentes"""
    
    print("🔍 Verificando requisitos del sistema...")
    
    # Verificar estructura de archivos
    required_files = [
        "web_dashboard/app.py",
        "web_dashboard/database/signal_tracker.py",
        "web_dashboard/templates/dashboard.html"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
    
    if missing_files:
        print("❌ Archivos faltantes:")
        for file_path in missing_files:
            print(f"   - {file_path}")
        print("\n🔧 Solución: Ejecuta primero 'python scripts/integrate_feedback_system.py'")
        return False
    
    # Verificar dependencias Python
    required_packages = ['flask', 'flask_socketio', 'pandas']
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'flask_socketio':
                import flask_socketio
                import pkg_resources
                version = pkg_resources.get_distribution("flask-socketio").version
                print(f"   ✅ {package}: Versión {version}")
            elif package == 'flask':
                import flask
                print(f"   ✅ {package}: Versión {flask.__version__}")
            elif package == 'pandas':
                import pandas
                print(f"   ✅ {package}: Versión {pandas.__version__}")
        except ImportError:
            missing_packages.append(package)
            print(f"   ❌ {package}: No instalado")
    
    if missing_packages:
        print(f"\n💡 Instalar dependencias faltantes:")
        if 'flask_socketio' in missing_packages:
            print("   pip install flask flask-socketio")
        if 'pandas' in missing_packages:
            print("   pip install pandas")
        return False
    
    # Verificar base de datos
    db_path = "web_dashboard/database/signals.db"
    if Path(db_path).exists():
        print(f"   ✅ Base de datos: {db_path} (existe)")
    else:
        print(f"   ℹ️ Base de datos: {db_path} (se creará automáticamente)")
    
    print("✅ Todos los requisitos están presentes")
    return True

def create_missing_dirs():
    """Crea directorios faltantes si es necesario"""
    
    dirs_to_create = [
        "web_dashboard/templates",
        "web_dashboard/static/css", 
        "web_dashboard/static/js",
        "web_dashboard/database"
    ]
    
    for dir_path in dirs_to_create:
        Path(dir_path).mkdir(parents=True, exist_ok=True)

def test_signal_tracker():
    """Prueba rápida del SignalTracker antes de iniciar el dashboard"""
    
    print("🧪 Probando SignalTracker...")
    
    try:
        # Cambiar al directorio web_dashboard temporalmente
        original_dir = os.getcwd()
        os.chdir("web_dashboard")
        
        # Agregar al path e importar
        sys.path.append(os.getcwd())
        sys.path.append(os.getcwd())  # Ensure 'web_dashboard' is in the module search path
        from web_dashboard.database.signal_tracker import SignalTracker
        
        # Crear instancia de prueba
        tracker = SignalTracker()
        
        # Volver al directorio original
        os.chdir(original_dir)
        
        print("   ✅ SignalTracker funciona correctamente")
        return True
        
    except Exception as e:
        original_dir = os.getcwd()  # Define original_dir al inicio de la función
        os.chdir(original_dir)  # Asegurar que volvamos al directorio original
        print(f"   ❌ Error en SignalTracker: {e}")
        return False

def start_dashboard():
    """Inicia el dashboard web"""
    
    print("\n🚀 INICIANDO NVBOT3 DASHBOARD")
    print("=" * 50)
    print("🌐 URL del Dashboard: http://localhost:5000")
    print("📡 API de prueba: http://localhost:5000/api/test")
    print("⏹️  Presiona Ctrl+C para detener el servidor")
    print("=" * 50)
    
    original_dir = os.getcwd()  # Initialize with the current working directory
    try:
        # Cambiar al directorio del dashboard
        os.chdir("web_dashboard")
        
        # Iniciar Flask app
        print("\n📊 Iniciando servidor Flask...")
        subprocess.run([sys.executable, "app.py"], check=True)
        
    except KeyboardInterrupt:
        print("\n\n🛑 Dashboard detenido por el usuario")
        print("✅ Servidor web cerrado correctamente")
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error ejecutando app.py: {e}")
        print("🔧 Verifica que web_dashboard/app.py existe y es válido")
        
    except Exception as e:
        print(f"\n❌ Error inesperado: {e}")
        
    finally:
        # Volver al directorio original
        os.chdir(original_dir)

def show_usage_instructions():
    """Muestra instrucciones de uso después de iniciar"""
    
    print("\n📋 INSTRUCCIONES DE USO:")
    print("=" * 30)
    
    print("\n1. 🌐 ACCEDER AL DASHBOARD:")
    print("   Abre tu navegador y ve a: http://localhost:5000")
    
    print("\n2. 🤖 INTEGRAR CON TU BOT:")
    print("   Agrega al inicio de tu código nvbot3:")
    print("   from integration.nvbot3_feedback_bridge import track_signal")
    print("   ")
    print("   En tu función de predicción:")
    print("   if prediction['confidence'] > 0.75:")
    print("       track_signal(symbol, prediction, current_price)")
    
    print("\n3. 📊 MONITOREAR SEÑALES:")
    print("   - Las señales aparecerán automáticamente en el dashboard")
    print("   - Puedes dar feedback sobre cada predicción")
    print("   - Ver estadísticas de performance en tiempo real")
    
    print("\n4. 🔄 ACTUALIZAR DATOS:")
    print("   - El dashboard se actualiza automáticamente")
    print("   - También puedes refrescar manualmente")

def main():
    """Función principal"""
    
    print("🎯 NVBOT3 DASHBOARD LAUNCHER")
    print("=" * 40)
    
    # Verificar que estamos en el directorio correcto
    if not Path("web_dashboard").exists():
        print("❌ Error: No se encuentra el directorio 'web_dashboard'")
        print("🔧 Ejecuta este script desde el directorio raíz de nvbot3")
        return
    
    # Crear directorios faltantes
    create_missing_dirs()
    
    # Verificar requisitos
    if not check_requirements():
        print("\n❌ No se puede iniciar el dashboard")
        print("🔧 Soluciona los problemas indicados arriba y vuelve a intentar")
        return
    
    # Probar SignalTracker
    if not test_signal_tracker():
        print("\n⚠️ Advertencia: Problemas con SignalTracker")
        print("   El dashboard iniciará pero podría no funcionar correctamente")
        print("\n¿Continuar de todas formas? (s/n): ", end="")
        if input().lower() not in ['s', 'si', 'y', 'yes']:
            return
    
    # Mostrar instrucciones
    show_usage_instructions()
    
    print("\n⏳ Iniciando en 3 segundos...")
    time.sleep(3)
    
    # Iniciar dashboard
    start_dashboard()

if __name__ == "__main__":
    main()