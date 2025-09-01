#!/usr/bin/env python3
# scripts/fix_import_errors.py
"""
Script para diagnosticar y solucionar errores de importación
en el sistema de retroalimentación de NvBot3
"""

import os
import sys
from pathlib import Path
import subprocess

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
    
    # Crear directorios necesarios
    dirs_to_create = [
        "web_dashboard",
        "web_dashboard/database", 
        "web_dashboard/templates",
        "web_dashboard/static/css",
        "web_dashboard/static/js",
        "integration",
        "scripts"
    ]
    
    for directory in dirs_to_create:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"   ✅ Directorio: {directory}")
    
    # Crear archivos __init__.py
    init_files = {
        "web_dashboard/__init__.py": "# NvBot3 Web Dashboard Package\n__version__ = '1.0.0'\n",
        "web_dashboard/database/__init__.py": "# NvBot3 Database Package\n",
        "integration/__init__.py": "# NvBot3 Integration Package\n"
    }
    
    for filepath, content in init_files.items():
        path = Path(filepath)
        if not path.exists():
            with open(path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"   ✅ Creado: {filepath}")
        else:
            print(f"   ℹ️ Ya existe: {filepath}")

    # Crear archivo CSS básico si no existe
    css_path = Path("web_dashboard/static/css/styles.css")
    if not css_path.exists():
        css_content = """/* Estilos básicos para NvBot3 Dashboard */
body {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    background-color: #f8f9fa;
    margin: 0;
    padding: 0;
}

.container {
    max-width: 1200px;
    margin: 0 auto;
    padding: 20px;
}

.signal-card {
    background: white;
    border-radius: 10px;
    padding: 20px;
    margin-bottom: 20px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
}

.signal-status {
    padding: 5px 10px;
    border-radius: 5px;
    color: white;
    font-weight: bold;
}

.status-monitoring { background-color: #17a2b8; }
.status-completed { background-color: #28a745; }
.status-failed { background-color: #dc3545; }
"""
        with open(css_path, 'w', encoding='utf-8') as f:
            f.write(css_content)
        print(f"   ✅ Creado: {css_path}")

    # Crear archivo JS básico si no existe
    js_path = Path("web_dashboard/static/js/scripts.js")
    if not js_path.exists():
        js_content = """// Scripts para NvBot3 Dashboard
function refreshData() {
    console.log('Refrescando datos...');
    location.reload();
}

function submitFeedback(signalId, feedbackType) {
    const data = {
        signal_id: signalId,
        feedback_type: feedbackType,
        comments: prompt('Comentarios adicionales (opcional):') || ''
    };
    
    fetch('/api/feedback', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(data)
    })
    .then(response => response.json())
    .then(data => {
        if (data.status === 'success') {
            alert('¡Feedback guardado correctamente!');
            location.reload();
        } else {
            alert('Error: ' + data.message);
        }
    })
    .catch(error => {
        console.error('Error:', error);
        alert('Error enviando feedback');
    });
}

// Auto-refresh cada 30 segundos
setInterval(refreshData, 30000);
"""
        with open(js_path, 'w', encoding='utf-8') as f:
            f.write(js_content)
        print(f"   ✅ Creado: {js_path}")

def test_imports():
    """Prueba las importaciones críticas"""
    
    print("🧪 Probando importaciones...")
    
    # Test 1: SignalTracker
    try:
        original_dir = os.getcwd()
        os.chdir('web_dashboard')
        sys.path.append(os.getcwd())
        
        from database.signal_tracker import SignalTracker
        tracker = SignalTracker()
        os.chdir(original_dir)
        
        print("   ✅ SignalTracker: Importación e inicialización exitosa")
        return True
        
    except ImportError as e:
        os.chdir(original_dir)
        print(f"   ❌ SignalTracker: Error de importación - {e}")
        return False
    except Exception as e:
        os.chdir(original_dir)
        print(f"   ⚠️ SignalTracker: Importación OK, error en inicialización - {e}")
        return True  # Importación funciona

def test_integration_bridge():
    """Prueba el bridge de integración"""
    
    print("🌉 Probando bridge de integración...")
    
    try:
        sys.path.append('integration')
        from integration.nvbot3_feedback_bridge import track_signal, get_tracking_stats
        
        # Intentar una función básica
        stats = get_tracking_stats()
        print("   ✅ Bridge: Importación y funciones básicas OK")
        print(f"   📊 Test stats: {stats}")
        return True
        
    except ImportError as e:
        print(f"   ❌ Bridge: Error de importación - {e}")
        return False
    except Exception as e:
        print(f"   ⚠️ Bridge: Error en ejecución - {e}")
        return False

def fix_permission_issues():
    """Intenta solucionar problemas de permisos"""
    
    print("🔑 Verificando permisos...")
    
    directories = [
        "web_dashboard",
        "web_dashboard/database", 
        "integration",
        "scripts"
    ]
    
    all_permissions_ok = True
    
    for directory in directories:
        path = Path(directory)
        if path.exists():
            # Verificar lectura
            if os.access(path, os.R_OK):
                print(f"   ✅ Lectura OK: {directory}")
            else:
                print(f"   ❌ Sin permisos de lectura: {directory}")
                all_permissions_ok = False
            
            # Verificar escritura
            if os.access(path, os.W_OK):
                print(f"   ✅ Escritura OK: {directory}")
            else:
                print(f"   ❌ Sin permisos de escritura: {directory}")
                all_permissions_ok = False
        else:
            print(f"   ⚠️ Directorio no existe: {directory}")
    
    return all_permissions_ok

def install_missing_dependencies():
    """Verifica e instala dependencias faltantes"""
    
    print("📦 Verificando dependencias...")
    
    # Dependencias críticas
    dependencies = {
        'sqlite3': 'sqlite3',  # Incluido en Python estándar
        'pandas': 'pandas',
        'flask': 'flask',
        'flask_socketio': 'flask-socketio'
    }
    
    missing_packages = []
    
    for import_name, package_name in dependencies.items():
        try:
            if import_name == 'flask_socketio':
                import flask_socketio
                print(f"   ✅ flask-socketio: Versión {flask_socketio.__version__}")
            elif import_name == 'sqlite3':
                import sqlite3
                print(f"   ✅ sqlite3: Incluido en Python estándar")
            elif import_name == 'pandas':
                import pandas as pd
                print(f"   ✅ pandas: Versión {pd.__version__}")
            elif import_name == 'flask':
                import flask
                print(f"   ✅ flask: Versión {flask.__version__}")
                
        except ImportError:
            print(f"   ❌ {package_name}: Faltante")
            missing_packages.append(package_name)
    
    if missing_packages:
        print(f"\n💡 Para instalar dependencias faltantes:")
        print(f"   pip install {' '.join(missing_packages)}")
        
        # Intentar instalación automática
        print(f"\n🔄 ¿Intentar instalación automática? (s/n): ", end="")
        try:
            response = input().lower()
            if response in ['s', 'si', 'y', 'yes']:
                print("📥 Instalando dependencias...")
                cmd = [sys.executable, "-m", "pip", "install"] + missing_packages
                result = subprocess.run(cmd, capture_output=True, text=True)
                
                if result.returncode == 0:
                    print("   ✅ Instalación exitosa")
                    return True
                else:
                    print(f"   ❌ Error en instalación: {result.stderr}")
                    return False
            else:
                print("   ℹ️ Instalación manual requerida")
                return False
        except KeyboardInterrupt:
            print("\n   ⏹️ Instalación cancelada por el usuario")
            return False
    
    return True

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
        "scripts/start_dashboard.py": "Script de inicio",
        "scripts/test_feedback_system.py": "Script de pruebas"
    }
    
    all_files_exist = True
    for filepath, description in critical_files.items():
        exists = check_file_exists(filepath, description)
        if not exists:
            all_files_exist = False
    
    # 2. Crear archivos faltantes
    print("\n2. 🔧 Archivos y Directorios:")
    create_missing_files()
    
    # 3. Verificar permisos
    print("\n3. 🔑 Permisos:")
    permissions_ok = fix_permission_issues()
    
    # 4. Verificar dependencias
    print("\n4. 📦 Dependencias:")
    dependencies_ok = install_missing_dependencies()
    
    # 5. Probar importaciones
    print("\n5. 🧪 Pruebas de Importación:")
    signal_tracker_ok = test_imports()
    bridge_ok = test_integration_bridge()
    
    # 6. Resumen final
    print("\n" + "=" * 50)
    print("📊 RESUMEN DEL DIAGNÓSTICO")
    print("=" * 50)
    
    issues = []
    if not all_files_exist:
        issues.append("Archivos faltantes")
    if not permissions_ok:
        issues.append("Problemas de permisos")
    if not dependencies_ok:
        issues.append("Dependencias faltantes")
    if not signal_tracker_ok:
        issues.append("Error en SignalTracker")
    if not bridge_ok:
        issues.append("Error en Bridge")
    
    if not issues:
        print("🎉 ¡TODO ESTÁ FUNCIONANDO CORRECTAMENTE!")
        print("✅ Todos los archivos existen")
        print("✅ Todas las importaciones funcionan")
        print("✅ Dependencias instaladas")
        print("✅ Sistema listo para usar")
        
        print("\n🚀 PRÓXIMOS PASOS:")
        print("   1. Ejecutar: python scripts/test_feedback_system.py")
        print("   2. Ejecutar: python scripts/start_dashboard.py")
        print("   3. Integrar: from integration.nvbot3_feedback_bridge import track_signal")
        
        return True
    else:
        print("⚠️ SE ENCONTRARON ALGUNOS PROBLEMAS:")
        for issue in issues:
            print(f"   ❌ {issue}")
        
        print("\n🔧 SOLUCIONES APLICADAS:")
        print("   ✅ Archivos faltantes creados")
        print("   ✅ Directorios estructurados")
        print("   ✅ Permisos verificados")
        
        if dependencies_ok and signal_tracker_ok and bridge_ok:
            print("   ✅ Funcionalidad básica operativa")
            print("\n💡 El sistema debería funcionar ahora")
            return True
        else:
            print("\n💡 ACCIONES RECOMENDADAS:")
            if not dependencies_ok:
                print("   1. Instalar dependencias: pip install flask flask-socketio pandas")
            print("   2. Ejecutar nuevamente: python scripts/fix_import_errors.py")
            print("   3. Probar el sistema: python scripts/test_feedback_system.py")
            return False

def quick_fix():
    """Solución rápida para problemas comunes"""
    
    print("⚡ SOLUCIÓN RÁPIDA DE PROBLEMAS")
    print("=" * 40)
    
    # Crear estructura básica
    create_missing_files()
    
    # Instalar dependencias
    print("\n📥 Instalando dependencias básicas...")
    basic_deps = ['flask', 'flask-socketio', 'pandas']
    
    try:
        cmd = [sys.executable, "-m", "pip", "install"] + basic_deps
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("   ✅ Dependencias instaladas")
        else:
            print(f"   ⚠️ Algunos problemas en la instalación: {result.stderr}")
    except Exception as e:
        print(f"   ❌ Error instalando dependencias: {e}")
    
    # Verificar nuevamente
    print("\n🔄 Verificando después de las correcciones...")
    signal_works = test_imports()
    bridge_works = test_integration_bridge()
    
    if signal_works and bridge_works:
        print("\n🎉 ¡CORRECCIÓN EXITOSA!")
        print("✅ Sistema funcionando correctamente")
        return True
    else:
        print("\n⚠️ Aún hay algunos problemas")
        print("🔧 Ejecuta el diagnóstico completo para más detalles")
        return False

def main():
    """Función principal"""
    
    print("🛠️ REPARADOR DE NVBOT3 FEEDBACK SYSTEM")
    print("=" * 45)
    
    # Verificar directorio actual
    if not Path(".").resolve().name in ['nvbot3', 'NvBot3'] and not Path("web_dashboard").exists():
        print("⚠️ Advertencia: No se detectó estructura de nvbot3")
        print("   Asegúrate de ejecutar desde el directorio raíz del proyecto")
        print("")
    
    print("Selecciona una opción:")
    print("1. 🔍 Diagnóstico completo")
    print("2. ⚡ Solución rápida")
    print("3. 🧪 Solo probar importaciones")
    print("4. 📦 Solo verificar dependencias")
    
    try:
        choice = input("\nOpción (1-4): ").strip()
        
        if choice == "1":
            success = comprehensive_diagnosis()
        elif choice == "2":
            success = quick_fix()
        elif choice == "3":
            print("\n🧪 Probando importaciones...")
            signal_ok = test_imports()
            bridge_ok = test_integration_bridge()
            success = signal_ok and bridge_ok
        elif choice == "4":
            print("\n📦 Verificando dependencias...")
            success = install_missing_dependencies()
        else:
            print("❌ Opción inválida")
            return
        
        if success:
            print("\n✅ Proceso completado exitosamente")
        else:
            print("\n⚠️ Se encontraron algunos problemas - revisa los mensajes anteriores")
            
    except KeyboardInterrupt:
        print("\n\n⏹️ Proceso cancelado por el usuario")
    except Exception as e:
        print(f"\n❌ Error inesperado: {e}")

if __name__ == "__main__":
    main()