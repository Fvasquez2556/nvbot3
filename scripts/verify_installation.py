#!/usr/bin/env python3
# scripts/verify_installation.py
"""
Script de verificación final para el sistema NvBot3 Feedback
Ejecutar después de la instalación para confirmar que todo funciona correctamente
"""

import os
import sys
import time
import json
from pathlib import Path
import subprocess
import importlib.util

def print_header():
    """Imprime el header de verificación"""
    header = """
╔══════════════════════════════════════════════════════════════╗
║            🔍 VERIFICACIÓN FINAL - NVBOT3 FEEDBACK          ║
║                Sistema de Retroalimentación                  ║
╚══════════════════════════════════════════════════════════════╝
"""
    print(header)

def check_file_structure():
    """Verifica que todos los archivos necesarios estén presentes"""
    
    print("📁 VERIFICANDO ESTRUCTURA DE ARCHIVOS")
    print("-" * 50)
    
    required_files = {
        # Archivos principales
        "web_dashboard/app.py": "Servidor Flask principal",
        "web_dashboard/database/signal_tracker.py": "Sistema de tracking",
        "web_dashboard/templates/dashboard.html": "Interfaz web",
        "integration/nvbot3_feedback_bridge.py": "Bridge de integración",
        
        # Scripts de utilidad
        "scripts/start_dashboard.py": "Script de inicio",
        "scripts/test_feedback_system.py": "Script de pruebas",
        "scripts/fix_import_errors.py": "Script de reparación",
        "scripts/full_setup_and_run.py": "Instalador maestro",
        
        # Archivos de configuración
        "requirements.txt": "Lista de dependencias"
    }
    
    missing_files = []
    existing_files = []
    
    for filepath, description in required_files.items():
        path = Path(filepath)
        if path.exists():
            size = path.stat().st_size
            print(f"   ✅ {description}: {filepath} ({size:,} bytes)")
            existing_files.append(filepath)
        else:
            print(f"   ❌ {description}: {filepath} - FALTANTE")
            missing_files.append(filepath)
    
    # Verificar directorios
    required_dirs = [
        "web_dashboard/static/css",
        "web_dashboard/static/js",
        "examples"
    ]
    
    for directory in required_dirs:
        path = Path(directory)
        if path.exists():
            print(f"   ✅ Directorio: {directory}")
        else:
            print(f"   ⚠️ Directorio faltante: {directory}")
            path.mkdir(parents=True, exist_ok=True)
            print(f"   ✅ Directorio creado: {directory}")
    
    return len(missing_files) == 0, missing_files, existing_files

def check_dependencies():
    """Verifica que todas las dependencias estén instaladas"""
    
    print("\n📦 VERIFICANDO DEPENDENCIAS")
    print("-" * 50)
    
    dependencies = {
        'flask': 'Framework web principal',
        'flask_socketio': 'WebSocket support',
        'pandas': 'Análisis de datos',
        'sqlite3': 'Base de datos (built-in)'
    }
    
    installed = []
    missing = []
    
    for package, description in dependencies.items():
        try:
            if package == 'flask_socketio':
                import flask_socketio
                version = getattr(flask_socketio, '__version__', 'unknown')
                print(f"   ✅ {description}: {package} v{version}")
            elif package == 'flask':
                import flask
                print(f"   ✅ {description}: {package} v{flask.__version__}")
            elif package == 'pandas':
                import pandas as pd
                print(f"   ✅ {description}: {package} v{pd.__version__}")
            elif package == 'sqlite3':
                import sqlite3
                print(f"   ✅ {description}: {package} (built-in)")
            
            installed.append(package)
            
        except ImportError:
            print(f"   ❌ {description}: {package} - NO INSTALADO")
            missing.append(package)
    
    return len(missing) == 0, missing, installed

def test_imports():
    """Prueba las importaciones críticas del sistema"""
    
    print("\n🧪 PROBANDO IMPORTACIONES CRÍTICAS")
    print("-" * 50)
    
    tests = []
    
    # Test 1: SignalTracker
    try:
        original_dir = os.getcwd()
        os.chdir('web_dashboard')
        sys.path.append(os.getcwd())
        
        from database.signal_tracker import SignalTracker
        tracker = SignalTracker()
        
        os.chdir(original_dir)
        print("   ✅ SignalTracker: Importación e inicialización exitosa")
        tests.append(("SignalTracker", True))
        
    except Exception as e:
        os.chdir(original_dir)
        print(f"   ❌ SignalTracker: Error - {e}")
        tests.append(("SignalTracker", False))
    
    # Test 2: Bridge de integración  
    try:
        from integration.nvbot3_feedback_bridge import track_signal, get_system_status
        
        status = get_system_status()
        print("   ✅ Integration Bridge: Importación exitosa")
        print(f"      📊 Estado del sistema: {status.get('tracking_enabled', 'unknown')}")
        tests.append(("Integration Bridge", True))
        
    except Exception as e:
        print(f"   ❌ Integration Bridge: Error - {e}")
        tests.append(("Integration Bridge", False))
    
    # Test 3: Flask App
    try:
        original_dir = os.getcwd()
        os.chdir('web_dashboard')
        sys.path.append(os.getcwd())
        
        import app
        print("   ✅ Flask App: Importación exitosa")
        tests.append(("Flask App", True))
        
        os.chdir(original_dir)
        
    except Exception as e:
        os.chdir(original_dir)
        print(f"   ❌ Flask App: Error - {e}")
        tests.append(("Flask App", False))
    
    passed = sum(1 for _, success in tests if success)
    return passed == len(tests), tests

def test_database_functionality():
    """Prueba la funcionalidad de la base de datos"""
    
    print("\n💾 PROBANDO FUNCIONALIDAD DE BASE DE DATOS")
    print("-" * 50)
    
    try:
        from integration.nvbot3_feedback_bridge import track_signal, get_tracking_stats, get_active_signals
        
        # Test 1: Crear señal de prueba
        test_signal = {
            'type': 'test_signal',
            'predicted_change': 5.0,
            'confidence': 0.80
        }
        
        signal_id = track_signal('TEST_VERIFICATION', test_signal, 50000.0)
        
        if signal_id:
            print("   ✅ Creación de señal: Exitosa")
            
            # Test 2: Obtener estadísticas
            stats = get_tracking_stats()
            print(f"   ✅ Estadísticas: {stats}")
            
            # Test 3: Obtener señales activas
            active = get_active_signals()
            print(f"   ✅ Señales activas: {len(active)} encontradas")
            
            return True, "Todas las funciones de base de datos funcionan"
        else:
            return False, "No se pudo crear señal de prueba"
            
    except Exception as e:
        return False, f"Error en funcionalidad de BD: {e}"

def test_web_server():
    """Prueba que el servidor web se pueda iniciar"""
    
    print("\n🌐 PROBANDO SERVIDOR WEB")
    print("-" * 50)
    
    try:
        # Importar y verificar Flask app
        original_dir = os.getcwd()
        os.chdir('web_dashboard')
        sys.path.append(os.getcwd())
        
        import app
        flask_app = app.app
        
        # Verificar rutas
        routes = [rule.rule for rule in flask_app.url_map.iter_rules()]
        required_routes = ['/', '/api/signals/active', '/api/feedback', '/api/test']
        
        missing_routes = [r for r in required_routes if r not in routes]
        
        os.chdir(original_dir)
        
        if not missing_routes:
            print("   ✅ Servidor Flask: Configuración correcta")
            print(f"   ✅ Rutas disponibles: {len(routes)}")
            for route in required_routes:
                print(f"      - {route}")
            return True, "Servidor web listo"
        else:
            return False, f"Rutas faltantes: {missing_routes}"
            
    except Exception as e:
        os.chdir(original_dir)
        return False, f"Error en servidor web: {e}"

def run_integration_demo():
    """Ejecuta una demostración completa del sistema"""
    
    print("\n🎬 EJECUTANDO DEMOSTRACIÓN COMPLETA")
    print("-" * 50)
    
    try:
        from integration.nvbot3_feedback_bridge import track_signal, update_price, get_tracking_stats
        
        # Datos de demostración
        demo_signals = [
            {
                'symbol': 'BTCUSDT',
                'prediction': {'type': 'momentum_alto', 'predicted_change': 4.5, 'confidence': 0.85},
                'price': 67250.0
            },
            {
                'symbol': 'ETHUSDT', 
                'prediction': {'type': 'rebound', 'predicted_change': 3.2, 'confidence': 0.78},
                'price': 3180.0
            }
        ]
        
        signals_created = 0
        
        for demo in demo_signals:
            print(f"   🧪 Creando señal demo: {demo['symbol']}")
            
            signal_id = track_signal(demo['symbol'], demo['prediction'], demo['price'])
            
            if signal_id:
                signals_created += 1
                print(f"      ✅ Señal creada: {signal_id}")
                
                # Simular actualización de precio
                new_price = demo['price'] * 1.02
                update_price(demo['symbol'], new_price)
                print(f"      📈 Precio actualizado: ${new_price:,.2f}")
            else:
                print(f"      ❌ Error creando señal para {demo['symbol']}")
        
        # Mostrar estadísticas finales
        if signals_created > 0:
            stats = get_tracking_stats()
            print(f"\n   📊 RESULTADO DE LA DEMOSTRACIÓN:")
            print(f"      Señales creadas: {signals_created}")
            print(f"      Total en sistema: {stats.get('total_signals', 0)}")
            
            return True, f"Demostración exitosa: {signals_created} señales creadas"
        else:
            return False, "No se pudieron crear señales de demostración"
            
    except Exception as e:
        return False, f"Error en demostración: {e}"

def generate_verification_report():
    """Genera un reporte completo de verificación"""
    
    print_header()
    
    # Variables para el reporte
    overall_score = 0
    max_score = 0
    issues = []
    successes = []
    
    # Test 1: Estructura de archivos
    files_ok, missing_files, existing_files = check_file_structure()
    max_score += 20
    if files_ok:
        overall_score += 20
        successes.append("✅ Todos los archivos necesarios están presentes")
    else:
        overall_score += 15  # Puntuación parcial
        issues.append(f"❌ Archivos faltantes: {len(missing_files)}")
    
    # Test 2: Dependencias
    deps_ok, missing_deps, installed_deps = check_dependencies()
    max_score += 15
    if deps_ok:
        overall_score += 15
        successes.append("✅ Todas las dependencias están instaladas")
    else:
        issues.append(f"❌ Dependencias faltantes: {missing_deps}")
    
    # Test 3: Importaciones
    imports_ok, import_tests = test_imports()
    max_score += 20
    if imports_ok:
        overall_score += 20
        successes.append("✅ Todas las importaciones funcionan correctamente")
    else:
        failed_imports = [name for name, success in import_tests if not success]
        issues.append(f"❌ Importaciones fallidas: {failed_imports}")
        overall_score += len([s for _, s in import_tests if s]) * 5  # Puntuación parcial
    
    # Test 4: Base de datos
    db_ok, db_msg = test_database_functionality()
    max_score += 20
    if db_ok:
        overall_score += 20
        successes.append("✅ Funcionalidad de base de datos operativa")
    else:
        issues.append(f"❌ Problema con base de datos: {db_msg}")
    
    # Test 5: Servidor web
    web_ok, web_msg = test_web_server()
    max_score += 15
    if web_ok:
        overall_score += 15
        successes.append("✅ Servidor web configurado correctamente")
    else:
        issues.append(f"❌ Problema con servidor web: {web_msg}")
    
    # Test 6: Demostración completa
    demo_ok, demo_msg = run_integration_demo()
    max_score += 10
    if demo_ok:
        overall_score += 10
        successes.append("✅ Demostración del sistema exitosa")
    else:
        issues.append(f"❌ Problema en demostración: {demo_msg}")
    
    # Calcular porcentaje
    percentage = (overall_score / max_score) * 100
    
    # Reporte final
    print("\n" + "=" * 70)
    print("📋 REPORTE FINAL DE VERIFICACIÓN")
    print("=" * 70)
    
    print(f"🎯 Puntuación Total: {overall_score}/{max_score} ({percentage:.1f}%)")
    
    if percentage >= 90:
        status = "🎉 EXCELENTE"
        color = "verde"
    elif percentage >= 75:
        status = "✅ BUENO" 
        color = "amarillo"
    elif percentage >= 50:
        status = "⚠️ ACEPTABLE"
        color = "naranja"
    else:
        status = "❌ NECESITA TRABAJO"
        color = "rojo"
    
    print(f"📊 Estado General: {status}")
    
    # Mostrar éxitos
    if successes:
        print(f"\n✅ ASPECTOS FUNCIONANDO CORRECTAMENTE ({len(successes)}):")
        for success in successes:
            print(f"   {success}")
    
    # Mostrar problemas
    if issues:
        print(f"\n⚠️ PROBLEMAS ENCONTRADOS ({len(issues)}):")
        for issue in issues:
            print(f"   {issue}")
    
    # Recomendaciones
    print(f"\n💡 RECOMENDACIONES:")
    
    if percentage >= 90:
        print("   🚀 ¡Sistema completamente funcional!")
        print("   📋 Próximos pasos:")
        print("      1. Ejecutar: python scripts/start_dashboard.py")
        print("      2. Abrir navegador: http://localhost:5000")
        print("      3. Integrar con tu nvbot3:")
        print("         from integration.nvbot3_feedback_bridge import track_signal")
        
    elif percentage >= 75:
        print("   ✅ Sistema funcional con problemas menores")
        print("   🔧 Ejecutar: python scripts/fix_import_errors.py")
        
    elif percentage >= 50:
        print("   ⚠️ Sistema parcialmente funcional")
        if missing_deps:
            print(f"   📦 Instalar dependencias: pip install {' '.join(missing_deps)}")
        print("   🔧 Ejecutar: python scripts/full_setup_and_run.py")
        
    else:
        print("   ❌ Sistema requiere instalación completa")
        print("   🔄 Ejecutar instalación completa:")
        print("      1. python scripts/full_setup_and_run.py")
        print("      2. python scripts/fix_import_errors.py")
        print("      3. python scripts/verify_installation.py")
    
    # Información adicional
    print(f"\n📁 Archivos verificados: {len(existing_files)}")
    print(f"📦 Dependencias instaladas: {len(installed_deps)}")
    print(f"🧪 Pruebas ejecutadas: 6")
    
    print("\n" + "=" * 70)
    
    return percentage >= 75

def main():
    """Función principal de verificación"""
    
    # Verificar que estamos en el directorio correcto
    if not Path("web_dashboard").exists() and not Path("integration").exists():
        print("❌ Error: No se encuentra la estructura de nvbot3")
        print("🔧 Ejecuta este script desde el directorio raíz del proyecto nvbot3")
        return False
    
    # Ejecutar verificación completa
    success = generate_verification_report()
    
    # Mensaje final
    if success:
        print("🎉 ¡VERIFICACIÓN COMPLETADA EXITOSAMENTE!")
        print("✅ El sistema NvBot3 Feedback está listo para usar")
        return True
    else:
        print("⚠️ La verificación encontró algunos problemas")
        print("🔧 Sigue las recomendaciones mostradas arriba")
        return False

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n⏹️ Verificación cancelada por el usuario")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error inesperado durante la verificación: {e}")
        sys.exit(1)