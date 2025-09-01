#!/usr/bin/env python3
# scripts/full_setup_and_run.py
"""
Script maestro para configurar y ejecutar todo el sistema de retroalimentación
Ejecutar UNA SOLA VEZ para configuración completa
"""

import subprocess
import sys
import time
import os
from pathlib import Path
import json

def print_banner():
    """Muestra el banner de inicio"""
    banner = """
╔══════════════════════════════════════════════════════════════╗
║              🚀 NVBOT3 FEEDBACK SYSTEM INSTALLER            ║
║                   Configuración Automática                   ║
╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)

def check_python_version():
    """Verifica que la versión de Python sea compatible"""
    print("🐍 Verificando versión de Python...")
    
    version = sys.version_info
    if version.major == 3 and version.minor >= 8:
        print(f"   ✅ Python {version.major}.{version.minor}.{version.micro} - Compatible")
        return True
    else:
        print(f"   ❌ Python {version.major}.{version.minor}.{version.micro} - Se requiere Python 3.8+")
        return False

def check_virtual_environment():
    """Verifica si estamos en un entorno virtual"""
    print("🔧 Verificando entorno virtual...")
    
    # Varias formas de detectar venv
    in_venv = (
        hasattr(sys, 'real_prefix') or 
        (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix) or
        'nvbot3_env' in sys.executable
    )
    
    if in_venv:
        print(f"   ✅ Entorno virtual activo: {sys.executable}")
        return True
    else:
        print("   ⚠️ No se detectó entorno virtual")
        print("   💡 Recomendación: Activar nvbot3_env antes de continuar")
        
        response = input("   ¿Continuar de todas formas? (s/n): ").lower()
        return response in ['s', 'si', 'y', 'yes']

def run_command(command, description="", check_success=True):
    """Ejecuta un comando y maneja errores"""
    print(f"\n🔄 {description}")
    print(f"   Comando: {command}")
    
    try:
        # Ejecutar comando
        result = subprocess.run(
            command, 
            shell=True, 
            check=check_success, 
            capture_output=True, 
            text=True,
            timeout=300  # 5 minutos timeout
        )
        
        if result.stdout:
            print(f"   📝 Output: {result.stdout.strip()}")
        
        print(f"   ✅ {description} - COMPLETADO")
        return True, result.stdout
        
    except subprocess.CalledProcessError as e:
        print(f"   ❌ Error en {description}:")
        print(f"   📝 Error: {e.stderr}")
        return False, e.stderr
    except subprocess.TimeoutExpired:
        print(f"   ⏰ Timeout en {description} (más de 5 minutos)")
        return False, "Timeout"
    except Exception as e:
        print(f"   ❌ Error inesperado en {description}: {e}")
        return False, str(e)

def install_dependencies():
    """Instala todas las dependencias necesarias"""
    print("\n📦 INSTALACIÓN DE DEPENDENCIAS")
    print("-" * 40)
    
    # Lista de dependencias críticas
    dependencies = [
        'flask==2.3.2',
        'flask-socketio==5.3.4', 
        'pandas>=2.0.0',
        'python-socketio==5.8.0'
    ]
    
    # Actualizar pip primero
    success, _ = run_command(
        f'"{sys.executable}" -m pip install --upgrade pip',
        "Actualizando pip"
    )
    
    if not success:
        print("   ⚠️ No se pudo actualizar pip, continuando...")
    
    # Instalar dependencias una por una para mejor diagnóstico
    installed_successfully = []
    failed_installations = []
    
    for dep in dependencies:
        success, output = run_command(
            f'"{sys.executable}" -m pip install {dep}',
            f"Instalando {dep}",
            check_success=False
        )
        
        if success:
            installed_successfully.append(dep)
        else:
            failed_installations.append(dep)
    
    # Resumen de instalación
    print(f"\n📊 Resumen de instalación:")
    print(f"   ✅ Exitosas: {len(installed_successfully)}")
    print(f"   ❌ Fallidas: {len(failed_installations)}")
    
    if failed_installations:
        print("   ⚠️ Dependencias con problemas:")
        for dep in failed_installations:
            print(f"      - {dep}")
        
        # Intentar instalación alternativa
        print("\n🔄 Intentando instalación alternativa...")
        alt_command = f'"{sys.executable}" -m pip install flask flask-socketio pandas --no-cache-dir'
        success, _ = run_command(alt_command, "Instalación alternativa", check_success=False)
        
        if success:
            print("   ✅ Instalación alternativa exitosa")
            return True
    
    return len(failed_installations) == 0

def create_project_structure():
    """Crea la estructura completa del proyecto"""
    print("\n🏗️ CREANDO ESTRUCTURA DEL PROYECTO")
    print("-" * 40)
    
    # Estructura de directorios
    directories = [
        "web_dashboard",
        "web_dashboard/templates",
        "web_dashboard/static",
        "web_dashboard/static/css",
        "web_dashboard/static/js",
        "web_dashboard/database",
        "integration",
        "scripts",
        "examples"
    ]
    
    created_dirs = 0
    for directory in directories:
        try:
            Path(directory).mkdir(parents=True, exist_ok=True)
            print(f"   ✅ {directory}")
            created_dirs += 1
        except Exception as e:
            print(f"   ❌ Error creando {directory}: {e}")
    
    print(f"   📊 Directorios creados/verificados: {created_dirs}/{len(directories)}")
    return created_dirs == len(directories)

def run_setup_scripts():
    """Ejecuta los scripts de configuración"""
    print("\n⚙️ EJECUTANDO CONFIGURACIÓN")
    print("-" * 40)
    
    scripts_to_run = [
        ("scripts/fix_import_errors.py", "Reparando imports y creando archivos base"),
        ("scripts/test_feedback_system.py", "Probando sistema y creando datos demo")
    ]
    
    success_count = 0
    
    for script_path, description in scripts_to_run:
        if Path(script_path).exists():
            success, output = run_command(
                f'"{sys.executable}" {script_path}',
                description,
                check_success=False
            )
            if success:
                success_count += 1
        else:
            print(f"   ⚠️ Script no encontrado: {script_path}")
    
    return success_count > 0

def verify_installation():
    """Verifica que todo esté instalado correctamente"""
    print("\n🔍 VERIFICACIÓN DE INSTALACIÓN")
    print("-" * 40)
    
    # Verificar importaciones críticas
    verification_tests = [
        ("import flask", "Flask framework"),
        ("import flask_socketio", "Flask-SocketIO"),
        ("import pandas", "Pandas"),
        ("import sqlite3", "SQLite3")
    ]
    
    passed_tests = 0
    
    for test_import, description in verification_tests:
        try:
            exec(test_import)
            print(f"   ✅ {description}")
            passed_tests += 1
        except ImportError as e:
            print(f"   ❌ {description}: {e}")
    
    # Verificar archivos críticos
    critical_files = [
        "web_dashboard/database/signal_tracker.py",
        "integration/nvbot3_feedback_bridge.py",
        "web_dashboard/app.py"
    ]
    
    files_exist = 0
    for file_path in critical_files:
        if Path(file_path).exists():
            print(f"   ✅ {file_path}")
            files_exist += 1
        else:
            print(f"   ❌ Faltante: {file_path}")
    
    total_score = passed_tests + files_exist
    max_score = len(verification_tests) + len(critical_files)
    
    print(f"\n📊 Puntuación: {total_score}/{max_score}")
    return total_score >= (max_score * 0.8)  # 80% o más

def create_usage_guide():
    """Crea una guía de uso rápida"""
    
    guide_content = """# 🤖 GUÍA RÁPIDA - NVBOT3 FEEDBACK SYSTEM

## 📋 CONFIGURACIÓN COMPLETADA ✅

### 🚀 PASOS SIGUIENTES:

#### 1. Iniciar el Dashboard
```bash
python scripts/start_dashboard.py
```
Luego abrir: http://localhost:5000

#### 2. Integrar con tu Bot
Agregar al inicio de tu archivo principal de nvbot3:

```python
# Importar sistema de tracking
from integration.nvbot3_feedback_bridge import track_signal, update_price

# En tu función de predicción:
def tu_funcion_de_prediccion():
    # ... tu código existente ...
    
    # Cuando generes una predicción:
    if prediction['confidence'] > 0.75:
        track_signal(symbol, prediction, current_price)
    
    # Cuando recibas nuevos precios:
    update_price(symbol, new_price)
```

#### 3. Ejemplo de Integración Completa

```python
# ejemplo_nvbot3_con_tracking.py
from integration.nvbot3_feedback_bridge import track_signal, update_price

def main_loop():
    symbols = ['BTCUSDT', 'ETHUSDT']
    
    for symbol in symbols:
        # Tu código existente de análisis...
        market_data = get_market_data(symbol)
        prediction = generate_prediction(symbol, market_data)
        
        # NUEVO: Trackear señal si tiene buena confianza
        if prediction['confidence'] > 0.7:
            signal_id = track_signal(
                symbol=symbol, 
                prediction_data=prediction, 
                current_price=market_data['close']
            )
            print(f"📊 Señal trackeada: {signal_id}")
        
        # NUEVO: Actualizar precio para tracking
        update_price(symbol, market_data['close'])
```

#### 4. Monitoreo y Feedback
- Dashboard web muestra todas las señales en tiempo real
- Puedes dar feedback sobre cada predicción (✅ exitosa, ❌ fallida)
- El sistema aprende de tu feedback para mejorar

#### 5. Archivos Importantes
- `web_dashboard/app.py` - Servidor web principal
- `integration/nvbot3_feedback_bridge.py` - Conexión con tu bot
- `web_dashboard/database/signals.db` - Base de datos de señales

## 🛠️ SOLUCIÓN DE PROBLEMAS

### Error: "Module not found"
```bash
python scripts/fix_import_errors.py
```

### Dashboard no inicia
1. Verificar que Flask esté instalado: `pip list | grep Flask`
2. Probar: `python scripts/test_feedback_system.py`
3. Revisar logs en terminal

### Base de datos corrupta
- Borrar archivo `web_dashboard/database/signals.db`
- Se recreará automáticamente

## 📞 SOPORTE
Si encuentras problemas, ejecuta:
```bash
python scripts/fix_import_errors.py
```

¡Sistema listo para usar! 🎉
"""
    
    # Guardar guía
    guide_path = Path("GUIA_NVBOT3_FEEDBACK.md")
    with open(guide_path, 'w', encoding='utf-8') as f:
        f.write(guide_content)
    
    print(f"   ✅ Guía creada: {guide_path}")
    return True

def main():
    """Función principal del instalador"""
    
    print_banner()
    
    # Verificaciones previas
    print("🔍 VERIFICACIONES PREVIAS")
    print("=" * 30)
    
    if not check_python_version():
        print("❌ Versión de Python incompatible. Proceso abortado.")
        sys.exit(1)
    
    if not check_virtual_environment():
        print("⏹️ Proceso cancelado por el usuario.")
        sys.exit(0)
    
    # Registro de tiempo de inicio
    start_time = time.time()
    
    # Proceso de instalación paso a paso
    steps = [
        ("📦 Instalando dependencias", install_dependencies),
        ("🏗️ Creando estructura", create_project_structure),
        ("⚙️ Ejecutando configuración", run_setup_scripts),
        ("🔍 Verificando instalación", verify_installation),
        ("📚 Creando guía de uso", create_usage_guide)
    ]
    
    completed_steps = 0
    failed_steps = []
    
    for step_name, step_function in steps:
        print(f"\n{step_name}")
        print("=" * len(step_name))
        
        try:
            success = step_function()
            if success:
                completed_steps += 1
                print(f"   ✅ {step_name} - COMPLETADO")
            else:
                failed_steps.append(step_name)
                print(f"   ⚠️ {step_name} - COMPLETADO CON ADVERTENCIAS")
                
        except Exception as e:
            failed_steps.append(step_name)
            print(f"   ❌ {step_name} - ERROR: {e}")
    
    # Cálculo de tiempo transcurrido
    elapsed_time = time.time() - start_time
    
    # Resultado final
    print("\n" + "=" * 60)
    print("🎯 RESULTADO FINAL DE LA INSTALACIÓN")
    print("=" * 60)
    
    print(f"⏱️ Tiempo transcurrido: {elapsed_time:.1f} segundos")
    print(f"✅ Pasos completados: {completed_steps}/{len(steps)}")
    
    if failed_steps:
        print(f"⚠️ Pasos con problemas: {len(failed_steps)}")
        for step in failed_steps:
            print(f"   - {step}")
    
    if completed_steps >= len(steps) * 0.8:  # 80% o más
        print("\n🎉 ¡INSTALACIÓN COMPLETADA EXITOSAMENTE!")
        print("=" * 45)
        
        print("📋 PRÓXIMOS PASOS:")
        print("   1. 🌐 Iniciar dashboard: python scripts/start_dashboard.py")
        print("   2. 🔗 Abrir navegador: http://localhost:5000")
        print("   3. 🤖 Integrar con tu bot (ver GUIA_NVBOT3_FEEDBACK.md)")
        print("   4. 📊 Ejecutar tu nvbot3 como siempre")
        
        print("\n📚 RECURSOS:")
        print("   - Guía completa: GUIA_NVBOT3_FEEDBACK.md")
        print("   - Probar sistema: python scripts/test_feedback_system.py")
        print("   - Solucionar problemas: python scripts/fix_import_errors.py")
        
        print("\n🚀 ¡Sistema listo para usar!")
        
    else:
        print("\n⚠️ INSTALACIÓN PARCIALMENTE COMPLETADA")
        print("=" * 40)
        
        print("🔧 ACCIONES RECOMENDADAS:")
        print("   1. Ejecutar: python scripts/fix_import_errors.py")
        print("   2. Instalar manualmente: pip install flask flask-socketio pandas")
        print("   3. Volver a ejecutar este script")
        
        print("💡 El sistema podría funcionar con limitaciones")
        
    print("\n" + "=" * 60)
    
    # Pregunta final
    if completed_steps >= len(steps) * 0.8:
        try:
            print("\n¿Deseas iniciar el dashboard ahora? (s/n): ", end="")
            response = input().lower()
            
            if response in ['s', 'si', 'y', 'yes']:
                print("\n🚀 Iniciando dashboard...")
                time.sleep(2)
                
                # Cambiar al directorio y ejecutar dashboard
                try:
                    subprocess.run([sys.executable, "scripts/start_dashboard.py"], check=True)
                except KeyboardInterrupt:
                    print("\n⏹️ Dashboard cerrado por el usuario")
                except Exception as e:
                    print(f"\n❌ Error iniciando dashboard: {e}")
                    print("💡 Ejecuta manualmente: python scripts/start_dashboard.py")
            
        except KeyboardInterrupt:
            print("\n\n👋 ¡Hasta luego! Sistema instalado correctamente.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⏹️ Instalación cancelada por el usuario")
        print("🔄 Puedes reanudar ejecutando nuevamente este script")
    except Exception as e:
        print(f"\n❌ Error inesperado: {e}")
        print("🔧 Intenta ejecutar: python scripts/fix_import_errors.py")