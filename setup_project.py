#!/usr/bin/env python3
"""
NvBot3 - Script de Setup y Configuración
========================================

Script para configurar el entorno virtual y todas las dependencias necesarias
para el proyecto NvBot3.

Uso:
    python setup_project.py

Autor: NvBot3 Team
Fecha: Agosto 2025
"""

import os
import sys
import subprocess
import platform
from pathlib import Path


def run_command(command, description, check=True):
    """Ejecutar comando del sistema con logging."""
    print(f"🔄 {description}...")
    try:
        if platform.system() == "Windows":
            result = subprocess.run(command, shell=True, check=check, 
                                  capture_output=True, text=True)
        else:
            result = subprocess.run(command, shell=True, check=check,
                                  capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"✅ {description} - Completado")
            return True
        else:
            print(f"❌ {description} - Error: {result.stderr}")
            return False
            
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} - Error: {e}")
        return False


def check_python_version():
    """Verificar que Python 3.9+ está instalado."""
    version = sys.version_info
    if version.major == 3 and version.minor >= 9:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro} - Compatible")
        return True
    else:
        print(f"❌ Python {version.major}.{version.minor}.{version.micro} - Se requiere Python 3.9+")
        return False


def setup_virtual_environment():
    """Crear y configurar entorno virtual."""
    print("\n🔧 Configurando Entorno Virtual")
    print("=" * 40)
    
    # Verificar si ya existe
    if Path("nvbot3_env").exists():
        print("⚠️  El entorno virtual ya existe")
        response = input("¿Deseas recrearlo? (y/N): ").lower()
        if response == 'y':
            if platform.system() == "Windows":
                run_command("rmdir /s /q nvbot3_env", "Eliminando entorno existente", check=False)
            else:
                run_command("rm -rf nvbot3_env", "Eliminando entorno existente", check=False)
        else:
            print("✅ Usando entorno virtual existente")
            return True
    
    # Crear entorno virtual
    success = run_command("python -m venv nvbot3_env", "Creando entorno virtual")
    if not success:
        return False
    
    # Activar y actualizar pip
    if platform.system() == "Windows":
        activate_cmd = "nvbot3_env\\Scripts\\activate"
        pip_cmd = "nvbot3_env\\Scripts\\pip"
    else:
        activate_cmd = "source nvbot3_env/bin/activate"
        pip_cmd = "nvbot3_env/bin/pip"
    
    success = run_command(f"{pip_cmd} install --upgrade pip", "Actualizando pip")
    return success


def install_dependencies():
    """Instalar dependencias desde requirements.txt."""
    print("\n📦 Instalando Dependencias")
    print("=" * 30)
    
    if not Path("requirements.txt").exists():
        print("❌ Archivo requirements.txt no encontrado")
        return False
    
    if platform.system() == "Windows":
        pip_cmd = "nvbot3_env\\Scripts\\pip"
    else:
        pip_cmd = "nvbot3_env/bin/pip"
    
    success = run_command(f"{pip_cmd} install -r requirements.txt", 
                         "Instalando paquetes de Python")
    return success


def verify_installation():
    """Verificar que las dependencias principales están instaladas."""
    print("\n🔍 Verificando Instalación")
    print("=" * 30)
    
    if platform.system() == "Windows":
        python_cmd = "nvbot3_env\\Scripts\\python"
    else:
        python_cmd = "nvbot3_env/bin/python"
    
    test_imports = [
        "numpy", "pandas", "sklearn", "xgboost", 
        "ccxt", "tqdm", "yaml", "tensorflow"
    ]
    
    import_cmd = f"{python_cmd} -c \"import {', '.join(test_imports)}; print('Todas las dependencias importadas correctamente')\""
    
    success = run_command(import_cmd, "Verificando imports de dependencias")
    return success


def create_env_file():
    """Crear archivo .env desde .env.example si no existe."""
    print("\n⚙️  Configurando Variables de Entorno")
    print("=" * 40)
    
    if not Path(".env").exists():
        if Path(".env.example").exists():
            # Copiar .env.example a .env
            with open(".env.example", 'r') as src:
                content = src.read()
            
            with open(".env", 'w') as dst:
                dst.write(content)
            
            print("✅ Archivo .env creado desde .env.example")
            print("⚠️  IMPORTANTE: Edita el archivo .env con tus valores reales")
        else:
            print("❌ Archivo .env.example no encontrado")
            return False
    else:
        print("✅ Archivo .env ya existe")
    
    return True


def show_activation_instructions():
    """Mostrar instrucciones para activar el entorno."""
    print("\n🎯 Instrucciones de Activación")
    print("=" * 35)
    
    if platform.system() == "Windows":
        print("Para activar el entorno virtual, ejecuta:")
        print("    nvbot3_env\\Scripts\\activate")
        print("\nPara desactivar:")
        print("    deactivate")
    else:
        print("Para activar el entorno virtual, ejecuta:")
        print("    source nvbot3_env/bin/activate")
        print("\nPara desactivar:")
        print("    deactivate")
    
    print("\n📝 Próximos pasos:")
    print("1. Activa el entorno virtual")
    print("2. Edita el archivo .env con tus API keys de Binance")
    print("3. Ejecuta: python scripts/download_historical_data.py")


def main():
    """Función principal del setup."""
    print("🤖 NvBot3 - Setup del Proyecto")
    print("=" * 50)
    
    # Verificar Python
    if not check_python_version():
        return False
    
    # Setup entorno virtual
    if not setup_virtual_environment():
        return False
    
    # Instalar dependencias
    if not install_dependencies():
        return False
    
    # Verificar instalación
    if not verify_installation():
        return False
    
    # Crear archivo .env
    create_env_file()
    
    # Mostrar instrucciones
    show_activation_instructions()
    
    print("\n🎉 ¡Setup completado exitosamente!")
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
