#!/usr/bin/env python3
"""
🚀 Wrapper para ejecutar verify_dual_strategy_data.py con debugging
"""

import sys
import os
from pathlib import Path

# Agregar scripts al path
sys.path.insert(0, str(Path('scripts')))

# Importar y ejecutar
try:
    print("🔧 Cargando verificador dual strategy...")
    
    # Cargar el módulo
    with open('scripts/verify_dual_strategy_data.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Crear un namespace para la ejecución
    namespace = {}
    
    # Ejecutar el contenido en el namespace
    exec(content, namespace)
    
    print("✅ Módulo cargado exitosamente")
    print("🔍 Iniciando verificación...")
    
    # Ejecutar la función main
    if 'main' in namespace:
        exit_code = namespace['main']()
        print(f"📋 Verificación completada con código: {exit_code}")
    else:
        print("❌ Función main no encontrada")
        
except Exception as e:
    print(f"❌ Error ejecutando verificador: {e}")
    import traceback
    traceback.print_exc()
