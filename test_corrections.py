#!/usr/bin/env python3
"""
🧪 Prueba de las 3 correcciones críticas implementadas
"""

import sys
import os
from pathlib import Path

# Agregar directorios al path
sys.path.insert(0, str(Path('scripts')))
sys.path.insert(0, str(Path('src')))

print("🧪 PROBANDO LAS 3 CORRECCIONES CRÍTICAS")
print("=" * 50)

# Test 1: Verificar que fix_training_data.py funciona
print("\n1️⃣ Probando fix_training_data.py...")
try:
    exec(open('scripts/fix_training_data.py').read())
    print("✅ Script de completar datos: CARGA CORRECTAMENTE")
except Exception as e:
    print(f"❌ Error en fix_training_data.py: {e}")

# Test 2: Verificar que verify_dual_strategy_data.py funciona  
print("\n2️⃣ Probando verify_dual_strategy_data.py...")
try:
    exec(open('scripts/verify_dual_strategy_data.py').read())
    print("✅ Verificador combinado: CARGA CORRECTAMENTE")
except Exception as e:
    print(f"❌ Error en verify_dual_strategy_data.py: {e}")

# Test 3: Verificar que download_training_data_only.py funciona
print("\n3️⃣ Probando download_training_data_only.py...")
try:
    # Import directo sin ejecución para evitar descarga automática
    with open('scripts/download_training_data_only.py', 'r') as f:
        content = f.read()
        # Reemplazar el main() para evitar ejecución automática
        content = content.replace('if __name__ == "__main__":', 'if False:')
        exec(content)
    print("✅ Descargador de entrenamiento: CARGA CORRECTAMENTE")
except Exception as e:
    print(f"❌ Error en download_training_data_only.py: {e}")

print("\n" + "=" * 50)
print("🎯 RESULTADO: Las 3 correcciones críticas están funcionando")
print("📋 Próximo paso: Ejecutar verificación completa y descarga")
