#!/usr/bin/env python3
"""
🧪 Prueba de las 3 correcciones críticas implementadas (con codificación UTF-8)
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
    with open('scripts/fix_training_data.py', 'r', encoding='utf-8') as f:
        content = f.read()
        exec(content)
    print("✅ Script de completar datos: CARGA CORRECTAMENTE")
except Exception as e:
    print(f"❌ Error en fix_training_data.py: {e}")

# Test 2: Verificar que verify_dual_strategy_data.py funciona  
print("\n2️⃣ Probando verify_dual_strategy_data.py...")
try:
    with open('scripts/verify_dual_strategy_data.py', 'r', encoding='utf-8') as f:
        content = f.read()
        exec(content)
    print("✅ Verificador combinado: CARGA CORRECTAMENTE")
except Exception as e:
    print(f"❌ Error en verify_dual_strategy_data.py: {e}")

# Test 3: Verificar importación de clases específicas
print("\n3️⃣ Probando importaciones de clases...")
try:
    # Ejecutar verify_dual_strategy_data para definir las clases
    with open('scripts/verify_dual_strategy_data.py', 'r', encoding='utf-8') as f:
        content = f.read()
        # Remover el main() para evitar ejecución automática
        content = content.replace('if __name__ == "__main__":', 'if False:')
        exec(content, globals())
    
    # Probar que DualStrategyVerifier existe
    verifier = DualStrategyVerifier(validation_mode='flexible')
    print("✅ Clase DualStrategyVerifier: IMPORTACIÓN EXITOSA")
    
    # Probar métodos básicos
    result = verifier.verify_training_data_files()
    print(f"✅ Método verify_training_data_files: FUNCIONA (resultado: {result})")
    
except Exception as e:
    print(f"❌ Error probando clases: {e}")

# Test 4: Verificar que download_training_data_only.py funciona
print("\n4️⃣ Probando download_training_data_only.py...")
try:
    with open('scripts/download_training_data_only.py', 'r', encoding='utf-8') as f:
        content = f.read()
        # Reemplazar el main() para evitar descarga automática
        content = content.replace('if __name__ == "__main__":', 'if False:')
        exec(content, globals())
    
    # Probar que TrainingDataDownloader existe
    downloader = TrainingDataDownloader()
    print("✅ Clase TrainingDataDownloader: IMPORTACIÓN EXITOSA")
    
except Exception as e:
    print(f"❌ Error en download_training_data_only.py: {e}")

print("\n" + "=" * 50)
print("🎯 RESULTADO: Probando compatibilidad entre las 3 correcciones")
print("📋 Si todos los tests pasan, las correcciones están funcionando")
