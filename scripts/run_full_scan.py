#!/usr/bin/env python3
"""
🤖 NvBot3 - Sistema de Escaneo Completo
======================================
Script automatizado para ejecutar el escaneo completo de las 30 monedas
y mostrar estadísticas resumidas.
"""

import sys
import os
import time
import logging
from datetime import datetime

# Agregar src al path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

def main():
    """Ejecuta el escaneo completo y muestra estadísticas"""
    
    print("🤖 === NVBOT3 FULL SCAN AUTOMATION ===")
    print("🎯 Ejecutando escaneo completo de las 30 monedas")
    print("=" * 60)
    
    start_time = time.time()
    
    try:
        # Importar módulos necesarios
        import subprocess
        import json
        import yaml
        
        # Cargar configuración directamente del YAML
        config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'training_config.yaml')
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        symbols = []
        for tier in ['tier_1', 'tier_2', 'tier_3']:
            if tier in config.get('data', {}).get('symbols', {}):
                symbols.extend(config['data']['symbols'][tier])
        
        timeframes = config.get('data', {}).get('timeframes', ['5m', '1h'])
        
        print(f"📊 Configuración cargada:")
        print(f"   • {len(symbols)} símbolos configurados")
        print(f"   • {len(timeframes)} timeframes")
        print()
        
        # Ejecutar signal_generator usando subprocess
        print("🚀 Ejecutando signal_generator.py...")
        script_path = os.path.join(os.path.dirname(__file__), 'signal_generator.py')
        
        # Configurar entorno con UTF-8
        env = os.environ.copy()
        env['PYTHONIOENCODING'] = 'utf-8'
        
        # Ejecutar el comando
        result = subprocess.run([
            sys.executable, script_path, '--mode', 'scan'
        ], capture_output=True, text=True, encoding='utf-8', 
           cwd=os.path.dirname(os.path.dirname(__file__)), env=env)
        
        if result.returncode == 0:
            print("✅ Signal generator ejecutado exitosamente")
            print("\n📝 Output del signal generator:")
            print("-" * 40)
            output_lines = result.stdout.split('\n')
            signal_count = 0
            for line in output_lines:
                if '🚨 SIGNAL:' in line:
                    signal_count += 1
                    print(f"   {line}")
                elif 'Signal scan completed:' in line:
                    print(f"   {line}")
            
            print(f"\n🎯 Total señales detectadas: {signal_count}")
        else:
            print("❌ Error ejecutando signal_generator")
            print(f"Error: {result.stderr}")
            return False
        
        # Calcular estadísticas
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        # Mostrar resumen
        print("\n" + "=" * 60)
        print("📊 RESUMEN DEL ESCANEO")
        print("=" * 60)
        print(f"⏱️  Tiempo total: {elapsed_time:.1f} segundos")
        
        print("\n" + "=" * 60)
        print("✅ Escaneo completado exitosamente")
        print("🌐 Dashboard disponible en: http://localhost:5000")
        print("=" * 60)
        
    except Exception as e:
        print(f"❌ Error durante el escaneo: {e}")
        logging.error(f"Error en run_full_scan: {e}", exc_info=True)
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
