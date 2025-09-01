#!/usr/bin/env python3
# scripts/integrate_feedback_system.py
"""
Script para integrar automáticamente el sistema de retroalimentación
con tu nvbot3 existente SIN modificar tu código principal
"""

import os
import sys
from pathlib import Path

def create_integration_bridge():
    """Crea el módulo de integración principal"""
    
    print("🌉 Creando bridge de integración...")
    
    # Asegurar que existe el directorio integration
    integration_dir = Path("integration")
    integration_dir.mkdir(exist_ok=True)
    
    # Crear __init__.py
    init_file = integration_dir / "__init__.py"
    with open(init_file, 'w', encoding='utf-8') as f:
        f.write("# NvBot3 Integration Package\n__version__ = '1.0.0'\n")
    
    # Código completo del bridge
    bridge_code = '''"""
Puente de integración entre nvbot3 existente y sistema de retroalimentación
Importar este módulo en tu código principal para activar tracking automático

EJEMPLO DE USO:
from integration.nvbot3_feedback_bridge import track_signal, update_price

# En tu función de predicción:
if prediction['confidence'] > 0.75:
    signal_id = track_signal(symbol, prediction, current_price)

# Al recibir nuevos precios:
update_price(symbol, new_price)
"""

import sys
import os
from datetime import datetime

# CONFIGURACIÓN DE RUTAS
current_dir = os.path.dirname(os.path.abspath(__file__))
web_dashboard_path = os.path.join(current_dir, '..', 'web_dashboard')
sys.path.append(web_dashboard_path)

# VARIABLES GLOBALES
TRACKER = None
TRACKING_ENABLED = False
DEBUG_MODE = True

def log_debug(message):
    """Log de depuración"""
    if DEBUG_MODE:
        timestamp = datetime.now().strftime('%H:%M:%S')
        print(f"[{timestamp}] 🔧 NvBot3-Feedback: {message}")

def init_tracker():
    """Inicializa el tracker de forma segura"""
    global TRACKER, TRACKING_ENABLED
    
    if TRACKER is not None:
        return TRACKER  # Ya inicializado
    
    try:
        # Importar SignalTracker
        from database.signal_tracker import SignalTracker
        
        # Crear instancia
        TRACKER = SignalTracker()
        TRACKING_ENABLED = True
        
        log_debug("Sistema de tracking inicializado correctamente")
        return TRACKER
        
    except ImportError as e:
        log_debug(f"Error importando SignalTracker: {e}")
        log_debug("Verifica que web_dashboard/database/signal_tracker.py existe")
        TRACKING_ENABLED = False
        return None
        
    except Exception as e:
        log_debug(f"Error inicializando SignalTracker: {e}")
        TRACKING_ENABLED = False
        return None

def track_signal(symbol, prediction_data, current_price):
    """
    Función principal para trackear señales del nvbot3
    
    Args:
        symbol (str): Símbolo de trading (ej: 'BTCUSDT')
        prediction_data (dict): Datos de predicción con claves:
            - 'type': Tipo de señal ('momentum_alto', 'rebound', etc.)
            - 'predicted_change': Cambio porcentual esperado
            - 'confidence': Nivel de confianza (0-1)
        current_price (float): Precio actual del activo
    
    Returns:
        str: ID de la señal si se guarda exitosamente, None si hay error
    
    Ejemplo de uso:
        prediction = {
            'type': 'momentum_alto',
            'predicted_change': 5.2,
            'confidence': 0.85
        }
        signal_id = track_signal('BTCUSDT', prediction, 67250.0)
    """
    
    # Inicializar tracker si es necesario
    if not TRACKING_ENABLED:
        init_tracker()
        if not TRACKING_ENABLED:
            log_debug(f"Tracking deshabilitado - no se guardará señal de {symbol}")
            return None
    
    try:
        # Validar datos de entrada
        if not symbol or not isinstance(symbol, str):
            log_debug("Error: símbolo inválido")
            return None
        
        if not prediction_data or not isinstance(prediction_data, dict):
            log_debug("Error: prediction_data debe ser un diccionario")
            return None
        
        if current_price is None or current_price <= 0:
            log_debug("Error: current_price debe ser un número positivo")
            return None
        
        # Extraer campos de predicción
        signal_type = prediction_data.get('type', 'unknown')
        predicted_change = float(prediction_data.get('predicted_change', 0))
        confidence = float(prediction_data.get('confidence', 0))
        
        # Preparar datos para el tracker
        signal_data = {
            'type': signal_type,
            'predicted_change': predicted_change,
            'confidence': confidence,
            'entry_price': current_price,
            'timestamp': datetime.now().isoformat()
        }
        
        # Guardar señal usando SignalTracker
        signal_id = TRACKER.save_new_signal(symbol, signal_data)
        
        if signal_id:
            log_debug(f"📊 Señal guardada: {symbol} | {signal_type} | Confianza: {confidence:.2f} | ID: {signal_id}")
            return signal_id
        else:
            log_debug(f"⚠️ No se pudo guardar señal para {symbol}")
            return None
            
    except Exception as e:
        log_debug(f"❌ Error guardando señal para {symbol}: {e}")
        return None

def update_price(symbol, price):
    """
    Actualiza el precio para todas las señales activas de un símbolo
    Llamar esta función cada vez que recibas un nuevo precio
    
    Args:
        symbol (str): Símbolo del activo
        price (float): Precio actual
    
    Ejemplo:
        update_price('BTCUSDT', 68500.0)
    """
    
    if not TRACKING_ENABLED:
        init_tracker()
        if not TRACKING_ENABLED:
            return
    
    try:
        if not symbol or price <= 0:
            log_debug(f"Datos inválidos para actualización de precio: {symbol}, {price}")
            return
        
        TRACKER.update_price_tracking(symbol, price)
        log_debug(f"📈 Precio actualizado: {symbol} = ${price:,.2f}")
        
    except Exception as e:
        log_debug(f"❌ Error actualizando precio de {symbol}: {e}")

def get_tracking_stats(days=30):
    """
    Obtiene estadísticas de performance del tracking
    
    Args:
        days (int): Número de días para las estadísticas
        
    Returns:
        dict: Estadísticas de las señales
    """
    
    if not TRACKING_ENABLED:
        init_tracker()
        if not TRACKING_ENABLED:
            return {"error": "Sistema de tracking no disponible"}
    
    try:
        stats = TRACKER.get_performance_stats(days=days)
        log_debug(f"📊 Estadísticas obtenidas: {stats}")
        return stats
    except Exception as e:
        log_debug(f"❌ Error obteniendo estadísticas: {e}")
        return {"error": str(e)}

def get_active_signals():
    """
    Obtiene todas las señales actualmente siendo monitoreadas
    
    Returns:
        list: Lista de señales activas
    """
    
    if not TRACKING_ENABLED:
        init_tracker()
        if not TRACKING_ENABLED:
            return []
    
    try:
        signals = TRACKER.get_active_signals()
        log_debug(f"📋 Señales activas obtenidas: {len(signals)}")
        return signals
    except Exception as e:
        log_debug(f"❌ Error obteniendo señales activas: {e}")
        return []

def save_feedback(signal_id, feedback_data):
    """
    Guarda feedback manual sobre una señal
    
    Args:
        signal_id (str): ID de la señal
        feedback_data (dict): Datos del feedback
    """
    
    if not TRACKING_ENABLED:
        init_tracker()
        if not TRACKING_ENABLED:
            return False
    
    try:
        TRACKER.save_user_feedback(signal_id, feedback_data)
        log_debug(f"💬 Feedback guardado para señal {signal_id}")
        return True
    except Exception as e:
        log_debug(f"❌ Error guardando feedback: {e}")
        return False

def enable_debug(enabled=True):
    """Habilita/deshabilita logs de depuración"""
    global DEBUG_MODE
    DEBUG_MODE = enabled
    log_debug(f"Debug mode {'habilitado' if enabled else 'deshabilitado'}")

def get_system_status():
    """
    Obtiene el estado del sistema de tracking
    
    Returns:
        dict: Estado del sistema
    """
    
    status = {
        'tracking_enabled': TRACKING_ENABLED,
        'tracker_initialized': TRACKER is not None,
        'debug_mode': DEBUG_MODE,
        'python_version': sys.version,
        'timestamp': datetime.now().isoformat()
    }
    
    if TRACKING_ENABLED and TRACKER:
        try:
            # Intentar obtener estadísticas básicas
            stats = TRACKER.get_performance_stats(days=1)
            status['recent_signals'] = stats.get('total_signals', 0)
            status['database_accessible'] = True
        except Exception as e:
            status['database_accessible'] = False
            status['database_error'] = str(e)
    
    return status

# Aliases para compatibilidad
manual_price_update = update_price  # Alias del nombre original
get_stats = get_tracking_stats      # Alias más corto

# Función de ejemplo y demostración
def run_integration_demo():
    """
    Ejecuta una demostración del sistema de integración
    Útil para probar que todo funcione correctamente
    """
    
    print("🤖 DEMOSTRACIÓN DEL SISTEMA DE INTEGRACIÓN NVBOT3")
    print("=" * 55)
    
    # Mostrar estado del sistema
    status = get_system_status()
    print(f"📊 Estado del sistema:")
    for key, value in status.items():
        print(f"   {key}: {value}")
    
    if not TRACKING_ENABLED:
        print("\\n❌ Sistema de tracking no disponible")
        print("💡 Ejecuta: python scripts/fix_import_errors.py")
        return
    
    print("\\n🧪 Ejecutando pruebas de integración...")
    
    # Datos de prueba similares a nvbot3 real
    test_scenarios = [
        {
            'symbol': 'BTCUSDT',
            'prediction': {
                'type': 'momentum_alto',
                'predicted_change': 4.8,
                'confidence': 0.87
            },
            'price': 67250.0
        },
        {
            'symbol': 'ETHUSDT',
            'prediction': {
                'type': 'rebound',
                'predicted_change': 3.2,
                'confidence': 0.73
            },
            'price': 3180.0
        }
    ]
    
    signals_created = []
    
    # Crear señales de prueba
    for i, scenario in enumerate(test_scenarios):
        print(f"\\n📡 Prueba {i+1}: {scenario['symbol']}")
        
        signal_id = track_signal(
            scenario['symbol'],
            scenario['prediction'],
            scenario['price']
        )
        
        if signal_id:
            signals_created.append(signal_id)
            print(f"   ✅ Señal creada: {signal_id}")
            
            # Simular actualización de precio
            new_price = scenario['price'] * 1.02  # +2%
            update_price(scenario['symbol'], new_price)
            print(f"   📈 Precio actualizado: ${new_price:,.2f}")
        else:
            print(f"   ❌ Error creando señal")
    
    # Mostrar estadísticas finales
    print(f"\\n📊 RESULTADO DE LA DEMOSTRACIÓN:")
    print(f"   Señales creadas: {len(signals_created)}")
    
    if signals_created:
        stats = get_tracking_stats()
        print(f"   Estadísticas: {stats}")
        
        active_signals = get_active_signals()
        print(f"   Señales activas: {len(active_signals)}")
        
        print("\\n🎉 ¡Integración funcionando correctamente!")
        print("💡 Ahora puedes integrar estas funciones en tu código nvbot3")
    else:
        print("\\n⚠️ No se pudieron crear señales de prueba")
        print("🔧 Verifica la configuración del sistema")

if __name__ == "__main__":
    print("🔧 NVBOT3 FEEDBACK INTEGRATION BRIDGE")
    print("Ejecutando demostración del sistema...")
    print()
    
    run_integration_demo()
'''
    
    # Escribir el archivo bridge
    bridge_file = integration_dir / "nvbot3_feedback_bridge.py"
    with open(bridge_file, 'w', encoding='utf-8') as f:
        f.write(bridge_code)
    
    print(f"   ✅ Bridge creado: {bridge_file}")
    return True

def create_startup_script():
    """Crea script de inicio del dashboard"""
    
    print("🚀 Creando script de inicio...")
    
    # Asegurar que existe el directorio scripts
    scripts_dir = Path("scripts")
    scripts_dir.mkdir(exist_ok=True)
    
    # Código del script de inicio (ya lo tenemos en otros artifacts)
    startup_code = '''#!/usr/bin/env python3
# scripts/start_dashboard.py
"""
Script simplificado para iniciar el dashboard
Creado automáticamente por integrate_feedback_system.py
"""

import sys
import os
import subprocess
from pathlib import Path

def main():
    print("🌐 Iniciando NvBot3 Dashboard...")
    
    # Verificar que existe la aplicación
    if not Path("web_dashboard/app.py").exists():
        print("❌ Error: No se encontró web_dashboard/app.py")
        print("🔧 Ejecuta: python scripts/integrate_feedback_system.py")
        return
    
    # Cambiar al directorio del dashboard
    original_dir = os.getcwd()
    
    try:
        os.chdir("web_dashboard")
        print("📊 Iniciando servidor Flask en http://localhost:5000")
        print("⏹️  Presiona Ctrl+C para detener")
        
        # Ejecutar la aplicación Flask
        subprocess.run([sys.executable, "app.py"], check=True)
        
    except KeyboardInterrupt:
        print("\\n🛑 Dashboard detenido")
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        os.chdir(original_dir)

if __name__ == "__main__":
    main()
'''
    
    startup_file = scripts_dir / "start_dashboard.py"
    with open(startup_file, 'w', encoding='utf-8') as f:
        f.write(startup_code)
    
    print(f"   ✅ Script de inicio creado: {startup_file}")
    return True

def create_example_integration():
    """Crea ejemplo de cómo integrar con nvbot3 existente"""
    
    print("📚 Creando ejemplos de integración...")
    
    # Asegurar que existe el directorio examples
    examples_dir = Path("examples")
    examples_dir.mkdir(exist_ok=True)
    
    # Código de ejemplo completo
    example_code = '''#!/usr/bin/env python3
# examples/nvbot3_con_tracking.py
"""
EJEMPLO: Cómo integrar el sistema de feedback con tu nvbot3 existente

ANTES de usar este ejemplo:
1. Ejecuta: python scripts/integrate_feedback_system.py
2. Instala: pip install flask flask-socketio pandas
3. Inicia dashboard: python scripts/start_dashboard.py (en otra terminal)

DESPUÉS de integrar:
- Todas las señales con buena confianza se guardarán automáticamente
- Puedes ver las señales en: http://localhost:5000
- Puedes dar feedback sobre cada predicción
- El sistema aprende de tu feedback
"""

import asyncio
import time
from datetime import datetime
import random

# PASO 1: IMPORTAR EL SISTEMA DE TRACKING
# Esta es la ÚNICA línea que necesitas agregar a tu código existente
from integration.nvbot3_feedback_bridge import track_signal, update_price

# Simulación de tu código nvbot3 existente
# (Reemplaza estas funciones con tu código real)

def get_market_data(symbol):
    """
    Placeholder - reemplazar con tu función real de obtener datos de mercado
    """
    # Simulación de datos de mercado
    base_prices = {
        'BTCUSDT': 67250.0,
        'ETHUSDT': 3180.0,
        'ADAUSDT': 0.45
    }
    
    base_price = base_prices.get(symbol, 50000.0)
    current_price = base_price * (1 + random.uniform(-0.02, 0.02))  # ±2% variación
    
    return {
        'symbol': symbol,
        'close': current_price,
        'volume': random.uniform(1000000, 5000000),
        'high': current_price * 1.01,
        'low': current_price * 0.99,
        'timestamp': datetime.now()
    }

def analyze_market_conditions(symbol, market_data):
    """
    Placeholder - reemplazar con tu análisis real
    """
    # Simulación de análisis técnico
    conditions = ['momentum_alto', 'rebound', 'consolidacion', 'regime']
    return {
        'trend': random.choice(['bullish', 'bearish', 'sideways']),
        'volatility': random.uniform(0.1, 0.8),
        'signal_type': random.choice(conditions),
        'strength': random.uniform(0.3, 0.9)
    }

def generate_prediction(symbol, market_data, market_conditions):
    """
    Placeholder - reemplazar con tu lógica de predicción real
    """
    # Simulación de tu sistema de predicción
    signal_types = ['momentum_alto', 'rebound', 'consolidacion']
    
    prediction = {
        'type': random.choice(signal_types),
        'predicted_change': round(random.uniform(1.5, 8.0), 2),
        'confidence': round(random.uniform(0.4, 0.95), 2),
        'timeframe': random.choice([120, 240, 360]),  # minutos
        'entry_price': market_data['close'],
        'stop_loss': market_data['close'] * 0.95,
        'take_profit': market_data['close'] * 1.05
    }
    
    return prediction

# FUNCIÓN PRINCIPAL DE TU BOT (MODIFICADA MÍNIMAMENTE)
async def nvbot3_main_loop():
    """
    Tu loop principal de trading - MODIFICADO MÍNIMAMENTE
    Solo se agregaron 2 líneas para el tracking automático
    """
    
    print("🤖 Iniciando NvBot3 con sistema de tracking")
    print("📊 Dashboard disponible en: http://localhost:5000")
    print("-" * 50)
    
    symbols = ['BTCUSDT', 'ETHUSDT', 'ADAUSDT']
    
    while True:
        try:
            for symbol in symbols:
                print(f"\\n📡 Analizando {symbol}...")
                
                # TU CÓDIGO EXISTENTE - SIN CAMBIOS
                market_data = get_market_data(symbol)
                market_conditions = analyze_market_conditions(symbol, market_data)
                prediction = generate_prediction(symbol, market_data, market_conditions)
                
                print(f"   💹 Precio: ${market_data['close']:,.2f}")
                print(f"   🎯 Predicción: {prediction['type']}")
                print(f"   📊 Confianza: {prediction['confidence']:.2f}")
                print(f"   📈 Cambio esperado: {prediction['predicted_change']:+.2f}%")
                
                # NUEVA LÍNEA 1: TRACKEAR SEÑALES CON BUENA CONFIANZA
                # Esta es la integración principal - solo 1 línea
                if prediction['confidence'] > 0.70:  # Threshold ajustable
                    signal_id = track_signal(symbol, prediction, market_data['close'])
                    if signal_id:
                        print(f"   📝 Señal trackeada: {signal_id}")
                
                # NUEVA LÍNEA 2: ACTUALIZAR PRECIO PARA TRACKING
                # Esto mantiene actualizado el sistema de tracking
                update_price(symbol, market_data['close'])
                
                # TU CÓDIGO EXISTENTE CONTINÚA SIN CAMBIOS
                # Aquí irían tus órdenes de trading, stop losses, etc.
                
                # Simulación de espera entre análisis
                await asyncio.sleep(2)  # 2 segundos entre símbolos
            
            print("\\n⏳ Esperando próximo ciclo de análisis...")
            await asyncio.sleep(30)  # 30 segundos entre ciclos completos
            
        except KeyboardInterrupt:
            print("\\n🛑 Bot detenido por el usuario")
            break
        except Exception as e:
            print(f"❌ Error en el loop principal: {e}")
            await asyncio.sleep(10)  # Esperar antes de reintentar

# FUNCIÓN DE MONITOREO OPCIONAL
def mostrar_estadisticas_tracking():
    """
    Función opcional para mostrar estadísticas del tracking
    """
    try:
        from integration.nvbot3_feedback_bridge import get_tracking_stats, get_active_signals
        
        stats = get_tracking_stats(days=7)  # Últimos 7 días
        active = get_active_signals()
        
        print("\\n📊 ESTADÍSTICAS DE TRACKING:")
        print(f"   Señales totales (7 días): {stats.get('total_signals', 0)}")
        print(f"   Tasa de éxito: {stats.get('success_rate', 0):.1f}%")
        print(f"   Confianza promedio: {stats.get('average_confidence', 0):.2f}")
        print(f"   Señales activas: {len(active)}")
        
    except Exception as e:
        print(f"⚠️ Error obteniendo estadísticas: {e}")

# PUNTO DE ENTRADA PRINCIPAL
if __name__ == "__main__":
    print("🚀 NVBOT3 CON SISTEMA DE TRACKING INTEGRADO")
    print("=" * 50)
    
    # Mostrar estadísticas iniciales
    mostrar_estadisticas_tracking()
    
    print("\\n🎯 Iniciando trading automático...")
    print("💡 Tip: Abre http://localhost:5000 para ver las señales en tiempo real")
    
    try:
        # Ejecutar el loop principal
        asyncio.run(nvbot3_main_loop())
    except KeyboardInterrupt:
        print("\\n👋 ¡Bot detenido correctamente!")
    except Exception as e:
        print(f"\\n❌ Error fatal: {e}")
        print("🔧 Verifica la configuración del sistema")

# INSTRUCCIONES PARA INTEGRACIÓN REAL:
"""
PARA INTEGRAR CON TU CÓDIGO REAL:

1. IMPORTAR (al inicio de tu archivo):
   from integration.nvbot3_feedback_bridge import track_signal, update_price

2. TRACKEAR SEÑALES (en tu función de predicción):
   if prediction['confidence'] > 0.75:  # Threshold que prefieras
       track_signal(symbol, prediction, current_price)

3. ACTUALIZAR PRECIOS (cada vez que obtengas nuevos precios):
   update_price(symbol, new_price)

¡Eso es todo! Solo 2-3 líneas de código adicional.

BENEFICIOS:
- Tracking automático de todas las señales
- Interfaz web para ver resultados en tiempo real  
- Sistema de feedback para mejorar predicciones
- Base de datos histórica de performance
- Análisis estadístico automático

DASHBOARD WEB:
- http://localhost:5000 - Ver señales activas
- http://localhost:5000/api/test - Probar API
"""
'''
    
    example_file = examples_dir / "nvbot3_con_tracking.py"
    with open(example_file, 'w', encoding='utf-8') as f:
        f.write(example_code)
    
    print(f"   ✅ Ejemplo completo creado: {example_file}")
    return True

def create_webapp_structure():
    """Crea la estructura básica de la aplicación web"""
    
    print("🌐 Creando estructura de la aplicación web...")
    
    # Crear directorios necesarios
    web_dirs = [
        "web_dashboard",
        "web_dashboard/templates",
        "web_dashboard/static/css",
        "web_dashboard/static/js",
        "web_dashboard/database"
    ]
    
    for directory in web_dirs:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"   ✅ {directory}")
    
    # Crear archivo básico de Flask si no existe
    app_file = Path("web_dashboard/app.py")
    if not app_file.exists():
        print("   ⚠️ web_dashboard/app.py no existe")
        print("   💡 Este archivo debe ser creado por separado con el código Flask completo")
        
        # Crear un app.py mínimo para que no falle
        minimal_app = '''# web_dashboard/app.py - VERSIÓN MÍNIMA
# REEMPLAZAR con la versión completa del artifact

from flask import Flask
import sys
import os

app = Flask(__name__)

@app.route('/')
def dashboard():
    return """
    <h1>NvBot3 Dashboard</h1>
    <p>Sistema de retroalimentación inicializado</p>
    <p>Reemplaza este archivo con la versión completa del app.py</p>
    """

@app.route('/api/test')
def test():
    return {"status": "Sistema funcionando", "message": "Reemplaza app.py con la versión completa"}

if __name__ == '__main__':
    print("🌐 Dashboard mínimo iniciado en http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)
'''
        
        with open(app_file, 'w', encoding='utf-8') as f:
            f.write(minimal_app)
        
        print(f"   ✅ Archivo mínimo creado: {app_file}")
    
    return True

def create_installation_summary():
    """Crea un resumen de instalación"""
    
    summary = """# 📋 RESUMEN DE INSTALACIÓN - NVBOT3 FEEDBACK SYSTEM

## ✅ ARCHIVOS CREADOS:

### 🌉 Integración:
- `integration/nvbot3_feedback_bridge.py` - Bridge principal para conectar con tu bot
- `integration/__init__.py` - Package initialization

### 🚀 Scripts:
- `scripts/start_dashboard.py` - Iniciar el dashboard web
- `scripts/integrate_feedback_system.py` - Este script de configuración

### 📚 Ejemplos:
- `examples/nvbot3_con_tracking.py` - Ejemplo completo de integración

### 🌐 Estructura Web:
- `web_dashboard/` - Directorio principal de la aplicación web
- `web_dashboard/templates/` - Templates HTML
- `web_dashboard/static/css/` - Estilos CSS  
- `web_dashboard/static/js/` - Scripts JavaScript
- `web_dashboard/database/` - Base de datos y modelos

## 📋 PRÓXIMOS PASOS:

### 1. 📦 Instalar dependencias:
```bash
pip install flask flask-socketio pandas
```

### 2. 🔧 Completar archivos faltantes:
- Copia `signal_tracker.py` completo en `web_dashboard/database/`
- Copia `app.py` completo en `web_dashboard/`
- Copia `dashboard.html` completo en `web_dashboard/templates/`

### 3. 🧪 Probar el sistema:
```bash
python scripts/test_feedback_system.py
```

### 4. 🌐 Iniciar el dashboard:
```bash  
python scripts/start_dashboard.py
```

### 5. 🤖 Integrar con tu bot:
```python
# Al inicio de tu archivo nvbot3:
from integration.nvbot3_feedback_bridge import track_signal, update_price

# En tu función de predicción:
if prediction['confidence'] > 0.75:
    track_signal(symbol, prediction, current_price)

# Al recibir nuevos precios:
update_price(symbol, new_price)
```

## 🎯 RESULTADO ESPERADO:
- Dashboard web funcionando en http://localhost:5000
- Tracking automático de todas las señales con buena confianza
- Interfaz para dar feedback sobre predicciones
- Base de datos histórica de performance
- Estadísticas en tiempo real

## 🔧 SOLUCIÓN DE PROBLEMAS:
- Error de imports: `python scripts/fix_import_errors.py`
- Sistema no funciona: Verificar que todos los archivos estén completos
- Dashboard no inicia: Revisar dependencias de Flask

¡Sistema listo para integración! 🎉
"""
    
    summary_file = Path("INSTALACION_COMPLETADA.md")
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write(summary)
    
    print(f"   ✅ Resumen creado: {summary_file}")
    return True

def main():
    """Función principal de integración"""
    
    print("🔧 INTEGRADOR AUTOMÁTICO - NVBOT3 FEEDBACK SYSTEM")
    print("=" * 60)
    
    # Verificar entorno actual
    if 'nvbot3_env' not in sys.executable:
        print("⚠️ Advertencia: Entorno virtual nvbot3_env no detectado")
        print("   Recomendado: nvbot3_env\\Scripts\\activate")
        print()
    
    print(f"🐍 Python: {sys.version}")
    print(f"📂 Directorio: {os.getcwd()}")
    print()
    
    # Pasos de integración
    integration_steps = [
        ("🌉 Creando bridge de integración", create_integration_bridge),
        ("🚀 Creando script de inicio", create_startup_script),
        ("📚 Creando ejemplo de integración", create_example_integration),
        ("🌐 Creando estructura web", create_webapp_structure),
        ("📋 Creando resumen de instalación", create_installation_summary)
    ]
    
    completed_steps = 0
    total_steps = len(integration_steps)
    
    for step_name, step_function in integration_steps:
        print(f"{step_name}")
        print("-" * len(step_name))
        
        try:
            success = step_function()
            if success:
                completed_steps += 1
                print(f"   ✅ Completado exitosamente\n")
            else:
                print(f"   ⚠️ Completado con advertencias\n")
        except Exception as e:
            print(f"   ❌ Error: {e}\n")
    
    # Resumen final
    print("=" * 60)
    print("🎯 RESUMEN DE INTEGRACIÓN")
    print("=" * 60)
    
    success_rate = (completed_steps / total_steps) * 100
    print(f"✅ Pasos completados: {completed_steps}/{total_steps} ({success_rate:.0f}%)")
    
    if completed_steps == total_steps:
        print("\n🎉 ¡INTEGRACIÓN COMPLETADA EXITOSAMENTE!")
        print("=" * 45)
        
        print("\n📋 ARCHIVOS CREADOS:")
        print("   🌉 integration/nvbot3_feedback_bridge.py")
        print("   🚀 scripts/start_dashboard.py")  
        print("   📚 examples/nvbot3_con_tracking.py")
        print("   📋 INSTALACION_COMPLETADA.md")
        
        print("\n📋 PRÓXIMOS PASOS OBLIGATORIOS:")
        print("   1. 📦 Instalar dependencias: pip install flask flask-socketio pandas")
        print("   2. 📁 Completar archivos faltantes (signal_tracker.py, app.py, etc.)")
        print("   3. 🧪 Probar: python examples/nvbot3_con_tracking.py")
        print("   4. 🌐 Dashboard: python scripts/start_dashboard.py")
        
        print("\n🤖 INTEGRACIÓN CON TU BOT:")
        print("   ✏️ Agregar 1 línea al inicio: from integration.nvbot3_feedback_bridge import track_signal")
        print("   ✏️ Agregar en predicción: track_signal(symbol, prediction, price)")
        print("   ✅ ¡Tu bot tendrá tracking automático!")
        
    else:
        print(f"\n⚠️ INTEGRACIÓN PARCIALMENTE COMPLETADA ({success_rate:.0f}%)")
        print("🔧 Algunos pasos tuvieron problemas, pero el sistema básico está listo")
    
    print(f"\n📖 Lee INSTALACION_COMPLETADA.md para instrucciones detalladas")
    print("🆘 Si hay problemas: python scripts/fix_import_errors.py")
    
    return completed_steps == total_steps

if __name__ == "__main__":
    success = main()
    
    if success:
        print("\n🚀 ¡Integración lista! Procede con la instalación de dependencias.")
    else:
        print("\n⚠️ Revisa los errores mostrados y vuelve a intentar.")