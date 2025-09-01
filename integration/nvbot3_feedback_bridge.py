# integration/nvbot3_feedback_bridge.py
"""
Puente de integración entre nvbot3 existente y sistema de retroalimentación
Importar este módulo en tu código principal para activar tracking automático
"""

import sys
import os
from datetime import datetime

# CORRECCIÓN: Ruta simplificada y más robusta
current_dir = os.path.dirname(os.path.abspath(__file__))
web_dashboard_path = os.path.join(current_dir, '..', 'web_dashboard')
sys.path.append(web_dashboard_path)

project_root = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(project_root)
try:
    from web_dashboard.database.signal_tracker import SignalTracker
except ImportError as e:
    print(f"WARNING: Error importando SignalTracker: {e}")

# Variables globales para el tracker
TRACKER = None
TRACKING_ENABLED = False

def init_tracker():
    """Inicializa el tracker de forma segura"""
    global TRACKER, TRACKING_ENABLED

    if TRACKER is not None:
        print("DEBUG: TRACKER ya inicializado.")
        return TRACKER  # Ya inicializado

    try:
        print("DEBUG: Intentando importar SignalTracker...")
        sys.path.append(os.path.abspath(os.path.join(current_dir, '..', 'web_dashboard', 'database')))
        from web_dashboard.database.signal_tracker import SignalTracker
        print("DEBUG: SignalTracker importado correctamente.")
        TRACKER = SignalTracker()
        if TRACKER:
            TRACKING_ENABLED = True
            print("SUCCESS: Sistema de tracking NvBot3 inicializado correctamente")
        else:
            TRACKING_ENABLED = False
            print("WARNING: TRACKER no pudo inicializarse correctamente")
        return TRACKER
    except ImportError as e:
        print(f"WARNING: Error importando SignalTracker: {e}")
        TRACKING_ENABLED = False
        return None
    except Exception as e:
        print(f"ERROR: Error inicializando SignalTracker: {e}")
        TRACKING_ENABLED = False
        return None

# Ensure TRACKER is initialized before accessing its methods
def ensure_tracker_initialized():
    global TRACKER, TRACKING_ENABLED
    if TRACKER is None:
        init_tracker()

def track_signal(symbol, prediction_data, current_price):
    """
    Función principal para trackear señales
    
    Uso en tu código nvbot3:
    from integration.nvbot3_feedback_bridge import track_signal
    
    # Después de generar una predicción:
    if prediction['confidence'] > 0.75:
        track_signal(symbol, prediction, current_price)
    
    Args:
        symbol (str): Símbolo de trading (ej: 'BTCUSDT')
        prediction_data (dict): Datos de predicción con 'type', 'predicted_change', 'confidence'
        current_price (float): Precio actual del activo
    
    Returns:
        str: ID de la señal si se guarda exitosamente, None si hay error
    """
    
    ensure_tracker_initialized()
    if not TRACKING_ENABLED:
        return None
    
    try:
        # Preparar datos para el tracker
        signal_data = {
            'type': prediction_data.get('type', 'unknown'),
            'predicted_change': prediction_data.get('predicted_change', 0),
            'confidence': prediction_data.get('confidence', 0),
            'entry_price': current_price,
            'timestamp': datetime.now().isoformat()
        }
        
        # Guardar la señal
        if TRACKER is not None:
            signal_id = TRACKER.save_new_signal(symbol, signal_data)
        else:
            print("WARNING: TRACKER no está inicializado. No se puede guardar la señal.")
            return None
        
        if signal_id:
            print(f"SUCCESS: Señal guardada: {symbol} - {prediction_data.get('type')} - Confianza: {prediction_data.get('confidence', 0):.2f}")
            return signal_id
        else:
            print(f"WARNING: No se pudo guardar la señal para {symbol}")
            return None
            
    except Exception as e:
        print(f"ERROR: Error guardando señal para {symbol}: {e}")
        return None

def update_price(symbol, price):
    """
    Actualiza el precio para todas las señales activas de un símbolo
    Llamar esta función cada vez que obtengas nuevo precio de mercado
    
    Args:
        symbol (str): Símbolo del activo
        price (float): Precio actual
    """
    
    ensure_tracker_initialized()
    if not TRACKING_ENABLED:
        return
    
    try:
        if TRACKER is not None:
            TRACKER.update_price_tracking(symbol, price)
        else:
            print("WARNING: TRACKER no está inicializado. No se puede actualizar el precio.")
    except Exception as e:
        print(f"ERROR: Error actualizando precio de {symbol}: {e}")

def get_tracking_stats():
    """
    Obtiene estadísticas de performance del tracking
    
    Returns:
        dict: Estadísticas de las señales
    """
    
    ensure_tracker_initialized()
    if not TRACKING_ENABLED:
        return {"error": "Sistema de tracking no disponible"}
    
    try:
        if TRACKER is not None:
            stats = TRACKER.get_performance_stats(days=30)
        else:
            print("WARNING: TRACKER no está inicializado. No se pueden obtener estadísticas.")
            stats = {"error": "TRACKER no inicializado"}
        return stats
    except Exception as e:
        print(f"ERROR: Error obteniendo estadísticas: {e}")
        return {"error": str(e)}

def get_active_signals():
    """
    Obtiene todas las señales actualmente siendo monitoreadas
    
    Returns:
        list: Lista de señales activas
    """
    
    ensure_tracker_initialized()
    if not TRACKING_ENABLED:
        return []
    
    try:
        if TRACKER is not None:
            signals = TRACKER.get_active_signals()
        else:
            print("WARNING: TRACKER no está inicializado.")
            signals = []
        return signals
    except Exception as e:
        print(f"ERROR: Error obteniendo señales activas: {e}")
        return []

def manual_price_update(symbol, price):
    """Alias para update_price (compatibilidad)"""
    return update_price(symbol, price)

# Función de ejemplo para mostrar cómo integrar
def example_integration():
    """
    Ejemplo de cómo integrar este sistema con tu código nvbot3 existente
    """
    
    print("🤖 Ejemplo de integración con NvBot3:")
    print("=" * 50)
    
    # Simulación de datos de tu nvbot3
    symbol = "BTCUSDT"
    current_price = 67250.0
    
    # Simulación de una predicción de tu bot
    prediction = {
        'type': 'momentum_alto',
        'predicted_change': 5.2,
        'confidence': 0.85
    }
    
    # Trackear la señal
    signal_id = track_signal(symbol, prediction, current_price)
    
    if signal_id:
        print(f"SUCCESS: Señal trackeada con ID: {signal_id}")
        
        # Simular actualización de precio después de 10 minutos
        new_price = 68500.0
        update_price(symbol, new_price)
        print(f"SUCCESS: Precio actualizado: {new_price}")
        
        # Obtener estadísticas
        stats = get_tracking_stats()
        print(f"INFO: Estadísticas: {stats}")
        
    else:
        print("ERROR: Error trackeando señal")

if __name__ == "__main__":
    print("Probando sistema de integración NvBot3...")
    example_integration()