"""
Puente de integración entre nvbot3 existente y sistema de retroalimentación
Importar este módulo en tu código principal para activar tracking automático
"""

import sys
import os
from datetime import datetime

# Agregar path del sistema web
current_dir = os.path.dirname(os.path.abspath(__file__))
web_dashboard_database_path = os.path.join(current_dir, '..', 'web_dashboard', 'database')
if web_dashboard_database_path not in sys.path:
    sys.path.append(web_dashboard_database_path)

try:
    from web_dashboard.database.signal_tracker import SignalTracker
    tracker = SignalTracker()
except Exception as e:
    print(f"Error al inicializar SignalTracker: {e}")

def track_signal(symbol, prediction_data, current_price):
    """
    Función para guardar una señal en la base de datos
    """
    try:
        tracker.save_new_signal(symbol, {
            'prediction_data': prediction_data,
            'current_price': current_price,
            'timestamp': datetime.now().isoformat()
        })
        print(f"Señal guardada para {symbol}")
    except Exception as e:
        print(f"Error al guardar señal: {e}")

def get_tracking_stats():
    """
    Obtener estadísticas de tracking
    """
    try:
        return tracker.get_performance_stats()
    except Exception as e:
        print(f"Error al obtener estadísticas: {e}")
        return {}

def manual_price_update(symbol, price):
    """
    Actualizar manualmente el precio de un símbolo
    """
    try:
        tracker.update_price_tracking(symbol, price)
        print(f"Precio actualizado para {symbol}")
    except Exception as e:
        print(f"Error al actualizar precio: {e}")
