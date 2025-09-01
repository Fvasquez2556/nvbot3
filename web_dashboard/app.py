# web_dashboard/app.py
from flask import Flask, render_template, request, jsonify
import sqlite3
import json
from datetime import datetime
import sys
import os

# CORRECCIÓN: Ruta de importación simplificada
sys.path.append(os.path.dirname(__file__))
from database.signal_tracker import SignalTracker

# Configuración de Flask
app = Flask(__name__)
app.config['SECRET_KEY'] = 'nvbot3-secret-key-2025'

# Inicializar componentes
try:
    signal_tracker = SignalTracker()
    print("✅ SignalTracker inicializado correctamente")
except Exception as e:
    print(f"❌ Error inicializando SignalTracker: {e}")
    signal_tracker = None

@app.route('/')
def dashboard():
    """Página principal del dashboard"""
    try:
        if signal_tracker:
            active_signals = signal_tracker.get_active_signals()
            stats = signal_tracker.get_performance_stats(days=30)
        else:
            active_signals = []
            stats = {"total_signals": 0, "success_rate": 0, "average_confidence": 0}
        
        return render_template('dashboard.html', 
                             active_signals=active_signals, 
                             stats=stats,
                             current_time=datetime.now())
    except Exception as e:
        print(f"❌ Error en dashboard: {e}")
        return f"Error cargando dashboard: {e}", 500

@app.route('/api/signals/active')
def get_active_signals():
    """API endpoint para obtener señales activas"""
    try:
        if signal_tracker:
            signals = signal_tracker.get_active_signals()
            return jsonify({"status": "success", "data": signals})
        else:
            return jsonify({"status": "error", "message": "SignalTracker no disponible"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

@app.route('/api/feedback', methods=['POST'])
def submit_feedback():
    """Recibir retroalimentación del usuario"""
    try:
        if not signal_tracker:
            return jsonify({'status': 'error', 'message': 'SignalTracker no disponible'})
        
        data = request.get_json()
        signal_id = data.get('signal_id')
        feedback_type = data.get('feedback_type')
        
        if not signal_id or not feedback_type:
            return jsonify({'status': 'error', 'message': 'Datos incompletos'})
        
        feedback_data = {
            'type': feedback_type,
            'result': data.get('result', ''),
            'actual_change': data.get('actual_change'),
            'notes': data.get('comments', '')
        }
        
        signal_tracker.save_user_feedback(signal_id, feedback_data)
        return jsonify({'status': 'success', 'message': 'Feedback guardado correctamente'})
        
    except Exception as e:
        print(f"❌ Error guardando feedback: {e}")
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/api/test')
def test_endpoint():
    """Endpoint de prueba para verificar que la API funciona"""
    return jsonify({
        "status": "success", 
        "message": "NvBot3 Dashboard API funcionando correctamente",
        "timestamp": datetime.now().isoformat(),
        "tracker_status": "disponible" if signal_tracker else "no disponible"
    })

if __name__ == '__main__':
    print("🚀 Iniciando NvBot3 Dashboard...")
    print("🌐 Dashboard disponible en: http://localhost:5000")
    print("📡 API disponible en: http://localhost:5000/api/test")
    print("⏹️  Presiona Ctrl+C para detener")
    
    app.run(debug=True, host='0.0.0.0', port=5000)