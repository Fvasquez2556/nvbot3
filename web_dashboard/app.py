from flask import Flask, render_template, request, jsonify
import sqlite3
import json
from datetime import datetime
import sys
import os

# Agregar path del database
sys.path.append(os.path.join(os.path.dirname(__file__), 'database'))
from database.signal_tracker import SignalTracker

# Configuración de Flask
app = Flask(__name__)
app.config['SECRET_KEY'] = 'nvbot3-secret-key-2025'

# Inicializar componentes
signal_tracker = SignalTracker()

@app.route('/')
def dashboard():
    """Página principal del dashboard"""
    active_signals = signal_tracker.get_active_signals()
    stats = signal_tracker.get_performance_stats(days=30)
    return render_template('dashboard.html', active_signals=active_signals, stats=stats)

@app.route('/api/signals/active')
def get_active_signals():
    """API endpoint para obtener señales activas"""
    signals = signal_tracker.get_active_signals()
    return jsonify(signals)

@app.route('/api/feedback', methods=['POST'])
def submit_feedback():
    """Recibir retroalimentación del usuario"""
    data = request.get_json()
    signal_id = data.get('signal_id')
    feedback_type = data.get('feedback_type')
    feedback_data = {
        'feedback_type': feedback_type,
        'comments': data.get('comments', '')
    }
    try:
        signal_tracker.save_user_feedback(signal_id, feedback_data)
        return jsonify({'status': 'success', 'message': 'Feedback guardado correctamente'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
