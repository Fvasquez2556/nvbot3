# web_dashboard/app.py
from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit
import sqlite3
import json
from datetime import datetime, timedelta
import sys
import os
import time
import threading

# CORRECCIÓN: Ruta de importación simplificada
sys.path.append(os.path.dirname(__file__))
from database.signal_tracker import SignalTracker

# Configuración de Flask con SocketIO para tiempo real
app = Flask(__name__)
app.config['SECRET_KEY'] = 'nvbot3-secret-key-2025'
socketio = SocketIO(app, cors_allowed_origins="*")

# Inicializar componentes
try:
    signal_tracker = SignalTracker()
    print("SUCCESS: SignalTracker inicializado correctamente")
except Exception as e:
    print(f"ERROR: Error inicializando SignalTracker: {e}")
    signal_tracker = None

# Cache global para datos del dashboard
dashboard_cache = {
    'top_momentum': [],
    'rebounds': [],
    'trending': [],
    'scanned_symbols': [],
    'last_update': None,
    'statistics': {}
}

@app.route('/')
def dashboard():
    """Dashboard unificado NvBot3 con tiempo real y análisis histórico"""
    try:
        if signal_tracker:
            # Obtener señales categorizadas para el panel de tiempo real
            categorized_data = get_categorized_signals()
            stats = signal_tracker.get_performance_stats(days=30)
            
            # Obtener top 5 monedas con mayor confianza
            top_confidence_coins = get_top_confidence_coins()
            
            # Obtener señales activas para el panel de análisis histórico
            active_signals = signal_tracker.get_active_signals()
            
        else:
            categorized_data = {
                'momentum': [],
                'rebounds': [],
                'trending': [],
                'scanned_symbols': []
            }
            stats = {"total_signals": 0, "success_rate": 0, "average_confidence": 0}
            top_confidence_coins = []
            active_signals = []
        
        return render_template('nvbot3_dashboard.html', 
                             categorized_signals=categorized_data,
                             top_confidence_coins=top_confidence_coins,
                             stats=stats,
                             active_signals=active_signals,
                             current_time=datetime.now())
    except Exception as e:
        print(f"ERROR: Error en dashboard: {e}")
        return f"Error cargando dashboard: {e}", 500

def get_categorized_signals():
    """Obtener señales categorizadas por tipo"""
    try:
        if not signal_tracker:
            return {'momentum': [], 'rebounds': [], 'trending': [], 'scanned_symbols': []}
        
        # Obtener señales activas
        active_signals = signal_tracker.get_active_signals()
        
        # Categorizar señales
        categorized = {
            'momentum': [],
            'rebounds': [], 
            'trending': [],
            'scanned_symbols': set()
        }
        
        # ENHANCED: Group signals by symbol and category to avoid duplicates
        symbol_categories = {
            'momentum': {},
            'rebounds': {},
            'trending': {}
        }
        
        for signal in active_signals:
            symbol = signal.get('symbol', '')
            # FIXED: Use signal_type directly from database instead of nested prediction
            signal_type = signal.get('signal_type', '')
            # FIXED: Use confidence_score from database
            confidence = signal.get('confidence_score', 0)
            # ENHANCED: Check signal status for lifecycle filtering
            signal_status = signal.get('signal_status', 'active')
            
            # Only include active and target_reached signals in main display
            if signal_status not in ['active', 'target_reached']:
                continue
            
            # Agregar a símbolos escaneados
            categorized['scanned_symbols'].add(symbol)
            
            # ENHANCED: Prepare signal data with progress information
            reference_price = signal.get('reference_price', signal.get('entry_price', 0))
            current_price = signal.get('current_price', signal.get('entry_price', 0))
            
            # Calculate real-time progress from reference price
            if reference_price and reference_price > 0 and current_price:
                current_progress = round(((current_price - reference_price) / reference_price) * 100, 2)
            else:
                current_progress = signal.get('current_progress', 0.0)
            
            signal_data = {
                'symbol': symbol,
                'confidence_score': confidence,
                'predicted_change': signal.get('predicted_change', 0),
                'timestamp': signal.get('entry_timestamp', ''),
                'current_price': current_price,
                'entry_price': signal.get('entry_price', 0),
                'reference_price': reference_price,
                'current_progress': current_progress,
                'signal_type': signal_type,
                'signal_status': signal.get('signal_status', 'active'),
                'created_at': signal.get('created_at', '')
            }
            
            # Categorizar por tipo de señal (keep highest confidence per symbol)
            if signal_type in ['momentum', 'momentum_advanced', 'momentum_alto']:
                if symbol not in symbol_categories['momentum'] or confidence > symbol_categories['momentum'][symbol]['confidence_score']:
                    symbol_categories['momentum'][symbol] = signal_data
                    
            elif signal_type == 'rebound':
                if symbol not in symbol_categories['rebounds'] or confidence > symbol_categories['rebounds'][symbol]['confidence_score']:
                    symbol_categories['rebounds'][symbol] = signal_data
                    
            elif signal_type in ['regime', 'consolidacion']:
                signal_data['trend_direction'] = 'bullish'  # FIXED: Default to bullish for regime signals
                if symbol not in symbol_categories['trending'] or confidence > symbol_categories['trending'][symbol]['confidence_score']:
                    symbol_categories['trending'][symbol] = signal_data
        
        # Convert dictionaries to lists
        categorized['momentum'] = list(symbol_categories['momentum'].values())
        categorized['rebounds'] = list(symbol_categories['rebounds'].values())
        categorized['trending'] = list(symbol_categories['trending'].values())
        
        # Convertir set a lista para scanned_symbols
        categorized['scanned_symbols'] = sorted(list(categorized['scanned_symbols']))
        
        # Ordenar por confianza (mayor a menor)
        for category in ['momentum', 'rebounds', 'trending']:
            categorized[category].sort(key=lambda x: x.get('confidence_score', 0), reverse=True)
        
        return categorized
        
    except Exception as e:
        print(f"ERROR: Error obteniendo señales categorizadas: {e}")
        return {'momentum': [], 'rebounds': [], 'trending': [], 'scanned_symbols': []}

def get_top_confidence_coins():
    """Obtener top 5 monedas con mayor confianza de subida (sin duplicados por símbolo)"""
    try:
        if not signal_tracker:
            return []
        
        active_signals = signal_tracker.get_active_signals()
        
        # FIXED: Group by symbol and get highest confidence per symbol
        symbol_best_signals = {}
        
        for signal in active_signals:
            symbol = signal.get('symbol', '')
            signal_type = signal.get('signal_type', '')
            confidence = signal.get('confidence_score', 0)
            
            # Consider momentum, rebound as positive signals
            if signal_type in ['momentum', 'momentum_advanced', 'momentum_alto', 'rebound']:
                # Check if this symbol already exists with lower confidence
                if symbol not in symbol_best_signals or confidence > symbol_best_signals[symbol]['confidence']:
                    # ENHANCED: Calculate real progress for top coins
                    reference_price = signal.get('reference_price', signal.get('entry_price', 0))
                    current_price = signal.get('current_price', signal.get('entry_price', 0))
                    
                    if reference_price and reference_price > 0 and current_price:
                        current_progress = round(((current_price - reference_price) / reference_price) * 100, 2)
                    else:
                        current_progress = signal.get('current_progress', 0.0)
                    
                    symbol_best_signals[symbol] = {
                        'symbol': symbol,
                        'confidence_score': confidence,
                        'predicted_change': signal.get('predicted_change', 0),
                        'signal_type': signal_type,
                        'current_price': current_price,
                        'timestamp': signal.get('entry_timestamp', ''),
                        'signal_freshness': signal.get('created_at', ''),  # For tie-breaking
                        'entry_price': signal.get('entry_price', 0),
                        'reference_price': reference_price,
                        'current_progress': current_progress,
                        'signal_status': signal.get('signal_status', 'active')
                    }
        
        # Convert to list and apply multi-criteria ranking
        unique_signals = list(symbol_best_signals.values())
        
        # Enhanced ranking: confidence (70%), signal freshness (20%), predicted change (10%)
        def ranking_score(signal):
            confidence_weight = 0.7
            freshness_weight = 0.2
            change_weight = 0.1
            
            confidence_score = signal['confidence_score']
            
            # Calculate freshness score (more recent = higher score)
            try:
                from datetime import datetime
                signal_time = datetime.fromisoformat(signal['signal_freshness'].replace('Z', '+00:00')) if signal.get('signal_freshness') else datetime.min
                now = datetime.now()
                hours_old = (now - signal_time).total_seconds() / 3600
                freshness_score = max(0, 1 - (hours_old / 24))  # Decay over 24 hours
            except:
                freshness_score = 0.5  # Default for parsing errors
            
            change_score = min(1.0, signal.get('predicted_change', 0) / 10.0)  # Normalize to 0-1
            
            total_score = (confidence_weight * confidence_score + 
                          freshness_weight * freshness_score + 
                          change_weight * change_score)
            
            return total_score
        
        # Sort by ranking score and return top 5
        unique_signals.sort(key=ranking_score, reverse=True)
        return unique_signals[:5]
        
    except Exception as e:
        print(f"ERROR: Error obteniendo top coins: {e}")
        return []

@app.route('/api/signals/categorized')
def get_categorized_signals_api():
    """API endpoint para obtener señales categorizadas"""
    try:
        categorized_data = get_categorized_signals()
        return jsonify({"status": "success", "data": categorized_data})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

@app.route('/api/signals/top-confidence')
def get_top_confidence_api():
    """API endpoint para obtener top monedas con mayor confianza"""
    try:
        top_coins = get_top_confidence_coins()
        return jsonify({"status": "success", "data": top_coins})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

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

@app.route('/api/realtime/stats')
def get_realtime_stats():
    """Obtener estadísticas en tiempo real"""
    try:
        if not signal_tracker:
            return jsonify({"status": "error", "message": "SignalTracker no disponible"})
        
        # Estadísticas básicas
        stats = signal_tracker.get_performance_stats(days=1)  # Últimas 24 horas
        
        # Agregar información adicional
        categorized = get_categorized_signals()
        
        enhanced_stats = {
            **stats,
            'momentum_signals': len(categorized['momentum']),
            'rebound_signals': len(categorized['rebounds']),
            'trend_signals': len(categorized['trending']),
            'total_scanned_symbols': len(categorized['scanned_symbols']),
            'last_update': datetime.now().isoformat(),
            'active_categories': {
                'momentum': len(categorized['momentum']) > 0,
                'rebounds': len(categorized['rebounds']) > 0,
                'trending': len(categorized['trending']) > 0
            }
        }
        
        return jsonify({"status": "success", "data": enhanced_stats})
        
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

# WebSocket events para actualizaciones en tiempo real
@socketio.on('connect')
def handle_connect():
    """Manejar nueva conexión WebSocket"""
    print('Cliente conectado al dashboard en tiempo real')
    emit('status', {'message': 'Conectado al dashboard NvBot3'})
    
    # Enviar datos iniciales
    try:
        categorized_data = get_categorized_signals()
        top_coins = get_top_confidence_coins()
        emit('initial_data', {
            'categorized_signals': categorized_data,
            'top_confidence_coins': top_coins,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        emit('error', {'message': f'Error cargando datos iniciales: {e}'})

@socketio.on('request_update')
def handle_update_request():
    """Manejar solicitud de actualización manual"""
    try:
        categorized_data = get_categorized_signals()
        top_coins = get_top_confidence_coins()
        stats = signal_tracker.get_performance_stats(days=1) if signal_tracker else {}
        
        emit('data_update', {
            'categorized_signals': categorized_data,
            'top_confidence_coins': top_coins,
            'stats': stats,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        emit('error', {'message': f'Error actualizando datos: {e}'})

def broadcast_new_signal(signal_data):
    """Transmitir nueva señal a todos los clientes conectados"""
    try:
        socketio.emit('new_signal', {
            'signal': signal_data,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        print(f"ERROR: Error transmitiendo señal: {e}")

# Función para actualizar dashboard periódicamente
def periodic_dashboard_update():
    """Actualizar dashboard cada 30 segundos"""
    cleanup_counter = 0
    
    while True:
        try:
            time.sleep(30)  # Actualizar cada 30 segundos
            
            # Obtener datos actualizados
            categorized_data = get_categorized_signals()
            top_coins = get_top_confidence_coins()
            
            # Transmitir a todos los clientes conectados
            socketio.emit('periodic_update', {
                'categorized_signals': categorized_data,
                'top_confidence_coins': top_coins,
                'timestamp': datetime.now().isoformat()
            })
            
            # ENHANCED: Periodic cleanup every hour (120 cycles * 30 seconds = 1 hour)
            cleanup_counter += 1
            if cleanup_counter >= 120:  # 1 hour
                try:
                    if signal_tracker:
                        # Clean up signals older than 7 days
                        cleaned_count = signal_tracker.cleanup_old_signals(max_age_days=7)
                        if cleaned_count > 0:
                            print(f"AUTO-CLEANUP: Removed {cleaned_count} old signals")
                        
                        # Get lifecycle summary for logging
                        summary = signal_tracker.get_signal_lifecycle_summary()
                        active_count = summary.get('total_active_signals', 0)
                        print(f"LIFECYCLE: {active_count} active signals currently tracked")
                        
                except Exception as cleanup_error:
                    print(f"ERROR: Auto-cleanup failed: {cleanup_error}")
                
                cleanup_counter = 0  # Reset counter
            
        except Exception as e:
            print(f"ERROR: Error en actualización periódica: {e}")
            time.sleep(60)  # Esperar más en caso de error

# Iniciar thread de actualización periódica
update_thread = threading.Thread(target=periodic_dashboard_update, daemon=True)
update_thread.start()

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
        print(f"ERROR: Error guardando feedback: {e}")
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/api/lifecycle/summary')
def get_lifecycle_summary():
    """ENHANCED: Get signal lifecycle summary"""
    try:
        if not signal_tracker:
            return jsonify({"status": "error", "message": "SignalTracker no disponible"})
        
        summary = signal_tracker.get_signal_lifecycle_summary()
        return jsonify({"status": "success", "data": summary})
        
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

@app.route('/api/lifecycle/cleanup', methods=['POST'])
def cleanup_old_signals():
    """ENHANCED: Clean up old/expired signals"""
    try:
        if not signal_tracker:
            return jsonify({"status": "error", "message": "SignalTracker no disponible"})
        
        data = request.get_json() if request.is_json else {}
        max_age_days = data.get('max_age_days', 30)
        
        # Validate max_age_days
        if not isinstance(max_age_days, int) or max_age_days < 1:
            return jsonify({"status": "error", "message": "max_age_days must be a positive integer"})
        
        cleaned_count = signal_tracker.cleanup_old_signals(max_age_days)
        
        return jsonify({
            "status": "success", 
            "message": f"Cleaned up {cleaned_count} old signals",
            "cleaned_count": cleaned_count
        })
        
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

@app.route('/api/performance/enhanced')
def get_enhanced_performance():
    """ENHANCED: Get enhanced performance statistics with lifecycle data"""
    try:
        if not signal_tracker:
            return jsonify({"status": "error", "message": "SignalTracker no disponible"})
        
        days = request.args.get('days', 7, type=int)
        days = max(1, min(365, days))  # Limit between 1-365 days
        
        stats = signal_tracker.get_performance_stats(days=days)
        lifecycle_summary = signal_tracker.get_signal_lifecycle_summary()
        
        enhanced_stats = {
            **stats,
            "lifecycle_summary": lifecycle_summary,
            "query_period_days": days,
            "timestamp": datetime.now().isoformat()
        }
        
        return jsonify({"status": "success", "data": enhanced_stats})
        
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

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
    print("INICIANDO NvBot3 Dashboard Mejorado...")
    print("🌐 Dashboard disponible en: http://localhost:5000")
    print("API categorizada en: http://localhost:5000/api/signals/categorized")
    print("Top confianza en: http://localhost:5000/api/signals/top-confidence")
    print("Stats tiempo real en: http://localhost:5000/api/realtime/stats")
    print("WebSocket habilitado para actualizaciones en tiempo real")
    print("Presiona Ctrl+C para detener")
    
    # Usar SocketIO en lugar de app.run para soporte WebSocket
    socketio.run(app, debug=True, host='0.0.0.0', port=5000)