# web_dashboard/database/signal_tracker.py
import sqlite3
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import pandas as pd
import os

class SignalTracker:
    """
    Sistema de tracking que se integra con nvbot3 existente
    Guarda cada predicción y monitorea automáticamente los resultados
    """
    
    def __init__(self, db_path: Optional[str] = None):
        if db_path is None:
            # Crear base de datos en el directorio actual del archivo
            current_dir = os.path.dirname(os.path.abspath(__file__))
            db_path = os.path.join(current_dir, "signals.db")
        
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Crea las tablas necesarias si no existen"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Tabla principal de señales
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS signals (
                signal_id TEXT PRIMARY KEY,
                symbol TEXT NOT NULL,
                signal_type TEXT NOT NULL,
                entry_price REAL NOT NULL,
                entry_timestamp DATETIME NOT NULL,
                predicted_change REAL NOT NULL,
                confidence_score REAL NOT NULL,
                expected_timeframe INTEGER NOT NULL,
                status TEXT DEFAULT 'monitoring',
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
            ''')
            
            # Tabla de tracking de precios
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS price_tracking (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                signal_id TEXT NOT NULL,
                timestamp DATETIME NOT NULL,
                price REAL NOT NULL,
                change_percent REAL NOT NULL,
                minutes_elapsed INTEGER NOT NULL,
                FOREIGN KEY (signal_id) REFERENCES signals (signal_id)
            )
            ''')
            
            # Tabla de retroalimentación del usuario
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                signal_id TEXT NOT NULL,
                feedback_type TEXT NOT NULL,
                actual_result TEXT,
                actual_change REAL,
                time_to_target INTEGER,
                user_notes TEXT,
                feedback_timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (signal_id) REFERENCES signals (signal_id)
            )
            ''')
            
            conn.commit()
            conn.close()
            print("SUCCESS: Base de datos SignalTracker inicializada correctamente")
            
        except Exception as e:
            print(f"ERROR: Error inicializando base de datos: {e}")
    
    def save_new_signal(self, symbol: str, signal_data: Dict) -> str:
        """
        Guarda una nueva señal del nvbot3
        Acepta tanto el formato nuevo como el formato legacy
        """
        try:
            signal_id = f"{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Manejar diferentes formatos de entrada
            if 'prediction_data' in signal_data:
                # Formato legacy del bridge
                pred_data = signal_data['prediction_data']
                signal_type = pred_data.get('type', 'unknown')
                predicted_change = pred_data.get('predicted_change', 0) or pred_data.get('change', 0)
                confidence = pred_data.get('confidence', 0)
                entry_price = signal_data.get('current_price', 0)
            else:
                # Formato directo
                signal_type = signal_data.get('type', 'unknown')
                predicted_change = signal_data.get('predicted_change', 0)
                confidence = signal_data.get('confidence', 0)
                entry_price = signal_data.get('entry_price', 0)
            
            # Determinar timeframe basado en tipo de señal
            timeframe_mapping = {
                'momentum': 240,
                'momentum_alto': 240,
                'rebound': 120,
                'rebote_pequeño': 120,
                'regime': 480,
                'consolidacion': 360,
            }
            expected_timeframe = timeframe_mapping.get(signal_type, 240)
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
            INSERT INTO signals (
                signal_id, symbol, signal_type, entry_price, entry_timestamp,
                predicted_change, confidence_score, expected_timeframe, status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                signal_id,
                symbol,
                signal_type,
                entry_price,
                datetime.now(),
                predicted_change,
                confidence,
                expected_timeframe,
                'monitoring'
            ))
            
            conn.commit()
            conn.close()
            
            print(f"SUCCESS: Señal guardada: {signal_id}")
            return signal_id
            
        except Exception as e:
            print(f"ERROR: Error guardando señal: {e}")
            return ""
    
    def update_price_tracking(self, symbol: str, current_price: float):
        """Actualiza el tracking de precio para todas las señales activas de un símbolo"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Buscar señales activas de este símbolo
            cursor.execute('''
            SELECT signal_id, entry_price, entry_timestamp, expected_timeframe
            FROM signals 
            WHERE symbol = ? AND status = 'monitoring'
            ''', (symbol,))
            
            active_signals = cursor.fetchall()
            
            for signal_id, entry_price, entry_timestamp, expected_timeframe in active_signals:
                try:
                    entry_time = datetime.strptime(entry_timestamp, '%Y-%m-%d %H:%M:%S.%f')
                except ValueError:
                    # Intentar otro formato si el primero falla
                    entry_time = datetime.strptime(entry_timestamp, '%Y-%m-%d %H:%M:%S')
                
                minutes_elapsed = int((datetime.now() - entry_time).total_seconds() / 60)
                
                # NaN protection for percentage calculation
                if entry_price and entry_price != 0 and current_price is not None:
                    change_percent = ((current_price - entry_price) / entry_price) * 100
                else:
                    change_percent = 0.0
                    print(f"⚠️ Warning: Invalid prices for {symbol} - entry: {entry_price}, current: {current_price}")
                
                # Guardar punto de tracking
                cursor.execute('''
                INSERT INTO price_tracking (
                    signal_id, timestamp, price, change_percent, minutes_elapsed
                ) VALUES (?, ?, ?, ?, ?)
                ''', (signal_id, datetime.now(), current_price, change_percent, minutes_elapsed))
                
                # Si ya pasó el tiempo esperado, marcar como completado
                if minutes_elapsed >= expected_timeframe:
                    cursor.execute('''
                    UPDATE signals SET status = 'completed' WHERE signal_id = ?
                    ''', (signal_id,))
            
            conn.commit()
            conn.close()
            
            if active_signals:
                print(f"💹 Precio actualizado para {len(active_signals)} señales de {symbol}")
                
        except Exception as e:
            print(f"ERROR: Error actualizando precio de {symbol}: {e}")
    
    def get_active_signals(self) -> List[Dict]:
        """Obtiene todas las señales que están siendo monitoreadas"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            query = '''
            SELECT s.*, pt.price as current_price, pt.change_percent as current_change,
                   pt.minutes_elapsed
            FROM signals s
            LEFT JOIN price_tracking pt ON s.signal_id = pt.signal_id
            WHERE s.status IN ('monitoring', 'completed')
            AND (pt.id IS NULL OR pt.id = (
                SELECT MAX(id) FROM price_tracking WHERE signal_id = s.signal_id
            ))
            ORDER BY s.entry_timestamp DESC
            '''
            
            df = pd.read_sql(query, conn)
            conn.close()
            
            return df.to_dict('records')
            
        except Exception as e:
            print(f"ERROR: Error obteniendo señales activas: {e}")
            return []
    
    def save_user_feedback(self, signal_id: str, feedback_data: Dict):
        """Guarda la retroalimentación del usuario sobre una señal"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
            INSERT INTO user_feedback (
                signal_id, feedback_type, actual_result, actual_change,
                time_to_target, user_notes
            ) VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                signal_id,
                feedback_data['type'],
                feedback_data.get('result', ''),
                feedback_data.get('actual_change'),
                feedback_data.get('time_to_target'),
                feedback_data.get('notes', '')
            ))
            
            # Marcar señal como con feedback
            cursor.execute('''
            UPDATE signals SET status = 'feedback_received' WHERE signal_id = ?
            ''', (signal_id,))
            
            conn.commit()
            conn.close()
            
            print(f"💬 Feedback guardado para {signal_id}")
            
        except Exception as e:
            print(f"ERROR: Error guardando feedback: {e}")
    
    def generate_smart_comment(self, signal_id: str) -> str:
        """Genera comentarios inteligentes basados en el comportamiento del precio"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            query = '''
            SELECT pt.*, s.entry_price, s.predicted_change, s.expected_timeframe
            FROM price_tracking pt
            JOIN signals s ON pt.signal_id = s.signal_id
            WHERE pt.signal_id = ?
            ORDER BY pt.minutes_elapsed
            '''
            
            df = pd.read_sql(query, conn, params=(signal_id,))
            conn.close()
            
            if len(df) == 0:
                return "Monitoreando señal..."
            
            # Analizar patrones
            max_change = df['change_percent'].max()
            min_change = df['change_percent'].min()
            final_change = df['change_percent'].iloc[-1]
            time_elapsed = df['minutes_elapsed'].iloc[-1]
            
            # NaN protection for comment generation
            if pd.isna(final_change) or pd.isna(max_change) or pd.isna(min_change):
                return "⏳ Monitoreando evolución de la señal..."
            
            # Generar comentario inteligente
            if abs(final_change) < 1.0:
                return f"Precio estable ({final_change:+.2f}%) después de {time_elapsed} min"
            elif final_change > 2.0 and min_change < -1.0:
                return f"Bajó hasta {min_change:.1f}%, luego subió {final_change:+.1f}%. Patrón dip-and-rally."
            else:
                return f"Cambio actual {final_change:+.1f}% en {time_elapsed}min. Máximo: {max_change:+.1f}%"
                
        except Exception as e:
            print(f"ERROR: Error generando comentario: {e}")
            return "Error generando comentario"
    
    def get_performance_stats(self, days: int = 30) -> Dict:
        """Calcula estadísticas de performance de las señales"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            query = '''
            SELECT s.signal_type, s.confidence_score, s.predicted_change,
                   uf.actual_change, uf.feedback_type
            FROM signals s
            JOIN user_feedback uf ON s.signal_id = uf.signal_id
            WHERE s.entry_timestamp >= datetime('now', '-{} days')
            '''.format(days)
            
            df = pd.read_sql(query, conn)
            conn.close()
            
            if len(df) == 0:
                return {"total_signals": 0, "success_rate": 0, "average_confidence": 0}
            
            # NaN protection for statistics
            total_signals = len(df)
            success_count = len(df[df['feedback_type'] == 'success']) if total_signals > 0 else 0
            success_rate = (success_count / total_signals * 100) if total_signals > 0 else 0.0
            average_confidence = df['confidence_score'].mean() if total_signals > 0 else 0.0
            
            stats = {
                "total_signals": total_signals,
                "success_rate": success_rate if not pd.isna(success_rate) else 0.0,
                "average_confidence": average_confidence if not pd.isna(average_confidence) else 0.0,
            }
            
            return stats
            
        except Exception as e:
            print(f"ERROR: Error calculando estadísticas: {e}")
            return {"total_signals": 0, "success_rate": 0, "average_confidence": 0}

# Test de inicialización
if __name__ == "__main__":
    try:
        tracker = SignalTracker()
        print("SUCCESS: SignalTracker inicializado correctamente")
        
        # Test básico
        test_data = {
            'type': 'momentum_alto',
            'predicted_change': 5.2,
            'confidence': 0.85,
            'entry_price': 67250.0
        }
        
        signal_id = tracker.save_new_signal('BTCUSDT', test_data)
        if signal_id:
            print(f"✅ Test exitoso: {signal_id}")
        
    except Exception as e:
        print(f"ERROR: Error en test: {e}")