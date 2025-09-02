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
            
            # ENHANCED: Add reference_price and initial progress tracking
            reference_price = signal_data.get('reference_price', entry_price)  # Use provided or default to entry_price
            current_progress = 0.0  # Always start at 0% progress
            signal_status = 'active'  # New enhanced status system
            
            cursor.execute('''
            INSERT INTO signals (
                signal_id, symbol, signal_type, entry_price, entry_timestamp,
                predicted_change, confidence_score, expected_timeframe, status,
                reference_price, current_progress, signal_status, last_updated
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                signal_id,
                symbol,
                signal_type,
                entry_price,
                datetime.now(),
                predicted_change,
                confidence,
                expected_timeframe,
                'monitoring',  # Keep legacy status for compatibility
                reference_price,
                current_progress,
                signal_status,
                datetime.now()
            ))
            
            conn.commit()
            conn.close()
            
            print(f"SUCCESS: Señal guardada: {signal_id}")
            return signal_id
            
        except Exception as e:
            print(f"ERROR: Error guardando señal: {e}")
            return ""
    
    def update_price_tracking(self, symbol: str, current_price: float):
        """ENHANCED: Actualiza el tracking con progreso real desde reference_price"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # ENHANCED: Buscar señales activas con reference_price
            cursor.execute('''
            SELECT signal_id, entry_price, reference_price, entry_timestamp, 
                   expected_timeframe, signal_status, predicted_change
            FROM signals 
            WHERE symbol = ? AND signal_status IN ('active', 'target_reached')
            ''', (symbol,))
            
            active_signals = cursor.fetchall()
            
            for signal_id, entry_price, reference_price, entry_timestamp, expected_timeframe, signal_status, predicted_change in active_signals:
                try:
                    if isinstance(entry_timestamp, str):
                        try:
                            entry_time = datetime.strptime(entry_timestamp, '%Y-%m-%d %H:%M:%S.%f')
                        except ValueError:
                            entry_time = datetime.strptime(entry_timestamp, '%Y-%m-%d %H:%M:%S')
                    else:
                        entry_time = entry_timestamp
                except Exception:
                    entry_time = datetime.now()
                
                minutes_elapsed = int((datetime.now() - entry_time).total_seconds() / 60)
                
                # ENHANCED: Calculate real progress from reference_price (not entry_price)
                use_reference_price = reference_price if reference_price and reference_price > 0 else entry_price
                
                if use_reference_price and use_reference_price != 0 and current_price is not None:
                    # Real progress calculation from reference point
                    current_progress = ((current_price - use_reference_price) / use_reference_price) * 100
                    # Legacy change_percent for compatibility
                    change_percent = ((current_price - entry_price) / entry_price) * 100 if entry_price > 0 else 0.0
                else:
                    current_progress = 0.0
                    change_percent = 0.0
                    print(f"WARNING: Invalid prices for {symbol} - reference: {use_reference_price}, current: {current_price}")
                
                # Determine new signal status based on lifecycle rules
                new_signal_status = self._determine_signal_status(
                    current_progress, predicted_change, minutes_elapsed, expected_timeframe, signal_status
                )
                
                # Update signal with current progress and status
                cursor.execute('''
                UPDATE signals 
                SET current_progress = ?, signal_status = ?, last_updated = ?
                WHERE signal_id = ?
                ''', (round(current_progress, 3), new_signal_status, datetime.now(), signal_id))
                
                # Update legacy status for compatibility
                if new_signal_status in ['expired', 'stopped_out']:
                    cursor.execute('''
                    UPDATE signals 
                    SET status = 'completed'
                    WHERE signal_id = ?
                    ''', (signal_id,))
                elif new_signal_status == 'target_reached':
                    cursor.execute('''
                    UPDATE signals 
                    SET status = 'completed'
                    WHERE signal_id = ?
                    ''', (signal_id,))
                
                # Guardar punto de tracking (mantener compatibilidad)
                cursor.execute('''
                INSERT INTO price_tracking (
                    signal_id, timestamp, price, change_percent, minutes_elapsed
                ) VALUES (?, ?, ?, ?, ?)
                ''', (signal_id, datetime.now(), current_price, change_percent, minutes_elapsed))
                
                # Log significant status changes
                if new_signal_status != signal_status:
                    print(f"STATUS: Signal {signal_id} status changed: {signal_status} -> {new_signal_status}")
            
            conn.commit()
            conn.close()
            
            if active_signals:
                print(f"INFO: Precio actualizado para {len(active_signals)} señales de {symbol}")
                
        except Exception as e:
            print(f"ERROR: Error actualizando precio de {symbol}: {e}")

    def _determine_signal_status(self, current_progress: float, predicted_change: float, 
                               minutes_elapsed: int, expected_timeframe: int, current_status: str) -> str:
        """ENHANCED: Determine signal status based on lifecycle rules"""
        
        # If already in terminal state, don't change
        if current_status in ['expired', 'stopped_out', 'completed']:
            return current_status
        
        # Rule 1: Time-based expiration
        if minutes_elapsed >= expected_timeframe:
            if abs(current_progress) >= 2.0:  # Decent movement
                return 'target_reached'
            else:
                return 'expired'
        
        # Rule 2: Target achievement (dynamic based on prediction)
        target_threshold = max(3.0, abs(predicted_change) * 0.6)  # At least 3% or 60% of prediction
        if abs(current_progress) >= target_threshold:
            return 'target_reached'
        
        # Rule 3: Stop-out conditions
        stop_loss_threshold = max(2.5, abs(predicted_change) * 0.8)  # Adaptive stop-loss
        if predicted_change > 0 and current_progress <= -stop_loss_threshold:
            return 'stopped_out'
        elif predicted_change < 0 and current_progress >= stop_loss_threshold:
            return 'stopped_out'
        
        # Rule 4: Extended monitoring for strong signals
        if current_status == 'target_reached' and minutes_elapsed < (expected_timeframe * 1.5):
            # Continue monitoring strong signals a bit longer
            return 'target_reached'
        
        # Default: keep active
        return 'active'
    
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
        """ENHANCED: Calcula estadísticas de performance con lifecycle tracking"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Get basic stats with lifecycle information
            lifecycle_query = '''
            SELECT signal_status, COUNT(*) as count,
                   AVG(current_progress) as avg_progress,
                   AVG(confidence_score) as avg_confidence
            FROM signals 
            WHERE entry_timestamp >= datetime('now', '-{} days')
            GROUP BY signal_status
            '''.format(days)
            
            lifecycle_df = pd.read_sql(lifecycle_query, conn)
            
            # Get overall stats
            overall_query = '''
            SELECT COUNT(*) as total_signals,
                   AVG(confidence_score) as average_confidence,
                   AVG(current_progress) as average_progress,
                   COUNT(CASE WHEN signal_status = 'target_reached' THEN 1 END) as target_reached_count,
                   COUNT(CASE WHEN signal_status = 'stopped_out' THEN 1 END) as stopped_out_count,
                   COUNT(CASE WHEN signal_status = 'expired' THEN 1 END) as expired_count,
                   COUNT(CASE WHEN signal_status = 'active' THEN 1 END) as active_count
            FROM signals 
            WHERE entry_timestamp >= datetime('now', '-{} days')
            '''.format(days)
            
            overall_df = pd.read_sql(overall_query, conn)
            
            # Get user feedback stats (legacy compatibility)
            feedback_query = '''
            SELECT s.signal_type, s.confidence_score, s.predicted_change,
                   uf.actual_change, uf.feedback_type
            FROM signals s
            JOIN user_feedback uf ON s.signal_id = uf.signal_id
            WHERE s.entry_timestamp >= datetime('now', '-{} days')
            '''
            
            feedback_df = pd.read_sql(feedback_query, conn)
            conn.close()
            
            # Build enhanced statistics
            stats = {}
            
            if not overall_df.empty:
                row = overall_df.iloc[0]
                total_signals = int(row['total_signals']) if not pd.isna(row['total_signals']) else 0
                
                # Calculate success rate based on lifecycle
                target_reached = int(row['target_reached_count']) if not pd.isna(row['target_reached_count']) else 0
                success_rate = (target_reached / total_signals * 100) if total_signals > 0 else 0.0
                
                stats.update({
                    "total_signals": total_signals,
                    "success_rate": round(success_rate, 2),
                    "average_confidence": round(float(row['average_confidence']) if not pd.isna(row['average_confidence']) else 0.0, 3),
                    "average_progress": round(float(row['average_progress']) if not pd.isna(row['average_progress']) else 0.0, 2),
                    
                    # Lifecycle breakdown
                    "lifecycle_stats": {
                        "active": int(row['active_count']) if not pd.isna(row['active_count']) else 0,
                        "target_reached": target_reached,
                        "stopped_out": int(row['stopped_out_count']) if not pd.isna(row['stopped_out_count']) else 0,
                        "expired": int(row['expired_count']) if not pd.isna(row['expired_count']) else 0,
                    }
                })
                
                # Stop-out rate (risk metric)
                stopped_out = int(row['stopped_out_count']) if not pd.isna(row['stopped_out_count']) else 0
                stop_out_rate = (stopped_out / total_signals * 100) if total_signals > 0 else 0.0
                stats["stop_out_rate"] = round(stop_out_rate, 2)
                
            else:
                stats = {
                    "total_signals": 0, 
                    "success_rate": 0.0, 
                    "average_confidence": 0.0,
                    "average_progress": 0.0,
                    "stop_out_rate": 0.0,
                    "lifecycle_stats": {
                        "active": 0, "target_reached": 0, "stopped_out": 0, "expired": 0
                    }
                }
            
            # Add legacy feedback stats if available
            if not feedback_df.empty:
                total_feedback = len(feedback_df)
                success_feedback = len(feedback_df[feedback_df['feedback_type'] == 'success'])
                stats["feedback_success_rate"] = round((success_feedback / total_feedback * 100), 2) if total_feedback > 0 else 0.0
                stats["total_with_feedback"] = total_feedback
            else:
                stats["feedback_success_rate"] = 0.0
                stats["total_with_feedback"] = 0
            
            return stats
            
        except Exception as e:
            print(f"ERROR: Error calculando estadísticas: {e}")
            return {
                "total_signals": 0, "success_rate": 0.0, "average_confidence": 0.0,
                "average_progress": 0.0, "stop_out_rate": 0.0,
                "lifecycle_stats": {"active": 0, "target_reached": 0, "stopped_out": 0, "expired": 0}
            }
    
    def cleanup_old_signals(self, max_age_days: int = 30) -> int:
        """ENHANCED: Clean up old/expired signals and optimize database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cutoff_date = datetime.now() - timedelta(days=max_age_days)
            
            # Get count of signals to be cleaned up
            cursor.execute('''
            SELECT COUNT(*) FROM signals 
            WHERE entry_timestamp < ? AND signal_status IN ('expired', 'stopped_out', 'completed')
            ''', (cutoff_date,))
            
            cleanup_count = cursor.fetchone()[0]
            
            if cleanup_count > 0:
                # Archive old signals to a backup before deletion (optional)
                cursor.execute('''
                CREATE TABLE IF NOT EXISTS signals_archive AS 
                SELECT * FROM signals WHERE 1=0
                ''')
                
                # Move old signals to archive
                cursor.execute('''
                INSERT INTO signals_archive 
                SELECT * FROM signals 
                WHERE entry_timestamp < ? AND signal_status IN ('expired', 'stopped_out', 'completed')
                ''', (cutoff_date,))
                
                # Delete old price tracking data
                cursor.execute('''
                DELETE FROM price_tracking 
                WHERE signal_id IN (
                    SELECT signal_id FROM signals 
                    WHERE entry_timestamp < ? AND signal_status IN ('expired', 'stopped_out', 'completed')
                )
                ''', (cutoff_date,))
                
                # Delete old feedback data
                cursor.execute('''
                DELETE FROM user_feedback 
                WHERE signal_id IN (
                    SELECT signal_id FROM signals 
                    WHERE entry_timestamp < ? AND signal_status IN ('expired', 'stopped_out', 'completed')
                )
                ''', (cutoff_date,))
                
                # Delete old signals
                cursor.execute('''
                DELETE FROM signals 
                WHERE entry_timestamp < ? AND signal_status IN ('expired', 'stopped_out', 'completed')
                ''', (cutoff_date,))
                
                # Optimize database
                cursor.execute('VACUUM')
                
                conn.commit()
                print(f"SUCCESS: Cleaned up {cleanup_count} old signals (older than {max_age_days} days)")
            else:
                print(f"INFO: No old signals to clean up (older than {max_age_days} days)")
            
            conn.close()
            return cleanup_count
            
        except Exception as e:
            print(f"ERROR: Error durante cleanup: {e}")
            return 0
    
    def get_signal_lifecycle_summary(self) -> Dict:
        """ENHANCED: Get comprehensive signal lifecycle summary"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Get lifecycle summary with timing
            query = '''
            SELECT 
                signal_status,
                COUNT(*) as count,
                AVG(current_progress) as avg_progress,
                AVG(confidence_score) as avg_confidence,
                AVG(CAST((julianday('now') - julianday(entry_timestamp)) * 24 * 60 AS INTEGER)) as avg_age_minutes,
                MIN(current_progress) as min_progress,
                MAX(current_progress) as max_progress
            FROM signals 
            WHERE entry_timestamp >= datetime('now', '-7 days')
            GROUP BY signal_status
            ORDER BY count DESC
            '''
            
            df = pd.read_sql(query, conn)
            conn.close()
            
            summary = {
                "status_breakdown": [],
                "total_active_signals": 0,
                "avg_signal_age_hours": 0.0,
                "performance_summary": {}
            }
            
            total_signals = 0
            total_age_minutes = 0
            
            for _, row in df.iterrows():
                status_info = {
                    "status": row['signal_status'],
                    "count": int(row['count']),
                    "avg_progress": round(float(row['avg_progress']) if not pd.isna(row['avg_progress']) else 0.0, 2),
                    "avg_confidence": round(float(row['avg_confidence']) if not pd.isna(row['avg_confidence']) else 0.0, 3),
                    "avg_age_minutes": int(row['avg_age_minutes']) if not pd.isna(row['avg_age_minutes']) else 0,
                    "progress_range": {
                        "min": round(float(row['min_progress']) if not pd.isna(row['min_progress']) else 0.0, 2),
                        "max": round(float(row['max_progress']) if not pd.isna(row['max_progress']) else 0.0, 2)
                    }
                }
                
                summary["status_breakdown"].append(status_info)
                
                if row['signal_status'] == 'active':
                    summary["total_active_signals"] = int(row['count'])
                
                total_signals += int(row['count'])
                total_age_minutes += int(row['avg_age_minutes']) * int(row['count'])
            
            # Calculate overall averages
            if total_signals > 0:
                summary["avg_signal_age_hours"] = round((total_age_minutes / total_signals) / 60.0, 1)
            
            # Performance summary
            target_reached = sum(s["count"] for s in summary["status_breakdown"] if s["status"] == "target_reached")
            stopped_out = sum(s["count"] for s in summary["status_breakdown"] if s["status"] == "stopped_out")
            
            summary["performance_summary"] = {
                "total_signals": total_signals,
                "success_signals": target_reached,
                "failed_signals": stopped_out,
                "success_rate": round((target_reached / total_signals * 100), 2) if total_signals > 0 else 0.0
            }
            
            return summary
            
        except Exception as e:
            print(f"ERROR: Error obteniendo resumen de lifecycle: {e}")
            return {
                "status_breakdown": [], 
                "total_active_signals": 0, 
                "avg_signal_age_hours": 0.0,
                "performance_summary": {"total_signals": 0, "success_signals": 0, "failed_signals": 0, "success_rate": 0.0}
            }

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
            print(f"SUCCESS: Test exitoso: {signal_id}")
        
    except Exception as e:
        print(f"ERROR: Error en test: {e}")