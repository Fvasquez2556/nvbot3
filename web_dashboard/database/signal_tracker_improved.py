#!/usr/bin/env python3
"""
Improved Thread-Safe SignalTracker for NvBot3
=============================================

Enhanced version with connection pooling, proper transaction management,
and thread-safe operations for high-concurrency signal tracking.
"""

import sqlite3
import json
import threading
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import pandas as pd
import os

from .db_connection_pool import get_db_pool, db_operation, close_db_pool

logger = logging.getLogger(__name__)

class ImprovedSignalTracker:
    """
    Thread-safe signal tracker with connection pooling and proper concurrency control
    """
    
    def __init__(self, db_path: Optional[str] = None):
        if db_path is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            db_path = os.path.join(current_dir, "signals.db")
        
        self.db_path = db_path
        self._db_pool = get_db_pool(db_path)
        self._lock = threading.RLock()  # Reentrant lock for thread safety
        
        # Initialize database schema
        self.init_database()
        
        logger.info("ImprovedSignalTracker initialized with connection pooling")
    
    def init_database(self):
        """Initialize database schema with proper constraints and indexes"""
        try:
            operations = [
                '''CREATE TABLE IF NOT EXISTS signals (
                    signal_id TEXT PRIMARY KEY,
                    symbol TEXT NOT NULL,
                    signal_type TEXT NOT NULL,
                    entry_price REAL NOT NULL,
                    entry_timestamp DATETIME NOT NULL,
                    predicted_change REAL NOT NULL,
                    confidence_score REAL NOT NULL,
                    expected_timeframe INTEGER NOT NULL DEFAULT 60,
                    status TEXT DEFAULT 'monitoring',
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )''',
                
                '''CREATE TABLE IF NOT EXISTS price_tracking (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    signal_id TEXT NOT NULL,
                    timestamp DATETIME NOT NULL,
                    price REAL NOT NULL,
                    change_percent REAL NOT NULL,
                    minutes_elapsed INTEGER NOT NULL,
                    FOREIGN KEY (signal_id) REFERENCES signals (signal_id) ON DELETE CASCADE
                )''',
                
                '''CREATE TABLE IF NOT EXISTS user_feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    signal_id TEXT NOT NULL,
                    feedback_type TEXT NOT NULL,
                    actual_result TEXT,
                    actual_change REAL,
                    time_to_target INTEGER,
                    user_notes TEXT,
                    feedback_timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (signal_id) REFERENCES signals (signal_id) ON DELETE CASCADE
                )''',
                
                # Create indexes for better performance
                'CREATE INDEX IF NOT EXISTS idx_signals_symbol ON signals(symbol)',
                'CREATE INDEX IF NOT EXISTS idx_signals_status ON signals(status)',
                'CREATE INDEX IF NOT EXISTS idx_signals_timestamp ON signals(entry_timestamp)',
                'CREATE INDEX IF NOT EXISTS idx_price_tracking_signal ON price_tracking(signal_id)',
                'CREATE INDEX IF NOT EXISTS idx_price_tracking_timestamp ON price_tracking(timestamp)',
                'CREATE INDEX IF NOT EXISTS idx_user_feedback_signal ON user_feedback(signal_id)'
            ]
            
            self._db_pool.execute_transaction(operations, readonly=False)
            logger.info("Database schema initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            raise e
    
    def save_new_signal(self, symbol: str, signal_data: Dict) -> str:
        """Save a new signal with proper transaction management"""
        try:
            signal_id = f"{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:23]}"  # Include microseconds for uniqueness
            
            # Handle different input formats
            if 'prediction_data' in signal_data:
                pred_data = signal_data['prediction_data']
                signal_type = pred_data.get('type', 'unknown')
                predicted_change = pred_data.get('predicted_change', 0) or pred_data.get('change', 0)
                confidence = pred_data.get('confidence', 0)
                entry_price = signal_data.get('current_price', 0)
            else:
                signal_type = signal_data.get('type', 'unknown')
                predicted_change = signal_data.get('predicted_change', 0)
                confidence = signal_data.get('confidence', 0)
                entry_price = signal_data.get('entry_price', 0)
            
            expected_timeframe = signal_data.get('expected_timeframe', 60)  # Default 1 hour
            entry_timestamp = datetime.now().isoformat()
            
            # Use connection pool for the operation
            operations = [
                ("BEGIN IMMEDIATE", ()),
                ('''INSERT INTO signals (
                    signal_id, symbol, signal_type, entry_price, entry_timestamp,
                    predicted_change, confidence_score, expected_timeframe, status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'monitoring')''', (
                    signal_id, symbol, signal_type, entry_price, entry_timestamp,
                    predicted_change, confidence, expected_timeframe
                )),
                ('''INSERT INTO price_tracking (
                    signal_id, timestamp, price, change_percent, minutes_elapsed
                ) VALUES (?, ?, ?, 0.0, 0)''', (signal_id, entry_timestamp, entry_price))
            ]
            
            self._db_pool.execute_transaction(operations, readonly=False)
            logger.info(f"Signal saved successfully: {signal_id}")
            return signal_id
                
        except Exception as e:
            logger.error(f"Error in save_new_signal: {e}")
            raise e
    
    @db_operation(readonly=False)
    def update_price_tracking(self, conn: sqlite3.Connection, symbol: str, price: float):
        """Update price tracking for all active signals of a symbol"""
        try:
            cursor = conn.cursor()
            cursor.execute("BEGIN IMMEDIATE")
            
            try:
                # Get active signals for the symbol
                cursor.execute('''
                    SELECT signal_id, entry_price, entry_timestamp 
                    FROM signals 
                    WHERE symbol = ? AND status = 'monitoring'
                ''', (symbol,))
                
                signals = cursor.fetchall()
                current_time = datetime.now()
                
                for signal_id, entry_price, entry_timestamp_str in signals:
                    # Calculate metrics
                    entry_timestamp = datetime.fromisoformat(entry_timestamp_str)
                    minutes_elapsed = int((current_time - entry_timestamp).total_seconds() / 60)
                    change_percent = ((price - entry_price) / entry_price * 100) if entry_price > 0 else 0
                    
                    # Insert price tracking record
                    cursor.execute('''
                        INSERT INTO price_tracking (
                            signal_id, timestamp, price, change_percent, minutes_elapsed
                        ) VALUES (?, ?, ?, ?, ?)
                    ''', (signal_id, current_time.isoformat(), price, change_percent, minutes_elapsed))
                    
                    # Update signal status if target reached or timeout
                    if abs(change_percent) >= 5.0:  # Target reached
                        cursor.execute('''
                            UPDATE signals SET status = 'completed' WHERE signal_id = ?
                        ''', (signal_id,))
                    elif minutes_elapsed >= 240:  # 4 hours timeout
                        cursor.execute('''
                            UPDATE signals SET status = 'completed' WHERE signal_id = ?
                        ''', (signal_id,))
                
                conn.commit()
                logger.debug(f"Price tracking updated for {len(signals)} signals of {symbol}")
                
            except Exception as e:
                conn.rollback()
                logger.error(f"Failed to update price tracking: {e}")
                raise e
                
        except Exception as e:
            logger.error(f"Error in update_price_tracking: {e}")
            raise e
    
    def get_active_signals(self) -> List[Dict]:
        """Get all active signals with latest price information"""
        try:
            query = '''
            SELECT s.*, pt.price as current_price, pt.change_percent as current_change,
                   pt.minutes_elapsed, pt.timestamp as last_update
            FROM signals s
            LEFT JOIN price_tracking pt ON s.signal_id = pt.signal_id
            WHERE s.status IN ('monitoring', 'completed')
            AND (pt.id IS NULL OR pt.id = (
                SELECT MAX(id) FROM price_tracking WHERE signal_id = s.signal_id
            ))
            ORDER BY s.entry_timestamp DESC
            '''
            
            # Use connection pool for read operation
            with self._db_pool.get_connection(readonly=True) as conn:
                cursor = conn.cursor()
                cursor.execute(query)
                columns = [desc[0] for desc in cursor.description]
                results = cursor.fetchall()
                
                signals = []
                for row in results:
                    signal_dict = dict(zip(columns, row))
                    signals.append(signal_dict)
                
                return signals
            
        except Exception as e:
            logger.error(f"Error getting active signals: {e}")
            return []
    
    @db_operation(readonly=False)
    def save_user_feedback(self, conn: sqlite3.Connection, signal_id: str, feedback_data: Dict):
        """Save user feedback for a signal"""
        try:
            cursor = conn.cursor()
            cursor.execute("BEGIN IMMEDIATE")
            
            try:
                cursor.execute('''
                    INSERT INTO user_feedback (
                        signal_id, feedback_type, actual_result, actual_change,
                        time_to_target, user_notes
                    ) VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    signal_id,
                    feedback_data.get('type', 'manual'),
                    feedback_data.get('result', ''),
                    feedback_data.get('actual_change'),
                    feedback_data.get('time_to_target'),
                    feedback_data.get('notes', '')
                ))
                
                # Update signal status
                cursor.execute('''
                    UPDATE signals SET status = 'feedback_received' WHERE signal_id = ?
                ''', (signal_id,))
                
                conn.commit()
                logger.info(f"Feedback saved for signal: {signal_id}")
                
            except Exception as e:
                conn.rollback()
                logger.error(f"Failed to save feedback: {e}")
                raise e
                
        except Exception as e:
            logger.error(f"Error in save_user_feedback: {e}")
            raise e
    
    def get_performance_stats(self, days: int = 30) -> Dict[str, Any]:
        """Get performance statistics for the last N days"""
        try:
            with self._db_pool.get_connection(readonly=True) as conn:
                cursor = conn.cursor()
                
                # Basic signal stats
                cursor.execute('''
                    SELECT COUNT(*) as total,
                           AVG(confidence_score) as avg_confidence,
                           signal_type,
                           status
                    FROM signals 
                    WHERE entry_timestamp >= datetime('now', '-{} days')
                    GROUP BY signal_type, status
                '''.format(days))
                
                stats_data = cursor.fetchall()
                
                # Feedback-based success rate
                cursor.execute('''
                    SELECT COUNT(*) as total_feedback,
                           SUM(CASE WHEN feedback_type = 'success' THEN 1 ELSE 0 END) as successes
                    FROM user_feedback uf
                    JOIN signals s ON uf.signal_id = s.signal_id
                    WHERE s.entry_timestamp >= datetime('now', '-{} days')
                '''.format(days))
                
                feedback_data = cursor.fetchone()
                
                # Compile statistics
                total_signals = sum(row[0] for row in stats_data)
                avg_confidence = sum(row[0] * row[1] for row in stats_data) / total_signals if total_signals > 0 else 0
                
                success_rate = 0.0
                if feedback_data and feedback_data[0] > 0:
                    success_rate = (feedback_data[1] / feedback_data[0]) * 100
                
                stats = {
                    "total_signals": total_signals,
                    "success_rate": success_rate,
                    "average_confidence": avg_confidence,
                    "feedback_count": feedback_data[0] if feedback_data else 0,
                    "period_days": days,
                    "by_type": {}
                }
                
                # Group by signal type
                for row in stats_data:
                    signal_type = row[2]
                    if signal_type not in stats["by_type"]:
                        stats["by_type"][signal_type] = {}
                    
                    stats["by_type"][signal_type][row[3]] = row[0]  # status -> count
                
                return stats
            
        except Exception as e:
            logger.error(f"Error getting performance stats: {e}")
            return {"total_signals": 0, "success_rate": 0, "average_confidence": 0}
    
    @db_operation(readonly=False)
    def cleanup_old_signals(self, conn: sqlite3.Connection, days: int = 30):
        """Clean up old completed signals to maintain database performance"""
        try:
            cursor = conn.cursor()
            cursor.execute("BEGIN IMMEDIATE")
            
            try:
                # Delete old price tracking records first (foreign key constraint)
                cursor.execute('''
                    DELETE FROM price_tracking 
                    WHERE signal_id IN (
                        SELECT signal_id FROM signals 
                        WHERE status = 'completed' 
                        AND entry_timestamp < datetime('now', '-{} days')
                    )
                '''.format(days))
                
                # Delete old signals (keep feedback for analysis)
                cursor.execute('''
                    DELETE FROM signals 
                    WHERE status = 'completed' 
                    AND entry_timestamp < datetime('now', '-{} days')
                '''.format(days))
                
                deleted_count = cursor.rowcount
                conn.commit()
                
                logger.info(f"Cleaned up {deleted_count} old signals")
                return deleted_count
                
            except Exception as e:
                conn.rollback()
                logger.error(f"Failed to cleanup old signals: {e}")
                raise e
                
        except Exception as e:
            logger.error(f"Error in cleanup_old_signals: {e}")
            return 0
    
    def close(self):
        """Close the tracker and cleanup resources"""
        logger.info("Closing ImprovedSignalTracker")
        # Connection pool will be closed globally
    
    def __del__(self):
        """Cleanup on destruction"""
        try:
            self.close()
        except:
            pass

# Factory function for backward compatibility
def create_signal_tracker(db_path: Optional[str] = None) -> ImprovedSignalTracker:
    """Create a new ImprovedSignalTracker instance"""
    return ImprovedSignalTracker(db_path)

if __name__ == "__main__":
    # Test the improved signal tracker
    print("Testing ImprovedSignalTracker...")
    
    try:
        tracker = ImprovedSignalTracker()
        
        # Test signal creation
        test_signal_data = {
            'type': 'momentum_test',
            'predicted_change': 5.2,
            'confidence': 0.85,
            'entry_price': 67250.0,
            'expected_timeframe': 60
        }
        
        signal_id = tracker.save_new_signal('TESTBTC', test_signal_data)
        print(f"Created test signal: {signal_id}")
        
        # Test price update
        tracker.update_price_tracking('TESTBTC', 68000.0)
        print("Updated price tracking")
        
        # Test getting active signals
        signals = tracker.get_active_signals()
        print(f"Active signals: {len(signals)}")
        
        # Test statistics
        stats = tracker.get_performance_stats()
        print(f"Performance stats: {stats}")
        
        print("✅ All tests passed!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
    finally:
        close_db_pool()