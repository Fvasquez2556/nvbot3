#!/usr/bin/env python3
"""
Database Migration Script for NvBot3 Improved Database System
============================================================

Migrates existing SignalTracker implementation to the new improved
connection pooling system with minimal disruption.
"""

import os
import sys
import shutil
import sqlite3
import logging
from datetime import datetime
from pathlib import Path

# Add project paths
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / 'web_dashboard'))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

from typing import Optional

def backup_database(db_path: str) -> Optional[str]:
    """Create a backup of the existing database"""
    if not os.path.exists(db_path):
        logger.warning(f"Database not found at {db_path}")
        return None
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    backup_path = f"{db_path}_backup_{timestamp}.db"
    
    try:
        shutil.copy2(db_path, backup_path)
        logger.info(f"Database backed up to: {backup_path}")
        return backup_path
    except Exception as e:
        logger.error(f"Failed to backup database: {e}")
        return None

def verify_database_schema(db_path: str) -> bool:
    """Verify that the database has the expected schema"""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Check for required tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = {row[0] for row in cursor.fetchall()}
        
        required_tables = {'signals', 'price_tracking', 'user_feedback'}
        missing_tables = required_tables - tables
        
        if missing_tables:
            logger.warning(f"Missing tables: {missing_tables}")
            return False
        
        # Check signals table structure
        cursor.execute("PRAGMA table_info(signals)")
        signals_columns = {row[1] for row in cursor.fetchall()}
        
        required_signals_columns = {
            'signal_id', 'symbol', 'signal_type', 'entry_price', 
            'entry_timestamp', 'predicted_change', 'confidence_score', 
            'expected_timeframe', 'status', 'created_at'
        }
        
        missing_columns = required_signals_columns - signals_columns
        if missing_columns:
            logger.warning(f"Missing columns in signals table: {missing_columns}")
            return False
        
        conn.close()
        logger.info("Database schema verification passed")
        return True
        
    except Exception as e:
        logger.error(f"Error verifying database schema: {e}")
        return False

def optimize_database(db_path: str) -> bool:
    """Optimize the database with improved settings"""
    try:
        conn = sqlite3.connect(db_path, timeout=30.0)
        cursor = conn.cursor()
        
        logger.info("Applying database optimizations...")
        
        # Enable WAL mode for better concurrency
        cursor.execute("PRAGMA journal_mode=WAL")
        logger.info("✓ Enabled WAL mode")
        
        # Set busy timeout
        cursor.execute("PRAGMA busy_timeout=30000")
        logger.info("✓ Set busy timeout to 30 seconds")
        
        # Enable foreign keys
        cursor.execute("PRAGMA foreign_keys=ON")
        logger.info("✓ Enabled foreign key constraints")
        
        # Optimize synchronous mode
        cursor.execute("PRAGMA synchronous=NORMAL")
        logger.info("✓ Set synchronous mode to NORMAL")
        
        # Set cache size (10MB)
        cursor.execute("PRAGMA cache_size=10000")
        logger.info("✓ Set cache size to 10MB")
        
        # Enable auto vacuum
        cursor.execute("PRAGMA auto_vacuum=INCREMENTAL")
        logger.info("✓ Enabled incremental auto vacuum")
        
        # Create indexes for performance
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_signals_symbol ON signals(symbol)",
            "CREATE INDEX IF NOT EXISTS idx_signals_status ON signals(status)",
            "CREATE INDEX IF NOT EXISTS idx_signals_timestamp ON signals(entry_timestamp)",
            "CREATE INDEX IF NOT EXISTS idx_price_tracking_signal ON price_tracking(signal_id)",
            "CREATE INDEX IF NOT EXISTS idx_price_tracking_timestamp ON price_tracking(timestamp)",
            "CREATE INDEX IF NOT EXISTS idx_user_feedback_signal ON user_feedback(signal_id)"
        ]
        
        for index_sql in indexes:
            cursor.execute(index_sql)
        
        logger.info("✓ Created performance indexes")
        
        # Analyze tables for query optimization
        cursor.execute("ANALYZE")
        logger.info("✓ Updated table statistics")
        
        conn.commit()
        conn.close()
        
        logger.info("Database optimization completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error optimizing database: {e}")
        return False

def test_improved_system() -> bool:
    """Test the improved database system"""
    try:
        logger.info("Testing improved database system...")
        
        # Test connection pool
        from web_dashboard.database.db_connection_pool import get_db_pool
        pool = get_db_pool()
        
        # Test basic operation
        operations = [
            "SELECT COUNT(*) FROM signals",
            "SELECT COUNT(*) FROM price_tracking",
            "SELECT COUNT(*) FROM user_feedback"
        ]
        
        results = pool.execute_transaction(operations, readonly=True)
        
        logger.info("✓ Connection pool working")
        logger.info(f"  Signals: {results[0][0]}")
        logger.info(f"  Price tracking: {results[1][0]}")
        logger.info(f"  User feedback: {results[2][0]}")
        
        # Test improved tracker
        from web_dashboard.database.signal_tracker_improved import ImprovedSignalTracker
        tracker = ImprovedSignalTracker()
        
        # Test getting active signals
        signals = tracker.get_active_signals()
        logger.info(f"✓ Retrieved {len(signals)} active signals")
        
        # Test stats
        stats = tracker.get_performance_stats()
        logger.info(f"✓ Performance stats: {stats.get('total_signals', 0)} total signals")
        
        logger.info("Improved system test completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error testing improved system: {e}")
        return False

def update_app_py() -> bool:
    """Update app.py to use improved tracker"""
    try:
        app_py_path = project_root / 'web_dashboard' / 'app.py'
        
        if not app_py_path.exists():
            logger.warning("app.py not found, skipping update")
            return True
        
        # Read current content
        with open(app_py_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Create backup
        backup_path = app_py_path.with_suffix('.py.backup')
        shutil.copy2(app_py_path, backup_path)
        logger.info(f"Created backup: {backup_path}")
        
        # Update import statement
        updated_content = content.replace(
            'from database.signal_tracker import SignalTracker',
            '''# Import improved tracker with fallback
try:
    from database.signal_tracker_improved import ImprovedSignalTracker as SignalTracker
    print("✓ Using ImprovedSignalTracker with connection pooling")
except ImportError:
    from database.signal_tracker import SignalTracker
    print("⚠ Using legacy SignalTracker")'''
        )
        
        # Write updated content
        with open(app_py_path, 'w', encoding='utf-8') as f:
            f.write(updated_content)
        
        logger.info("✓ Updated app.py to use improved tracker")
        return True
        
    except Exception as e:
        logger.error(f"Error updating app.py: {e}")
        return False

def main():
    """Main migration function"""
    print("🚀 NvBot3 Database Migration to Improved System")
    print("=" * 60)
    
    # Find database path
    db_path = project_root / 'web_dashboard' / 'database' / 'signals.db'
    
    if not db_path.exists():
        logger.error(f"Database not found at {db_path}")
        return False
    
    logger.info(f"Found database: {db_path}")
    
    # Step 1: Backup
    print("\n📦 Step 1: Creating database backup...")
    backup_path = backup_database(str(db_path))
    if not backup_path:
        logger.error("Failed to create backup, aborting migration")
        return False
    
    # Step 2: Verify schema
    print("\n🔍 Step 2: Verifying database schema...")
    if not verify_database_schema(str(db_path)):
        logger.error("Database schema verification failed")
        return False
    
    # Step 3: Optimize database
    print("\n⚡ Step 3: Optimizing database settings...")
    if not optimize_database(str(db_path)):
        logger.error("Database optimization failed")
        return False
    
    # Step 4: Test improved system
    print("\n🧪 Step 4: Testing improved system...")
    if not test_improved_system():
        logger.error("Improved system test failed")
        return False
    
    # Step 5: Update app.py
    print("\n🔄 Step 5: Updating application code...")
    if not update_app_py():
        logger.warning("Failed to update app.py automatically")
    
    print("\n✅ Migration completed successfully!")
    print("=" * 60)
    print("🎯 Benefits of the improved system:")
    print("  • Thread-safe database operations")
    print("  • Connection pooling for better performance")
    print("  • Automatic retry logic for locked databases")
    print("  • WAL mode for improved concurrency")
    print("  • Optimized indexes for faster queries")
    print("  • Proper transaction management")
    print("  • Database operation queuing")
    print()
    print("📁 Files created:")
    print(f"  • {project_root}/web_dashboard/database/db_connection_pool.py")
    print(f"  • {project_root}/web_dashboard/database/signal_tracker_improved.py")
    print(f"  • {project_root}/integration/nvbot3_feedback_bridge_improved.py")
    print()
    print(f"💾 Database backup: {backup_path}")
    print()
    print("🚀 Next steps:")
    print("  1. Update signal_generator.py to use improved bridge:")
    print("     from integration.nvbot3_feedback_bridge_improved import track_signal")
    print("  2. Restart your dashboard and signal generator")
    print("  3. Monitor logs for any issues")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)