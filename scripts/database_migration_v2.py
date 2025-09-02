#!/usr/bin/env python3
"""
Database Migration v2 for NvBot3 Signal Enhancement
==================================================

Adds reference_price field and signal lifecycle improvements:
- reference_price: Price when signal was first created
- current_progress: Real-time progress calculation
- signal_status: Enhanced lifecycle management
"""

import sqlite3
import os
import sys
import shutil
from datetime import datetime
from pathlib import Path

# Add project paths
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.append(str(project_root))

def backup_database(db_path: str) -> str:
    """Create a backup of the existing database"""
    if not os.path.exists(db_path):
        print(f"Database not found at {db_path}")
        return None
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    backup_path = f"{db_path}_migration_v2_backup_{timestamp}.db"
    
    try:
        shutil.copy2(db_path, backup_path)
        print(f"SUCCESS: Database backed up to: {backup_path}")
        return backup_path
    except Exception as e:
        print(f"ERROR: Failed to backup database: {e}")
        return None

def migrate_database_schema(db_path: str) -> bool:
    """Add new fields for enhanced signal tracking"""
    try:
        conn = sqlite3.connect(db_path, timeout=30.0)
        cursor = conn.cursor()
        
        print("Checking current schema...")
        
        # Check if reference_price already exists
        cursor.execute("PRAGMA table_info(signals)")
        columns = {row[1]: row[2] for row in cursor.fetchall()}
        
        migrations_needed = []
        
        # Check for reference_price field
        if 'reference_price' not in columns:
            migrations_needed.append({
                'field': 'reference_price',
                'sql': 'ALTER TABLE signals ADD COLUMN reference_price REAL DEFAULT 0.0',
                'update': '''UPDATE signals SET reference_price = entry_price 
                           WHERE reference_price IS NULL OR reference_price = 0.0'''
            })
        
        # Check for current_progress field
        if 'current_progress' not in columns:
            migrations_needed.append({
                'field': 'current_progress',
                'sql': 'ALTER TABLE signals ADD COLUMN current_progress REAL DEFAULT 0.0',
                'update': None
            })
        
        # Check for signal_status field (enhanced version)
        if 'signal_status' not in columns:
            migrations_needed.append({
                'field': 'signal_status',
                'sql': 'ALTER TABLE signals ADD COLUMN signal_status TEXT DEFAULT \'active\'',
                'update': '''UPDATE signals SET signal_status = 
                           CASE 
                               WHEN status = 'monitoring' THEN 'active'
                               WHEN status = 'completed' THEN 'expired'
                               WHEN status = 'feedback_received' THEN 'completed'
                               ELSE 'active'
                           END'''
            })
        
        # Check for last_updated field
        if 'last_updated' not in columns:
            migrations_needed.append({
                'field': 'last_updated',
                'sql': 'ALTER TABLE signals ADD COLUMN last_updated DATETIME DEFAULT CURRENT_TIMESTAMP',
                'update': 'UPDATE signals SET last_updated = created_at WHERE last_updated IS NULL'
            })
        
        if not migrations_needed:
            print("SUCCESS: Database schema is already up to date")
            conn.close()
            return True
        
        print(f"Applying {len(migrations_needed)} schema migrations...")
        
        for migration in migrations_needed:
            try:
                # Add the field
                print(f"   Adding {migration['field']}...")
                cursor.execute(migration['sql'])
                
                # Update existing records if needed
                if migration['update']:
                    cursor.execute(migration['update'])
                    print(f"   Updated existing records for {migration['field']}")
                
            except Exception as e:
                print(f"ERROR: Error applying migration for {migration['field']}: {e}")
                conn.rollback()
                conn.close()
                return False
        
        # Create new indexes for performance
        print("Creating performance indexes...")
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_signals_symbol_status ON signals(symbol, signal_status)",
            "CREATE INDEX IF NOT EXISTS idx_signals_confidence_desc ON signals(confidence_score DESC)",
            "CREATE INDEX IF NOT EXISTS idx_signals_last_updated ON signals(last_updated DESC)",
            "CREATE INDEX IF NOT EXISTS idx_signals_reference_price ON signals(reference_price)"
        ]
        
        for index_sql in indexes:
            try:
                cursor.execute(index_sql)
            except Exception as e:
                print(f"WARNING: Could not create index: {e}")
        
        # Commit all changes
        conn.commit()
        print("SUCCESS: Database schema migration completed successfully")
        
        # Verify the changes
        cursor.execute("PRAGMA table_info(signals)")
        new_columns = {row[1]: row[2] for row in cursor.fetchall()}
        
        print("Updated schema:")
        for field_name, field_type in new_columns.items():
            status = "NEW" if field_name in [m['field'] for m in migrations_needed] else ""
            print(f"   {field_name:<20} {field_type:<10} {status}")
        
        conn.close()
        return True
        
    except Exception as e:
        print(f"ERROR: Error during database migration: {e}")
        return False

def update_existing_signals(db_path: str) -> bool:
    """Update existing signals with reference prices and initial progress"""
    try:
        conn = sqlite3.connect(db_path, timeout=30.0)
        cursor = conn.cursor()
        
        print("Updating existing signals...")
        
        # Update signals that don't have reference_price set
        cursor.execute('''
            UPDATE signals 
            SET reference_price = entry_price,
                current_progress = 0.0,
                last_updated = CURRENT_TIMESTAMP
            WHERE reference_price IS NULL 
               OR reference_price = 0.0 
               OR reference_price = entry_price
        ''')
        
        updated_count = cursor.rowcount
        
        # Calculate current progress for signals with price tracking
        cursor.execute('''
            UPDATE signals 
            SET current_progress = ROUND(
                CASE 
                    WHEN reference_price > 0 AND pt.price IS NOT NULL 
                    THEN ((pt.price - reference_price) / reference_price * 100)
                    ELSE 0.0 
                END, 3
            ),
            last_updated = CURRENT_TIMESTAMP
            FROM (
                SELECT signal_id, price 
                FROM price_tracking pt1 
                WHERE pt1.id = (
                    SELECT MAX(id) FROM price_tracking pt2 
                    WHERE pt2.signal_id = pt1.signal_id
                )
            ) pt
            WHERE signals.signal_id = pt.signal_id
        ''')
        
        progress_updated = cursor.rowcount
        
        conn.commit()
        conn.close()
        
        print(f"SUCCESS: Updated {updated_count} signals with reference prices")
        print(f"SUCCESS: Updated {progress_updated} signals with current progress")
        
        return True
        
    except Exception as e:
        print(f"ERROR: Error updating existing signals: {e}")
        return False

def main():
    """Main migration function"""
    print("NvBot3 Database Migration v2")
    print("=" * 50)
    print("Adding enhanced signal tracking features:")
    print("- reference_price: Price at signal creation")
    print("- current_progress: Real-time progress calculation")
    print("- signal_status: Enhanced lifecycle management")
    print("- Performance indexes")
    print()
    
    # Find database
    db_path = project_root / 'web_dashboard' / 'database' / 'signals.db'
    
    if not db_path.exists():
        print(f"ERROR: Database not found at {db_path}")
        return False
    
    print(f"Database: {db_path}")
    
    # Step 1: Backup
    print("\nStep 1: Creating backup...")
    backup_path = backup_database(str(db_path))
    if not backup_path:
        print("ERROR: Backup failed, aborting migration")
        return False
    
    # Step 2: Apply schema migration
    print("\nStep 2: Applying schema migrations...")
    if not migrate_database_schema(str(db_path)):
        print("ERROR: Schema migration failed")
        return False
    
    # Step 3: Update existing data
    print("\nStep 3: Updating existing signals...")
    if not update_existing_signals(str(db_path)):
        print("ERROR: Data update failed")
        return False
    
    # Step 4: Verify migration
    print("\nStep 4: Verifying migration...")
    try:
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        
        # Check record counts
        cursor.execute("SELECT COUNT(*) FROM signals")
        total_signals = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM signals WHERE reference_price > 0")
        with_reference = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM signals WHERE signal_status = 'active'")
        active_signals = cursor.fetchone()[0]
        
        print(f"   Total signals: {total_signals}")
        print(f"   With reference price: {with_reference}")
        print(f"   Active signals: {active_signals}")
        
        conn.close()
        
    except Exception as e:
        print(f"WARNING: Verification warning: {e}")
    
    print("\nMigration completed successfully!")
    print("=" * 50)
    print("Summary of changes:")
    print("- Added reference_price field (tracks initial signal price)")
    print("- Added current_progress field (real-time progress calculation)")
    print("- Added signal_status field (enhanced lifecycle)")
    print("- Added last_updated field (timestamp tracking)")
    print("- Created performance indexes")
    print("- Updated existing signals with reference data")
    print()
    print(f"Backup saved: {backup_path}")
    print()
    print("Next steps:")
    print("1. Update signal tracking code to use reference_price")
    print("2. Implement progress calculation in dashboard")
    print("3. Add signal lifecycle management")
    print("4. Test the enhanced functionality")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)