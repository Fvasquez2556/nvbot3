#!/usr/bin/env python3
# scripts/fix_database_schema.py
"""
Corrector de esquema de base de datos para NvBot3 Feedback System
Soluciona el error: table signals has no column named signal_id
"""

import sqlite3
import os
import sys
from pathlib import Path
from datetime import datetime

def backup_existing_database():
    """Hace backup de la base de datos existente"""
    
    db_path = Path("web_dashboard/database/signals.db")
    
    if db_path.exists():
        backup_path = f"web_dashboard/database/signals_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.db"
        
        try:
            import shutil
            shutil.copy2(db_path, backup_path)
            print(f"✅ Backup creado: {backup_path}")
            return backup_path
        except Exception as e:
            print(f"⚠️ No se pudo crear backup: {e}")
            return None
    else:
        print("ℹ️ No existe base de datos previa")
        return None

def inspect_current_database():
    """Inspecciona la estructura actual de la base de datos"""
    
    db_path = Path("web_dashboard/database/signals.db")
    
    if not db_path.exists():
        print("ℹ️ No existe base de datos actual")
        return {}
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Obtener tablas existentes
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        
        print(f"📋 Tablas encontradas: {[t[0] for t in tables]}")
        
        table_info = {}
        
        for table in tables:
            table_name = table[0]
            cursor.execute(f"PRAGMA table_info({table_name});")
            columns = cursor.fetchall()
            table_info[table_name] = columns
            
            print(f"\n📊 Tabla '{table_name}':")
            for col in columns:
                print(f"   - {col[1]} ({col[2]})")
        
        conn.close()
        return table_info
        
    except Exception as e:
        print(f"❌ Error inspeccionando base de datos: {e}")
        return {}

def create_correct_database_schema():
    """Crea la base de datos con el esquema correcto"""
    
    # Asegurar que existe el directorio
    db_dir = Path("web_dashboard/database")
    db_dir.mkdir(parents=True, exist_ok=True)
    
    db_path = db_dir / "signals.db"
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Eliminar tablas existentes si tienen problemas
        print("🗑️ Eliminando tablas problemáticas...")
        cursor.execute("DROP TABLE IF EXISTS signals;")
        cursor.execute("DROP TABLE IF EXISTS price_tracking;")
        cursor.execute("DROP TABLE IF EXISTS user_feedback;")
        
        print("🏗️ Creando tablas con esquema correcto...")
        
        # Tabla principal de señales (ESQUEMA CORRECTO)
        cursor.execute('''
        CREATE TABLE signals (
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
        print("   ✅ Tabla 'signals' creada")
        
        # Tabla de tracking de precios
        cursor.execute('''
        CREATE TABLE price_tracking (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            signal_id TEXT NOT NULL,
            timestamp DATETIME NOT NULL,
            price REAL NOT NULL,
            change_percent REAL NOT NULL,
            minutes_elapsed INTEGER NOT NULL,
            FOREIGN KEY (signal_id) REFERENCES signals (signal_id)
        )
        ''')
        print("   ✅ Tabla 'price_tracking' creada")
        
        # Tabla de retroalimentación del usuario
        cursor.execute('''
        CREATE TABLE user_feedback (
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
        print("   ✅ Tabla 'user_feedback' creada")
        
        # Crear índices para mejor performance
        cursor.execute("CREATE INDEX idx_signals_symbol ON signals(symbol);")
        cursor.execute("CREATE INDEX idx_signals_status ON signals(status);")
        cursor.execute("CREATE INDEX idx_price_tracking_signal_id ON price_tracking(signal_id);")
        cursor.execute("CREATE INDEX idx_user_feedback_signal_id ON user_feedback(signal_id);")
        print("   ✅ Índices creados")
        
        conn.commit()
        conn.close()
        
        print("✅ Base de datos recreada correctamente")
        return True
        
    except Exception as e:
        print(f"❌ Error creando base de datos: {e}")
        return False

def test_new_database():
    """Prueba que la nueva base de datos funcione correctamente"""
    
    print("\n🧪 PROBANDO NUEVA BASE DE DATOS")
    print("-" * 40)
    
    original_dir = os.getcwd()  # Initialize with the current working directory
    try:
        # Cambiar al directorio correcto temporalmente
        os.chdir("web_dashboard")
        sys.path.append(os.getcwd())
        
        from web_dashboard.database.signal_tracker import SignalTracker
        
        # Crear instancia
        tracker = SignalTracker()
        print("   ✅ SignalTracker inicializado")
        
        # Probar crear una señal
        test_signal_data = {
            'type': 'test_corrected',
            'predicted_change': 3.5,
            'confidence': 0.75,
            'entry_price': 50000.0
        }
        
        signal_id = tracker.save_new_signal('TESTFIX', test_signal_data)
        
        if signal_id:
            print(f"   ✅ Señal de prueba creada: {signal_id}")
            
            # Probar actualizar precio
            tracker.update_price_tracking('TESTFIX', 51000.0)
            print("   ✅ Actualización de precio exitosa")
            
            # Probar obtener señales
            active = tracker.get_active_signals()
            print(f"   ✅ Señales activas obtenidas: {len(active)}")
            
            # Probar estadísticas
            stats = tracker.get_performance_stats()
            print(f"   ✅ Estadísticas: {stats}")
            
            os.chdir(original_dir)
            return True
        else:
            print("   ❌ No se pudo crear señal de prueba")
            os.chdir(original_dir)
            return False      
    except Exception as e:
        os.chdir(original_dir)
        print(f"   ❌ Error probando base de datos: {e}")
        return False

def main():
    """Función principal del corrector"""
    
    print("🛠️ CORRECTOR DE ESQUEMA DE BASE DE DATOS - NVBOT3")
    print("=" * 60)
    
    print("Problema detectado: 'table signals has no column named signal_id'")
    print("Solución: Recrear base de datos con esquema correcto\n")
    
    # Paso 1: Inspeccionar base de datos actual
    print("🔍 INSPECCIONANDO BASE DE DATOS ACTUAL")
    print("-" * 45)
    current_schema = inspect_current_database()
    
    # Paso 2: Hacer backup si existe
    print("\n💾 CREANDO BACKUP")
    print("-" * 20)
    backup_created = backup_existing_database()
    
    # Paso 3: Recrear base de datos
    print("\n🏗️ RECREANDO BASE DE DATOS")
    print("-" * 30)
    db_created = create_correct_database_schema()
    
    if not db_created:
        print("❌ Error recreando base de datos. Proceso abortado.")
        return False
    
    # Paso 4: Probar nueva base de datos
    test_success = test_new_database()
    
    # Resultado final
    print("\n" + "=" * 60)
    print("📊 RESULTADO DE LA CORRECCIÓN")
    print("=" * 60)
    
    if test_success:
        print("🎉 ¡CORRECCIÓN EXITOSA!")
        print("✅ Base de datos recreada correctamente")
        print("✅ Esquema de tablas corregido")
        print("✅ Funcionalidad verificada")
        
        if backup_created:
            print(f"💾 Backup disponible: {backup_created}")
        
        print("\n🚀 PRÓXIMOS PASOS:")
        print("   1. Ejecutar: python scripts/test_feedback_system.py")
        print("   2. Iniciar dashboard: python scripts/start_dashboard.py")
        print("   3. Abrir navegador: http://localhost:5000")
        
        return True
    else:
        print("❌ LA CORRECCIÓN TUVO PROBLEMAS")
        print("🔧 Intenta ejecutar nuevamente o contacta soporte")
        
        if backup_created:
            print(f"💾 Backup disponible para restaurar: {backup_created}")
        
        return False

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n⏹️ Corrección cancelada por el usuario")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error inesperado: {e}")
        sys.exit(1)