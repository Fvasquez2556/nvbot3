#!/usr/bin/env python3
# scripts/test_feedback_system.py
"""
Script para probar que el sistema de retroalimentación funciona correctamente
Genera señales de prueba y verifica que todo el flujo funcione
"""
import sys
import time
import random
from datetime import datetime, timedelta
import json
import os

# Agregar paths necesarios
sys.path.append('.')
sys.path.append('integration')
sys.path.append('web_dashboard')

def test_signal_tracker_direct():
    """Prueba directa del SignalTracker"""
    
    print("🧪 PRUEBA 1: SignalTracker directo")
    print("-" * 40)
    
    try:
        sys.path.append('web_dashboard')
        from web_dashboard.database.signal_tracker import SignalTracker
        
        tracker = SignalTracker()
        print("   ✅ SignalTracker inicializado correctamente")
        
        # Crear señal de prueba
        test_signal_data = {
            'type': 'momentum_alto',
            'predicted_change': 4.5,
            'confidence': 0.82,
            'entry_price': 67250.0
        }
        
        signal_id = tracker.save_new_signal('BTCUSDT', test_signal_data)
        
        if signal_id:
            print(f"   ✅ Señal de prueba creada: {signal_id}")
            
            # Simular actualización de precio
            tracker.update_price_tracking('BTCUSDT', 67800.0)
            print("   ✅ Precio actualizado correctamente")
            
            # Obtener señales activas
            active_signals = tracker.get_active_signals()
            print(f"   ✅ Señales activas obtenidas: {len(active_signals)}")
            
            return True
        else:
            print("   ❌ No se pudo crear la señal de prueba")
            return False
            
    except Exception as e:
        print(f"   ❌ Error en prueba directa: {e}")
        return False

def test_integration_bridge():
    """Prueba el puente de integración"""
    
    print("\n🌉 PRUEBA 2: Bridge de integración")
    print("-" * 40)
    
    try:
        from integration.nvbot3_feedback_bridge import track_signal, get_tracking_stats, update_price
        
        print("   ✅ Bridge importado correctamente")
        
        # Generar datos de prueba similares a nvbot3
        symbols = ['BTCUSDT', 'ETHUSDT', 'ADAUSDT']
        signal_types = ['momentum_alto', 'rebound', 'consolidacion']
        
        signals_created = 0
        
        for i in range(3):
            symbol = random.choice(symbols)
            signal_type = random.choice(signal_types)
            
            # Simular predicción del bot
            prediction = {
                'type': signal_type,
                'predicted_change': round(random.uniform(2.0, 8.0), 2),
                'confidence': round(random.uniform(0.6, 0.95), 2)
            }
            
            current_price = random.uniform(45000, 70000) if 'BTC' in symbol else random.uniform(2000, 4000)
            
            # Trackear la señal
            signal_id = track_signal(symbol, prediction, current_price)
            
            if signal_id:
                signals_created += 1
                print(f"   ✅ Señal {i+1} creada: {symbol} - {signal_type}")
                
                # Simular actualización de precio después de un tiempo
                time.sleep(0.5)  # Pequeña pausa
                new_price = current_price * (1 + random.uniform(-0.05, 0.05))
                update_price(symbol, new_price)
                
            else:
                print(f"   ❌ Error creando señal {i+1}")
        
        print(f"   📊 Señales creadas exitosamente: {signals_created}/3")
        
        # Obtener estadísticas
        stats = get_tracking_stats()
        print(f"   📈 Estadísticas obtenidas: {stats}")
        
        return signals_created > 0
        
    except Exception as e:
        print(f"   ❌ Error en bridge de integración: {e}")
        return False

def test_flask_app():
    """Prueba que la aplicación Flask se puede importar"""
    
    print("\n🌐 PRUEBA 3: Aplicación Flask")
    print("-" * 40)
    
    original_dir = os.getcwd()  # Save the current working directory

    try:
        # Cambiar temporalmente al directorio web_dashboard
        os.chdir('web_dashboard')
        sys.path.append(os.getcwd())

        # Intentar importar la app Flask
        from web_dashboard.app import app
        print("   ✅ Aplicación Flask importada correctamente")

        # Verificar que tiene las rutas principales
        routes = [rule.rule for rule in app.url_map.iter_rules()]
        required_routes = ['/', '/api/signals/active', '/api/feedback']

        missing_routes = [route for route in required_routes if route not in routes]

        if not missing_routes:
            return True
        else:
            print(f"   ⚠️ Rutas faltantes: {missing_routes}")
            return False

    except Exception as e:
        print(f"   ❌ Error probando Flask app: {e}")
        return False

    finally:
        # Restore the original directory
        os.chdir(original_dir)

def generate_demo_data():
    """Genera datos de demostración más realistas"""
    
    print("\n📊 GENERANDO DATOS DE DEMOSTRACIÓN")
    print("-" * 40)
    
    try:
        from integration.nvbot3_feedback_bridge import track_signal, update_price
        
        # Datos de demo más realistas
        demo_scenarios = [
            {
                'symbol': 'BTCUSDT',
                'prediction': {'type': 'momentum_alto', 'predicted_change': 6.2, 'confidence': 0.89},
                'entry_price': 67450.0,
                'price_updates': [67800, 68200, 67900, 68500]  # Simula precio subiendo
            },
            {
                'symbol': 'ETHUSDT', 
                'prediction': {'type': 'rebound', 'predicted_change': 3.8, 'confidence': 0.76},
                'entry_price': 3250.0,
                'price_updates': [3280, 3320, 3290, 3380]  # Rebote exitoso
            },
            {
                'symbol': 'ADAUSDT',
                'prediction': {'type': 'consolidacion', 'predicted_change': 1.5, 'confidence': 0.65},
                'entry_price': 0.45,
                'price_updates': [0.452, 0.448, 0.451, 0.449]  # Consolidación
            }
        ]
        
        signals_created = 0
        
        for scenario in demo_scenarios:
            # Crear la señal inicial
            signal_id = track_signal(
                scenario['symbol'], 
                scenario['prediction'], 
                scenario['entry_price']
            )
            
            if signal_id:
                signals_created += 1
                print(f"   ✅ Demo: {scenario['symbol']} - {scenario['prediction']['type']}")
                
                # Simular evolución del precio a lo largo del tiempo
                for i, price in enumerate(scenario['price_updates']):
                    time.sleep(0.2)  # Pequeña pausa entre actualizaciones
                    update_price(scenario['symbol'], price)
                    
                    change_pct = ((price - scenario['entry_price']) / scenario['entry_price']) * 100
                    print(f"      📈 Actualización {i+1}: ${price:.2f} ({change_pct:+.2f}%)")
            
            time.sleep(1)  # Pausa entre señales
        
        print(f"   🎯 Señales de demostración creadas: {signals_created}")
        return True
        
    except Exception as e:
        print(f"   ❌ Error generando datos de demo: {e}")
        return False

def comprehensive_test():
    """Ejecuta todas las pruebas del sistema"""
    
    print("🚀 PRUEBA COMPLETA DEL SISTEMA DE RETROALIMENTACIÓN")
    print("=" * 60)
    
    # Verificar entorno
    if 'nvbot3_env' not in sys.executable:
        print("⚠️ Advertencia: Entorno virtual nvbot3_env no detectado")
        print("   Recomendación: Activa el entorno antes de las pruebas")
    
    print(f"🐍 Python: {sys.version}")
    print(f"📂 Directorio: {sys.path[0]}")
    
    # Ejecutar pruebas en secuencia
    test_results = {
        'SignalTracker directo': test_signal_tracker_direct(),
        'Bridge de integración': test_integration_bridge(), 
        'Aplicación Flask': test_flask_app(),
        'Datos de demostración': generate_demo_data()
    }
    
    # Resumen de resultados
    print("\n" + "=" * 60)
    print("📋 RESUMEN DE PRUEBAS")
    print("=" * 60)
    
    passed = 0
    total = len(test_results)
    
    for test_name, result in test_results.items():
        status = "✅ PASÓ" if result else "❌ FALLÓ"
        print(f"   {status} - {test_name}")
        if result:
            passed += 1
    
    success_rate = (passed / total) * 100
    print(f"\n📊 Resultado: {passed}/{total} pruebas exitosas ({success_rate:.1f}%)")
    
    if passed == total:
        print("\n🎉 ¡TODAS LAS PRUEBAS PASARON!")
        print("✅ El sistema está listo para usar")
        print("\n📋 Próximos pasos:")
        print("   1. Ejecutar: python scripts/start_dashboard.py")
        print("   2. Abrir: http://localhost:5000")
        print("   3. Integrar con tu código nvbot3")
    else:
        print(f"\n⚠️ {total - passed} pruebas fallaron")
        print("🔧 Revisa los errores mostrados arriba")
        print("💡 Ejecuta: python scripts/fix_import_errors.py")
    
    return passed == total

if __name__ == "__main__":
    success = comprehensive_test()
    
    if success:
        print("\n🚀 Sistema listo - puedes proceder con la integración")
        exit(0)
    else:
        print("\n❌ Se encontraron problemas - revisa la configuración")
        exit(1)