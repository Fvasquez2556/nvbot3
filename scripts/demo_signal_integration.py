#!/usr/bin/env python3
"""
Simple Demo: Test Signal Integration
===================================

A simple script to demonstrate and test the signal integration functionality
without complex dependencies.
"""

import sys
import os
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import time
from datetime import datetime

def test_basic_integration():
    """Test basic integration components"""
    print("=== Testing Basic Integration ===")
    
    # Test 1: Import integration bridge
    try:
        from integration.nvbot3_feedback_bridge import track_signal, update_price, init_tracker
        print("✅ Successfully imported integration bridge")
    except Exception as e:
        print(f"❌ Failed to import integration bridge: {e}")
        return False
    
    # Test 2: Initialize tracker
    try:
        init_tracker()
        print("✅ Successfully initialized tracker")
    except Exception as e:
        print(f"❌ Failed to initialize tracker: {e}")
        return False
    
    # Test 3: Test price update
    try:
        update_price("BTCUSDT", 45000.0)
        print("✅ Successfully updated price")
    except Exception as e:
        print(f"❌ Failed to update price: {e}")
        return False
    
    # Test 4: Test signal tracking
    try:
        test_prediction = {
            'type': 'momentum',
            'prediction': 1,
            'confidence': 0.85,
            'predicted_change': 5.2,
            'timestamp': datetime.now().isoformat(),
            'model_description': 'Test Momentum Model'
        }
        
        signal_id = track_signal("BTCUSDT", test_prediction, 45000.0)
        
        if signal_id:
            print(f"✅ Successfully tracked signal (ID: {signal_id})")
        else:
            print("❌ Signal tracking returned None")
            return False
            
    except Exception as e:
        print(f"❌ Failed to track signal: {e}")
        return False
    
    # Test 5: Verify database connection
    try:
        from web_dashboard.database.signal_tracker import SignalTracker
        tracker = SignalTracker()
        active_signals = tracker.get_active_signals()
        print(f"✅ Database connected. Found {len(active_signals)} active signals")
    except Exception as e:
        print(f"❌ Failed to connect to database: {e}")
        return False
    
    return True

def test_signal_generator_basic():
    """Test signal generator without exchange connection"""
    print("\n=== Testing Signal Generator (Basic) ===")
    
    try:
        from scripts.signal_generator import NvBot3SignalGenerator
        generator = NvBot3SignalGenerator()
        print("✅ Successfully created signal generator")
    except Exception as e:
        print(f"❌ Failed to create signal generator: {e}")
        return False
    
    # Test feature calculation with dummy data
    try:
        dates = pd.date_range('2025-08-01', periods=50, freq='5T')
        dummy_data = pd.DataFrame({
            'open': np.random.uniform(40000, 45000, 50),
            'high': np.random.uniform(45000, 46000, 50),
            'low': np.random.uniform(39000, 40000, 50),
            'close': np.random.uniform(40000, 45000, 50),
            'volume': np.random.uniform(100, 1000, 50),
            'symbol': 'BTCUSDT',
            'timeframe': '5m'
        }, index=dates)
        
        features_df = generator.calculate_features(dummy_data)
        
        if len(features_df.columns) > len(dummy_data.columns):
            print(f"✅ Features calculated successfully: {len(features_df.columns)} features")
        else:
            print("❌ No additional features calculated")
            return False
            
    except Exception as e:
        print(f"❌ Feature calculation failed: {e}")
        return False
    
    return True

def simulate_signal_generation():
    """Simulate signal generation and tracking"""
    print("\n=== Simulating Complete Signal Flow ===")
    
    try:
        from integration.nvbot3_feedback_bridge import track_signal, update_price
        
        # Simulate different types of signals
        signals_to_test = [
            {
                'symbol': 'BTCUSDT',
                'price': 45000.0,
                'prediction': {
                    'type': 'momentum',
                    'prediction': 1,
                    'confidence': 0.85,
                    'predicted_change': 5.2,
                    'timestamp': datetime.now().isoformat(),
                    'model_description': 'Momentum Model Test'
                }
            },
            {
                'symbol': 'ETHUSDT',
                'price': 2800.0,
                'prediction': {
                    'type': 'rebound',
                    'prediction': 1,
                    'confidence': 0.78,
                    'predicted_change': 2.5,
                    'timestamp': datetime.now().isoformat(),
                    'model_description': 'Rebound Model Test'
                }
            },
            {
                'symbol': 'BNBUSDT',
                'price': 320.0,
                'prediction': {
                    'type': 'regime',
                    'prediction': 2,  # Bull regime
                    'confidence': 0.72,
                    'predicted_change': 0.0,
                    'timestamp': datetime.now().isoformat(),
                    'model_description': 'Regime Model Test'
                }
            }
        ]
        
        tracked_signals = []
        
        for signal_data in signals_to_test:
            # Update price
            update_price(signal_data['symbol'], signal_data['price'])
            
            # Track signal
            signal_id = track_signal(
                signal_data['symbol'], 
                signal_data['prediction'], 
                signal_data['price']
            )
            
            if signal_id:
                tracked_signals.append(signal_id)
                print(f"✅ Tracked {signal_data['prediction']['type']} signal for {signal_data['symbol']} (ID: {signal_id})")
            else:
                print(f"❌ Failed to track signal for {signal_data['symbol']}")
        
        print(f"\n📊 Successfully tracked {len(tracked_signals)} signals")
        
        # Verify signals are in database
        from web_dashboard.database.signal_tracker import SignalTracker
        tracker = SignalTracker()
        active_signals = tracker.get_active_signals()
        
        print(f"📈 Total active signals in database: {len(active_signals)}")
        
        # Show recent signals
        if active_signals:
            print("\n🔍 Recent signals:")
            for signal in active_signals[-3:]:  # Show last 3
                print(f"   - {signal.get('symbol')} {signal.get('type')}: {signal.get('confidence', 0):.2f} confidence")
        
        return True
        
    except Exception as e:
        print(f"❌ Simulation failed: {e}")
        return False

def main():
    """Main demo function"""
    print("NVBOT3 SIGNAL INTEGRATION DEMO")
    print("=" * 50)
    
    all_tests_passed = True
    
    # Run tests
    test_results = [
        ("Basic Integration", test_basic_integration),
        ("Signal Generator Basic", test_signal_generator_basic),
        ("Signal Flow Simulation", simulate_signal_generation)
    ]
    
    for test_name, test_func in test_results:
        print(f"\nRunning: {test_name}")
        try:
            result = test_func()
            if not result:
                all_tests_passed = False
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            all_tests_passed = False
        
        time.sleep(1)  # Brief pause between tests
    
    # Summary
    print("\n" + "=" * 50)
    print("DEMO RESULTS SUMMARY")
    print("=" * 50)
    
    if all_tests_passed:
        print("ALL TESTS PASSED!")
        print("\nSignal integration is working correctly!")
        print("\nNext steps:")
        print("   1. Start web dashboard: python scripts/start_dashboard.py")
        print("   2. Run signal generator: python scripts/signal_generator.py --mode scan")
        print("   3. Monitor signals in web dashboard")
        return 0
    else:
        print("SOME TESTS FAILED")
        print("\nTroubleshooting needed:")
        print("   - Check that web_dashboard database is accessible")
        print("   - Ensure integration bridge is properly configured")
        print("   - Verify all dependencies are installed")
        return 1

if __name__ == "__main__":
    exit(main())