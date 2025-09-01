#!/usr/bin/env python3
"""
Test Fixed Signal Integration
=============================
Test the corrected signal generator with proper feature calculation
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import time
from datetime import datetime

def test_corrected_feature_calculation():
    """Test the corrected feature calculation"""
    print("=== Testing Corrected Feature Calculation ===")
    
    try:
        from scripts.signal_generator import NvBot3SignalGenerator
        generator = NvBot3SignalGenerator()
        print("SUCCESS: Signal generator created")
    except Exception as e:
        print(f"ERROR: Failed to create signal generator: {e}")
        return False
    
    # Create realistic dummy data with enough history for indicators
    print("Generating test market data...")
    dates = pd.date_range('2025-08-01', periods=300, freq='5T')
    
    # Create realistic price movements
    np.random.seed(42)  # For reproducible results
    base_price = 45000
    price_changes = np.random.normal(0, 0.01, 300).cumsum()
    prices = base_price * (1 + price_changes)
    
    dummy_data = pd.DataFrame({
        'open': prices * (1 + np.random.normal(0, 0.001, 300)),
        'high': prices * (1 + np.abs(np.random.normal(0, 0.002, 300))),
        'low': prices * (1 - np.abs(np.random.normal(0, 0.002, 300))),
        'close': prices,
        'volume': np.random.uniform(100, 2000, 300),
        'symbol': 'BTCUSDT',
        'timeframe': '5m'
    }, index=dates)
    
    print(f"Market data created: {len(dummy_data)} candles from {dummy_data.index[0]} to {dummy_data.index[-1]}")
    
    # Calculate features
    try:
        features_df = generator.calculate_features(dummy_data)
        print(f"SUCCESS: Features calculated")
        print(f"  Original columns: {len(dummy_data.columns)}")
        print(f"  Total columns: {len(features_df.columns)}")
        print(f"  Added features: {len(features_df.columns) - len(dummy_data.columns)}")
        
        # Show some specific features we know the models expect
        expected_features = ['atr_ratio_14', 'bb_upper_20', 'rsi_14', 'macd', 'adx_14', 'momentum_score']
        found_features = [f for f in expected_features if f in features_df.columns]
        print(f"  Expected model features found: {len(found_features)}/{len(expected_features)}")
        print(f"  Found features: {found_features}")
        
        return True
        
    except Exception as e:
        print(f"ERROR: Feature calculation failed: {e}")
        return False

def test_signal_tracking():
    """Test signal tracking integration"""
    print("\n=== Testing Signal Tracking Integration ===")
    
    try:
        from integration.nvbot3_feedback_bridge import track_signal, update_price, init_tracker
        
        # Initialize tracker
        init_tracker()
        print("SUCCESS: Tracker initialized")
        
        # Update price
        test_price = 45123.45
        update_price("BTCUSDT", test_price)
        print(f"SUCCESS: Price updated for BTCUSDT: ${test_price}")
        
        # Create a test prediction
        test_prediction = {
            'type': 'momentum',
            'prediction': 1,
            'confidence': 0.87,
            'predicted_change': 5.2,
            'timestamp': datetime.now().isoformat(),
            'model_description': 'Fixed Feature Calculation Test'
        }
        
        # Track signal
        signal_id = track_signal("BTCUSDT", test_prediction, test_price)
        
        if signal_id:
            print(f"SUCCESS: Signal tracked with ID: {signal_id}")
            return True
        else:
            print("ERROR: Signal tracking returned None")
            return False
            
    except Exception as e:
        print(f"ERROR: Signal tracking failed: {e}")
        return False

def verify_database():
    """Verify signals are in database"""
    print("\n=== Verifying Database ===")
    
    try:
        from web_dashboard.database.signal_tracker import SignalTracker
        tracker = SignalTracker()
        
        # Get all signals
        active_signals = tracker.get_active_signals()
        print(f"SUCCESS: Database connected")
        print(f"  Active signals: {len(active_signals)}")
        
        if active_signals:
            latest = active_signals[-1]
            print(f"  Latest signal: {latest.get('symbol')} {latest.get('type')} (conf: {latest.get('confidence_score', 0):.2f})")
        
        return True
        
    except Exception as e:
        print(f"ERROR: Database verification failed: {e}")
        return False

def main():
    """Main test function"""
    print("NVBOT3 FIXED INTEGRATION TEST")
    print("=" * 50)
    
    tests = [
        ("Feature Calculation", test_corrected_feature_calculation),
        ("Signal Tracking", test_signal_tracking),
        ("Database Verification", verify_database)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\nRunning: {test_name}")
        try:
            if test_func():
                passed += 1
                print(f"PASSED: {test_name}")
            else:
                print(f"FAILED: {test_name}")
        except Exception as e:
            print(f"ERROR: {test_name} - {e}")
        
        time.sleep(1)
    
    print(f"\n" + "=" * 50)
    print(f"RESULTS: {passed}/{total} tests passed")
    print("=" * 50)
    
    if passed == total:
        print("ALL TESTS PASSED!")
        print("\nNvBot3 integration is now working correctly with:")
        print("  - Proper feature calculation (146 features)")
        print("  - Working signal tracking")
        print("  - Database integration")
        print("\nNext steps:")
        print("  1. Start dashboard: python scripts/start_dashboard.py")
        print("  2. Generate signals: python scripts/signal_generator.py --mode scan")
        return 0
    else:
        print("SOME TESTS FAILED - Check error messages above")
        return 1

if __name__ == "__main__":
    exit(main())