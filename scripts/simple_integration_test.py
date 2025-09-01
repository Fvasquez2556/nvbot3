#!/usr/bin/env python3
"""
Simple Integration Test - NvBot3
===============================

Basic test of signal integration without Unicode characters.
"""

import sys
import os
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import time
from datetime import datetime

def test_integration():
    """Test the signal integration"""
    print("TESTING SIGNAL INTEGRATION")
    print("=" * 40)
    
    success_count = 0
    total_tests = 5
    
    # Test 1: Import integration bridge
    print("\n1. Testing integration bridge import...")
    try:
        from integration.nvbot3_feedback_bridge import track_signal, update_price, init_tracker
        print("   SUCCESS: Integration bridge imported")
        success_count += 1
    except Exception as e:
        print(f"   FAILED: {e}")
    
    # Test 2: Initialize tracker
    print("\n2. Testing tracker initialization...")
    try:
        init_tracker()
        print("   SUCCESS: Tracker initialized")
        success_count += 1
    except Exception as e:
        print(f"   FAILED: {e}")
    
    # Test 3: Test signal tracking
    print("\n3. Testing signal tracking...")
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
            print(f"   SUCCESS: Signal tracked with ID {signal_id}")
            success_count += 1
        else:
            print("   FAILED: Signal tracking returned None")
    except Exception as e:
        print(f"   FAILED: {e}")
    
    # Test 4: Test price update
    print("\n4. Testing price updates...")
    try:
        update_price("BTCUSDT", 45000.0)
        print("   SUCCESS: Price updated")
        success_count += 1
    except Exception as e:
        print(f"   FAILED: {e}")
    
    # Test 5: Test database connection
    print("\n5. Testing database connection...")
    try:
        from web_dashboard.database.signal_tracker import SignalTracker
        tracker = SignalTracker()
        active_signals = tracker.get_active_signals()
        print(f"   SUCCESS: Database connected. {len(active_signals)} active signals")
        success_count += 1
    except Exception as e:
        print(f"   FAILED: {e}")
    
    # Results
    print("\n" + "=" * 40)
    print("INTEGRATION TEST RESULTS")
    print("=" * 40)
    print(f"Tests passed: {success_count}/{total_tests}")
    print(f"Success rate: {(success_count/total_tests)*100:.1f}%")
    
    if success_count == total_tests:
        print("\nALL TESTS PASSED!")
        print("\nIntegration is working correctly.")
        print("\nNext steps:")
        print("1. Start dashboard: python scripts/start_dashboard.py")
        print("2. Run signal generator: python scripts/signal_generator.py --mode scan")
        return True
    else:
        print(f"\n{total_tests - success_count} tests failed.")
        print("Integration needs troubleshooting.")
        return False

def test_signal_generator():
    """Test signal generator basic functionality"""
    print("\n\nTESTING SIGNAL GENERATOR")
    print("=" * 40)
    
    try:
        from scripts.signal_generator import NvBot3SignalGenerator
        generator = NvBot3SignalGenerator()
        print("SUCCESS: Signal generator created")
        
        # Test feature calculation with dummy data
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
            print(f"SUCCESS: Features calculated - {len(features_df.columns)} total features")
            return True
        else:
            print("FAILED: No additional features calculated")
            return False
            
    except Exception as e:
        print(f"FAILED: {e}")
        return False

def main():
    """Main test function"""
    print("NVBOT3 SIGNAL INTEGRATION TEST")
    print("=" * 50)
    
    # Run integration test
    integration_ok = test_integration()
    
    # Run signal generator test
    generator_ok = test_signal_generator()
    
    # Final results
    print("\n" + "=" * 50)
    print("FINAL TEST RESULTS")
    print("=" * 50)
    
    if integration_ok and generator_ok:
        print("SUCCESS: All components working correctly!")
        print("\nThe signal generator can now track signals to the dashboard.")
        print("\nTo use the system:")
        print("1. Start web dashboard: python scripts/start_dashboard.py")
        print("2. In another terminal, run: python scripts/signal_generator.py --mode scan")
        print("3. Check signals in the web dashboard")
        return 0
    else:
        print("FAILED: Some components need attention")
        if not integration_ok:
            print("- Integration bridge needs fixing")
        if not generator_ok:
            print("- Signal generator needs fixing")
        return 1

if __name__ == "__main__":
    exit(main())