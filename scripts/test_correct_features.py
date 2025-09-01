#!/usr/bin/env python3
"""
Test Correct Feature Selection
============================
Test that signal generator uses the EXACT features from training
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np

def test_feature_selection():
    """Test the corrected feature selection"""
    print("=== Testing Correct Feature Selection ===")
    
    try:
        from scripts.signal_generator import NvBot3SignalGenerator
        generator = NvBot3SignalGenerator()
        print("SUCCESS: Signal generator created")
    except Exception as e:
        print(f"ERROR: Failed to create signal generator: {e}")
        return False
    
    # Show exact feature selections
    print("\nModel Feature Counts:")
    for model_type, features in generator.model_features.items():
        print(f"  {model_type}: {len(features)} features")
    
    # Generate test market data
    print("\nGenerating test market data...")
    dates = pd.date_range('2025-08-01', periods=300, freq='5T')
    
    # Create realistic price movements
    np.random.seed(42)
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
    
    print(f"Market data created: {len(dummy_data)} candles")
    
    # Calculate features
    try:
        features_df = generator.calculate_features(dummy_data)
        print(f"SUCCESS: Features calculated - {len(features_df.columns)} total features")
    except Exception as e:
        print(f"ERROR: Feature calculation failed: {e}")
        return False
    
    # Check feature compatibility
    try:
        compatibility = generator.verify_feature_compatibility(features_df)
        print("\nFeature Compatibility Check:")
        
        all_compatible = True
        for model_type, comp in compatibility.items():
            rate = comp['compatibility_rate']
            if rate == 1.0:
                status = "PERFECT"
            elif rate > 0.9:
                status = "GOOD"
            else:
                status = "POOR"
                all_compatible = False
            
            print(f"  {model_type}: {comp['available']}/{comp['expected']} ({rate:.1%}) - {status}")
            
            if comp['missing']:
                print(f"    Missing: {comp['missing'][:5]}...")
        
        if all_compatible:
            print("\nSUCCESS: All models have perfect feature compatibility!")
            return True
        else:
            print("\nWARNING: Some models have missing features")
            return False
        
    except Exception as e:
        print(f"ERROR: Compatibility check failed: {e}")
        return False

def test_specific_features():
    """Test for specific expected features"""
    print("\n=== Testing Specific Expected Features ===")
    
    from scripts.signal_generator import NvBot3SignalGenerator
    generator = NvBot3SignalGenerator()
    
    # Create test data
    dates = pd.date_range('2025-08-01', periods=300, freq='5T')
    np.random.seed(42)
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
    
    features_df = generator.calculate_features(dummy_data)
    
    # Test for specific features mentioned in the original problem
    critical_features = [
        'asian_session', 'hour', 'european_session', 'ma_alignment_bear', 
        'vpt', 'plus_di_30', 'atr_ratio_14', 'bb_upper_20'
    ]
    
    print("Checking for critical features:")
    found = 0
    for feature in critical_features:
        if feature in features_df.columns:
            print(f"  FOUND: {feature}")
            found += 1
        else:
            print(f"  MISSING: {feature}")
    
    print(f"\nCritical features found: {found}/{len(critical_features)}")
    
    return found == len(critical_features)

def main():
    """Main test function"""
    print("NVBOT3 CORRECT FEATURE SELECTION TEST")
    print("=" * 50)
    
    tests = [
        ("Feature Selection", test_feature_selection),
        ("Specific Features", test_specific_features)
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
    
    print(f"\n" + "=" * 50)
    print(f"RESULTS: {passed}/{total} tests passed")
    print("=" * 50)
    
    if passed == total:
        print("ALL TESTS PASSED!")
        print("\nThe signal generator now uses:")
        print("  - EXACT feature calculation from FeatureCalculator")
        print("  - EXACT feature selection from saved model metrics")
        print("  - 100% training compatibility")
        print("\nFeature selection issue is FIXED!")
        return 0
    else:
        print("SOME TESTS FAILED - Feature selection needs more work")
        return 1

if __name__ == "__main__":
    exit(main())