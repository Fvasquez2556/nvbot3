#!/usr/bin/env python3
"""
Script to test FeatureCalculator and see what feature names are currently generated.
Creates sample OHLCV data and runs the calculate_features method.
"""

import pandas as pd
import numpy as np
import sys
import os

# Add src to path
sys.path.append('src')

from data.feature_calculator import FeatureCalculator

def create_sample_ohlcv_data(n_rows=300):
    """Create sample OHLCV data for testing."""
    np.random.seed(42)  # For reproducible results
    
    # Generate sample price data with some realistic patterns
    base_price = 50000  # Starting price
    returns = np.random.normal(0.001, 0.02, n_rows)  # Daily returns
    prices = [base_price]
    
    for ret in returns[1:]:
        new_price = prices[-1] * (1 + ret)
        prices.append(new_price)
    
    prices = np.array(prices)
    
    # Create OHLCV data
    data = []
    for i in range(n_rows):
        price = prices[i]
        
        # Create realistic OHLC based on price
        high = price * (1 + np.random.uniform(0, 0.03))
        low = price * (1 - np.random.uniform(0, 0.03))
        open_price = price * (1 + np.random.uniform(-0.01, 0.01))
        close_price = prices[i]  # Use the actual price as close
        volume = np.random.uniform(1000000, 10000000)
        
        data.append({
            'timestamp': pd.Timestamp('2024-01-01') + pd.Timedelta(hours=i),
            'open': open_price,
            'high': high,
            'low': low,
            'close': close_price,
            'volume': volume
        })
    
    df = pd.DataFrame(data)
    df.set_index('timestamp', inplace=True)
    return df

def main():
    """Main function to test feature calculation."""
    print("Testing FeatureCalculator with sample data...")
    
    # Create sample data
    print("Creating sample OHLCV data...")
    df = create_sample_ohlcv_data(300)
    print(f"Sample data created: {len(df)} rows")
    print(f"Date range: {df.index.min()} to {df.index.max()}")
    print(f"Price range: {df['close'].min():.2f} to {df['close'].max():.2f}")
    print()
    
    # Create feature calculator (but we'll use it without file I/O)
    calculator = FeatureCalculator()
    
    # Test each feature category separately to isolate issues
    print("Testing feature calculations...")
    
    try:
        # Test momentum features
        print("1. Testing momentum features...")
        df_momentum = calculator.calculate_momentum_features(df.copy())
        momentum_features = [col for col in df_momentum.columns if col not in df.columns]
        print(f"   Added {len(momentum_features)} momentum features")
        
        # Test rebound features
        print("2. Testing rebound features...")
        df_rebound = calculator.calculate_rebound_features(df_momentum.copy())
        rebound_features = [col for col in df_rebound.columns if col not in df_momentum.columns]
        print(f"   Added {len(rebound_features)} rebound features")
        
        # Test regime features
        print("3. Testing regime features...")
        df_regime = calculator.calculate_regime_features(df_rebound.copy())
        regime_features = [col for col in df_regime.columns if col not in df_rebound.columns]
        print(f"   Added {len(regime_features)} regime features")
        
        # Test additional features
        print("4. Testing additional features...")
        df_additional = calculator.calculate_additional_features(df_regime.copy())
        additional_features = [col for col in df_additional.columns if col not in df_regime.columns]
        print(f"   Added {len(additional_features)} additional features")
        
        print()
        print("FEATURE SUMMARY:")
        print("=" * 80)
        
        all_features = [col for col in df_additional.columns if col not in df.columns]
        print(f"Total features generated: {len(all_features)}")
        print()
        
        # Categorize and display features
        print("MOMENTUM FEATURES:")
        for feat in momentum_features[:10]:
            print(f"  - {feat}")
        if len(momentum_features) > 10:
            print(f"  ... and {len(momentum_features) - 10} more")
        print()
        
        print("REBOUND FEATURES:")
        for feat in rebound_features[:10]:
            print(f"  - {feat}")
        if len(rebound_features) > 10:
            print(f"  ... and {len(rebound_features) - 10} more")
        print()
        
        print("REGIME FEATURES:")
        for feat in regime_features[:10]:
            print(f"  - {feat}")
        if len(regime_features) > 10:
            print(f"  ... and {len(regime_features) - 10} more")
        print()
        
        print("ADDITIONAL FEATURES:")
        for feat in additional_features[:10]:
            print(f"  - {feat}")
        if len(additional_features) > 10:
            print(f"  ... and {len(additional_features) - 10} more")
        print()
        
        print("ALL FEATURES (alphabetical):")
        print("=" * 50)
        for i, feat in enumerate(sorted(all_features)):
            print(f"{i+1:3d}. {feat}")
        
        # Compare with training data expectations
        print()
        print("COMPARISON WITH TRAINING DATA:")
        print("=" * 50)
        expected_count = 146  # From training data analysis
        actual_count = len(all_features)
        print(f"Expected features in training data: {expected_count}")
        print(f"Generated features by calculator: {actual_count}")
        print(f"Difference: {actual_count - expected_count}")
        
        if actual_count != expected_count:
            print("WARNING: Feature count mismatch detected!")
            if actual_count < expected_count:
                print("The FeatureCalculator generates FEWER features than expected.")
            else:
                print("The FeatureCalculator generates MORE features than expected.")
        else:
            print("SUCCESS: Feature count matches training data!")
        
        print()
        print("DATA QUALITY CHECK:")
        print("=" * 30)
        print(f"Rows with NaN values: {df_additional.isnull().any(axis=1).sum()}")
        print(f"Total NaN values: {df_additional.isnull().sum().sum()}")
        print(f"Features with NaN: {df_additional.isnull().any().sum()}")
        
        # Check which features have NaN values
        features_with_nan = df_additional.columns[df_additional.isnull().any()].tolist()
        if features_with_nan:
            print("\nFeatures with NaN values:")
            for feat in features_with_nan[:10]:
                nan_count = df_additional[feat].isnull().sum()
                print(f"  - {feat}: {nan_count} NaN values")
            if len(features_with_nan) > 10:
                print(f"  ... and {len(features_with_nan) - 10} more")
        
    except Exception as e:
        print(f"ERROR during feature calculation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()