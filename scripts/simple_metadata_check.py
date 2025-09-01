#!/usr/bin/env python3
"""
Simple script to examine key metadata from model files, focusing on feature information.
"""

import pickle
import os
from datetime import datetime

def examine_key_info(filepath):
    """Extract key information from a metrics file"""
    print(f"\n{'='*50}")
    print(f"FILE: {os.path.basename(filepath)}")
    print(f"{'='*50}")
    
    try:
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        if isinstance(data, dict):
            # Key information we're looking for
            symbol = data.get('symbol', 'Unknown')
            timeframe = data.get('timeframe', 'Unknown')
            model_type = data.get('model_type', 'Unknown')
            success = data.get('success', 'Unknown')
            
            print(f"Symbol: {symbol}")
            print(f"Timeframe: {timeframe}")
            print(f"Model Type: {model_type}")
            print(f"Training Success: {success}")
            
            # Feature information
            selected_features = data.get('selected_features', [])
            if selected_features:
                print(f"Number of Features: {len(selected_features)}")
                print(f"Feature Names: {selected_features}")
            else:
                print("No selected_features found")
                
            # Validation results summary
            validation_results = data.get('validation_results', [])
            if validation_results:
                print(f"Number of Validation Folds: {len(validation_results)}")
                if len(validation_results) > 0:
                    first_result = validation_results[0]
                    if isinstance(first_result, dict):
                        print(f"Sample Sizes: Train={first_result.get('n_train_samples', 'N/A')}, Test={first_result.get('n_test_samples', 'N/A')}")
            
            # Summary information
            summary = data.get('summary', {})
            if summary:
                print(f"Average Test Accuracy: {summary.get('avg_test_accuracy', 'N/A')}")
                print(f"Overfitting Detected: {summary.get('is_overfitting', 'N/A')}")
            
            # Look for any timestamp/date related info
            for key, value in data.items():
                if any(word in key.lower() for word in ['time', 'date', 'created', 'timestamp']):
                    print(f"{key}: {value}")
                    
            # Check if model object has any useful attributes
            best_model = data.get('best_model', None)
            if best_model:
                print(f"Model Type: {type(best_model)}")
                # Try to get basic model info without triggering the error
                if hasattr(best_model, 'n_features_in_'):
                    print(f"Model expects {best_model.n_features_in_} features")
                elif hasattr(best_model, 'feature_importances_'):
                    print(f"Feature importances available: {len(best_model.feature_importances_)} features")
                    
        else:
            print(f"Data is not a dictionary: {type(data)}")
            
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return None
        
    return data

def main():
    """Check key files for feature information"""
    
    # Target files to examine
    files_to_check = [
        "E:\\nvbot3\\data\\models\\BTCUSDT_5m_momentum_metrics.pkl",
        "E:\\nvbot3\\data\\models\\ETHUSDT_1h_momentum_metrics.pkl", 
        "E:\\nvbot3\\data\\models\\ADAUSDT_5m_rebound_metrics.pkl",
        "E:\\nvbot3\\data\\models\\ALL_SYMBOLS_1h_momentum_metrics.pkl"
    ]
    
    print("EXAMINING MODEL METADATA - FOCUS ON FEATURES")
    print("Looking for feature count discrepancies...")
    
    feature_counts = {}
    
    for filepath in files_to_check:
        if os.path.exists(filepath):
            data = examine_key_info(filepath)
            if data and isinstance(data, dict):
                features = data.get('selected_features', [])
                if features:
                    key = f"{data.get('symbol', 'UNK')}_{data.get('timeframe', 'UNK')}_{data.get('model_type', 'UNK')}"
                    feature_counts[key] = len(features)
        else:
            print(f"\nFile not found: {filepath}")
    
    print(f"\n{'='*60}")
    print("FEATURE COUNT SUMMARY")
    print(f"{'='*60}")
    for key, count in feature_counts.items():
        print(f"{key}: {count} features")
        
    if feature_counts:
        unique_counts = set(feature_counts.values())
        if len(unique_counts) > 1:
            print(f"\nWARNING: Different feature counts detected: {sorted(unique_counts)}")
        else:
            print(f"\nAll models have the same feature count: {list(unique_counts)[0]}")

if __name__ == "__main__":
    main()