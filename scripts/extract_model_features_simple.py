#!/usr/bin/env python3
"""
Simple and robust model feature extraction script.
"""

import pickle
import json
import os
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')

def extract_features():
    """Extract features from all model metrics files."""
    
    model_mapping = {"models": {}}
    
    models_dir = Path("data/models")
    metric_files = list(models_dir.glob("*_metrics.pkl"))
    
    print(f"Found {len(metric_files)} metric files")
    
    successful = 0
    failed = 0
    
    for metric_file in metric_files:
        try:
            # Load file
            with open(metric_file, 'rb') as f:
                data = pickle.load(f)
            
            if not isinstance(data, dict):
                print(f"SKIP {metric_file.name}: Contains {type(data)}, not dict")
                failed += 1
                continue
            
            # Get features
            features = data.get('selected_features', [])
            if not features:
                print(f"SKIP {metric_file.name}: No selected_features found")
                failed += 1
                continue
            
            # Parse filename
            name = metric_file.stem.replace('_metrics', '')
            
            # Store
            model_mapping["models"][name] = {
                "filename": metric_file.name,
                "feature_count": len(features),
                "selected_features": features,
                "model_type": data.get('model_type', 'unknown'),
                "symbol": data.get('symbol', 'unknown'),
                "timeframe": data.get('timeframe', 'unknown')
            }
            
            print(f"SUCCESS {name}: {len(features)} features")
            successful += 1
            
        except Exception as e:
            print(f"ERROR {metric_file.name}: {e}")
            failed += 1
    
    print(f"\nResults: {successful} successful, {failed} failed")
    
    # Save to JSON
    with open("model_feature_mapping.json", 'w') as f:
        json.dump(model_mapping, f, indent=2)
    
    print(f"Saved to model_feature_mapping.json")
    
    return model_mapping

if __name__ == "__main__":
    mapping = extract_features()
    
    # Show summary
    print(f"\nSUMMARY:")
    for name, info in mapping["models"].items():
        print(f"  {name}: {info['feature_count']} features")