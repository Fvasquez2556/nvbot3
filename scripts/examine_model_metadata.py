#!/usr/bin/env python3
"""
Script to examine model metadata files and understand what information is stored
about trained models, particularly around feature counts and configurations.
"""

import pickle
import os
from datetime import datetime
from pathlib import Path

def examine_metrics_file(filepath):
    """Examine a single metrics file and return its contents"""
    print(f"\n{'='*60}")
    print(f"EXAMINING: {os.path.basename(filepath)}")
    print(f"{'='*60}")
    
    try:
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        print(f"Data type: {type(data)}")
        
        if isinstance(data, dict):
            print(f"\nKeys in metadata: {list(data.keys())}")
            
            for key, value in data.items():
                print(f"\n{key}:")
                if isinstance(value, (list, tuple)):
                    print(f"  Type: {type(value)} with {len(value)} items")
                    if len(value) > 0:
                        print(f"  First few items: {value[:5] if len(value) > 5 else value}")
                        if key.lower().find('feature') != -1:
                            print(f"  All features: {value}")
                elif isinstance(value, dict):
                    print(f"  Type: dict with keys: {list(value.keys())}")
                    for subkey, subvalue in value.items():
                        if isinstance(subvalue, (int, float, str)):
                            print(f"    {subkey}: {subvalue}")
                        else:
                            print(f"    {subkey}: {type(subvalue)} - {str(subvalue)[:100]}")
                elif isinstance(value, (int, float, str, bool)):
                    print(f"  {value}")
                elif hasattr(value, '__dict__'):
                    print(f"  Type: {type(value)}")
                    try:
                        attrs = vars(value)
                        for attr_name, attr_value in attrs.items():
                            if not attr_name.startswith('_'):
                                print(f"    {attr_name}: {attr_value}")
                    except:
                        print(f"  Could not inspect object attributes")
                else:
                    print(f"  Type: {type(value)} - {str(value)[:100]}")
        
        elif isinstance(data, (list, tuple)):
            print(f"List/tuple with {len(data)} items")
            for i, item in enumerate(data[:3]):  # Show first 3 items
                print(f"  Item {i}: {type(item)} - {str(item)[:100]}")
        
        else:
            print(f"Data content: {str(data)[:200]}")
            
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        
    return data

def main():
    """Examine multiple metadata files to understand the structure"""
    
    # Files to examine - select a variety
    files_to_examine = [
        "E:\\nvbot3\\data\\models\\BTCUSDT_5m_momentum_metrics.pkl",
        "E:\\nvbot3\\data\\models\\ETHUSDT_1h_momentum_metrics.pkl", 
        "E:\\nvbot3\\data\\models\\ALL_SYMBOLS_1h_momentum_metrics.pkl",
        "E:\\nvbot3\\data\\models\\ADAUSDT_5m_rebound_metrics.pkl"
    ]
    
    all_data = {}
    
    print("EXAMINING MODEL METADATA FILES")
    print("Looking for feature count, feature names, timestamps, and configuration info")
    
    for filepath in files_to_examine:
        if os.path.exists(filepath):
            data = examine_metrics_file(filepath)
            all_data[os.path.basename(filepath)] = data
        else:
            print(f"\nFile not found: {filepath}")
    
    # Summary analysis
    print(f"\n{'='*60}")
    print("SUMMARY ANALYSIS")
    print(f"{'='*60}")
    
    for filename, data in all_data.items():
        print(f"\n{filename}:")
        if isinstance(data, dict):
            # Look for feature-related information
            feature_keys = [k for k in data.keys() if 'feature' in k.lower()]
            if feature_keys:
                print(f"  Feature-related keys: {feature_keys}")
                for fkey in feature_keys:
                    fdata = data[fkey]
                    if isinstance(fdata, (list, tuple)):
                        print(f"    {fkey}: {len(fdata)} items")
                    else:
                        print(f"    {fkey}: {fdata}")
            
            # Look for timestamp/date information
            time_keys = [k for k in data.keys() if any(word in k.lower() for word in ['time', 'date', 'created', 'trained'])]
            if time_keys:
                print(f"  Time-related keys: {time_keys}")
                for tkey in time_keys:
                    print(f"    {tkey}: {data[tkey]}")
            
            # Look for model configuration
            config_keys = [k for k in data.keys() if any(word in k.lower() for word in ['config', 'param', 'model', 'n_'])]
            if config_keys:
                print(f"  Configuration keys: {config_keys}")
                for ckey in config_keys:
                    cdata = data[ckey]
                    if isinstance(cdata, dict):
                        print(f"    {ckey}: {list(cdata.keys())}")
                    else:
                        print(f"    {ckey}: {cdata}")

if __name__ == "__main__":
    main()