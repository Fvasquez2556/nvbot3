#!/usr/bin/env python3
"""
Comprehensive analysis of all model metadata files to understand feature count patterns.
"""

import pickle
import os
import glob
from collections import defaultdict

def extract_model_info(filepath):
    """Extract key information from a model metadata file"""
    try:
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        if not isinstance(data, dict):
            return None
            
        return {
            'symbol': data.get('symbol', 'Unknown'),
            'timeframe': data.get('timeframe', 'Unknown'),
            'model_type': data.get('model_type', 'Unknown'),
            'success': data.get('success', False),
            'feature_count': len(data.get('selected_features', [])),
            'features': data.get('selected_features', []),
            'model_class': str(type(data.get('best_model', None))),
            'n_features_expected': getattr(data.get('best_model', None), 'n_features_in_', None),
            'filepath': filepath
        }
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return None

def main():
    """Analyze all model metadata files"""
    
    # Find all metrics files
    pattern = "E:\\nvbot3\\data\\models\\*_metrics.pkl"
    metrics_files = glob.glob(pattern)
    
    print(f"Found {len(metrics_files)} model metadata files\n")
    
    # Extract information from all files
    models_info = []
    for filepath in metrics_files:
        info = extract_model_info(filepath)
        if info:
            models_info.append(info)
    
    # Group by different criteria
    by_feature_count = defaultdict(list)
    by_model_type = defaultdict(list)
    by_symbol_timeframe = defaultdict(list)
    
    for info in models_info:
        key = f"{info['symbol']}_{info['timeframe']}_{info['model_type']}"
        by_feature_count[info['feature_count']].append(key)
        by_model_type[info['model_type']].append((key, info['feature_count']))
        by_symbol_timeframe[f"{info['symbol']}_{info['timeframe']}"].append((info['model_type'], info['feature_count']))
    
    print("="*60)
    print("FEATURE COUNT DISTRIBUTION")
    print("="*60)
    for feature_count in sorted(by_feature_count.keys()):
        models = by_feature_count[feature_count]
        print(f"\n{feature_count} features ({len(models)} models):")
        for model in sorted(models):
            print(f"  - {model}")
    
    print("\n" + "="*60)
    print("BY MODEL TYPE")
    print("="*60)
    for model_type in sorted(by_model_type.keys()):
        models = by_model_type[model_type]
        print(f"\n{model_type.upper()} models:")
        feature_counts = defaultdict(int)
        for model_key, feature_count in models:
            feature_counts[feature_count] += 1
            print(f"  {model_key}: {feature_count} features")
        
        print(f"  Summary: {dict(feature_counts)}")
    
    print("\n" + "="*60)
    print("FEATURE COUNT VARIATIONS BY SYMBOL/TIMEFRAME")
    print("="*60)
    inconsistencies = []
    for combo, models in by_symbol_timeframe.items():
        feature_counts = set([count for _, count in models])
        if len(feature_counts) > 1:
            inconsistencies.append((combo, models))
            print(f"\nINCONSISTENCY - {combo}:")
            for model_type, feature_count in models:
                print(f"  {model_type}: {feature_count} features")
    
    if not inconsistencies:
        print("No feature count inconsistencies found within symbol/timeframe combinations")
    
    # Look for specific feature differences
    print("\n" + "="*60)
    print("DETAILED FEATURE COMPARISON")
    print("="*60)
    
    # Compare features between different counts
    features_by_count = defaultdict(set)
    sample_models = {}
    
    for info in models_info:
        count = info['feature_count']
        features_by_count[count].update(info['features'])
        if count not in sample_models:
            sample_models[count] = info
    
    counts = sorted(features_by_count.keys())
    if len(counts) > 1:
        print(f"\nComparing feature sets:")
        for i, count1 in enumerate(counts):
            for count2 in counts[i+1:]:
                features1 = features_by_count[count1]
                features2 = features_by_count[count2]
                
                only_in_1 = features1 - features2
                only_in_2 = features2 - features1
                common = features1 & features2
                
                print(f"\n{count1} vs {count2} features:")
                print(f"  Common features: {len(common)}")
                print(f"  Only in {count1}-feature models: {sorted(only_in_1) if only_in_1 else 'None'}")
                print(f"  Only in {count2}-feature models: {sorted(only_in_2) if only_in_2 else 'None'}")
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    total_models = len(models_info)
    successful_models = len([m for m in models_info if m['success']])
    
    print(f"Total models analyzed: {total_models}")
    print(f"Successful models: {successful_models}")
    print(f"Feature count range: {min(by_feature_count.keys())} - {max(by_feature_count.keys())}")
    
    feature_count_distribution = {count: len(models) for count, models in by_feature_count.items()}
    print(f"Feature count distribution: {feature_count_distribution}")
    
    print("\nMost common feature count:", max(feature_count_distribution.items(), key=lambda x: x[1]))
    
    return models_info

if __name__ == "__main__":
    models_info = main()