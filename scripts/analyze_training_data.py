#!/usr/bin/env python3
"""
Script to analyze training data structure in data/processed/ directory.
Examines CSV files ending with '_with_targets.csv' to understand the data format.
"""

import os
import pandas as pd
import glob
from typing import Dict, List, Tuple

def analyze_csv_header(file_path: str) -> Dict:
    """Analyze CSV file header without loading the full dataset."""
    info = {
        'file_name': os.path.basename(file_path),
        'total_columns': 0,
        'feature_columns': [],
        'target_columns': [],
        'non_feature_columns': [],
        'first_few_features': [],
        'error': None
    }
    
    try:
        # Read only the header
        df = pd.read_csv(file_path, nrows=0)
        columns = list(df.columns)
        
        info['total_columns'] = len(columns)
        
        # Categorize columns
        target_columns = [col for col in columns if col.endswith('_target')]
        non_feature_columns = ['timestamp', 'datetime', 'symbol', 'timeframe', 'open', 'high', 'low', 'close', 'volume']
        
        # Identify non-feature columns that exist
        existing_non_features = [col for col in non_feature_columns if col in columns]
        
        # Feature columns are everything else that's not a target or basic OHLCV/metadata
        feature_columns = [col for col in columns 
                          if not col.endswith('_target') 
                          and col not in existing_non_features]
        
        info['feature_columns'] = feature_columns
        info['target_columns'] = target_columns
        info['non_feature_columns'] = existing_non_features
        info['first_few_features'] = feature_columns[:10]  # First 10 feature names
        
    except Exception as e:
        info['error'] = str(e)
    
    return info

def main():
    """Main function to analyze training data structure."""
    processed_dir = 'data/processed/'
    
    if not os.path.exists(processed_dir):
        print(f"ERROR: Processed data directory not found: {processed_dir}")
        return
    
    # Find all *_with_targets.csv files
    pattern = os.path.join(processed_dir, '*_with_targets.csv')
    target_files = glob.glob(pattern)
    
    if not target_files:
        print(f"ERROR: No '*_with_targets.csv' files found in {processed_dir}")
        return
    
    print(f"Found {len(target_files)} training data files to analyze...\n")
    
    # Analyze a representative sample (first 5 files from different symbols/timeframes)
    sample_files = []
    seen_patterns = set()
    
    for file_path in sorted(target_files):
        file_name = os.path.basename(file_path)
        # Extract symbol and timeframe pattern
        parts = file_name.replace('_with_targets.csv', '').split('_')
        if len(parts) >= 2:
            pattern = f"{parts[-1]}"  # Just the timeframe
            if pattern not in seen_patterns or len(sample_files) < 5:
                sample_files.append(file_path)
                seen_patterns.add(pattern)
        
        if len(sample_files) >= 10:  # Analyze first 10 for variety
            break
    
    results = []
    
    print("DETAILED ANALYSIS of Representative Files:")
    print("=" * 80)
    
    for file_path in sample_files:
        print(f"\nAnalyzing: {os.path.basename(file_path)}")
        result = analyze_csv_header(file_path)
        results.append(result)
        
        if result['error']:
            print(f"  ERROR: {result['error']}")
            continue
        
        print(f"  Total Columns: {result['total_columns']}")
        print(f"  Feature Columns: {len(result['feature_columns'])}")
        print(f"  Target Columns: {len(result['target_columns'])} - {result['target_columns']}")
        print(f"  Non-Feature Columns: {len(result['non_feature_columns'])} - {result['non_feature_columns']}")
        print(f"  First 10 Features: {result['first_few_features']}")
        
        if len(result['feature_columns']) > 10:
            print(f"  ... and {len(result['feature_columns']) - 10} more features")
    
    # Quick analysis of all files for pattern consistency
    print(f"\n\nQUICK ANALYSIS of All {len(target_files)} Files:")
    print("=" * 80)
    
    feature_counts = {}
    target_patterns = {}
    feature_name_samples = {}
    
    for i, file_path in enumerate(target_files):
        if i % 50 == 0:  # Progress indicator
            print(f"Processing file {i+1}/{len(target_files)}...")
        
        result = analyze_csv_header(file_path)
        
        if not result['error']:
            # Count feature columns
            feature_count = len(result['feature_columns'])
            if feature_count not in feature_counts:
                feature_counts[feature_count] = []
            feature_counts[feature_count].append(result['file_name'])
            
            # Track target patterns
            target_pattern = tuple(sorted(result['target_columns']))
            if target_pattern not in target_patterns:
                target_patterns[target_pattern] = []
            target_patterns[target_pattern].append(result['file_name'])
            
            # Sample feature names
            if feature_count not in feature_name_samples:
                feature_name_samples[feature_count] = result['first_few_features']
    
    # Summary statistics
    print(f"\nSUMMARY:")
    print(f"Total files analyzed: {len(target_files)}")
    print(f"Files with errors: {sum(1 for r in results if r.get('error'))}")
    
    print(f"\nFeature Count Distribution:")
    for count in sorted(feature_counts.keys()):
        files = feature_counts[count]
        print(f"  {count} features: {len(files)} files")
        if len(files) <= 5:
            print(f"    Files: {files}")
        else:
            print(f"    Files: {files[:3]} ... and {len(files)-3} more")
    
    print(f"\nTarget Column Patterns:")
    for i, (pattern, files) in enumerate(target_patterns.items()):
        print(f"  Pattern {i+1}: {list(pattern)} ({len(files)} files)")
        if len(files) <= 3:
            print(f"    Files: {files}")
    
    print(f"\nFeature Name Patterns (by feature count):")
    for count in sorted(feature_name_samples.keys())[:5]:  # Show top 5 patterns
        print(f"  {count} features - Sample names: {feature_name_samples[count]}")
    
    # Check for consistency across symbol/timeframe combinations
    print(f"\nFEATURE CONSISTENCY CHECK:")
    if len(set(feature_counts.keys())) == 1:
        print("  SUCCESS: All files have the same number of features")
    else:
        print("  WARNING: Files have different numbers of features")
        print(f"    Feature counts found: {sorted(feature_counts.keys())}")
    
    if len(target_patterns) == 1:
        print("  SUCCESS: All files have the same target column pattern")
    else:
        print("  WARNING: Files have different target column patterns")

if __name__ == "__main__":
    main()