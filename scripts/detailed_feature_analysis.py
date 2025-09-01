#!/usr/bin/env python3
"""
Detailed analysis of FeatureCalculator methods and feature naming conventions.
This script will extract all feature names from each method and analyze patterns.
"""

import re
import ast
import sys
from pathlib import Path

# Add src to path
sys.path.append('src')

def extract_feature_names_from_method(method_code):
    """Extract all feature names created in a method."""
    features = []
    
    # Pattern to match feature assignments like: result_df['feature_name'] = ...
    feature_pattern = r"result_df\[['\"](.*?)['\"]\]\s*="
    matches = re.findall(feature_pattern, method_code)
    features.extend(matches)
    
    # Pattern to match formatted feature names like: f'roc_{period}'
    formatted_pattern = r"f['\"]([^'\"]*?\{[^}]+\}[^'\"]*?)['\"]"
    formatted_matches = re.findall(formatted_pattern, method_code)
    
    # Process formatted strings to extract patterns
    for match in formatted_matches:
        # Simple replacement of common variables
        pattern = match.replace('{period}', '_XX').replace('{timeperiod}', '_XX')
        features.append(f"PATTERN: {pattern}")
    
    return features

def analyze_feature_calculator_methods():
    """Analyze all feature calculation methods."""
    
    print("DETAILED FEATURECALCULATOR ANALYSIS")
    print("=" * 80)
    
    # Read the FeatureCalculator file
    feature_calc_path = Path("src/data/feature_calculator.py")
    
    if not feature_calc_path.exists():
        print("ERROR: FeatureCalculator file not found!")
        return
    
    with open(feature_calc_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Find all method definitions
    methods = {
        'momentum': 'calculate_momentum_features',
        'rebound': 'calculate_rebound_features', 
        'regime': 'calculate_regime_features',
        'additional': 'calculate_additional_features'
    }
    
    total_features = 0
    all_feature_patterns = []
    
    for category, method_name in methods.items():
        print(f"\n{category.upper()} FEATURES ({method_name}):")
        print("-" * 60)
        
        # Extract method code
        method_start = content.find(f"def {method_name}(")
        if method_start == -1:
            print(f"ERROR: Method {method_name} not found!")
            continue
        
        # Find the end of the method (next def or class)
        method_end = content.find("\n    def ", method_start + 1)
        if method_end == -1:
            method_end = content.find("\nclass ", method_start + 1)
        if method_end == -1:
            method_end = len(content)
        
        method_code = content[method_start:method_end]
        
        # Extract features from this method
        features = extract_feature_names_from_method(method_code)
        
        print(f"Features found in {method_name}:")
        for i, feature in enumerate(features, 1):
            print(f"  {i:2d}. {feature}")
            
        total_features += len([f for f in features if not f.startswith('PATTERN:')])
        all_feature_patterns.extend(features)
        
        # Look for loops that generate multiple features
        period_loops = re.findall(r'for period in \[(.*?)\]:', method_code)
        if period_loops:
            print(f"\nPeriod loops found:")
            for loop in period_loops:
                periods = loop.replace(' ', '').split(',')
                print(f"  - Periods: {periods}")
    
    print(f"\nTOTAL FEATURE ANALYSIS:")
    print("=" * 50)
    print(f"Direct feature assignments found: {total_features}")
    
    # Analyze patterns
    patterns = [f for f in all_feature_patterns if f.startswith('PATTERN:')]
    direct_features = [f for f in all_feature_patterns if not f.startswith('PATTERN:')]
    
    print(f"Pattern-based features: {len(patterns)}")
    print(f"Direct features: {len(direct_features)}")
    
    print(f"\nCOMMON FEATURE PATTERNS:")
    pattern_counts = {}
    for pattern in patterns:
        base = pattern.replace('PATTERN: ', '').split('_')[0]
        pattern_counts[base] = pattern_counts.get(base, 0) + 1
    
    for pattern, count in sorted(pattern_counts.items()):
        print(f"  - {pattern}_*: {count} variations")

def analyze_period_configurations():
    """Analyze the period configurations used in FeatureCalculator."""
    print(f"\nPERIOD CONFIGURATIONS:")
    print("-" * 30)
    
    # From the FeatureCalculator __init__ method
    periods = {
        'short': [5, 7, 10, 14],
        'medium': [20, 26, 30], 
        'long': [50, 100, 200]
    }
    
    print("Standard periods defined:")
    for category, period_list in periods.items():
        print(f"  {category}: {period_list}")
    
    # Calculate expected features for some common indicators
    print(f"\nESTIMATED FEATURE COUNTS by indicator:")
    indicators_with_periods = {
        'ROC': [5, 10, 14, 20],  # 4 features
        'ADX + DI': [14, 20, 30],  # 3 * 3 = 9 features  
        'RSI': [7, 14, 21, 30],  # 4 features
        'Bollinger Bands': [20, 30],  # 2 * 4 = 8 features (upper, middle, lower, width, position)
        'ATR': [14, 20, 30],  # 3 * 2 = 6 features (atr, ratio)
        'SMA': [10, 20, 50, 100, 200],  # 5 features + 5 slopes = 10
        'EMA': [10, 20, 50, 100, 200],  # 5 features + 5 slopes = 10
    }
    
    total_estimated = 0
    for indicator, periods_used in indicators_with_periods.items():
        count = len(periods_used)
        if 'Bollinger' in indicator:
            count *= 5  # upper, middle, lower, width, position
        elif 'ADX' in indicator:
            count *= 3  # adx, plus_di, minus_di
        elif 'ATR' in indicator:
            count *= 2  # atr, ratio
        elif 'SMA' in indicator or 'EMA' in indicator:
            count *= 2  # value + slope
            
        print(f"  {indicator}: {count} features")
        total_estimated += count
    
    print(f"\nTotal estimated from major indicators: {total_estimated}")

if __name__ == "__main__":
    analyze_feature_calculator_methods()
    analyze_period_configurations()