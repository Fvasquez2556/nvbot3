#!/usr/bin/env python3
"""
Script to analyze all trained models in data/models/ directory.
Extracts and displays expected feature names from each model.
"""

import os
import pickle
import pandas as pd
from typing import Dict, List, Any, Optional
import numpy as np

def analyze_sklearn_model(model: Any, model_name: str) -> Dict[str, Any]:
    """Analyze sklearn-based models for feature information."""
    info = {
        'model_type': type(model).__name__,
        'feature_names': None,
        'n_features': None,
        'feature_importances': None,
        'additional_info': {}
    }
    
    # Check for feature_names_in_ (sklearn >= 1.0)
    if hasattr(model, 'feature_names_in_'):
        info['feature_names'] = list(model.feature_names_in_)
        info['n_features'] = len(model.feature_names_in_)
    
    # Check for n_features_in_
    elif hasattr(model, 'n_features_in_'):
        info['n_features'] = model.n_features_in_
        info['feature_names'] = [f'feature_{i}' for i in range(model.n_features_in_)]
    
    # Check for feature importances
    if hasattr(model, 'feature_importances_'):
        info['feature_importances'] = list(model.feature_importances_)
        if info['n_features'] is None:
            info['n_features'] = len(model.feature_importances_)
    
    # Additional model-specific attributes
    if hasattr(model, 'get_params'):
        try:
            params = model.get_params()
            info['additional_info']['params'] = params
        except:
            pass
    
    return info

def analyze_xgboost_model(model: Any, model_name: str) -> Dict[str, Any]:
    """Analyze XGBoost models for feature information."""
    info = {
        'model_type': 'XGBoost',
        'feature_names': None,
        'n_features': None,
        'feature_importances': None,
        'additional_info': {}
    }
    
    # XGBoost feature names
    if hasattr(model, 'feature_names_in_'):
        info['feature_names'] = list(model.feature_names_in_)
        info['n_features'] = len(model.feature_names_in_)
    elif hasattr(model, 'get_booster'):
        try:
            booster = model.get_booster()
            feature_names = booster.feature_names
            if feature_names:
                info['feature_names'] = feature_names
                info['n_features'] = len(feature_names)
        except:
            pass
    
    # Feature importances
    if hasattr(model, 'feature_importances_'):
        info['feature_importances'] = list(model.feature_importances_)
        if info['n_features'] is None:
            info['n_features'] = len(model.feature_importances_)
    
    return info

def analyze_keras_model(model: Any, model_name: str) -> Dict[str, Any]:
    """Analyze Keras/TensorFlow models for feature information."""
    info = {
        'model_type': 'Keras/TensorFlow',
        'feature_names': None,
        'n_features': None,
        'feature_importances': None,
        'additional_info': {}
    }
    
    try:
        # Get input shape
        if hasattr(model, 'input_shape'):
            input_shape = model.input_shape
            if input_shape and len(input_shape) > 1:
                info['n_features'] = input_shape[1]  # Assuming batch dimension is first
                info['feature_names'] = [f'feature_{i}' for i in range(input_shape[1])]
        
        # Additional model info
        if hasattr(model, 'summary'):
            info['additional_info']['layers'] = len(model.layers) if hasattr(model, 'layers') else None
    except:
        pass
    
    return info

def analyze_custom_model(model: Any, model_name: str) -> Dict[str, Any]:
    """Analyze custom model structures (dictionaries, etc.)."""
    info = {
        'model_type': type(model).__name__,
        'feature_names': None,
        'n_features': None,
        'feature_importances': None,
        'additional_info': {}
    }
    
    if isinstance(model, dict):
        # Check for common feature-related keys
        feature_keys = ['features', 'feature_names', 'columns', 'X_columns']
        for key in feature_keys:
            if key in model:
                if isinstance(model[key], (list, np.ndarray)):
                    info['feature_names'] = list(model[key])
                    info['n_features'] = len(model[key])
                    break
        
        # Store dictionary keys for debugging
        info['additional_info']['dict_keys'] = list(model.keys())
    
    return info

def analyze_model_file(file_path: str) -> Dict[str, Any]:
    """Analyze a single model file."""
    model_name = os.path.basename(file_path)
    
    try:
        with open(file_path, 'rb') as f:
            model = pickle.load(f)
        
        # Determine model type and analyze accordingly
        model_type = type(model).__name__.lower()
        
        if any(sklearn_type in model_type for sklearn_type in ['forest', 'tree', 'regressor', 'classifier', 'svm', 'linear']):
            return analyze_sklearn_model(model, model_name)
        elif 'xgb' in model_type or 'gradient' in model_type:
            return analyze_xgboost_model(model, model_name)
        elif 'keras' in model_type or 'tensorflow' in model_type or 'sequential' in model_type:
            return analyze_keras_model(model, model_name)
        else:
            return analyze_custom_model(model, model_name)
            
    except Exception as e:
        return {
            'model_type': 'ERROR',
            'feature_names': None,
            'n_features': None,
            'feature_importances': None,
            'additional_info': {'error': str(e)}
        }

def main():
    """Main function to analyze all models."""
    models_dir = 'data/models/'
    
    if not os.path.exists(models_dir):
        print(f"ERROR: Models directory not found: {models_dir}")
        return
    
    # Get all .pkl files
    model_files = [f for f in os.listdir(models_dir) if f.endswith('.pkl')]
    
    if not model_files:
        print(f"ERROR: No .pkl files found in {models_dir}")
        return
    
    print(f"Found {len(model_files)} model files to analyze...\n")
    
    results = {}
    
    for model_file in sorted(model_files):
        file_path = os.path.join(models_dir, model_file)
        print(f"Analyzing: {model_file}")
        
        result = analyze_model_file(file_path)
        results[model_file] = result
        
        # Display results for this model
        print(f"  Model Type: {result['model_type']}")
        print(f"  Features Count: {result['n_features']}")
        
        if result['feature_names']:
            print(f"  Feature Names (first 10): {result['feature_names'][:10]}")
            if len(result['feature_names']) > 10:
                print(f"    ... and {len(result['feature_names']) - 10} more")
        else:
            print(f"  Feature Names: Not available")
        
        if result['feature_importances']:
            print(f"  Has Feature Importances: Yes ({len(result['feature_importances'])} features)")
        else:
            print(f"  Has Feature Importances: No")
        
        if result['additional_info']:
            if 'error' in result['additional_info']:
                print(f"  ERROR: {result['additional_info']['error']}")
            elif 'dict_keys' in result['additional_info']:
                print(f"  Dictionary Keys: {result['additional_info']['dict_keys']}")
        
        print()
    
    # Summary
    print("SUMMARY:")
    print(f"Total models analyzed: {len(results)}")
    
    models_with_features = sum(1 for r in results.values() if r['feature_names'])
    models_with_importances = sum(1 for r in results.values() if r['feature_importances'])
    models_with_errors = sum(1 for r in results.values() if r['additional_info'].get('error'))
    
    print(f"Models with feature names: {models_with_features}")
    print(f"Models with feature importances: {models_with_importances}")
    print(f"Models with errors: {models_with_errors}")
    
    # Check for consistency in feature names
    feature_sets = []
    for model_file, result in results.items():
        if result['feature_names']:
            feature_sets.append((model_file, set(result['feature_names'])))
    
    if len(feature_sets) > 1:
        print(f"\nFeature Consistency Check:")
        base_features = feature_sets[0][1]
        consistent = True
        for model_file, features in feature_sets[1:]:
            if features != base_features:
                consistent = False
                print(f"  WARNING: {model_file} has different features than {feature_sets[0][0]}")
                diff = features.symmetric_difference(base_features)
                if diff:
                    print(f"      Different features: {list(diff)[:5]}{'...' if len(diff) > 5 else ''}")
        
        if consistent:
            print(f"  SUCCESS: All models have consistent feature sets")

if __name__ == "__main__":
    main()