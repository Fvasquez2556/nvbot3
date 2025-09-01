"""
Feature Selector for Model Compatibility

This module provides feature selection and compatibility layer between
FeatureCalculator output (141 features) and trained models (25-50 features).

Handles:
- Loading model feature requirements from model_feature_mapping.json
- Filtering FeatureCalculator output to match model expectations
- Validating feature compatibility before prediction
- Providing detailed error messages for missing features

Usage:
    selector = FeatureSelector()
    features_for_momentum = selector.select_features(all_features, "momentum", "BTCUSDT", "5m")
"""

import json
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple
import warnings

# Import error handling
from .feature_compatibility_errors import (
    FeatureMismatchError, 
    ModelLoadError, 
    MappingFileError,
    FeatureCompatibilityDiagnostics,
    GracefulDegradationManager,
    log_feature_compatibility_error
)

# Setup logging
logger = logging.getLogger(__name__)

# Maintain backward compatibility
class FeatureSelectorError(Exception):
    """Custom exception for FeatureSelector errors."""
    pass

class FeatureSelector:
    """
    Feature compatibility layer for trained models.
    
    Loads model feature requirements and filters FeatureCalculator output
    to match what each trained model expects.
    """
    
    def __init__(self, mapping_file: str = "model_feature_mapping.json"):
        """
        Initialize FeatureSelector.
        
        Args:
            mapping_file: Path to the model feature mapping JSON file
        """
        self.mapping_file = Path(mapping_file)
        self.model_mapping = {}
        self.diagnostics = FeatureCompatibilityDiagnostics(self)
        self.degradation_manager = GracefulDegradationManager()
        self.load_mapping()
        
    def load_mapping(self):
        """Load model feature mapping from JSON file."""
        
        if not self.mapping_file.exists():
            raise MappingFileError(str(self.mapping_file), "NOT_FOUND")
        
        try:
            with open(self.mapping_file, 'r') as f:
                self.model_mapping = json.load(f)
            
            if 'models' not in self.model_mapping:
                raise MappingFileError(str(self.mapping_file), "INVALID_FORMAT", "missing 'models' key")
            
            num_models = len(self.model_mapping['models'])
            logger.info(f"Loaded feature mapping for {num_models} models")
            
        except json.JSONDecodeError as e:
            raise MappingFileError(str(self.mapping_file), "INVALID_FORMAT", f"JSON decode error: {e}")
        except Exception as e:
            raise MappingFileError(str(self.mapping_file), "UNKNOWN", str(e))
    
    def get_model_key(self, model_type: str, symbol: str, timeframe: str) -> str:
        """
        Generate model key from components.
        
        Args:
            model_type: Type of model (momentum, rebound, regime, momentum_advanced)
            symbol: Trading symbol (BTCUSDT, ETHUSDT, etc. or ALL_SYMBOLS)
            timeframe: Timeframe (5m, 15m, 1h, 4h, 1d)
            
        Returns:
            Model key string used for lookup
        """
        return f"{symbol}_{timeframe}_{model_type}"
    
    def get_available_models(self) -> List[str]:
        """Get list of all available model keys."""
        return list(self.model_mapping['models'].keys())
    
    def get_model_info(self, model_type: str, symbol: str, timeframe: str) -> Dict:
        """
        Get model information and requirements.
        
        Args:
            model_type: Type of model 
            symbol: Trading symbol
            timeframe: Timeframe
            
        Returns:
            Dictionary with model information
            
        Raises:
            FeatureSelectorError: If model not found
        """
        model_key = self.get_model_key(model_type, symbol, timeframe)
        
        if model_key not in self.model_mapping['models']:
            # Try with ALL_SYMBOLS fallback
            fallback_key = self.get_model_key(model_type, "ALL_SYMBOLS", timeframe)
            if fallback_key in self.model_mapping['models']:
                logger.info(f"Using ALL_SYMBOLS model for {model_key}")
                return self.model_mapping['models'][fallback_key]
            
            available = self.get_available_models()
            raise FeatureSelectorError(
                f"Model not found: {model_key}\n"
                f"Available models: {available[:10]}..." if len(available) > 10 else f"Available models: {available}"
            )
        
        return self.model_mapping['models'][model_key]
    
    def get_required_features(self, model_type: str, symbol: str, timeframe: str) -> List[str]:
        """
        Get required feature names for a specific model.
        
        Args:
            model_type: Type of model
            symbol: Trading symbol
            timeframe: Timeframe
            
        Returns:
            List of required feature names in correct order
        """
        model_info = self.get_model_info(model_type, symbol, timeframe)
        return model_info['selected_features']
    
    def validate_features(self, 
                         available_features: List[str], 
                         required_features: List[str]) -> Tuple[List[str], List[str]]:
        """
        Validate that required features are available.
        
        Args:
            available_features: List of available feature names
            required_features: List of required feature names
            
        Returns:
            Tuple of (found_features, missing_features)
        """
        available_set = set(available_features)
        required_set = set(required_features)
        
        found = [f for f in required_features if f in available_set]
        missing = list(required_set - available_set)
        
        return found, missing
    
    def select_features(self, 
                       features_df: pd.DataFrame, 
                       model_type: str, 
                       symbol: str, 
                       timeframe: str,
                       strict: bool = True) -> pd.DataFrame:
        """
        Select and reorder features for a specific model.
        
        Args:
            features_df: DataFrame with all features from FeatureCalculator
            model_type: Type of model (momentum, rebound, regime, momentum_advanced)
            symbol: Trading symbol (BTCUSDT, etc.)
            timeframe: Timeframe (5m, 15m, 1h, 4h, 1d)
            strict: If True, raise error for missing features. If False, warn and continue.
            
        Returns:
            DataFrame with only required features in correct order
            
        Raises:
            FeatureSelectorError: If required features are missing and strict=True
        """
        # Get required features for this model
        required_features = self.get_required_features(model_type, symbol, timeframe)
        available_features = list(features_df.columns)
        
        # Validate features
        found_features, missing_features = self.validate_features(available_features, required_features)
        
        if missing_features:
            model_key = f"{symbol}_{timeframe}_{model_type}"
            
            if strict:
                # Raise detailed error with diagnostics
                error = FeatureMismatchError(
                    model_key, missing_features, 
                    len(available_features), len(required_features)
                )
                log_feature_compatibility_error(error, logger)
                raise error
            else:
                # Log warning with diagnostic information
                diagnosis = self.diagnostics.diagnose_feature_mismatch(available_features, required_features)
                logger.warning(f"Feature mismatch for {model_key}:")
                logger.warning(f"  Missing: {len(missing_features)} features")
                logger.warning(f"  Compatibility rate: {diagnosis['summary']['compatibility_rate']:.1%}")
                
                # Use degradation strategy
                degradation_result = self.degradation_manager.handle_compatibility_failure(
                    FeatureMismatchError(model_key, missing_features, len(available_features), len(required_features)),
                    {"available_features": available_features, "required_features": required_features}
                )
                logger.info(f"  Degradation strategy: {degradation_result['strategy']}")
                
                # Use only found features
                required_features = found_features
        
        # Select and reorder features
        try:
            selected_df = features_df[required_features].copy()
            logger.debug(f"Selected {len(required_features)} features for {model_type} model")
            return selected_df
            
        except KeyError as e:
            # This shouldn't happen after validation, but just in case
            raise FeatureSelectorError(f"Feature selection failed: {e}")
    
    def batch_select_features(self, 
                             features_df: pd.DataFrame,
                             model_specs: List[Dict]) -> Dict[str, pd.DataFrame]:
        """
        Select features for multiple models at once.
        
        Args:
            features_df: DataFrame with all features from FeatureCalculator
            model_specs: List of dictionaries with keys: model_type, symbol, timeframe
            
        Returns:
            Dictionary mapping model keys to selected feature DataFrames
        """
        results = {}
        
        for spec in model_specs:
            model_key = self.get_model_key(spec['model_type'], spec['symbol'], spec['timeframe'])
            try:
                selected_features = self.select_features(
                    features_df, 
                    spec['model_type'], 
                    spec['symbol'], 
                    spec['timeframe']
                )
                results[model_key] = selected_features
                logger.debug(f"Successfully selected features for {model_key}")
                
            except FeatureSelectorError as e:
                logger.error(f"Failed to select features for {model_key}: {e}")
                results[model_key] = None
        
        return results
    
    def get_feature_statistics(self) -> Dict:
        """
        Get statistics about feature requirements across all models.
        
        Returns:
            Dictionary with feature statistics
        """
        stats = {
            'total_models': len(self.model_mapping['models']),
            'by_model_type': {},
            'feature_counts': {'min': float('inf'), 'max': 0, 'avg': 0},
            'common_features': set(),
            'unique_features': set()
        }
        
        all_features = set()
        feature_counts = []
        
        # Analyze by model type
        for model_key, model_info in self.model_mapping['models'].items():
            model_type = model_info['model_type']
            feature_count = model_info['feature_count']
            features = set(model_info['selected_features'])
            
            # Update by model type
            if model_type not in stats['by_model_type']:
                stats['by_model_type'][model_type] = {
                    'count': 0, 
                    'feature_counts': []
                }
            
            stats['by_model_type'][model_type]['count'] += 1
            stats['by_model_type'][model_type]['feature_counts'].append(feature_count)
            
            # Update overall stats
            all_features.update(features)
            feature_counts.append(feature_count)
            stats['feature_counts']['min'] = min(stats['feature_counts']['min'], feature_count)
            stats['feature_counts']['max'] = max(stats['feature_counts']['max'], feature_count)
        
        # Calculate averages
        stats['feature_counts']['avg'] = sum(feature_counts) / len(feature_counts)
        
        for model_type, type_stats in stats['by_model_type'].items():
            counts = type_stats['feature_counts']
            type_stats['avg_features'] = sum(counts) / len(counts)
        
        # Find common and unique features
        stats['total_unique_features'] = len(all_features)
        stats['unique_features'] = list(all_features)
        
        return stats
    
    def print_summary(self):
        """Print a summary of loaded models and feature requirements."""
        
        stats = self.get_feature_statistics()
        
        print("="*60)
        print("FEATURE SELECTOR SUMMARY")
        print("="*60)
        
        print(f"Total models loaded: {stats['total_models']}")
        print(f"Feature count range: {stats['feature_counts']['min']}-{stats['feature_counts']['max']} "
              f"(avg: {stats['feature_counts']['avg']:.1f})")
        print(f"Total unique features across all models: {stats['total_unique_features']}")
        
        print(f"\nBy model type:")
        for model_type, type_stats in stats['by_model_type'].items():
            print(f"  {model_type.upper()}: {type_stats['count']} models, "
                  f"avg {type_stats['avg_features']:.0f} features")
        
        # Show some model examples
        print(f"\nExample models:")
        for i, (model_key, model_info) in enumerate(self.model_mapping['models'].items()):
            if i >= 5:  # Show only first 5
                break
            print(f"  {model_key}: {model_info['feature_count']} features")

def main():
    """Test function for FeatureSelector."""
    
    try:
        selector = FeatureSelector()
        selector.print_summary()
        print("\nFeatureSelector loaded successfully!")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()