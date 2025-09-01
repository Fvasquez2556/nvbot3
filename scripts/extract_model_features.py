#!/usr/bin/env python3
"""
Extract feature lists from all trained models and create model_feature_mapping.json

This script examines all trained model files in data/models/ to extract:
- Expected feature names (in exact order)
- Feature count for each model
- Model metadata and specifications

Creates a comprehensive mapping file for the FeatureSelector class.
"""

import pickle
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Any
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def extract_model_features():
    """Extract feature lists from all trained models."""
    
    model_mapping = {
        "metadata": {
            "extraction_date": str(Path(__file__).stat().st_mtime),
            "total_models": 0,
            "model_types": [],
            "symbols": [],
            "timeframes": []
        },
        "models": {}
    }
    
    models_dir = Path("data/models")
    if not models_dir.exists():
        logger.error(f"Models directory not found: {models_dir}")
        return None
    
    # Get all metric files (contain feature lists)
    metric_files = list(models_dir.glob("*_metrics.pkl"))
    logger.info(f"Found {len(metric_files)} model metric files")
    
    for metric_file in metric_files:
        try:
            # Parse filename to get model info
            filename = metric_file.stem  # Remove .pkl
            filename_clean = filename.replace("_metrics", "")
            
            # Parse: SYMBOL_TIMEFRAME_MODELTYPE or ALL_SYMBOLS_TIMEFRAME_MODELTYPE
            parts = filename_clean.split("_")
            if len(parts) >= 3:
                if parts[0] == "ALL" and parts[1] == "SYMBOLS":
                    symbol = "ALL_SYMBOLS"
                    timeframe = parts[2]
                    model_type = "_".join(parts[3:]) if len(parts) > 3 else parts[2]
                else:
                    symbol = parts[0]
                    timeframe = parts[1]
                    model_type = "_".join(parts[2:]) if len(parts) > 2 else "unknown"
            else:
                logger.warning(f"Could not parse filename: {filename}")
                continue
            
            # Load the metrics file
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                with open(metric_file, 'rb') as f:
                    metrics = pickle.load(f)
            
            # Verify it's a dictionary
            if not isinstance(metrics, dict):
                logger.warning(f"Metrics file {metric_file} contains {type(metrics)}, expected dict")
                continue
            
            # Extract feature information
            selected_features = metrics.get('selected_features', [])
            feature_count = len(selected_features)
            
            # Get additional model info if available
            model_info = {
                "filename": str(metric_file.name),
                "symbol": symbol,
                "timeframe": timeframe,
                "model_type": model_type,
                "feature_count": feature_count,
                "selected_features": selected_features,
                "success": metrics.get('success', False)
            }
            
            # Try to get model-specific info
            if 'best_model' in metrics and hasattr(metrics['best_model'], 'n_features_in_'):
                model_info["n_features_in"] = metrics['best_model'].n_features_in_
            
            if 'validation_results' in metrics:
                val_results = metrics['validation_results']
                model_info["validation_score"] = val_results.get('test_score', 0.0)
            
            # Add to mapping
            model_key = f"{symbol}_{timeframe}_{model_type}"
            model_mapping["models"][model_key] = model_info
            
            # Update metadata
            model_mapping["metadata"]["total_models"] += 1
            if model_type not in model_mapping["metadata"]["model_types"]:
                model_mapping["metadata"]["model_types"].append(model_type)
            if symbol not in model_mapping["metadata"]["symbols"]:
                model_mapping["metadata"]["symbols"].append(symbol)
            if timeframe not in model_mapping["metadata"]["timeframes"]:
                model_mapping["metadata"]["timeframes"].append(timeframe)
            
            logger.info(f"Extracted {feature_count} features from {model_key}")
            
        except Exception as e:
            # Try to see what the actual content type is for debugging
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    with open(metric_file, 'rb') as f:
                        test_metrics = pickle.load(f)
                logger.error(f"Error processing {metric_file}: {e} (Content type: {type(test_metrics)})")
            except Exception as e2:
                logger.error(f"Error processing {metric_file}: {e} (Could not reload: {e2})")
            continue
    
    # Sort metadata lists for consistency
    model_mapping["metadata"]["model_types"].sort()
    model_mapping["metadata"]["symbols"].sort()
    model_mapping["metadata"]["timeframes"].sort()
    
    return model_mapping

def analyze_feature_patterns(model_mapping: Dict):
    """Analyze patterns in the extracted features."""
    
    print("\n" + "="*60)
    print("FEATURE PATTERN ANALYSIS")
    print("="*60)
    
    # Group by model type
    by_model_type = {}
    for model_key, model_info in model_mapping["models"].items():
        model_type = model_info["model_type"]
        if model_type not in by_model_type:
            by_model_type[model_type] = []
        by_model_type[model_type].append(model_info)
    
    # Analyze each model type
    for model_type, models in by_model_type.items():
        print(f"\n{model_type.upper()} Models:")
        print(f"   Count: {len(models)}")
        
        # Feature count statistics
        feature_counts = [m["feature_count"] for m in models]
        print(f"   Feature counts: {min(feature_counts)} - {max(feature_counts)} (avg: {sum(feature_counts)/len(feature_counts):.1f})")
        
        # Check feature consistency within model type
        if len(models) > 1:
            first_features = models[0]["selected_features"]
            all_same = all(m["selected_features"] == first_features for m in models)
            print(f"   Feature consistency: {'All identical' if all_same else 'Different features'}")
        
        # Show sample features
        if models[0]["selected_features"]:
            sample_features = models[0]["selected_features"][:10]
            print(f"   Sample features: {sample_features}")
    
    # Overall statistics
    print(f"\nOVERALL STATISTICS:")
    print(f"   Total models: {model_mapping['metadata']['total_models']}")
    print(f"   Model types: {model_mapping['metadata']['model_types']}")
    print(f"   Symbols: {model_mapping['metadata']['symbols']}")
    print(f"   Timeframes: {model_mapping['metadata']['timeframes']}")

def save_mapping(model_mapping: Dict, output_file: str = "model_feature_mapping.json"):
    """Save the model mapping to JSON file."""
    
    try:
        with open(output_file, 'w') as f:
            json.dump(model_mapping, f, indent=2)
        
        logger.info(f"Model feature mapping saved to: {output_file}")
        logger.info(f"   Total models: {model_mapping['metadata']['total_models']}")
        logger.info(f"   File size: {os.path.getsize(output_file) / 1024:.1f} KB")
        
    except Exception as e:
        logger.error(f"Error saving mapping file: {e}")
        return False
    
    return True

def main():
    """Main execution function."""
    
    print("Extracting feature lists from trained models...")
    
    # Extract features from all models
    model_mapping = extract_model_features()
    
    if model_mapping is None:
        print("❌ Failed to extract model features")
        return 1
    
    # Analyze patterns
    analyze_feature_patterns(model_mapping)
    
    # Save mapping file
    success = save_mapping(model_mapping)
    
    if success:
        print(f"\nSUCCESS: model_feature_mapping.json created successfully!")
        print(f"   Ready for FeatureSelector implementation")
        return 0
    else:
        print(f"\nFAILED: Could not save mapping file")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)