"""
Feature Compatibility Error Handling Module

This module provides comprehensive error handling for the feature compatibility pipeline.
It includes custom exceptions, error diagnostics, and solution suggestions for common issues.

Key Components:
- Custom exception classes with detailed error information
- Error diagnostics for feature mismatches
- Automated solution suggestions
- Feature pipeline health checks
- Graceful degradation strategies
"""

import logging
from typing import Dict, List, Optional, Tuple, Any, Set
from pathlib import Path
import json

logger = logging.getLogger(__name__)

class FeatureCompatibilityError(Exception):
    """Base exception for feature compatibility issues."""
    
    def __init__(self, message: str, error_code: Optional[str] = None, suggestions: Optional[List[str]] = None):
        self.message = message
        self.error_code = error_code or "UNKNOWN_ERROR"
        self.suggestions = suggestions or []
        super().__init__(self.message)
    
    def get_full_error_info(self) -> Dict[str, Any]:
        """Get complete error information for debugging."""
        return {
            "error_code": self.error_code,
            "message": self.message,
            "suggestions": self.suggestions,
            "exception_type": self.__class__.__name__
        }

class FeatureMismatchError(FeatureCompatibilityError):
    """Raised when required features are missing from FeatureCalculator output."""
    
    def __init__(self, model_key: str, missing_features: List[str], 
                 available_count: int, required_count: int):
        self.model_key = model_key
        self.missing_features = missing_features
        self.available_count = available_count
        self.required_count = required_count
        
        message = (
            f"Feature mismatch for model {model_key}:\n"
            f"Missing {len(missing_features)} required features: {missing_features[:5]}"
            f"{'...' if len(missing_features) > 5 else ''}\n"
            f"Available: {available_count}, Required: {required_count}"
        )
        
        suggestions = self._generate_mismatch_suggestions()
        
        super().__init__(message, "FEATURE_MISMATCH", suggestions)
    
    def _generate_mismatch_suggestions(self) -> List[str]:
        """Generate specific suggestions for feature mismatch issues."""
        
        suggestions = []
        
        # Analyze missing features for patterns
        time_features = {'hour', 'day_of_week', 'asian_session', 'european_session', 'american_session'}
        missing_time = set(self.missing_features) & time_features
        
        if missing_time:
            suggestions.append(
                f"Time features missing: {list(missing_time)}. "
                f"Ensure your DataFrame has a proper DatetimeIndex when calling FeatureCalculator."
            )
        
        # Check for indicator features
        indicator_patterns = ['sma_', 'ema_', 'rsi_', 'macd_', 'bb_', 'atr_']
        missing_indicators = [f for f in self.missing_features 
                            if any(f.startswith(pattern) for pattern in indicator_patterns)]
        
        if missing_indicators:
            suggestions.append(
                f"Technical indicator features missing: {missing_indicators[:3]}... "
                f"Verify that all FeatureCalculator methods are called in sequence."
            )
        
        # General suggestions
        suggestions.extend([
            f"Update FeatureCalculator to generate all required features",
            f"Or retrain the model {self.model_key} with current FeatureCalculator output",
            f"Check that FeatureCalculator output matches training data structure"
        ])
        
        return suggestions

class ModelLoadError(FeatureCompatibilityError):
    """Raised when model files cannot be loaded or found."""
    
    def __init__(self, model_key: str, model_path: str, original_error: Optional[Exception] = None):
        self.model_key = model_key
        self.model_path = model_path
        self.original_error = original_error
        
        message = f"Failed to load model {model_key} from {model_path}"
        if original_error:
            message += f": {original_error}"
        
        suggestions = [
            f"Check that model file exists: {model_path}",
            f"Verify model file permissions and integrity",
            f"Try using ALL_SYMBOLS fallback model if symbol-specific model missing",
            f"Retrain models if they were created with incompatible Python/library versions"
        ]
        
        super().__init__(message, "MODEL_LOAD_ERROR", suggestions)

class MappingFileError(FeatureCompatibilityError):
    """Raised when model_feature_mapping.json has issues."""
    
    def __init__(self, mapping_path: str, issue_type: str, details: str = ""):
        self.mapping_path = mapping_path
        self.issue_type = issue_type
        
        message = f"Model mapping file error ({issue_type}): {mapping_path}"
        if details:
            message += f" - {details}"
        
        suggestions = []
        if issue_type == "NOT_FOUND":
            suggestions = [
                f"Run 'python scripts/extract_model_features_simple.py' to create mapping file",
                f"Ensure you have trained models in data/models/ directory",
                f"Check file permissions for creating mapping file"
            ]
        elif issue_type == "INVALID_FORMAT":
            suggestions = [
                f"Delete corrupted mapping file and regenerate",
                f"Check that model metadata files are valid pickle files",
                f"Verify JSON format of existing mapping file"
            ]
        
        super().__init__(message, "MAPPING_FILE_ERROR", suggestions)

class PredictionError(FeatureCompatibilityError):
    """Raised when model prediction fails despite feature compatibility."""
    
    def __init__(self, model_key: str, model_type: str, prediction_error: Exception):
        self.model_key = model_key
        self.model_type = model_type
        self.prediction_error = prediction_error
        
        message = f"Prediction failed for {model_key} ({model_type}): {prediction_error}"
        
        suggestions = [
            f"Check model file integrity and compatibility",
            f"Verify that feature data types match training expectations", 
            f"Ensure feature values are within reasonable ranges (no inf/nan)",
            f"Check model-specific requirements (e.g., sequence length for LSTM)"
        ]
        
        super().__init__(message, "PREDICTION_ERROR", suggestions)

class FeatureCompatibilityDiagnostics:
    """Diagnostic tools for feature compatibility issues."""
    
    def __init__(self, feature_selector=None):
        self.feature_selector = feature_selector
        
    def diagnose_feature_mismatch(self, available_features: List[str], 
                                required_features: List[str]) -> Dict[str, Any]:
        """Provide detailed diagnosis of feature mismatch."""
        
        available_set = set(available_features)
        required_set = set(required_features)
        
        found_features = list(required_set & available_set)
        missing_features = list(required_set - available_set)
        extra_features = list(available_set - required_set)
        
        # Categorize missing features
        time_features = {'hour', 'day_of_week', 'asian_session', 'european_session', 'american_session'}
        missing_time = list(set(missing_features) & time_features)
        
        indicator_patterns = ['sma_', 'ema_', 'rsi_', 'macd_', 'bb_', 'atr_', 'adx_', 'roc_']
        missing_indicators = [f for f in missing_features 
                            if any(f.startswith(pattern) for pattern in indicator_patterns)]
        
        pattern_features = ['doji', 'hammer', 'engulfing', 'stoch_', 'cci_', 'mfi_']
        missing_patterns = [f for f in missing_features 
                          if any(pattern in f for pattern in pattern_features)]
        
        diagnosis = {
            "summary": {
                "available_count": len(available_features),
                "required_count": len(required_features),
                "found_count": len(found_features),
                "missing_count": len(missing_features),
                "extra_count": len(extra_features),
                "compatibility_rate": len(found_features) / len(required_features) if required_features else 0
            },
            "missing_features": {
                "all": missing_features,
                "time_features": missing_time,
                "indicators": missing_indicators[:10],  # Limit for readability
                "patterns": missing_patterns[:10],
                "other": [f for f in missing_features 
                         if f not in missing_time and f not in missing_indicators and f not in missing_patterns][:10]
            },
            "found_features": found_features[:10],  # Sample of found features
            "extra_features": extra_features[:10],  # Sample of extra features
            "recommendations": self._generate_mismatch_recommendations(missing_time, missing_indicators, missing_patterns)
        }
        
        return diagnosis
    
    def _generate_mismatch_recommendations(self, missing_time: List[str], 
                                         missing_indicators: List[str], 
                                         missing_patterns: List[str]) -> List[str]:
        """Generate specific recommendations based on missing feature types."""
        
        recommendations = []
        
        if missing_time:
            recommendations.append(
                "TIME FEATURES: Ensure DataFrame has DatetimeIndex before feature calculation"
            )
        
        if missing_indicators:
            recommendations.append(
                "INDICATORS: Call all FeatureCalculator methods (momentum, rebound, regime, additional)"
            )
        
        if missing_patterns:
            recommendations.append(
                "PATTERNS: Ensure TA-Lib is properly installed and additional features are calculated"
            )
        
        recommendations.extend([
            "GENERAL: Compare current FeatureCalculator output with training data",
            "UPDATE: Consider retraining models with current feature set",
            "FALLBACK: Use feature selection to match training requirements"
        ])
        
        return recommendations
    
    def check_feature_pipeline_health(self, feature_calculator, sample_data) -> Dict[str, Any]:
        """Perform comprehensive health check of feature pipeline."""
        
        health_check = {
            "feature_calculator": {"status": "unknown", "details": {}},
            "feature_generation": {"status": "unknown", "details": {}},
            "feature_selector": {"status": "unknown", "details": {}},
            "overall_health": "unknown"
        }
        
        try:
            # Test FeatureCalculator
            test_features = self._test_feature_calculation(feature_calculator, sample_data)
            health_check["feature_calculator"]["status"] = "healthy"
            health_check["feature_calculator"]["details"] = {
                "total_features": len(test_features.columns),
                "sample_features": list(test_features.columns)[:10]
            }
            
            # Test FeatureSelector
            if self.feature_selector:
                selector_health = self._test_feature_selector(test_features)
                health_check["feature_selector"]["status"] = selector_health["status"]
                health_check["feature_selector"]["details"] = selector_health
            
            # Overall assessment
            if all(component["status"] == "healthy" for component in health_check.values() if isinstance(component, dict)):
                health_check["overall_health"] = "healthy"
            else:
                health_check["overall_health"] = "degraded"
                
        except Exception as e:
            health_check["overall_health"] = "critical"
            health_check["error"] = str(e)
        
        return health_check
    
    def _test_feature_calculation(self, feature_calculator, sample_data):
        """Test feature calculation pipeline."""
        
        temp_df = sample_data.copy()
        temp_df = feature_calculator.calculate_momentum_features(temp_df)
        temp_df = feature_calculator.calculate_rebound_features(temp_df)
        temp_df = feature_calculator.calculate_regime_features(temp_df)
        features_df = feature_calculator.calculate_additional_features(temp_df)
        
        # Clean features
        features_df.replace([float('inf'), float('-inf')], float('nan'), inplace=True)
        features_df = features_df.fillna(0)
        
        return features_df
    
    def _test_feature_selector(self, features_df):
        """Test feature selector with sample model."""
        
        try:
            # Check if feature_selector is available
            if self.feature_selector is None:
                return {
                    "status": "unhealthy",
                    "error": "FeatureSelector not initialized",
                    "selected_features": 0,
                    "models_available": 0
                }
            
            # Try to select features for a sample model
            selected = self.feature_selector.select_features(
                features_df, "momentum", "BTCUSDT", "5m"
            )
            
            return {
                "status": "healthy",
                "selected_features": len(selected.columns),
                "models_available": len(self.feature_selector.model_mapping.get("models", {}))
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e)
            }

class GracefulDegradationManager:
    """Manages graceful degradation when feature compatibility fails."""
    
    def __init__(self):
        self.fallback_strategies = {
            "missing_time_features": self._handle_missing_time_features,
            "missing_indicators": self._handle_missing_indicators,
            "model_load_failure": self._handle_model_load_error,
            "prediction_failure": self._handle_prediction_error
        }
    
    def handle_compatibility_failure(self, error: FeatureCompatibilityError, 
                                   context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Handle compatibility failures with appropriate fallback strategies."""
        
        context = context or {}
        
        if isinstance(error, FeatureMismatchError):
            return self._handle_feature_mismatch(error, context)
        elif isinstance(error, ModelLoadError):
            return self._handle_model_load_error(error, context)
        elif isinstance(error, PredictionError):
            return self._handle_prediction_error(error, context)
        else:
            return self._handle_generic_error(error, context)
    
    def _handle_feature_mismatch(self, error: FeatureMismatchError, 
                               context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle feature mismatch with fallback strategies."""
        
        # Try to identify the type of mismatch and apply appropriate fallback
        time_features = {'hour', 'day_of_week', 'asian_session', 'european_session', 'american_session'}
        missing_time = set(error.missing_features) & time_features
        
        if missing_time:
            return self._handle_missing_time_features(error, context)
        else:
            return self._handle_missing_indicators(error, context)
    
    def _handle_missing_time_features(self, error, context):
        """Handle missing time features by providing synthetic ones."""
        
        return {
            "strategy": "synthetic_time_features",
            "action": "Add synthetic time features based on data position",
            "success": False,  # Would need actual implementation
            "message": "Time features missing - recommend fixing DataFrame index"
        }
    
    def _handle_missing_indicators(self, error, context):
        """Handle missing technical indicators."""
        
        return {
            "strategy": "reduced_feature_set",
            "action": "Use only available features for prediction",
            "success": False,  # Would need actual implementation
            "message": "Technical indicators missing - recommend running all FeatureCalculator methods"
        }
    
    def _handle_model_load_error(self, error, context):
        """Handle model loading failures."""
        
        return {
            "strategy": "fallback_model",
            "action": "Try ALL_SYMBOLS model or skip this model type",
            "success": False,  # Would need actual implementation
            "message": f"Model load failed: {error.model_key}"
        }
    
    def _handle_prediction_error(self, error, context):
        """Handle prediction failures."""
        
        return {
            "strategy": "skip_model",
            "action": "Skip this model and continue with others",
            "success": True,
            "message": f"Skipping failed model: {error.model_key}"
        }
    
    def _handle_generic_error(self, error, context):
        """Handle generic compatibility errors."""
        
        return {
            "strategy": "log_and_continue",
            "action": "Log error and continue processing",
            "success": True,
            "message": f"Generic error handled: {error.message}"
        }

# Utility functions for error handling
def log_feature_compatibility_error(error: FeatureCompatibilityError, logger: Optional[logging.Logger] = None):
    """Log feature compatibility error with full details."""
    
    if logger is None:
        logger = logging.getLogger(__name__)
    
    error_info = error.get_full_error_info()
    
    logger.error(f"Feature Compatibility Error [{error_info['error_code']}]:")
    logger.error(f"  Message: {error_info['message']}")
    
    if error_info['suggestions']:
        logger.info("  Suggestions:")
        for i, suggestion in enumerate(error_info['suggestions'], 1):
            logger.info(f"    {i}. {suggestion}")

def create_error_report(errors: List[FeatureCompatibilityError], output_path: Optional[str] = None) -> Dict[str, Any]:
    """Create comprehensive error report for debugging."""
    
    report = {
        "timestamp": str(Path(__file__).stat().st_mtime),
        "total_errors": len(errors),
        "error_types": {},
        "errors": []
    }
    
    for error in errors:
        error_info = error.get_full_error_info()
        report["errors"].append(error_info)
        
        error_type = error_info["exception_type"]
        if error_type not in report["error_types"]:
            report["error_types"][error_type] = 0
        report["error_types"][error_type] += 1
    
    if output_path:
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
    
    return report