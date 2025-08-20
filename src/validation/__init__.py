"""
Módulo de validación anti-overfitting para NvBot3.
Sistema completo de validación temporal y detección de overfitting.
"""

from .temporal_validator import (
    TemporalValidator,
    CryptoTimeSeriesSplit
)

from .walk_forward_validator import (
    WalkForwardValidator,
    WalkForwardResult
)

from .overfitting_detector import (
    OverfittingDetector,
    OverfittingLevel,
    OverfittingReport
)

__all__ = [
    # Validación temporal
    'TemporalValidator',
    'CryptoTimeSeriesSplit',
    
    # Validación walk-forward
    'WalkForwardValidator', 
    'WalkForwardResult',
    
    # Detección de overfitting
    'OverfittingDetector',
    'OverfittingLevel',
    'OverfittingReport'
]
