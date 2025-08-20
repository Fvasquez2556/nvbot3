"""
Módulo de modelos regularizados para NvBot3.
Contiene implementaciones con anti-overfitting agresivo.
"""

from .regularized_models import (
    RegularizedXGBoost,
    RegularizedTimeSeriesModel,
    RegularizedLSTM,  # Alias para RegularizedTimeSeriesModel
    RegularizedEnsemble
)

__all__ = [
    'RegularizedXGBoost',
    'RegularizedTimeSeriesModel', 
    'RegularizedLSTM',
    'RegularizedEnsemble'
]
