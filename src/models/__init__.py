"""
Módulo de modelos regularizados para NvBot3.
Contiene implementaciones con anti-overfitting agresivo.
"""

from .regularized_models import (
    RegularizedXGBoost,
    RegularizedEnsemble
)

__all__ = [
    'RegularizedXGBoost',
    'RegularizedEnsemble'
]
