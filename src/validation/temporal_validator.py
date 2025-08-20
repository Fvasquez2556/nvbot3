"""
Sistema de validación temporal para prevenir overfitting en trading.
NUNCA usar random splits en datos financieros.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Tuple, Dict, List
import logging

class TemporalValidator:
    """
    Validador temporal estricto para datos de trading.
    Asegura que NUNCA se use información futura para predecir el pasado.
    """
    
    def __init__(self, train_ratio: float = 0.7, val_ratio: float = 0.15, test_ratio: float = 0.15):
        """
        Inicializar validador temporal.
        
        Args:
            train_ratio: Porcentaje para entrenamiento (datos más antiguos)
            val_ratio: Porcentaje para validación (datos intermedios)
            test_ratio: Porcentaje para testing (datos más recientes)
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios deben sumar 1.0"
        
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.logger = logging.getLogger(__name__)
    
    def temporal_split(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split temporal estricto - datos más antiguos para train, más recientes para test.
        
        Args:
            df: DataFrame con index temporal ordenado
            
        Returns:
            train_data, val_data, test_data
        """
        # Verificar que el DataFrame está ordenado temporalmente
        if not df.index.is_monotonic_increasing:
            self.logger.warning("DataFrame no está ordenado temporalmente. Ordenando...")
            df = df.sort_index()
        
        total_len = len(df)
        train_end = int(total_len * self.train_ratio)
        val_end = int(total_len * (self.train_ratio + self.val_ratio))
        
        train_data = df.iloc[:train_end].copy()
        val_data = df.iloc[train_end:val_end].copy()
        test_data = df.iloc[val_end:].copy()
        
        # Logging de fechas para verificación
        self.logger.info(f"Temporal Split ejecutado:")
        self.logger.info(f"  Train: {train_data.index[0]} a {train_data.index[-1]} ({len(train_data)} samples)")
        self.logger.info(f"  Val:   {val_data.index[0]} a {val_data.index[-1]} ({len(val_data)} samples)")
        self.logger.info(f"  Test:  {test_data.index[0]} a {test_data.index[-1]} ({len(test_data)} samples)")
        
        # Verificación crítica: no overlap temporal
        assert train_data.index[-1] < val_data.index[0], "❌ CRITICAL: Train y Val se superponen temporalmente!"
        assert val_data.index[-1] < test_data.index[0], "❌ CRITICAL: Val y Test se superponen temporalmente!"
        
        return train_data, val_data, test_data
    
    def validate_no_data_leakage(self, train_data: pd.DataFrame, val_data: pd.DataFrame, test_data: pd.DataFrame) -> bool:
        """
        Verificar que no existe data leakage temporal.
        
        Returns:
            True si no hay leakage, False si lo hay
        """
        checks = []
        
        # Check 1: Orden temporal estricto
        temporal_order = (train_data.index[-1] < val_data.index[0] and 
                         val_data.index[-1] < test_data.index[0])
        checks.append(("Orden temporal", temporal_order))
        
        # Check 2: No fechas duplicadas entre sets
        train_dates = set(train_data.index)
        val_dates = set(val_data.index)
        test_dates = set(test_data.index)
        
        no_overlap = (len(train_dates & val_dates) == 0 and 
                     len(val_dates & test_dates) == 0 and 
                     len(train_dates & test_dates) == 0)
        checks.append(("Sin overlap de fechas", no_overlap))
        
        # Check 3: Gaps temporales razonables
        train_to_val_gap = (val_data.index[0] - train_data.index[-1]).total_seconds() / 3600
        val_to_test_gap = (test_data.index[0] - val_data.index[-1]).total_seconds() / 3600
        
        reasonable_gaps = train_to_val_gap < 168 and val_to_test_gap < 168  # <1 semana de gap
        checks.append(("Gaps temporales razonables", reasonable_gaps))
        
        # Logging de resultados
        for check_name, result in checks:
            if result:
                self.logger.info(f"✅ {check_name}: PASS")
            else:
                self.logger.error(f"❌ {check_name}: FAIL")
        
        return all(result for _, result in checks)

class CryptoTimeSeriesSplit:
    """
    Cross-validation temporal para criptomonedas.
    Implementa Time Series Split respetando la naturaleza temporal de los datos.
    """
    
    def __init__(self, n_splits: int = 5, test_size_months: int = 2):
        """
        Inicializar Time Series Split para crypto.
        
        Args:
            n_splits: Número de splits para CV
            test_size_months: Tamaño del test set en meses
        """
        self.n_splits = n_splits
        self.test_size_months = test_size_months
        self.logger = logging.getLogger(__name__)
    
    def split(self, df: pd.DataFrame) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Generar splits temporales para cross-validation.
        
        Args:
            df: DataFrame con datos ordenados temporalmente
            
        Yields:
            (train_data, test_data) para cada split
        """
        splits = []
        total_len = len(df)
        
        # Calcular tamaño aproximado de test en samples
        # Asumiendo datos de 5 minutos: ~288 samples por día, ~8640 por mes
        samples_per_month = 8640  # Aproximado para 5m timeframe
        test_size = self.test_size_months * samples_per_month
        
        # Calcular incremento para cada split
        increment = (total_len - test_size) // self.n_splits
        
        for i in range(self.n_splits):
            # Definir índices de train y test
            train_start = 0
            train_end = increment * (i + 1)
            test_start = train_end
            test_end = min(train_end + test_size, total_len)
            
            # Crear splits
            train_data = df.iloc[train_start:train_end].copy()
            test_data = df.iloc[test_start:test_end].copy()
            
            # Verificar que tenemos suficientes datos
            if len(train_data) < 1000 or len(test_data) < 100:
                self.logger.warning(f"Split {i+1}: Datos insuficientes (Train:{len(train_data)}, Test:{len(test_data)})")
                continue
            
            # Verificar orden temporal
            if train_data.index[-1] >= test_data.index[0]:
                self.logger.error(f"Split {i+1}: Violación temporal detectada!")
                continue
            
            self.logger.info(f"Split {i+1}: Train({len(train_data)}) Test({len(test_data)})")
            splits.append((train_data, test_data))
        
        return splits
