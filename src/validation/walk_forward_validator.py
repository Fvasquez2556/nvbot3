"""
Walk-Forward Validation - El gold standard para validación en trading.
Simula el reentrenamiento periódico que ocurre en trading real.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import joblib
import logging
from dataclasses import dataclass

@dataclass
class WalkForwardResult:
    """Resultado de un período de walk-forward validation."""
    period_start: datetime
    period_end: datetime
    train_size: int
    test_size: int
    model_performance: Dict[str, float]
    predictions: np.ndarray
    actuals: np.ndarray
    market_regime: str
    confidence_scores: np.ndarray

class WalkForwardValidator:
    """
    Implementa Walk-Forward Validation para simular trading real.
    Reentrena el modelo periódicamente con nuevos datos.
    """
    
    def __init__(self, 
                 initial_train_months: int = 6,
                 test_months: int = 1,
                 retrain_frequency_months: int = 1,
                 min_train_samples: int = 5000):
        """
        Configurar Walk-Forward Validation.
        
        Args:
            initial_train_months: Meses iniciales para entrenamiento
            test_months: Meses para cada período de testing
            retrain_frequency_months: Frecuencia de reentrenamiento
            min_train_samples: Mínimo de samples para entrenar
        """
        self.initial_train_months = initial_train_months
        self.test_months = test_months
        self.retrain_frequency = retrain_frequency_months
        self.min_train_samples = min_train_samples
        self.logger = logging.getLogger(__name__)
    
    def validate(self, df: pd.DataFrame, model_class: Any, model_params: Dict) -> List[WalkForwardResult]:
        """
        Ejecutar Walk-Forward Validation completa.
        
        Args:
            df: DataFrame con features y targets
            model_class: Clase del modelo a entrenar
            model_params: Parámetros del modelo
            
        Returns:
            Lista de resultados por período
        """
        results = []
        
        # CORRECCIÓN CRÍTICA: Verificar y ajustar datos automáticamente
        total_samples = len(df)
        min_required = self.initial_train_months * 100  # 100 muestras mínimas por mes
        
        if total_samples < min_required:
            self.logger.warning(f"⚠️ Datos insuficientes para Walk-Forward: {total_samples} < {min_required}")
            self.logger.warning(f"🔧 Ajustando parámetros automáticamente...")
            
            # Ajustar parámetros automáticamente
            self.initial_train_months = max(1, total_samples // 200)  # Reducir train period
            self.test_months = max(1, total_samples // 400)           # Reducir test period
            self.retrain_frequency = max(1, total_samples // 600)     # Reducir reentrenamiento
            
            self.logger.warning(f"✅ Nuevos parámetros: train={self.initial_train_months}m, test={self.test_months}m, retrain={self.retrain_frequency}m")
        
        # Calcular samples por mes (aproximado para 5m timeframe)
        try:
            days_total = (df.index[-1] - df.index[0]).days
            samples_per_month = len(df) // max(1, (days_total / 30))  # Evitar división por cero
        except:
            # Fallback: asumir ~288 samples por día (5min timeframe)
            samples_per_month = 288 * 30  # ~8640 samples por mes
        
        initial_train_samples = int(self.initial_train_months * samples_per_month)
        test_samples = int(self.test_months * samples_per_month)
        step_samples = int(self.retrain_frequency * samples_per_month)
        
        # Verificar que tenemos suficientes datos (con ajuste automático)
        min_required_samples = initial_train_samples + test_samples
        if len(df) < min_required_samples:
            # Último recurso: usar la mitad de datos para train y la otra mitad para test
            initial_train_samples = len(df) // 2
            test_samples = len(df) // 4
            step_samples = len(df) // 8
            self.logger.warning(f"🚨 Ajuste de emergencia: train={initial_train_samples}, test={test_samples}")
        
        self.logger.info(f"Iniciando Walk-Forward Validation:")
        self.logger.info(f"  Datos totales: {len(df)} samples")
        self.logger.info(f"  Train inicial: {initial_train_samples} samples ({self.initial_train_months} meses)")
        self.logger.info(f"  Test size: {test_samples} samples ({self.test_months} meses)")
        self.logger.info(f"  Reentrenamiento cada: {step_samples} samples ({self.retrain_frequency} meses)")
        
        # Iterar sobre períodos de tiempo
        current_pos = initial_train_samples
        iteration = 0
        
        while current_pos + test_samples <= len(df):
            iteration += 1
            
            # Definir períodos de train y test
            train_start = max(0, current_pos - initial_train_samples)
            train_end = current_pos
            test_start = current_pos
            test_end = min(current_pos + test_samples, len(df))
            
            # Extraer datos
            train_data = df.iloc[train_start:train_end]
            test_data = df.iloc[test_start:test_end]
            
            # Verificar tamaño mínimo
            if len(train_data) < self.min_train_samples:
                self.logger.warning(f"Iteración {iteration}: Train data insuficiente ({len(train_data)} < {self.min_train_samples})")
                current_pos += step_samples
                continue
            
            self.logger.info(f"Iteración {iteration}: Train({train_data.index[0]} a {train_data.index[-1]}) Test({test_data.index[0]} a {test_data.index[-1]})")
            
            # Entrenar modelo
            try:
                result = self._train_and_evaluate_period(train_data, test_data, model_class, model_params, iteration)
                results.append(result)
                
            except Exception as e:
                self.logger.error(f"Error en iteración {iteration}: {str(e)}")
                
            # Avanzar al siguiente período
            current_pos += step_samples
        
        self.logger.info(f"Walk-Forward Validation completado: {len(results)} períodos evaluados")
        return results
    
    def _train_and_evaluate_period(self, 
                                  train_data: pd.DataFrame, 
                                  test_data: pd.DataFrame,
                                  model_class: Any,
                                  model_params: Dict,
                                  iteration: int) -> WalkForwardResult:
        """
        Entrenar y evaluar para un período específico.
        """
        # Separar features y targets
        feature_cols = [col for col in train_data.columns if col.startswith('feature_')]
        target_col = 'target'  # Asumiendo columna target
        
        X_train = train_data[feature_cols]
        y_train = train_data[target_col]
        X_test = test_data[feature_cols]
        y_test = test_data[target_col]
        
        # Entrenar modelo
        model = model_class(**model_params)
        
        # Early stopping si el modelo lo soporta
        if hasattr(model, 'fit') and 'eval_set' in model.fit.__code__.co_varnames:
            model.fit(X_train, y_train, 
                     eval_set=[(X_test, y_test)],
                     early_stopping_rounds=15,
                     verbose=False)
        else:
            model.fit(X_train, y_train)
        
        # Hacer predicciones
        predictions = model.predict(X_test)
        
        # Calcular confidence scores si es posible
        if hasattr(model, 'predict_proba'):
            confidence_scores = model.predict_proba(X_test)[:, 1]  # Probabilidad clase positiva
        else:
            confidence_scores = np.abs(predictions)  # Usar valor absoluto como proxy
        
        # Calcular métricas
        performance = self._calculate_metrics(np.array(y_test), predictions, confidence_scores)
        
        # Detectar régimen de mercado
        market_regime = self._detect_market_regime(test_data)
        
        return WalkForwardResult(
            period_start=test_data.index[0],
            period_end=test_data.index[-1],
            train_size=len(X_train),
            test_size=len(X_test),
            model_performance=performance,
            predictions=predictions,
            actuals=np.array(y_test.values),
            market_regime=market_regime,
            confidence_scores=confidence_scores
        )
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, confidence: np.ndarray) -> Dict[str, float]:
        """Calcular métricas específicas de trading."""
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        # Convertir a clasificación binaria si es necesario
        if len(np.unique(y_true)) == 2:
            y_pred_binary = (y_pred > 0.5).astype(int)
        else:
            y_pred_binary = (y_pred > np.median(y_pred)).astype(int)
            y_true_binary = (y_true > np.median(y_true)).astype(int)
            y_true = y_true_binary
        
        return {
            'accuracy': float(accuracy_score(y_true, y_pred_binary)),
            'precision': float(precision_score(y_true, y_pred_binary, zero_division=0)),
            'recall': float(recall_score(y_true, y_pred_binary, zero_division=0)),
            'f1_score': float(f1_score(y_true, y_pred_binary, zero_division=0)),
            'directional_accuracy': float(accuracy_score(y_true > 0, y_pred > 0)),
            'avg_confidence': float(np.mean(confidence)),
            'std_confidence': float(np.std(confidence))
        }
    
    def _detect_market_regime(self, data: pd.DataFrame) -> str:
        """Detectar régimen de mercado para el período."""
        if 'close' in data.columns:
            returns = data['close'].pct_change().dropna()
            volatility = returns.std()
            avg_return = returns.mean()
            
            if avg_return > 0.001 and volatility < 0.03:
                return 'bullish_stable'
            elif avg_return > 0.001 and volatility >= 0.03:
                return 'bullish_volatile'
            elif avg_return < -0.001 and volatility < 0.03:
                return 'bearish_stable'
            elif avg_return < -0.001 and volatility >= 0.03:
                return 'bearish_volatile'
            else:
                return 'sideways'
        
        return 'unknown'
    
    def analyze_results(self, results: List[WalkForwardResult]) -> Dict[str, Any]:
        """
        Analizar resultados de Walk-Forward Validation.
        """
        if not results:
            return {}
        
        # Extraer métricas
        accuracies = [r.model_performance['accuracy'] for r in results]
        directional_accuracies = [r.model_performance['directional_accuracy'] for r in results]
        regimes = [r.market_regime for r in results]
        
        # Análisis por régimen
        regime_performance = {}
        for regime in set(regimes):
            regime_results = [r for r in results if r.market_regime == regime]
            regime_accuracies = [r.model_performance['accuracy'] for r in regime_results]
            
            regime_performance[regime] = {
                'count': len(regime_results),
                'avg_accuracy': np.mean(regime_accuracies) if regime_accuracies else 0,
                'std_accuracy': np.std(regime_accuracies) if regime_accuracies else 0
            }
        
        return {
            'total_periods': len(results),
            'avg_accuracy': np.mean(accuracies),
            'std_accuracy': np.std(accuracies),
            'min_accuracy': np.min(accuracies),
            'max_accuracy': np.max(accuracies),
            'avg_directional_accuracy': np.mean(directional_accuracies),
            'regime_performance': regime_performance,
            'stability_score': 1 - (np.std(accuracies) / np.mean(accuracies)) if np.mean(accuracies) > 0 else 0,
            'periods_below_60pct': sum(1 for acc in accuracies if acc < 0.6),
            'consistent_performer': sum(1 for acc in accuracies if acc > 0.65) / len(accuracies)
        }
