"""
Detector automático de overfitting para modelos de trading.
Versión completamente limpia y robusta - NvBot3.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Union
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import logging
from dataclasses import dataclass
from enum import Enum

class OverfittingLevel(Enum):
    """Niveles de overfitting detectados."""
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    EXTREME = "extreme"

@dataclass
class OverfittingReport:
    """Reporte completo de overfitting."""
    level: OverfittingLevel
    score: float
    train_score: float
    val_score: float
    gap: float
    warnings: List[str]
    recommendations: List[str]
    metrics: Dict[str, Any]

class OverfittingDetector:
    """
    Detector robusto de overfitting para modelos de trading.
    Maneja de forma segura diferentes tipos de datos y errores.
    """
    
    def __init__(self):
        """Inicializar detector con configuración robusta."""
        self.logger = logging.getLogger(__name__)
        
        # Thresholds para clasificación de overfitting
        self.thresholds = {
            OverfittingLevel.NONE: 0.05,
            OverfittingLevel.LOW: 0.1,
            OverfittingLevel.MEDIUM: 0.2,
            OverfittingLevel.HIGH: 0.3,
            OverfittingLevel.EXTREME: 0.5
        }
        
        # Métricas de alerta
        self.alert_metrics = {
            'train_val_gap': 0.15,
            'validation_r2': 0.1,
            'learning_curve_slope': 0.9,
            'complexity_penalty': 0.8
        }
    
    def _safe_convert_to_array(self, data: Any) -> np.ndarray:
        """
        Convertir datos a numpy array de forma completamente segura.
        
        Args:
            data: Datos en cualquier formato (DataFrame, Series, list, array, etc.)
            
        Returns:
            np.ndarray: Array numpy válido
        """
        try:
            if data is None:
                return np.array([])
            
            if hasattr(data, 'values'):
                # DataFrame o Series de pandas
                return np.asarray(data.values)
            elif isinstance(data, (list, tuple)):
                # Lista o tupla
                return np.asarray(data)
            elif isinstance(data, np.ndarray):
                # Ya es array numpy
                return data
            else:
                # Cualquier otro tipo, forzar conversión
                return np.asarray(data)
                
        except Exception as e:
            self.logger.warning(f"Error convirtiendo datos a array: {e}")
            return np.array([0])  # Array fallback
    
    def _validate_inputs(self, X_train: np.ndarray, y_train: np.ndarray, 
                        X_val: np.ndarray, y_val: np.ndarray) -> bool:
        """
        Validar que los datos de entrada son válidos.
        
        Returns:
            bool: True si los datos son válidos, False si no
        """
        try:
            # Verificar que no están vacíos
            if X_train.size == 0 or y_train.size == 0 or X_val.size == 0 or y_val.size == 0:
                self.logger.error("Datos vacíos detectados")
                return False
            
            # Verificar dimensiones
            if X_train.ndim < 2:
                X_train = X_train.reshape(-1, 1)
            if X_val.ndim < 2:
                X_val = X_val.reshape(-1, 1)
            
            if y_train.ndim > 1:
                y_train = y_train.flatten()
            if y_val.ndim > 1:
                y_val = y_val.flatten()
            
            # Verificar que tenemos features
            if X_train.shape[1] == 0:
                self.logger.error("No hay features en X_train")
                return False
            
            # Verificar que las muestras coinciden
            if X_train.shape[0] != y_train.shape[0]:
                self.logger.error(f"Mismatch en train: X={X_train.shape[0]}, y={y_train.shape[0]}")
                return False
            
            if X_val.shape[0] != y_val.shape[0]:
                self.logger.error(f"Mismatch en val: X={X_val.shape[0]}, y={y_val.shape[0]}")
                return False
            
            # Verificar que número de features coincide
            if X_train.shape[1] != X_val.shape[1]:
                self.logger.error(f"Mismatch features: train={X_train.shape[1]}, val={X_val.shape[1]}")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error validando inputs: {e}")
            return False
    
    def _safe_model_score(self, model: Any, X: np.ndarray, y: np.ndarray) -> float:
        """
        Calcular score del modelo de forma segura.
        
        Returns:
            float: Score del modelo o valor por defecto si hay error
        """
        try:
            if hasattr(model, 'score'):
                score = model.score(X, y)
                # Verificar que el score es un número válido
                if np.isfinite(score):
                    return float(score)
                else:
                    self.logger.warning(f"Score no finito: {score}")
                    return 0.0
            else:
                self.logger.warning("Modelo no tiene método score")
                return 0.0
                
        except Exception as e:
            self.logger.warning(f"Error calculando score: {e}")
            return 0.0
    
    def _safe_model_predict(self, model: Any, X: np.ndarray) -> np.ndarray:
        """
        Hacer predicciones de forma segura.
        
        Returns:
            np.ndarray: Predicciones o array por defecto si hay error
        """
        try:
            if hasattr(model, 'predict'):
                predictions = model.predict(X)
                pred_array = np.asarray(predictions)
                
                # Verificar que las predicciones son válidas
                if pred_array.size == X.shape[0] and np.all(np.isfinite(pred_array)):
                    return pred_array
                else:
                    self.logger.warning(f"Predicciones inválidas: shape={pred_array.shape}, finite={np.all(np.isfinite(pred_array))}")
                    return np.zeros(X.shape[0])
            else:
                self.logger.warning("Modelo no tiene método predict")
                return np.zeros(X.shape[0])
                
        except Exception as e:
            self.logger.warning(f"Error en predicciones: {e}")
            return np.zeros(X.shape[0])
    
    def detect(self, model: Any, X_train: Union[np.ndarray, pd.DataFrame], y_train: Union[np.ndarray, pd.Series],
               X_val: Union[np.ndarray, pd.DataFrame], y_val: Union[np.ndarray, pd.Series],
               model_name: str = "Unknown") -> OverfittingReport:
        """
        Detectar overfitting en modelo entrenado de forma robusta.
        
        Args:
            model: Modelo entrenado con métodos predict y score
            X_train: Features de entrenamiento
            y_train: Target de entrenamiento  
            X_val: Features de validación
            y_val: Target de validación
            model_name: Nombre del modelo para logging
            
        Returns:
            OverfittingReport: Reporte completo de análisis
        """
        warnings_list = []
        recommendations = []
        metrics = {}
        
        try:
            self.logger.info(f"Iniciando análisis de overfitting para: {model_name}")
            
            # Convertir todos los datos de forma segura
            X_train_array = self._safe_convert_to_array(X_train)
            y_train_array = self._safe_convert_to_array(y_train)
            X_val_array = self._safe_convert_to_array(X_val)
            y_val_array = self._safe_convert_to_array(y_val)
            
            # Validar datos
            if not self._validate_inputs(X_train_array, y_train_array, X_val_array, y_val_array):
                raise ValueError("Datos de entrada inválidos")
            
            self.logger.info(f"Datos validados: train={X_train_array.shape}, val={X_val_array.shape}")
            
            # 1. Calcular scores básicos de forma segura
            train_score = self._safe_model_score(model, X_train_array, y_train_array)
            val_score = self._safe_model_score(model, X_val_array, y_val_array)
            gap = train_score - val_score
            
            metrics.update({
                'train_score': train_score,
                'val_score': val_score,
                'train_val_gap': gap
            })
            
            # 2. Obtener predicciones de forma segura
            y_train_pred = self._safe_model_predict(model, X_train_array)
            y_val_pred = self._safe_model_predict(model, X_val_array)
            
            # 3. Calcular métricas detalladas de forma segura
            try:
                train_mse = mean_squared_error(y_train_array, y_train_pred)
                val_mse = mean_squared_error(y_val_array, y_val_pred)
                train_mae = mean_absolute_error(y_train_array, y_train_pred)
                val_mae = mean_absolute_error(y_val_array, y_val_pred)
                
                # Ratios seguros (evitar división por cero)
                mse_ratio = val_mse / max(train_mse, 1e-8)
                mae_ratio = val_mae / max(train_mae, 1e-8)
                
                metrics.update({
                    'train_mse': train_mse,
                    'val_mse': val_mse,
                    'train_mae': train_mae,
                    'val_mae': val_mae,
                    'mse_ratio': mse_ratio,
                    'mae_ratio': mae_ratio
                })
                
            except Exception as e:
                self.logger.warning(f"Error calculando métricas detalladas: {e}")
                metrics.update({
                    'train_mse': 0.0, 'val_mse': 0.0, 'train_mae': 0.0, 'val_mae': 0.0,
                    'mse_ratio': 1.0, 'mae_ratio': 1.0
                })
            
            # 4. Análisis de varianza de predicciones
            try:
                train_pred_var = np.var(y_train_pred) if y_train_pred.size > 0 else 0.0
                val_pred_var = np.var(y_val_pred) if y_val_pred.size > 0 else 0.0
                variance_ratio = val_pred_var / max(train_pred_var, 1e-8)
                
                metrics.update({
                    'train_pred_variance': train_pred_var,
                    'val_pred_variance': val_pred_var,
                    'variance_ratio': variance_ratio
                })
                
            except Exception as e:
                self.logger.warning(f"Error calculando varianzas: {e}")
                metrics.update({
                    'train_pred_variance': 0.0,
                    'val_pred_variance': 0.0,
                    'variance_ratio': 1.0
                })
            
            # 5. Detectar señales de overfitting
            
            # Gap train-validation excesivo
            if gap > self.alert_metrics['train_val_gap']:
                warnings_list.append(f"Gap train-val excesivo: {gap:.3f} > {self.alert_metrics['train_val_gap']}")
                recommendations.append("Reducir complejidad del modelo o aumentar regularización")
            
            # Score de validación muy bajo
            if val_score < self.alert_metrics['validation_r2']:
                warnings_list.append(f"Score de validación muy bajo: {val_score:.3f}")
                recommendations.append("Revisar calidad de datos o aumentar tamaño de dataset")
            
            # Varianza de predicciones inconsistente
            variance_ratio = metrics.get('variance_ratio', 1.0)
            if variance_ratio > 2.0:
                warnings_list.append("Varianza de predicciones inconsistente entre train/val")
                recommendations.append("Verificar distribución de datos y normalización")
            
            # MSE ratio muy alto
            mse_ratio = metrics.get('mse_ratio', 1.0)
            if mse_ratio > 2.0:
                warnings_list.append(f"MSE en validación {mse_ratio:.2f}x mayor que en training")
                recommendations.append("Modelo memoriza training data - aumentar regularización")
            
            # 6. Determinar nivel de overfitting
            overfitting_level = self._classify_overfitting_level(gap, val_score, metrics)
            
            # 7. Calcular score de overfitting (0-1, donde 1 = overfitting extremo)
            overfitting_score = min(1.0, max(0.0, gap * 2.0 + (1.0 - max(val_score, 0)) * 0.5))
            
            # 8. Recomendaciones específicas según nivel
            recommendations.extend(self._get_level_recommendations(overfitting_level))
            
            # 9. Logging detallado
            self.logger.info(f"Análisis completado para {model_name}:")
            self.logger.info(f"  Nivel: {overfitting_level.value.upper()}")
            self.logger.info(f"  Score de overfitting: {overfitting_score:.3f}")
            self.logger.info(f"  Gap train-val: {gap:.3f}")
            self.logger.info(f"  Score validación: {val_score:.3f}")
            
            if warnings_list:
                self.logger.warning(f"  Advertencias ({len(warnings_list)}):")
                for warning in warnings_list:
                    self.logger.warning(f"    - {warning}")
            
            return OverfittingReport(
                level=overfitting_level,
                score=overfitting_score,
                train_score=train_score,
                val_score=val_score,
                gap=gap,
                warnings=warnings_list,
                recommendations=recommendations,
                metrics=metrics
            )
            
        except Exception as e:
            self.logger.error(f"Error crítico en detección de overfitting: {e}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            
            # Retornar reporte de error
            return OverfittingReport(
                level=OverfittingLevel.EXTREME,
                score=1.0,
                train_score=0.0,
                val_score=0.0,
                gap=1.0,
                warnings=[f"Error crítico en análisis: {str(e)}"],
                recommendations=["Verificar modelo y datos", "Revisar logs para detalles del error"],
                metrics={'error': str(e)}
            )
    
    def _classify_overfitting_level(self, gap: float, val_score: float, metrics: Dict[str, Any]) -> OverfittingLevel:
        """
        Clasificar nivel de overfitting basado en múltiples métricas.
        
        Args:
            gap: Diferencia train-val score
            val_score: Score en validación
            metrics: Métricas adicionales
            
        Returns:
            OverfittingLevel: Nivel de overfitting detectado
        """
        try:
            # Criterio principal: gap train-validation
            if gap <= self.thresholds[OverfittingLevel.NONE]:
                base_level = OverfittingLevel.NONE
            elif gap <= self.thresholds[OverfittingLevel.LOW]:
                base_level = OverfittingLevel.LOW
            elif gap <= self.thresholds[OverfittingLevel.MEDIUM]:
                base_level = OverfittingLevel.MEDIUM
            elif gap <= self.thresholds[OverfittingLevel.HIGH]:
                base_level = OverfittingLevel.HIGH
            else:
                base_level = OverfittingLevel.EXTREME
            
            # Ajustar basado en métricas adicionales
            mse_ratio = metrics.get('mse_ratio', 1.0)
            variance_ratio = metrics.get('variance_ratio', 1.0)
            
            # Penalizar si hay señales adicionales de overfitting
            penalty_factors = 0
            if mse_ratio > 2.0:
                penalty_factors += 1
            if variance_ratio > 2.0:
                penalty_factors += 1
            if val_score < self.alert_metrics['validation_r2']:
                penalty_factors += 1
            
            # Aumentar nivel según penalizaciones
            levels = list(OverfittingLevel)
            current_index = levels.index(base_level)
            new_index = min(len(levels) - 1, current_index + penalty_factors)
            
            return levels[new_index]
            
        except Exception as e:
            self.logger.warning(f"Error clasificando overfitting: {e}")
            return OverfittingLevel.EXTREME  # Conservador en caso de error
    
    def _get_level_recommendations(self, level: OverfittingLevel) -> List[str]:
        """
        Obtener recomendaciones específicas según nivel de overfitting.
        
        Args:
            level: Nivel de overfitting
            
        Returns:
            List[str]: Lista de recomendaciones
        """
        recommendations_map = {
            OverfittingLevel.NONE: [
                "Modelo bien regularizado - mantener configuración actual",
                "Considerar aumentar ligeramente la complejidad si el performance lo permite"
            ],
            OverfittingLevel.LOW: [
                "Overfitting mínimo - monitor continuo recomendado",
                "Validar con datos de períodos diferentes"
            ],
            OverfittingLevel.MEDIUM: [
                "Aumentar regularización L1/L2",
                "Reducir complejidad del modelo (max_depth, n_estimators)",
                "Implementar dropout o feature selection más agresivo"
            ],
            OverfittingLevel.HIGH: [
                "Regularización agresiva necesaria",
                "Implementar early stopping más estricto",
                "Reducir features o usar feature selection",
                "Aumentar datos de entrenamiento si es posible"
            ],
            OverfittingLevel.EXTREME: [
                "Modelo inviable - overfitting crítico",
                "Revisar arquitectura completa del modelo",
                "Aumentar significativamente los datos de entrenamiento",
                "Considerar modelos más simples (linear, ridge)",
                "Verificar data leakage en el pipeline"
            ]
        }
        
        return recommendations_map.get(level, ["Revisar configuración del modelo"])
    
    def batch_analysis(self, models: Dict[str, Any], X_train: np.ndarray, y_train: np.ndarray,
                      X_val: np.ndarray, y_val: np.ndarray) -> pd.DataFrame:
        """
        Analizar múltiples modelos y generar comparación.
        
        Args:
            models: Diccionario de modelos {nombre: modelo}
            X_train, y_train: Datos de entrenamiento
            X_val, y_val: Datos de validación
            
        Returns:
            pd.DataFrame: Comparación de modelos ordenada por mejor performance
        """
        try:
            results = []
            
            for model_name, model in models.items():
                try:
                    report = self.detect(model, X_train, y_train, X_val, y_val, model_name)
                    
                    results.append({
                        'model': model_name,
                        'overfitting_level': report.level.value,
                        'overfitting_score': report.score,
                        'train_score': report.train_score,
                        'val_score': report.val_score,
                        'gap': report.gap,
                        'warnings_count': len(report.warnings),
                        'mse_ratio': report.metrics.get('mse_ratio', 1.0),
                        'variance_ratio': report.metrics.get('variance_ratio', 1.0)
                    })
                    
                except Exception as e:
                    self.logger.error(f"Error analizando modelo {model_name}: {e}")
                    results.append({
                        'model': model_name,
                        'overfitting_level': 'error',
                        'overfitting_score': 1.0,
                        'train_score': 0.0,
                        'val_score': 0.0,
                        'gap': 1.0,
                        'warnings_count': 1,
                        'mse_ratio': 1.0,
                        'variance_ratio': 1.0
                    })
            
            if results:
                df = pd.DataFrame(results)
                # Ordenar por mejor performance (menor overfitting_score y mayor val_score)
                df = df.sort_values(['overfitting_score', 'val_score'], ascending=[True, False])
                return df
            else:
                return pd.DataFrame()
                
        except Exception as e:
            self.logger.error(f"Error en análisis batch: {e}")
            return pd.DataFrame()
