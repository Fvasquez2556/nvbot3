"""
Detector automático de overfitting para modelos de trading.
Sistema de alertas y métricas anti-overfitting.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import learning_curve, validation_curve
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import warnings
from dataclasses import dataclass
from enum import Enum

def safe_get_feature_names(data):
    """Extracción segura de nombres de features."""
    try:
        if hasattr(data, 'columns'):
            return list(data.columns)
        elif hasattr(data, 'shape') and len(data.shape) > 1:
            return [f'feature_{i}' for i in range(data.shape[1])]
        else:
            return ['feature_0']
    except:
        return ['unknown_feature']

def safe_array_conversion(data):
    """Conversión segura a numpy array."""
    try:
        if hasattr(data, 'values'):
            return np.array(data.values)
        else:
            return np.array(data)
    except:
        return np.array([0])

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
    metrics: Dict[str, float]

class OverfittingDetector:
    """
    Detector automático de overfitting con múltiples métricas.
    """
    
    def __init__(self, task_type: str = 'regression'):
        """
        Inicializar detector de overfitting.
        
        Args:
            task_type: 'regression' o 'classification'
        """
        self.task_type = task_type
        self.logger = logging.getLogger(__name__)
        
        # Umbrales de overfitting por nivel
        self.thresholds = {
            OverfittingLevel.NONE: 0.05,
            OverfittingLevel.LOW: 0.1,
            OverfittingLevel.MEDIUM: 0.2,
            OverfittingLevel.HIGH: 0.3,
            OverfittingLevel.EXTREME: 0.5
        }
        
        # Métricas de alerta
        self.alert_metrics = {
            'train_val_gap': 0.15,        # Gap máximo aceptable train-val
            'validation_r2': 0.1,         # R² mínimo en validación
            'learning_curve_slope': 0.9,  # Pendiente de curva de aprendizaje
            'complexity_penalty': 0.8     # Penalización por complejidad
        }
    
    def detect(self, model, X_train: np.ndarray, y_train: np.ndarray,
               X_val: np.ndarray, y_val: np.ndarray,
               model_name: str = "Unknown") -> OverfittingReport:
        """
        Detectar overfitting en modelo entrenado.
        
        Args:
            model: Modelo entrenado con métodos predict y score
            X_train: Features de entrenamiento
            y_train: Target de entrenamiento  
            X_val: Features de validación
            y_val: Target de validación
            model_name: Nombre del modelo para logging
            
        Returns:
            OverfittingReport con análisis completo
        """
        warnings_list = []
        recommendations = []
        metrics = {}
        
        try:
            # 1. Scores básicos
            train_score = model.score(X_train, y_train)
            val_score = model.score(X_val, y_val)
            gap = train_score - val_score
            
            metrics.update({
                'train_score': train_score,
                'val_score': val_score,
                'train_val_gap': gap
            })
            
            # 2. Predicciones para métricas adicionales
            y_train_pred = model.predict(X_train)
            y_val_pred = model.predict(X_val)
            
            # 3. Métricas detalladas
            train_mse = mean_squared_error(y_train, y_train_pred)
            val_mse = mean_squared_error(y_val, y_val_pred)
            train_mae = mean_absolute_error(y_train, y_train_pred)
            val_mae = mean_absolute_error(y_val, y_val_pred)
            
            metrics.update({
                'train_mse': train_mse,
                'val_mse': val_mse,
                'train_mae': train_mae,
                'val_mae': val_mae,
                'mse_ratio': val_mse / max(train_mse, 1e-8),
                'mae_ratio': val_mae / max(train_mae, 1e-8)
            })
            
            # 4. Análisis de varianza de predicciones
            train_pred_var = np.var(y_train_pred)
            val_pred_var = np.var(y_val_pred)
            train_target_var = np.var(y_train)
            val_target_var = np.var(y_val)
            
            metrics.update({
                'train_pred_variance': train_pred_var,
                'val_pred_variance': val_pred_var,
                'variance_ratio': val_pred_var / max(train_pred_var, 1e-8)
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
            if metrics['variance_ratio'] > 2.0:
                warnings_list.append("Varianza de predicciones inconsistente entre train/val")
                recommendations.append("Verificar distribución de datos y normalización")
            
            # MSE ratio muy alto
            if metrics['mse_ratio'] > 2.0:
                warnings_list.append(f"MSE en validación {metrics['mse_ratio']:.2f}x mayor que en training")
                recommendations.append("Modelo memoriza training data - aumentar regularización")
            
            # 6. Análisis de curva de aprendizaje (si es posible)
            try:
                learning_analysis = self._analyze_learning_curve(model, X_train, y_train, X_val, y_val)
                metrics.update(learning_analysis)
                
                if learning_analysis.get('learning_curve_slope', 0) > self.alert_metrics['learning_curve_slope']:
                    warnings_list.append("Curva de aprendizaje indica overfitting temprano")
                    recommendations.append("Implementar early stopping más agresivo")
                    
            except Exception as e:
                self.logger.warning(f"No se pudo analizar curva de aprendizaje: {e}")
            
            # 7. Determinear nivel de overfitting
            overfitting_level = self._classify_overfitting_level(gap, val_score, metrics)
            
            # 8. Calcular score de overfitting (0-1, donde 1 = overfitting extremo)
            overfitting_score = min(1.0, max(0.0, gap * 2.0 + (1.0 - max(val_score, 0)) * 0.5))
            
            # 9. Recomendaciones específicas según nivel
            recommendations.extend(self._get_level_recommendations(overfitting_level))
            
            # 10. Logging
            self.logger.info(f"Análisis de overfitting para {model_name}:")
            self.logger.info(f"  Nivel: {overfitting_level.value.upper()}")
            self.logger.info(f"  Score de overfitting: {overfitting_score:.3f}")
            self.logger.info(f"  Gap train-val: {gap:.3f}")
            self.logger.info(f"  Score validación: {val_score:.3f}")
            
            if warnings_list:
                self.logger.warning(f"  Advertencias: {len(warnings_list)}")
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
            self.logger.error(f"Error en detección de overfitting: {e}")
            return OverfittingReport(
                level=OverfittingLevel.EXTREME,
                score=1.0,
                train_score=0.0,
                val_score=0.0,
                gap=1.0,
                warnings=[f"Error en análisis: {e}"],
                recommendations=["Verificar modelo y datos"],
                metrics={}
            )
    
    def _analyze_learning_curve(self, model, X_train: np.ndarray, y_train: np.ndarray,
                               X_val: np.ndarray, y_val: np.ndarray) -> Dict[str, float]:
        """Analizar curva de aprendizaje para detectar overfitting."""
        
        # Tamaños de muestra para la curva
        train_sizes = np.linspace(0.1, 1.0, 10)
        
        try:
            # Solo si el modelo soporta fit incremental o podemos recrearlo
            if hasattr(model, 'n_estimators'):
                # Para modelos tipo ensemble, variar n_estimators
                param_range = [10, 25, 50, 75, 100]
                train_scores, val_scores = validation_curve(
                    model.__class__(**model.get_params()), X_train, y_train,
                    param_name='n_estimators', param_range=param_range,
                    cv=3, scoring='r2'
                )
            else:
                # Usar curva de aprendizaje estándar
                train_sizes_abs, train_scores, val_scores = learning_curve(
                    model.__class__(**model.get_params()), X_train, y_train,
                    train_sizes=train_sizes, cv=3, scoring='r2'
                )[:3]
            
            # Calcular métricas de la curva
            train_mean = np.mean(train_scores, axis=1)
            val_mean = np.mean(val_scores, axis=1)
            
            # Pendiente de la diferencia (indicador de overfitting)
            gap_evolution = train_mean - val_mean
            slope = np.polyfit(range(len(gap_evolution)), gap_evolution, 1)[0]
            
            return {
                'learning_curve_slope': float(abs(slope)),
                'final_train_cv': float(train_mean[-1]),
                'final_val_cv': float(val_mean[-1]),
                'curve_stability': float(np.std(gap_evolution))
            }
            
        except Exception as e:
            self.logger.warning(f"Análisis de curva de aprendizaje falló: {e}")
            return {}
    
    def _classify_overfitting_level(self, gap: float, val_score: float, 
                                  metrics: Dict[str, float]) -> OverfittingLevel:
        """Clasificar nivel de overfitting basado en métricas."""
        
        # Factores de overfitting
        factors = []
        
        # Factor 1: Gap train-validation
        if gap <= self.thresholds[OverfittingLevel.NONE]:
            factors.append(0)
        elif gap <= self.thresholds[OverfittingLevel.LOW]:
            factors.append(1)
        elif gap <= self.thresholds[OverfittingLevel.MEDIUM]:
            factors.append(2)
        elif gap <= self.thresholds[OverfittingLevel.HIGH]:
            factors.append(3)
        else:
            factors.append(4)
        
        # Factor 2: Score de validación
        if val_score >= 0.7:
            factors.append(0)
        elif val_score >= 0.5:
            factors.append(1)
        elif val_score >= 0.3:
            factors.append(2)
        elif val_score >= 0.1:
            factors.append(3)
        else:
            factors.append(4)
        
        # Factor 3: Ratio de MSE
        mse_ratio = metrics.get('mse_ratio', 1.0)
        if mse_ratio <= 1.2:
            factors.append(0)
        elif mse_ratio <= 1.5:
            factors.append(1)
        elif mse_ratio <= 2.0:
            factors.append(2)
        elif mse_ratio <= 3.0:
            factors.append(3)
        else:
            factors.append(4)
        
        # Promedio de factores
        avg_factor = np.mean(factors)
        
        if avg_factor <= 0.5:
            return OverfittingLevel.NONE
        elif avg_factor <= 1.5:
            return OverfittingLevel.LOW
        elif avg_factor <= 2.5:
            return OverfittingLevel.MEDIUM
        elif avg_factor <= 3.5:
            return OverfittingLevel.HIGH
        else:
            return OverfittingLevel.EXTREME
    
    def _get_level_recommendations(self, level: OverfittingLevel) -> List[str]:
        """Obtener recomendaciones específicas según nivel de overfitting."""
        
        recommendations = {
            OverfittingLevel.NONE: [
                "✅ Modelo bien regularizado - mantener configuración actual"
            ],
            OverfittingLevel.LOW: [
                "Monitorear performance en datos futuros",
                "Considerar validación cruzada más robusta"
            ],
            OverfittingLevel.MEDIUM: [
                "⚠️ Aumentar regularización (L1/L2)",
                "Reducir complejidad del modelo",
                "Implementar early stopping más agresivo",
                "Usar dropout o feature selection"
            ],
            OverfittingLevel.HIGH: [
                "🚨 ACCIÓN REQUERIDA: Overfitting significativo",
                "Reducir drásticamente complejidad del modelo",
                "Aumentar dataset de entrenamiento",
                "Implementar validación temporal estricta",
                "Considerar ensemble con alta regularización"
            ],
            OverfittingLevel.EXTREME: [
                "🔥 CRÍTICO: Modelo inutilizable en producción",
                "Reiniciar entrenamiento con máxima regularización",
                "Verificar calidad y distribución de datos", 
                "Usar modelos más simples (linear/ridge)",
                "Implementar walk-forward validation obligatorio"
            ]
        }
        
        return recommendations.get(level, ["Revisar configuración del detector"])
    
    def monitor_training(self, train_scores: List[float], val_scores: List[float],
                        model_name: str = "Unknown") -> Dict[str, Any]:
        """
        Monitorear entrenamiento en tiempo real.
        
        Args:
            train_scores: Lista de scores de entrenamiento por época
            val_scores: Lista de scores de validación por época
            model_name: Nombre del modelo
            
        Returns:
            Dict con análisis de entrenamiento
        """
        if len(train_scores) != len(val_scores):
            raise ValueError("train_scores y val_scores deben tener la misma longitud")
        
        if len(train_scores) < 3:
            return {"status": "insufficient_data", "message": "Necesarias al menos 3 épocas"}
        
        # Convertir a arrays
        train_arr = np.array(train_scores)
        val_arr = np.array(val_scores)
        gaps = train_arr - val_arr
        
        # Análisis de tendencias
        analysis = {
            'epochs': len(train_scores),
            'final_train': train_arr[-1],
            'final_val': val_arr[-1],
            'final_gap': gaps[-1],
            'max_gap': np.max(gaps),
            'gap_trend': np.polyfit(range(len(gaps)), gaps, 1)[0],  # Pendiente
            'val_trend': np.polyfit(range(len(val_arr)), val_arr, 1)[0],
            'early_stop_suggested': False,
            'best_epoch': np.argmax(val_arr)
        }
        
        # Detectar señales de early stopping
        if len(val_scores) >= 5:
            # Si validación no mejora en últimas 5 épocas
            recent_val = val_arr[-5:]
            if np.all(recent_val <= recent_val[0]):
                analysis['early_stop_suggested'] = True
                analysis['stop_reason'] = "validación sin mejora"
        
        # Si gap aumenta consistentemente
        if analysis['gap_trend'] > 0.01:  # Gap aumentando
            analysis['early_stop_suggested'] = True
            analysis['stop_reason'] = "gap train-val creciente"
        
        # Si validación está disminuyendo
        if analysis['val_trend'] < -0.01:
            analysis['early_stop_suggested'] = True
            analysis['stop_reason'] = "performance validación decreciente"
        
        # Logging de alertas
        if analysis['early_stop_suggested']:
            self.logger.warning(f"⚠️ Early stopping sugerido para {model_name}: {analysis['stop_reason']}")
            self.logger.warning(f"   Mejor época: {analysis['best_epoch']}")
        
        if analysis['final_gap'] > 0.2:
            self.logger.warning(f"🚨 Overfitting crítico en {model_name}: gap {analysis['final_gap']:.3f}")
        
        return analysis
    
    def create_overfitting_plot(self, train_scores: List[float], val_scores: List[float],
                               save_path: Optional[str] = None) -> None:
        """
        Crear gráfico de overfitting.
        
        Args:
            train_scores: Scores de entrenamiento
            val_scores: Scores de validación  
            save_path: Ruta para guardar el gráfico
        """
        try:
            epochs = range(1, len(train_scores) + 1)
            
            plt.figure(figsize=(10, 6))
            plt.plot(epochs, train_scores, 'b-', label='Training Score', linewidth=2)
            plt.plot(epochs, val_scores, 'r-', label='Validation Score', linewidth=2)
            
            # Área de overfitting
            plt.fill_between(epochs, train_scores, val_scores, 
                           where=np.array(train_scores) >= np.array(val_scores),
                           alpha=0.3, color='red', label='Overfitting Area')
            
            plt.xlabel('Época')
            plt.ylabel('Score')
            plt.title('Análisis de Overfitting')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Marcar mejor época
            best_epoch = np.argmax(val_scores)
            plt.axvline(x=float(best_epoch+1), color='green', linestyle='--', 
                       label=f'Mejor época: {best_epoch+1}')
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                self.logger.info(f"Gráfico guardado en: {save_path}")
            else:
                plt.show()
                
        except Exception as e:
            self.logger.error(f"Error creando gráfico: {e}")
    
    def batch_analysis(self, models: Dict[str, Any], X_train: np.ndarray, y_train: np.ndarray,
                      X_val: np.ndarray, y_val: np.ndarray) -> pd.DataFrame:
        """
        Análisis de overfitting para múltiples modelos.
        
        Args:
            models: Dict con nombre -> modelo entrenado
            X_train, y_train: Datos de entrenamiento
            X_val, y_val: Datos de validación
            
        Returns:
            DataFrame con comparación de modelos
        """
        results = []
        
        for name, model in models.items():
            try:
                report = self.detect(model, X_train, y_train, X_val, y_val, name)
                
                results.append({
                    'model': name,
                    'overfitting_level': report.level.value,
                    'overfitting_score': report.score,
                    'train_score': report.train_score,
                    'val_score': report.val_score,
                    'train_val_gap': report.gap,
                    'warnings_count': len(report.warnings),
                    'recommendation_count': len(report.recommendations)
                })
                
            except Exception as e:
                self.logger.error(f"Error analizando modelo {name}: {e}")
                results.append({
                    'model': name,
                    'overfitting_level': 'error',
                    'overfitting_score': 1.0,
                    'train_score': 0.0,
                    'val_score': 0.0, 
                    'train_val_gap': 1.0,
                    'warnings_count': 1,
                    'recommendation_count': 1
                })
        
        df = pd.DataFrame(results)
        
        if not df.empty:
            # Ordenar por mejor performance (menor overfitting, mayor val_score)
            df['rank_score'] = (1 - df['overfitting_score']) * 0.6 + df['val_score'] * 0.4
            df = df.sort_values('rank_score', ascending=False)
            
            self.logger.info(f"Análisis batch completado para {len(models)} modelos:")
            self.logger.info(f"  Mejor modelo: {df.iloc[0]['model']}")
            self.logger.info(f"  Nivel overfitting: {df.iloc[0]['overfitting_level']}")
            self.logger.info(f"  Score validación: {df.iloc[0]['val_score']:.3f}")
        
        return df
