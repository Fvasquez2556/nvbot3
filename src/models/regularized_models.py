"""
Modelos con regularización agresiva para prevenir overfitting.
Configuración específica para trading de criptomonedas.
Versión limpia y corregida - NvBot3.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple, Union, List
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import logging
import warnings

class RegularizedXGBoost:
    """
    XGBoost con regularización agresiva para trading.
    Versión corregida que maneja arrays y DataFrames de forma segura.
    """
    
    def __init__(self, task_type: str = 'momentum'):
        """
        Inicializar XGBoost regularizado.
        
        Args:
            task_type: 'momentum', 'regime', o 'rebound'
        """
        self.task_type = task_type
        self.scaler = StandardScaler()
        self.feature_selector = SelectKBest(score_func=f_regression, k=50)
        self.logger = logging.getLogger(__name__)
        
        # Parámetros anti-overfitting por tipo de tarea
        self.params = self._get_regularized_params(task_type)
        self.model = None
    
    def _get_regularized_params(self, task_type: str) -> Dict[str, Any]:
        """Obtener parámetros específicos anti-overfitting por tarea."""
        
        base_params = {
            'n_estimators': 100,        # Menos árboles para reducir overfitting
            'max_depth': 4,             # Árboles más simples
            'learning_rate': 0.05,      # Aprendizaje lento y estable
            'subsample': 0.7,           # Solo 70% de datos por árbol
            'colsample_bytree': 0.7,    # Solo 70% de features por árbol
            'colsample_bylevel': 0.8,   # Feature sampling por nivel
            'min_child_weight': 5,      # Mínimo peso por hoja
            'gamma': 1.0,               # Mínima ganancia para split
            'reg_alpha': 5,             # Regularización L1
            'reg_lambda': 5,            # Regularización L2
            'random_state': 42,
            'n_jobs': -1
        }
        
        # Parámetros específicos por tipo de tarea
        if task_type == 'momentum':
            base_params.update({
                'learning_rate': 0.03,
                'reg_alpha': 10,
                'reg_lambda': 10
            })
        elif task_type == 'regime':
            base_params.update({
                'learning_rate': 0.05,
                'reg_alpha': 10,
                'reg_lambda': 15
            })
        elif task_type == 'rebound':
            base_params.update({
                'learning_rate': 0.05,
                'reg_alpha': 10,
                'reg_lambda': 5
            })
        
        return base_params
    
    def _safe_convert_to_array(self, data):
        """Convertir datos a numpy array de forma segura."""
        if hasattr(data, 'values'):
            return data.values
        elif isinstance(data, (list, tuple)):
            return np.array(data)
        else:
            return np.array(data)
    
    def fit(self, X_train, y_train, X_val=None, y_val=None, eval_set=None, **kwargs):
        """
        Entrenar modelo XGBoost con manejo seguro de datos.
        
        Args:
            X_train: Features de entrenamiento (DataFrame o array)
            y_train: Target de entrenamiento (Series o array)
            X_val: Features de validación (opcional)
            y_val: Target de validación (opcional)
            eval_set: Lista de (X_val, y_val) para early stopping (opcional)
            **kwargs: Argumentos adicionales (ignorados para compatibilidad)
        """
        try:
            # Convertir datos de forma segura
            X_train_array = self._safe_convert_to_array(X_train)
            y_train_array = self._safe_convert_to_array(y_train)
            
            # Verificar que tenemos features
            if X_train_array.shape[1] == 0:
                raise ValueError(f"X_train no tiene features! Shape: {X_train_array.shape}")
            
            self.logger.info(f"Entrenando XGBoost {self.task_type} con {X_train_array.shape[1]} features")
            
            # Normalizar features
            X_train_scaled = self.scaler.fit_transform(X_train_array)
            
            # Selección de features para reducir dimensionalidad
            X_train_selected = self.feature_selector.fit_transform(X_train_scaled, y_train_array)
            
            # Preparar eval_set si está disponible
            eval_set_processed = None
            X_val_selected = None
            y_val_array = None
            
            if eval_set is not None and len(eval_set) > 0:
                try:
                    X_val_raw, y_val_raw = eval_set[0]
                    X_val_array = self._safe_convert_to_array(X_val_raw)
                    y_val_array = self._safe_convert_to_array(y_val_raw)
                    
                    X_val_scaled = self.scaler.transform(X_val_array)
                    X_val_selected = self.feature_selector.transform(X_val_scaled)
                    eval_set_processed = [(X_val_selected, y_val_array)]
                except Exception as e:
                    self.logger.warning(f"Error procesando eval_set: {e}")
            elif X_val is not None and y_val is not None:
                X_val_array = self._safe_convert_to_array(X_val)
                y_val_array = self._safe_convert_to_array(y_val)
                
                X_val_scaled = self.scaler.transform(X_val_array)
                X_val_selected = self.feature_selector.transform(X_val_scaled)
                eval_set_processed = [(X_val_selected, y_val_array)]
            
            # Crear modelo XGBoost con parámetros seguros
            try:
                self.model = xgb.XGBRegressor(**self.params)
            except Exception as e:
                self.logger.warning(f"Error con parámetros completos: {e}")
                # Fallback con parámetros mínimos
                minimal_params = {
                    'n_estimators': self.params.get('n_estimators', 100),
                    'max_depth': self.params.get('max_depth', 4),
                    'learning_rate': self.params.get('learning_rate', 0.05),
                    'random_state': self.params.get('random_state', 42)
                }
                self.model = xgb.XGBRegressor(**minimal_params)
                self.logger.info("Usando parámetros mínimos para XGBoost")
            
            # Preparar parámetros de entrenamiento
            fit_params = {}
            fit_params['verbose'] = False
            
            if eval_set_processed:
                fit_params['eval_set'] = eval_set_processed
                fit_params['early_stopping_rounds'] = 15
            
            # Entrenar modelo
            try:
                self.model.fit(X_train_selected, y_train_array, **fit_params)
            except Exception as e:
                self.logger.warning(f"Error en fit con eval_set: {e}")
                # Fallback: entrenar sin eval_set
                self.model.fit(X_train_selected, y_train_array, verbose=False)
            
            # Log de resultados
            selected_features = self.feature_selector.get_support()
            
            self.logger.info(f"Modelo {self.task_type} entrenado:")
            self.logger.info(f"  Features seleccionadas: {np.sum(selected_features)}/{len(selected_features)}")
            self.logger.info(f"  Early stopping en iteración: {getattr(self.model, 'best_iteration', 'N/A')}")
            
            # Calcular scores si es posible
            try:
                train_score = self.model.score(X_train_selected, y_train_array)
                self.logger.info(f"  Score en training: {train_score:.4f}")
                
                if eval_set_processed and X_val_selected is not None and y_val_array is not None:
                    val_score = self.model.score(X_val_selected, y_val_array)
                    overfitting_gap = train_score - val_score
                    
                    self.logger.info(f"  Score en validación: {val_score:.4f}")
                    self.logger.info(f"  Gap train-val: {overfitting_gap:.4f}")
                    
                    if overfitting_gap > 0.15:
                        self.logger.warning(f"⚠️  OVERFITTING DETECTADO: Gap {overfitting_gap:.4f} > 0.15!")
            except Exception as e:
                self.logger.warning(f"Error calculando scores: {e}")
                
        except Exception as e:
            self.logger.error(f"Error en entrenamiento XGBoost: {e}")
            raise
    
    def predict(self, X) -> np.ndarray:
        """Hacer predicciones con el modelo entrenado."""
        if self.model is None:
            raise ValueError("Modelo no ha sido entrenado")
        
        # Convertir a array de forma segura
        X_array = self._safe_convert_to_array(X)
        
        # Verificar que tenemos features
        if X_array.shape[1] == 0:
            raise ValueError(f"X no tiene features para predicción! Shape: {X_array.shape}")
        
        X_scaled = self.scaler.transform(X_array)
        X_selected = self.feature_selector.transform(X_scaled)
        return self.model.predict(X_selected)
    
    def score(self, X, y) -> float:
        """Calcular R² score para compatibilidad con sklearn."""
        if self.model is None:
            raise ValueError("Modelo no ha sido entrenado")
        
        predictions = self.predict(X)
        y_array = self._safe_convert_to_array(y)
        return r2_score(y_array, predictions)


class TemporalFeatureModel:
    """
    Modelo con features temporales específicos para series de tiempo.
    """
    
    def __init__(self, task_type: str = 'momentum', lookback_periods: Optional[List[int]] = None):
        """
        Inicializar modelo temporal.
        
        Args:
            task_type: Tipo de tarea
            lookback_periods: Períodos de lookback para features temporales
        """
        self.task_type = task_type
        self.lookback_periods = lookback_periods or [5, 10, 20, 50]
        self.scaler = RobustScaler()  # Más robusto para outliers
        self.feature_selector = SelectKBest(score_func=f_regression, k=30)
        self.logger = logging.getLogger(__name__)
        self.model = None
        self.feature_names_ = None
    
    def _create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Crear features temporales avanzados."""
        features = df.copy()
        
        # Features de momentum
        for period in self.lookback_periods:
            if 'price' in df.columns:
                features[f'returns_{period}'] = df['price'].pct_change(period)
                features[f'volatility_{period}'] = df['price'].pct_change().rolling(period).std()
                features[f'momentum_{period}'] = df['price'] / df['price'].shift(period) - 1
            
            if 'volume' in df.columns:
                features[f'volume_ma_{period}'] = df['volume'].rolling(period).mean()
                features[f'volume_ratio_{period}'] = df['volume'] / features[f'volume_ma_{period}']
        
        # Features de tendencia
        if 'price' in df.columns:
            features['trend_5_20'] = df['price'].rolling(5).mean() / df['price'].rolling(20).mean() - 1
            features['trend_10_50'] = df['price'].rolling(10).mean() / df['price'].rolling(50).mean() - 1
        
        # Limpiar NaN
        features = features.ffill().fillna(0)
        
        return features
    
    def fit(self, X_train, y_train, X_val=None, y_val=None, **kwargs):
        """Entrenar modelo temporal."""
        try:
            # Crear features temporales
            if isinstance(X_train, pd.DataFrame):
                X_temporal = self._create_temporal_features(X_train)
            else:
                # Si es array, convertir a DataFrame con nombres genéricos
                temp_df = pd.DataFrame(X_train, columns=[f'feature_{i}' for i in range(X_train.shape[1])])
                X_temporal = self._create_temporal_features(temp_df)
            
            # Excluir target si existe
            feature_cols = [col for col in X_temporal.columns if col != 'target']
            X_features = X_temporal[feature_cols]
            
            self.logger.info(f"Features temporales creadas: {len(feature_cols)}")
            
            # Convertir y procesar
            X_array = X_features.values
            y_array = np.array(y_train) if hasattr(y_train, '__iter__') else y_train
            
            # Normalizar
            X_scaled = self.scaler.fit_transform(X_array)
            
            # Selección de features
            X_selected = self.feature_selector.fit_transform(X_scaled, y_array)
            
            # Entrenar modelo Ridge (más estable para features temporales)
            self.model = Ridge(alpha=10.0, random_state=42)
            self.model.fit(X_selected, y_array)
            
            # Guardar nombres de features para logging
            self.feature_names_ = feature_cols
            selected_mask = self.feature_selector.get_support()
            selected_features = [name for name, selected in zip(feature_cols, selected_mask) if selected]
            
            self.logger.info(f"Modelo Temporal {self.task_type} entrenado:")
            self.logger.info(f"  Features temporales creadas: {len(feature_cols)}")
            self.logger.info(f"  Features seleccionadas: {len(selected_features)}")
            
            # Calcular score
            train_score = self.model.score(X_selected, y_array)
            self.logger.info(f"  Score en training: {train_score:.4f}")
            
        except Exception as e:
            self.logger.error(f"Error en entrenamiento Temporal: {e}")
            raise
    
    def predict(self, X) -> np.ndarray:
        """Hacer predicciones."""
        if self.model is None:
            raise ValueError("Modelo no ha sido entrenado")
        
        # Crear features temporales
        if isinstance(X, pd.DataFrame):
            X_temporal = self._create_temporal_features(X)
        else:
            temp_df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
            X_temporal = self._create_temporal_features(temp_df)
        
        # Usar los mismos features que en entrenamiento
        if self.feature_names_:
            feature_cols = [col for col in self.feature_names_ if col in X_temporal.columns]
            X_features = X_temporal[feature_cols]
        else:
            feature_cols = [col for col in X_temporal.columns if col != 'target']
            X_features = X_temporal[feature_cols]
        
        X_array = X_features.values
        X_scaled = self.scaler.transform(X_array)
        X_selected = self.feature_selector.transform(X_scaled)
        
        return self.model.predict(X_selected)
    
    def score(self, X, y) -> float:
        """Calcular R² score."""
        predictions = self.predict(X)
        y_array = np.array(y) if hasattr(y, '__iter__') else y
        return r2_score(y_array, predictions)


class RegularizedEnsemble:
    """
    Ensemble de modelos regularizados para máxima robustez.
    """
    
    def __init__(self, task_type: str = 'momentum'):
        """Inicializar ensemble regularizado."""
        self.task_type = task_type
        self.logger = logging.getLogger(__name__)
        self.models = {}
        self.weights = {}
        self.scaler = StandardScaler()
        
        # Definir modelos del ensemble
        self.model_configs = {
            'xgb': RegularizedXGBoost(task_type),
            'ridge': Ridge(alpha=10.0, random_state=42),
            'elastic': ElasticNet(alpha=1.0, l1_ratio=0.5, random_state=42),
            'rf': RandomForestRegressor(
                n_estimators=50, max_depth=5, min_samples_split=10,
                min_samples_leaf=5, random_state=42, n_jobs=-1
            ),
            'temporal': TemporalFeatureModel(task_type)
        }
    
    def fit(self, X_train, y_train, X_val=None, y_val=None, **kwargs):
        """Entrenar ensemble de modelos."""
        try:
            # Preparar datos
            X_array = np.array(X_train) if hasattr(X_train, 'values') else X_train
            y_array = np.array(y_train) if hasattr(y_train, 'values') else y_train
            
            # Normalizar para modelos que lo necesiten
            X_scaled = self.scaler.fit_transform(X_array)
            
            active_models = []
            
            # Entrenar cada modelo
            for name, model in self.model_configs.items():
                try:
                    self.logger.info(f"Entrenando modelo {name}...")
                    
                    if name == 'xgb' or name == 'temporal':
                        # Modelos que manejan sus propios datos
                        model.fit(X_train, y_train, X_val, y_val)
                    else:
                        # Modelos sklearn estándar
                        model.fit(X_scaled, y_array)
                    
                    self.models[name] = model
                    active_models.append(name)
                    
                except Exception as e:
                    self.logger.warning(f"Error entrenando {name}: {e}")
            
            if not active_models:
                raise ValueError("No se pudo entrenar ningún modelo del ensemble")
            
            # Pesos iguales para simplicidad (podría optimizarse)
            weight = 1.0 / len(active_models)
            self.weights = {name: weight for name in active_models}
            
            self.logger.info(f"Ensemble {self.task_type} entrenado:")
            self.logger.info(f"  Modelos activos: {active_models}")
            self.logger.info(f"  Pesos: {self.weights}")
            
        except Exception as e:
            self.logger.error(f"Error en entrenamiento Ensemble: {e}")
            raise
    
    def predict(self, X) -> np.ndarray:
        """Hacer predicciones con ensemble."""
        if not self.models:
            raise ValueError("Ensemble no ha sido entrenado")
        
        predictions = []
        weights = []
        
        X_array = np.array(X) if hasattr(X, 'values') else X
        X_scaled = self.scaler.transform(X_array)
        
        for name, model in self.models.items():
            try:
                if name == 'xgb' or name == 'temporal':
                    pred = model.predict(X)
                else:
                    pred = model.predict(X_scaled)
                
                predictions.append(pred)
                weights.append(self.weights[name])
                
            except Exception as e:
                self.logger.warning(f"Error en predicción {name}: {e}")
        
        if not predictions:
            raise ValueError("No se pudo hacer predicciones con ningún modelo")
        
        # Promedio ponderado
        weighted_pred = np.average(predictions, axis=0, weights=weights)
        return weighted_pred
    
    def score(self, X, y) -> float:
        """Calcular R² score del ensemble."""
        predictions = self.predict(X)
        y_array = np.array(y) if hasattr(y, 'values') else y
        return r2_score(y_array, predictions)
