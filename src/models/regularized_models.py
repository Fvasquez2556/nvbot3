"""
Modelos con regularización agresiva para prevenir overfitting.
Configuración específica para trading de criptomonedas.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple, Union
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVC, SVR
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import logging

class RegularizedXGBoost:
    """
    XGBoost con regularización agresiva para trading.
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
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'n_estimators': 100,        # ⬇️ Menos árboles para reducir overfitting
            'max_depth': 4,             # ⬇️ Árboles más simples
            'learning_rate': 0.05,      # ⬇️ Aprendizaje lento y estable
            'subsample': 0.7,           # 🎲 Solo 70% de datos por árbol
            'colsample_bytree': 0.7,    # 🎲 Solo 70% de features por árbol
            'colsample_bylevel': 0.8,   # 🎲 Feature sampling por nivel
            'reg_alpha': 10,            # 🛡️ L1 regularization fuerte
            'reg_lambda': 10,           # 🛡️ L2 regularization fuerte
            'min_child_weight': 10,     # 🛡️ Mínimo peso por hoja
            'gamma': 1,                 # 🛡️ Mínima ganancia para split
            'random_state': 42,
            'n_jobs': -1
        }
        
        # Ajustes específicos por tarea
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
    
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series, 
            X_val: Optional[pd.DataFrame] = None, y_val: Optional[pd.Series] = None):
        """
        Entrenar modelo con regularización y early stopping.
        """
        # Normalizar features
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Selección de features para reducir dimensionalidad
        X_train_selected = self.feature_selector.fit_transform(X_train_scaled, y_train)
        
        # Preparar eval_set para early stopping
        eval_set = []
        X_val_selected = None
        if X_val is not None and y_val is not None:
            X_val_scaled = self.scaler.transform(X_val)
            X_val_selected = self.feature_selector.transform(X_val_scaled)
            eval_set = [(X_val_selected, y_val)]
        
        # Crear y entrenar modelo
        self.model = xgb.XGBRegressor(**self.params)
        
        # CORRECCIÓN CRÍTICA: Early stopping solo si hay datos de validación
        fit_params = {
            'verbose': False
        }
        
        if eval_set:
            fit_params['eval_set'] = eval_set
            fit_params['early_stopping_rounds'] = 15  # Solo si hay eval_set
        
        self.model.fit(X_train_selected, y_train, **fit_params)
        
        # Log de importancia de features
        feature_importance = self.model.feature_importances_
        selected_features = self.feature_selector.get_support()
        
        self.logger.info(f"Modelo {self.task_type} entrenado:")
        self.logger.info(f"  Features seleccionadas: {np.sum(selected_features)}/{len(selected_features)}")
        self.logger.info(f"  Early stopping en iteración: {self.model.best_iteration if hasattr(self.model, 'best_iteration') else 'N/A'}")
        self.logger.info(f"  Score en training: {self.model.score(X_train_selected, y_train):.4f}")
        
        if eval_set and X_val_selected is not None:
            val_score = self.model.score(X_val_selected, y_val)
            train_score = self.model.score(X_train_selected, y_train)
            overfitting_gap = train_score - val_score
            
            self.logger.info(f"  Score en validación: {val_score:.4f}")
            self.logger.info(f"  Gap train-val: {overfitting_gap:.4f}")
            
            if overfitting_gap > 0.15:
                self.logger.warning(f"⚠️  OVERFITTING DETECTADO: Gap {overfitting_gap:.4f} > 0.15!")
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Hacer predicciones con el modelo entrenado."""
        if self.model is None:
            raise ValueError("Modelo no ha sido entrenado")
        
        X_scaled = self.scaler.transform(X)
        X_selected = self.feature_selector.transform(X_scaled)
        return self.model.predict(X_selected)
    
    def score(self, X: pd.DataFrame, y: pd.Series) -> float:
        """Calcular R² score para compatibilidad con sklearn."""
        if self.model is None:
            raise ValueError("Modelo no ha sido entrenado")
        
        predictions = self.predict(X)
        return r2_score(y, predictions)
    
    def get_feature_importance(self) -> pd.Series:
        """Obtener importancia de features seleccionadas."""
        if self.model is None:
            raise ValueError("Modelo no ha sido entrenado")
        
        selected_features = self.feature_selector.get_support()
        feature_names = [f"feature_{i}" for i in range(len(selected_features)) if selected_features[i]]
        
        return pd.Series(self.model.feature_importances_, index=feature_names).sort_values(ascending=False)

class RegularizedTimeSeriesModel:
    """
    Modelo temporal regularizado usando Gradient Boosting para patrones temporales.
    Alternativa robusta a LSTM sin dependencia de TensorFlow.
    """
    
    def __init__(self, sequence_length: int = 24, task_type: str = 'rebound'):
        """
        Inicializar modelo temporal regularizado.
        
        Args:
            sequence_length: Longitud de secuencia temporal
            task_type: Tipo de tarea ('rebound', 'momentum')
        """
        self.sequence_length = sequence_length
        self.task_type = task_type
        self.scaler = RobustScaler()  # Más robusto para datos financieros
        self.feature_selector = SelectKBest(score_func=f_regression, k=30)
        self.model = None
        self.logger = logging.getLogger(__name__)
        
        # Configurar modelo según tarea
        self._configure_model()
    
    def _configure_model(self):
        """Configurar modelo con regularización fuerte según tarea."""
        
        base_params = {
            'n_estimators': 50,          # ⬇️ Menos estimators para reducir overfitting
            'learning_rate': 0.05,       # ⬇️ Aprendizaje lento
            'max_depth': 3,              # ⬇️ Árboles simples
            'min_samples_split': 20,     # 🛡️ Mínimas muestras para split
            'min_samples_leaf': 10,      # 🛡️ Mínimas muestras por hoja
            'subsample': 0.7,            # 🎲 Solo 70% de datos por árbol
            'max_features': 'sqrt',      # 🎲 Feature subsampling
            'random_state': 42,
            'validation_fraction': 0.2,  # Para early stopping
            'n_iter_no_change': 10       # Early stopping
        }
        
        if self.task_type == 'rebound':
            # Para rebotes: mayor regularización
            base_params.update({
                'learning_rate': 0.03,
                'max_depth': 2,
                'min_samples_split': 30,
                'min_samples_leaf': 15
            })
        elif self.task_type == 'momentum':
            # Para momentum: balance regularización/expresividad
            base_params.update({
                'learning_rate': 0.05,
                'max_depth': 3,
                'min_samples_split': 20,
                'min_samples_leaf': 10
            })
        
        # Crear el modelo aquí
        self.model = GradientBoostingRegressor(**base_params)
    
    def _create_temporal_features(self, X: np.ndarray) -> np.ndarray:
        """Crear features temporales usando ventanas deslizantes."""
        
        n_samples, n_features = X.shape
        temporal_features = []
        
        for i in range(self.sequence_length, n_samples):
            # Ventana actual
            window = X[i-self.sequence_length:i]
            
            # Features estadísticas por ventana
            window_features = []
            
            for j in range(n_features):
                feature_series = window[:, j]
                
                # Estadísticas básicas
                window_features.extend([
                    np.mean(feature_series),      # Media
                    np.std(feature_series),       # Desviación
                    np.min(feature_series),       # Mínimo
                    np.max(feature_series),       # Máximo
                    feature_series[-1]            # Valor actual
                ])
                
                # Tendencia (diferencia primera)
                if len(feature_series) > 1:
                    diff = np.diff(feature_series)
                    window_features.extend([
                        np.mean(diff),            # Tendencia promedio
                        np.std(diff)              # Volatilidad de cambio
                    ])
                else:
                    window_features.extend([0.0, 0.0])
            
            temporal_features.append(window_features)
        
        return np.array(temporal_features)
    
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series,
            X_val: Optional[pd.DataFrame] = None, y_val: Optional[pd.Series] = None):
        """
        Entrenar modelo temporal con regularización.
        """
        # Normalizar datos
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Crear features temporales
        X_train_temporal = self._create_temporal_features(X_train_scaled)
        y_train_temporal = np.array(y_train.iloc[self.sequence_length:])
        
        # Selección de features para reducir dimensionalidad
        X_train_selected = self.feature_selector.fit_transform(X_train_temporal, y_train_temporal)
        
        # Preparar validación si está disponible
        X_val_selected = None
        y_val_temporal = None
        if X_val is not None and y_val is not None:
            X_val_scaled = self.scaler.transform(X_val)
            X_val_temporal = self._create_temporal_features(X_val_scaled)
            X_val_selected = self.feature_selector.transform(X_val_temporal)
            y_val_temporal = np.array(y_val.iloc[self.sequence_length:])
        
        # Entrenar modelo (modelo ya configurado en __init__)
        if self.model is not None:
            self.model.fit(X_train_selected, y_train_temporal)
            
            # Logging de resultados
            train_score = self.model.score(X_train_selected, y_train_temporal)
            
            self.logger.info(f"Modelo Temporal {self.task_type} entrenado:")
            self.logger.info(f"  Features temporales creadas: {X_train_temporal.shape[1]}")
            self.logger.info(f"  Features seleccionadas: {X_train_selected.shape[1]}")
            self.logger.info(f"  Score en training: {train_score:.4f}")
            
            if X_val_selected is not None and y_val_temporal is not None:
                val_score = self.model.score(X_val_selected, y_val_temporal)
                overfitting_gap = train_score - val_score
                
                self.logger.info(f"  Score en validación: {val_score:.4f}")
                self.logger.info(f"  Gap train-val: {overfitting_gap:.4f}")
                
                if overfitting_gap < 0.05:  # Diferencia pequeña = buen signo
                    self.logger.info("✅ Modelo bien regularizado")
                elif overfitting_gap > 0.2:
                    self.logger.warning(f"⚠️  OVERFITTING DETECTADO: Gap {overfitting_gap:.4f} > 0.2!")
        else:
            raise ValueError("Error: Modelo no se pudo configurar correctamente")
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Hacer predicciones con modelo temporal entrenado."""
        if self.model is None:
            raise ValueError("Modelo no ha sido entrenado")
        
        X_scaled = self.scaler.transform(X)
        X_temporal = self._create_temporal_features(X_scaled)
        X_selected = self.feature_selector.transform(X_temporal)
        
        return self.model.predict(X_selected)
    
    def score(self, X: pd.DataFrame, y: pd.Series) -> float:
        """Calcular R² score para compatibilidad con sklearn."""
        if self.model is None:
            raise ValueError("Modelo no ha sido entrenado")
        
        predictions = self.predict(X)
        # Ajustar tamaños para datos temporales
        min_len = min(len(predictions), len(y))
        return r2_score(y.iloc[-min_len:], predictions[-min_len:])
    
    def get_feature_importance(self) -> pd.Series:
        """Obtener importancia de features temporales."""
        if self.model is None:
            raise ValueError("Modelo no ha sido entrenado")
        
        selected_features = self.feature_selector.get_support()
        feature_names = [f"temporal_feature_{i}" for i in range(len(selected_features)) if selected_features[i]]
        
        return pd.Series(self.model.feature_importances_, index=feature_names).sort_values(ascending=False)


class RegularizedEnsemble:
    """
    Ensemble de modelos regularizados para máxima robustez.
    """
    
    def __init__(self, task_type: str = 'momentum'):
        """
        Inicializar ensemble de modelos.
        
        Args:
            task_type: Tipo de tarea ('momentum', 'regime', 'rebound')
        """
        self.task_type = task_type
        self.scaler = StandardScaler()
        self.models = {}
        self.weights = {}
        self.logger = logging.getLogger(__name__)
        
        self._initialize_models()
    
    def _initialize_models(self):
        """Inicializar conjunto de modelos complementarios."""
        
        # XGBoost regularizado
        self.models['xgb'] = RegularizedXGBoost(self.task_type)
        
        # Ridge con regularización fuerte
        self.models['ridge'] = Ridge(alpha=10.0, random_state=42)
        
        # Elastic Net para selección automática de features
        self.models['elastic'] = ElasticNet(alpha=1.0, l1_ratio=0.5, random_state=42)
        
        # Random Forest con alta regularización
        self.models['rf'] = RandomForestRegressor(
            n_estimators=50,
            max_depth=4,
            min_samples_split=20,
            min_samples_leaf=10,
            max_features='sqrt',
            random_state=42
        )
        
        # Modelo temporal
        self.models['temporal'] = RegularizedTimeSeriesModel(task_type=self.task_type)
    
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series,
            X_val: Optional[pd.DataFrame] = None, y_val: Optional[pd.Series] = None):
        """
        Entrenar ensemble de modelos con validación cruzada.
        """
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_train_df = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
        
        # Preparar validación
        X_val_scaled = None
        X_val_df = None
        if X_val is not None:
            X_val_scaled = self.scaler.transform(X_val)
            X_val_df = pd.DataFrame(X_val_scaled, columns=X_val.columns, index=X_val.index)
        
        val_scores = {}
        
        # Entrenar cada modelo
        model_names = list(self.models.keys())  # Crear lista inmutable
        for name in model_names:
            model = self.models[name]
            try:
                self.logger.info(f"Entrenando modelo {name}...")
                
                # CORRECCIÓN: Usar solo argumentos básicos para sklearn
                if name == 'temporal':
                    # Modelo temporal personalizado
                    model.fit(X_train_df, y_train, X_val_df, y_val)
                elif name == 'xgb':
                    # XGBoost personalizado
                    model.fit(X_train_df, y_train, X_val_df, y_val)
                else:
                    # Modelos sklearn estándar (Ridge, ElasticNet, RandomForest)
                    model.fit(X_train_scaled, y_train)
                
                # Calcular score de validación si está disponible
                if X_val is not None and y_val is not None:
                    if name == 'temporal':
                        val_pred = model.predict(X_val_df)
                        # Ajustar tamaños si es necesario
                        min_len = min(len(val_pred), len(y_val))
                        val_score = r2_score(y_val.iloc[-min_len:], val_pred[-min_len:])
                    elif name == 'xgb':
                        # RegularizedXGBoost usa su propio predict method
                        val_pred = model.predict(X_val_df)
                        val_score = r2_score(y_val, val_pred)
                    else:
                        val_score = model.score(X_val_scaled, y_val)
                    
                    val_scores[name] = val_score
                    self.logger.info(f"  {name}: Score validación = {val_score:.4f}")
                
            except Exception as e:
                self.logger.error(f"Error entrenando {name}: {e}")
                # Remover modelo que falló
                if name in self.models:
                    del self.models[name]
        
        # Calcular pesos basados en performance de validación
        if val_scores:
            # Pesos proporcionales al score (solo scores positivos)
            positive_scores = {k: max(v, 0.01) for k, v in val_scores.items()}
            total_score = sum(positive_scores.values())
            self.weights = {k: v/total_score for k, v in positive_scores.items()}
        else:
            # Pesos uniformes si no hay validación
            n_models = len(self.models)
            self.weights = {name: 1.0/n_models for name in self.models.keys()}
        
        self.logger.info(f"Ensemble {self.task_type} entrenado:")
        self.logger.info(f"  Modelos activos: {list(self.models.keys())}")
        self.logger.info(f"  Pesos: {self.weights}")
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Hacer predicciones con ensemble ponderado."""
        if not self.models:
            raise ValueError("No hay modelos entrenados")
        
        X_scaled = self.scaler.transform(X)
        X_df = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
        
        predictions = []
        weights = []
        
        for name, model in self.models.items():
            try:
                if name == 'temporal':
                    pred = model.predict(X_df)
                    # Ajustar tamaño de predicción
                    if len(pred) < len(X):
                        # Rellenar con última predicción
                        pred = np.concatenate([np.full(len(X) - len(pred), pred[0]), pred])
                    elif len(pred) > len(X):
                        pred = pred[-len(X):]
                else:
                    pred = model.predict(X_scaled)
                
                predictions.append(pred)
                weights.append(self.weights[name])
                
            except Exception as e:
                self.logger.warning(f"Error en predicción de {name}: {e}")
                continue
        
        if not predictions:
            raise ValueError("No se pudieron generar predicciones")
        
        # Promedio ponderado
        predictions = np.array(predictions)
        weights = np.array(weights)
        weights = weights / weights.sum()  # Renormalizar
        
        return np.average(predictions, axis=0, weights=weights)
    
    def score(self, X: pd.DataFrame, y: pd.Series) -> float:
        """Calcular R² score promedio del ensemble."""
        if not self.models:
            raise ValueError("No hay modelos entrenados en el ensemble")
        
        predictions = self.predict(X)
        return r2_score(y, predictions)

# Alias para compatibilidad
RegularizedLSTM = RegularizedTimeSeriesModel
