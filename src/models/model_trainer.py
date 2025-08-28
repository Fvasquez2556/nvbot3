"""
🤖 Model Trainer - NvBot3
Entrenador de modelos con sistema anti-overfitting para trading real

Modelos especializados:
🔥 Momentum Model (XGBoost): Detectar movimientos alcistas ≥5%
⚡ Rebound Model (Random Forest): Predecir rebotes 1-3%
📊 Regime Model (LSTM): Clasificar tendencia de mercado
🎯 Advanced Momentum (Ensemble): Momentum con filtros de volumen

Sistema Anti-Overfitting:
- Walk-Forward Analysis con múltiples períodos
- Cross-Validation temporal respetando secuencia
- Regularización adaptativa
- Feature Selection automática
- Validación Out-of-Sample estricta
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import numpy as np
import logging
import pickle
import joblib
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

# ML Libraries
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif, RFE
import xgboost as xgb
import tensorflow as tf
try:
    from tensorflow import keras  # type: ignore
    from tensorflow.keras import layers  # type: ignore
except ImportError:
    import keras  # type: ignore
    from keras import layers  # type: ignore

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

class AntiOverfittingValidator:
    """Sistema de validación anti-overfitting para trading"""
    
    def __init__(self, min_train_size: int = 5000, test_size: int = 1000):
        self.min_train_size = min_train_size
        self.test_size = test_size
        self.validation_results = {}
        
    def walk_forward_split(self, df: pd.DataFrame, n_splits: int = 5) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Walk-Forward Analysis: entrenamiento progresivo respetando temporalidad
        
        Cada split usa datos pasados para entrenar y datos futuros para validar
        """
        logger.info(f"🔄 Configurando Walk-Forward con {n_splits} splits")
        
        total_size = len(df)
        step_size = (total_size - self.min_train_size) // n_splits
        
        splits = []
        for i in range(n_splits):
            train_end = self.min_train_size + (i * step_size)
            test_start = train_end
            test_end = min(test_start + self.test_size, total_size)
            
            if test_end <= test_start:
                break
                
            train_idx = np.arange(0, train_end)
            test_idx = np.arange(test_start, test_end)
            
            splits.append((train_idx, test_idx))
            
            logger.info(f"   Split {i+1}: Train[0:{train_end}] Test[{test_start}:{test_end}]")
        
        return splits
    
    def temporal_cross_validation(self, df: pd.DataFrame, n_splits: int = 3) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Cross-validation temporal que respeta el orden cronológico"""
        tscv = TimeSeriesSplit(n_splits=n_splits, test_size=self.test_size)
        return list(tscv.split(df))
    
    def validate_model_performance(self, model, X_train, X_test, y_train, y_test, model_name: str) -> Dict:
        """Validación comprehensiva de performance del modelo"""
        try:
            # Entrenar modelo
            model.fit(X_train, y_train)
            
            # Predicciones
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)
            
            # Métricas de entrenamiento
            train_accuracy = (y_pred_train == y_train).mean()
            
            # Métricas de validación
            test_accuracy = (y_pred_test == y_test).mean()
            
            # Detectar overfitting
            overfitting_gap = train_accuracy - test_accuracy
            is_overfitting = overfitting_gap > 0.1  # Gap > 10%
            
            # Métricas adicionales para targets binarios
            additional_metrics = {}
            if len(np.unique(y_test)) == 2:
                try:
                    if hasattr(model, 'predict_proba'):
                        y_prob = model.predict_proba(X_test)[:, 1]
                        additional_metrics['auc_score'] = roc_auc_score(y_test, y_prob)
                    else:
                        additional_metrics['auc_score'] = 0.5
                except:
                    additional_metrics['auc_score'] = 0.5
            
            validation_result = {
                'model_name': model_name,
                'train_accuracy': float(train_accuracy),
                'test_accuracy': float(test_accuracy),
                'overfitting_gap': float(overfitting_gap),
                'is_overfitting': bool(is_overfitting),
                'n_train_samples': len(X_train),
                'n_test_samples': len(X_test),
                'target_distribution': dict(pd.Series(y_test).value_counts()),
                **additional_metrics
            }
            
            return validation_result
            
        except Exception as e:
            logger.error(f"❌ Error validando {model_name}: {e}")
            return {
                'model_name': model_name,
                'error': str(e),
                'is_overfitting': True  # Conservative approach
            }

class FeatureSelector:
    """Selector automático de features para evitar overfitting"""
    
    def __init__(self, max_features: int = 50):
        self.max_features = max_features
        self.selected_features = {}
        
    def select_features_univariate(self, X: pd.DataFrame, y: pd.Series, k: Optional[int] = None) -> List[str]:
        """Selección univariada con f_classif"""
        if k is None:
            k = min(self.max_features, X.shape[1])
            
        selector = SelectKBest(score_func=f_classif, k=k)
        selector.fit(X, y)
        
        feature_scores = pd.DataFrame({
            'feature': X.columns,
            'score': selector.scores_
        }).sort_values('score', ascending=False)
        
        selected_features = feature_scores.head(k)['feature'].tolist()
        
        logger.info(f"📊 Selección univariada: {len(selected_features)} features")
        
        return selected_features
    
    def select_features_rfe(self, X: pd.DataFrame, y: pd.Series, estimator, n_features: Optional[int] = None) -> List[str]:
        """Recursive Feature Elimination"""
        if n_features is None:
            n_features = min(self.max_features, X.shape[1] // 2)
            
        selector = RFE(estimator=estimator, n_features_to_select=n_features)
        selector.fit(X, y)
        
        selected_features = X.columns[selector.support_].tolist()
        
        logger.info(f"🔄 RFE: {len(selected_features)} features seleccionadas")
        
        return selected_features
    
    def select_features_importance(self, X: pd.DataFrame, y: pd.Series, model_type: str = 'xgb') -> List[str]:
        """Selección basada en importancia del modelo"""
        try:
            if model_type == 'xgb':
                model = xgb.XGBClassifier(random_state=42, eval_metric='logloss')
            else:
                model = RandomForestClassifier(n_estimators=100, random_state=42)
            
            model.fit(X, y)
            
            feature_importance = pd.DataFrame({
                'feature': X.columns,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            # Seleccionar top features
            n_select = min(self.max_features, len(X.columns))
            selected_features = feature_importance.head(n_select)['feature'].tolist()
            
            logger.info(f"⭐ Importancia {model_type.upper()}: {len(selected_features)} features")
            
            return selected_features
            
        except Exception as e:
            logger.error(f"❌ Error en selección por importancia: {e}")
            return X.columns.tolist()[:self.max_features]

class ModelTrainer:
    """Entrenador principal de modelos para NvBot3"""
    
    def __init__(self, data_path: str = "data/processed", models_path: str = "data/models"):
        self.data_path = Path(data_path)
        self.models_path = Path(models_path)
        self.models_path.mkdir(exist_ok=True)
        
        # Componentes del sistema
        self.validator = AntiOverfittingValidator()
        self.feature_selector = FeatureSelector()
        
        # Modelos entrenados
        self.trained_models = {}
        self.model_metrics = {}
        
        # Configuraciones por tipo de modelo
        self.model_configs = {
            'momentum': {
                'model_type': 'xgb',
                'target_column': 'momentum_target',
                'max_features': 40,
                'description': '🔥 Momentum Model: Detectar movimientos ≥5%'
            },
            'rebound': {
                'model_type': 'rf',
                'target_column': 'rebound_target',
                'max_features': 35,
                'description': '⚡ Rebound Model: Predecir rebotes 1-3%'
            },
            'regime': {
                'model_type': 'lstm',
                'target_column': 'regime_target',
                'max_features': 45,
                'description': '📊 Regime Model: Clasificar tendencia de mercado'
            },
            'momentum_advanced': {
                'model_type': 'ensemble',
                'target_column': 'momentum_advanced_target',
                'max_features': 50,
                'description': '🎯 Advanced Momentum: Momentum con filtros'
            }
        }
    
    def load_training_data(self, symbol: str, timeframe: str) -> Optional[pd.DataFrame]:
        """Cargar datos con features y targets para entrenamiento"""
        try:
            filename = f"{symbol}_{timeframe}_with_targets.csv"
            filepath = self.data_path / filename
            
            if not filepath.exists():
                logger.error(f"❌ Archivo no encontrado: {filepath}")
                return None
            
            df = pd.read_csv(filepath, index_col=0, parse_dates=True)
            
            # Verificar que tenemos todos los targets necesarios
            required_targets = ['momentum_target', 'rebound_target', 'regime_target', 'momentum_advanced_target']
            missing_targets = [col for col in required_targets if col not in df.columns]
            
            if missing_targets:
                logger.error(f"❌ Targets faltantes: {missing_targets}")
                return None
            
            logger.info(f"✅ Datos cargados: {len(df)} registros, {len(df.columns)} columnas")
            
            return df
            
        except Exception as e:
            logger.error(f"❌ Error cargando datos: {e}")
            return None
    
    def prepare_features_targets(self, df: pd.DataFrame, model_type: str) -> Tuple[pd.DataFrame, pd.Series]:
        """Preparar features y targets para un modelo específico"""
        config = self.model_configs[model_type]
        target_column = config['target_column']
        
        # Excluir columnas que no son features
        exclude_columns = [
            'symbol', 'timeframe', 'timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time',
            'quote_asset_volume', 'number_of_trades', 
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume',
            'momentum_target', 'rebound_target', 'regime_target', 'momentum_advanced_target'
        ]
        
        feature_columns = [col for col in df.columns if col not in exclude_columns]
        
        X = df[feature_columns].copy()
        y = df[target_column].copy()
        
        # Limpiar datos
        X = X.ffill().fillna(0)
        X = X.replace([np.inf, -np.inf], 0)
        
        logger.info(f"📊 Features preparadas: {len(feature_columns)} columnas")
        logger.info(f"🎯 Target '{target_column}': {y.value_counts().to_dict()}")
        
        return X, y
    
    def create_xgb_model(self, X_train: pd.DataFrame, y_train: pd.Series) -> xgb.XGBClassifier:
        """Crear modelo XGBoost optimizado"""
        
        # Detectar si es binario o multiclass
        n_classes = len(np.unique(y_train))
        
        if n_classes == 2:
            objective = 'binary:logistic'
            eval_metric = 'logloss'
        else:
            objective = 'multi:softprob'
            eval_metric = 'mlogloss'
        
        model = xgb.XGBClassifier(
            objective=objective,
            eval_metric=eval_metric,
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1
        )
        
        logger.info(f"🤖 XGBoost configurado: {objective}, {n_classes} clases")
        
        return model
    
    def create_rf_model(self, X_train: pd.DataFrame, y_train: pd.Series) -> RandomForestClassifier:
        """Crear modelo Random Forest optimizado"""
        
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=10,
            min_samples_leaf=5,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1,
            class_weight='balanced'
        )
        
        logger.info("🌲 Random Forest configurado")
        
        return model
    
    def create_lstm_model(self, X_train: pd.DataFrame, y_train: pd.Series, sequence_length: int = 20) -> Any:
        """Crear modelo LSTM optimizado para régimen de mercado"""
        
        n_features = X_train.shape[1]
        n_classes = len(np.unique(y_train))
        
        # 🛡️ CONFIGURACIÓN ANTI-OVERFITTING MÁXIMA PARA REGIME
        model = keras.Sequential([
            # Primera capa LSTM MUY PEQUEÑA (reducido de 64 a 24)
            layers.LSTM(24, return_sequences=True, input_shape=(sequence_length, n_features),
                       dropout=0.8,           # 🔼 DROPOUT MÁXIMO (era 0.3)
                       recurrent_dropout=0.6, # 🔼 RECURRENT DROPOUT ALTO (era 0.3)
                       kernel_regularizer=keras.regularizers.l2(0.1), # 🔼 L2 ALTO
                       recurrent_regularizer=keras.regularizers.l2(0.05)),
            layers.BatchNormalization(),
            
            # Segunda capa LSTM AÚN MÁS PEQUEÑA (reducido de 32 a 12)  
            layers.LSTM(12, return_sequences=False,
                       dropout=0.8,           # 🔼 DROPOUT MÁXIMO
                       recurrent_dropout=0.6, # 🔼 RECURRENT DROPOUT ALTO
                       kernel_regularizer=keras.regularizers.l2(0.1),
                       recurrent_regularizer=keras.regularizers.l2(0.05)),
            layers.BatchNormalization(),
            
            # Capas densas MÍNIMAS (reducido de 32 a 8)
            layers.Dense(8, activation='relu',
                        kernel_regularizer=keras.regularizers.l2(0.1)),
            layers.Dropout(0.7),              # 🔼 DROPOUT ALTÍSIMO (era 0.2)
            
            layers.Dense(4, activation='relu',
                        kernel_regularizer=keras.regularizers.l2(0.1)),
            layers.Dropout(0.6),
            
            # Output layer para 3 clases (Bear/Side/Bull)
            layers.Dense(n_classes, activation='softmax' if n_classes > 2 else 'sigmoid')
        ])
        
        # Optimizador SÚPER CONSERVADOR
        optimizer = keras.optimizers.Adam(
            learning_rate=0.0003,      # 🔽 LEARNING RATE MÁS LENTO (era 0.001)
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-7
        )
        
        loss = 'sparse_categorical_crossentropy' if n_classes > 2 else 'binary_crossentropy'
        
        model.compile(
            optimizer=optimizer,  # type: ignore
            loss=loss,
            metrics=['accuracy']
        )
        
        logger.info(f"🧠 LSTM ANTI-OVERFITTING configurado: {n_classes} clases")
        logger.info(f"   🔽 LSTM units: 24→12 (reducido drásticamente)")
        logger.info(f"   🔼 Dropout: 0.8/0.6 (máximo)")
        logger.info(f"   🔼 L2 regularization: 0.1/0.05 (alto)")
        logger.info(f"   🔽 Learning rate: 0.0003 (conservador)")
        
        return model
    
    def create_ensemble_model(self, X_train: pd.DataFrame, y_train: pd.Series) -> Dict[str, Any]:
        """Crear modelo ensemble combinando XGBoost y Random Forest"""
        
        xgb_model = self.create_xgb_model(X_train, y_train)
        rf_model = self.create_rf_model(X_train, y_train)
        
        ensemble = {
            'xgb': xgb_model,
            'rf': rf_model,
            'weights': [0.7, 0.3]  # XGBoost tiene más peso
        }
        
        logger.info("🎯 Ensemble configurado: XGBoost (70%) + Random Forest (30%)")
        
        return ensemble
    
    def train_single_model(self, df: pd.DataFrame, model_type: str, symbol: str, timeframe: str) -> Dict:
        """Entrenar un modelo específico con validación anti-overfitting"""
        
        logger.info(f"🤖 === ENTRENANDO {model_type.upper()} MODEL: {symbol}_{timeframe} ===")
        
        config = self.model_configs[model_type]
        logger.info(config['description'])
        
        # Preparar datos
        X, y = self.prepare_features_targets(df, model_type)
        
        # Verificar que tenemos suficientes datos
        if len(X) < self.validator.min_train_size:
            logger.error(f"❌ Insuficientes datos: {len(X)} < {self.validator.min_train_size}")
            return {'success': False, 'error': 'Insufficient data'}
        
        # Selección de features
        logger.info("🔍 Seleccionando features...")
        if model_type == 'regime':
            # 🎯 SELECCIÓN SUPER AGRESIVA PARA REGIME
            max_features_regime = min(25, config['max_features'])  # 🔽 REDUCIDO drásticamente
            selected_features = self.feature_selector.select_features_importance(X, y, 'xgb')
            selected_features = selected_features[:max_features_regime]  # Solo tomar las mejores 25
            logger.info(f"⭐ Régimen LSTM: {len(selected_features)} features (máximo anti-overfitting)")
        else:
            # Para otros modelos, usar selección univariada
            selected_features = self.feature_selector.select_features_univariate(X, y, config['max_features'])
        
        X_selected = X[selected_features]
        
        # Walk-Forward validation
        logger.info("🔄 Iniciando Walk-Forward Analysis...")
        splits = self.validator.walk_forward_split(X_selected)
        
        validation_results = []
        best_model = None
        best_score = -1
        scaler = None  # Inicializar para LSTM
        sequence_length = 20  # Valor por defecto
        
        for i, (train_idx, test_idx) in enumerate(splits):
            logger.info(f"📊 Validación {i+1}/{len(splits)}")
            
            X_train = X_selected.iloc[train_idx]
            X_test = X_selected.iloc[test_idx]
            y_train = y.iloc[train_idx]
            y_test = y.iloc[test_idx]
            
            # Normalización para LSTM
            if model_type == 'regime':
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                
                # 🎯 SECUENCIAS MÁS CORTAS PARA ANTI-OVERFITTING
                sequence_length = 12  # 🔽 REDUCIDO de 20 a 12
                X_train_lstm = self._create_sequences(X_train_scaled, sequence_length)
                X_test_lstm = self._create_sequences(X_test_scaled, sequence_length)
                
                # Ajustar targets para las secuencias LSTM
                # Las secuencias empiezan desde sequence_length, así que los targets también
                y_train_lstm = y_train.iloc[sequence_length:].values
                y_test_lstm = y_test.iloc[sequence_length:].values
                
                # Asegurar que las dimensiones coincidan exactamente
                min_train_samples = min(len(X_train_lstm), len(y_train_lstm))
                min_test_samples = min(len(X_test_lstm), len(y_test_lstm))
                
                X_train_lstm = X_train_lstm[:min_train_samples]
                y_train_lstm = y_train_lstm[:min_train_samples]
                X_test_lstm = X_test_lstm[:min_test_samples]
                y_test_lstm = y_test_lstm[:min_test_samples]
                
                # Crear y entrenar modelo LSTM
                model = self.create_lstm_model(X_selected, y, sequence_length)
                
                # 🎯 ENTRENAMIENTO ANTI-OVERFITTING REGIME
                model.fit(
                    X_train_lstm, y_train_lstm,
                    epochs=30,               # 🔽 REDUCIDO de 50 a 30
                    batch_size=12,           # 🔽 REDUCIDO de 32 a 12
                    validation_split=0.2,
                    verbose=0,
                    callbacks=[
                        keras.callbacks.EarlyStopping(
                            patience=5,      # 🔽 MUY AGRESIVO (era 10)
                            restore_best_weights=True,
                            min_delta=0.002  # type: ignore # 🔼 MAYOR MEJORA REQUERIDA
                        ),
                        keras.callbacks.ReduceLROnPlateau(
                            factor=0.2,     # 🔽 REDUCCIÓN DRÁSTICA (era 0.5)
                            patience=3,     # 🔽 IMPACIENTE (era 5)
                            min_lr=0.000001 # 🔽 LR SÚPER MÍNIMO
                        )
                    ]
                )
                
                # Validación especial para LSTM
                val_result = self._validate_lstm_model(model, X_train_lstm, X_test_lstm, y_train_lstm, y_test_lstm, f"{model_type}_{i}")
                
            elif model_type == 'ensemble':
                # Crear ensemble
                ensemble = self.create_ensemble_model(X_train, y_train)
                
                # Entrenar componentes
                ensemble['xgb'].fit(X_train, y_train)
                ensemble['rf'].fit(X_train, y_train)
                
                # Validación especial para ensemble
                val_result = self._validate_ensemble_model(ensemble, X_train, X_test, y_train, y_test, f"{model_type}_{i}")
                model = ensemble
                
            else:
                # Modelos tradicionales (XGB, RF)
                if config['model_type'] == 'xgb':
                    model = self.create_xgb_model(X_train, y_train)
                else:
                    model = self.create_rf_model(X_train, y_train)
                
                val_result = self.validator.validate_model_performance(
                    model, X_train, X_test, y_train, y_test, f"{model_type}_{i}"
                )
            
            validation_results.append(val_result)
            
            # Seleccionar mejor modelo
            if 'test_accuracy' in val_result and val_result['test_accuracy'] > best_score:
                best_score = val_result['test_accuracy']
                best_model = model
                
                # Guardar scaler para LSTM
                if model_type == 'regime':
                    best_model = {
                        'model': model,
                        'scaler': scaler,
                        'sequence_length': sequence_length
                    }
        
        # Resumen de validación
        avg_train_acc = np.mean([r.get('train_accuracy', 0) for r in validation_results])
        avg_test_acc = np.mean([r.get('test_accuracy', 0) for r in validation_results])
        avg_overfitting = np.mean([r.get('overfitting_gap', 0) for r in validation_results])
        
        training_result = {
            'success': True,
            'model_type': model_type,
            'symbol': symbol,
            'timeframe': timeframe,
            'best_model': best_model,
            'selected_features': selected_features,
            'validation_results': validation_results,
            'summary': {
                'avg_train_accuracy': float(avg_train_acc),
                'avg_test_accuracy': float(avg_test_acc),
                'avg_overfitting_gap': float(avg_overfitting),
                'is_overfitting': avg_overfitting > 0.1,
                'n_validations': len(validation_results),
                'best_test_score': float(best_score)
            }
        }
        
        # Guardar modelo
        model_key = f"{symbol}_{timeframe}_{model_type}"
        self.trained_models[model_key] = best_model
        self.model_metrics[model_key] = training_result
        
        # Guardar modelo individual
        self._save_single_model(symbol, timeframe, model_type)
        
        logger.info(f"✅ {model_type.upper()} entrenado: Test Acc={best_score:.3f}, Overfitting={avg_overfitting:.3f}")
        
        return training_result
    
    def _create_sequences(self, data: np.ndarray, sequence_length: int) -> np.ndarray:
        """Crear secuencias para LSTM"""
        if len(data) < sequence_length:
            logger.warning(f"⚠️ Datos insuficientes para secuencias: {len(data)} < {sequence_length}")
            return np.array([])
            
        sequences = []
        for i in range(sequence_length, len(data)):
            sequences.append(data[i-sequence_length:i])
        
        return np.array(sequences)
    
    def _validate_lstm_model(self, model, X_train, X_test, y_train, y_test, model_name: str) -> Dict:
        """Validación especial para modelos LSTM"""
        try:
            # Predicciones
            y_pred_train_proba = model.predict(X_train, verbose=0)
            y_pred_test_proba = model.predict(X_test, verbose=0)
            
            y_pred_train = np.argmax(y_pred_train_proba, axis=1)
            y_pred_test = np.argmax(y_pred_test_proba, axis=1)
            
            # Asegurar que las dimensiones coincidan
            min_train_len = min(len(y_pred_train), len(y_train))
            min_test_len = min(len(y_pred_test), len(y_test))
            
            y_pred_train = y_pred_train[:min_train_len]
            y_train_adj = y_train.iloc[:min_train_len].values if hasattr(y_train, 'iloc') else y_train[:min_train_len]
            
            y_pred_test = y_pred_test[:min_test_len]
            y_test_adj = y_test.iloc[:min_test_len].values if hasattr(y_test, 'iloc') else y_test[:min_test_len]
            
            # Métricas
            train_accuracy = (y_pred_train == y_train_adj).mean()
            test_accuracy = (y_pred_test == y_test_adj).mean()
            overfitting_gap = train_accuracy - test_accuracy
            
            return {
                'model_name': model_name,
                'train_accuracy': float(train_accuracy),
                'test_accuracy': float(test_accuracy),
                'overfitting_gap': float(overfitting_gap),
                'is_overfitting': overfitting_gap > 0.1,
                'n_train_samples': min_train_len,
                'n_test_samples': min_test_len
            }
            
        except Exception as e:
            logger.error(f"❌ Error validando LSTM: {e}")
            return {'model_name': model_name, 'error': str(e), 'is_overfitting': True}
    
    def _validate_ensemble_model(self, ensemble, X_train, X_test, y_train, y_test, model_name: str) -> Dict:
        """Validación especial para modelos ensemble"""
        try:
            # Predicciones de cada componente
            xgb_pred_train = ensemble['xgb'].predict_proba(X_train)
            rf_pred_train = ensemble['rf'].predict_proba(X_train)
            
            xgb_pred_test = ensemble['xgb'].predict_proba(X_test)
            rf_pred_test = ensemble['rf'].predict_proba(X_test)
            
            # Combinación ponderada
            weights = ensemble['weights']
            ensemble_pred_train = weights[0] * xgb_pred_train + weights[1] * rf_pred_train
            ensemble_pred_test = weights[0] * xgb_pred_test + weights[1] * rf_pred_test
            
            # Predicciones finales
            y_pred_train = np.argmax(ensemble_pred_train, axis=1)
            y_pred_test = np.argmax(ensemble_pred_test, axis=1)
            
            # Métricas
            train_accuracy = (y_pred_train == y_train).mean()
            test_accuracy = (y_pred_test == y_test).mean()
            overfitting_gap = train_accuracy - test_accuracy
            
            return {
                'model_name': model_name,
                'train_accuracy': float(train_accuracy),
                'test_accuracy': float(test_accuracy),
                'overfitting_gap': float(overfitting_gap),
                'is_overfitting': overfitting_gap > 0.1,
                'n_train_samples': len(X_train),
                'n_test_samples': len(X_test)
            }
            
        except Exception as e:
            logger.error(f"❌ Error validando ensemble: {e}")
            return {'model_name': model_name, 'error': str(e), 'is_overfitting': True}
    
    def train_all_models(self, symbol: str, timeframe: str) -> Dict:
        """Entrenar todos los modelos para un símbolo y timeframe"""
        logger.info(f"🚀 === ENTRENAMIENTO COMPLETO: {symbol}_{timeframe} ===")
        
        # Cargar datos
        df = self.load_training_data(symbol, timeframe)
        if df is None:
            return {'success': False, 'error': 'Could not load data'}
        
        results = {
            'symbol': symbol,
            'timeframe': timeframe,
            'models': {},
            'summary': {},
            'success': True
        }
        
        # Entrenar cada modelo
        for model_type in ['momentum', 'rebound', 'regime', 'momentum_advanced']:
            try:
                logger.info(f"\\n{'='*60}")
                result = self.train_single_model(df, model_type, symbol, timeframe)
                results['models'][model_type] = result
                
            except Exception as e:
                logger.error(f"❌ Error entrenando {model_type}: {e}")
                results['models'][model_type] = {'success': False, 'error': str(e)}
                results['success'] = False
        
        # Resumen general
        successful_models = [m for m in results['models'].values() if m.get('success', False)]
        
        results['summary'] = {
            'total_models': len(self.model_configs),
            'successful_models': len(successful_models),
            'success_rate': len(successful_models) / len(self.model_configs),
            'avg_test_accuracy': np.mean([m['summary']['best_test_score'] for m in successful_models]) if successful_models else 0,
            'models_with_overfitting': sum(1 for m in successful_models if m['summary']['is_overfitting'])
        }
        
        # Guardar modelos entrenados
        self.save_models(symbol, timeframe)
        
        logger.info(f"\\n🎯 === ENTRENAMIENTO COMPLETADO ===")
        logger.info(f"✅ Modelos exitosos: {results['summary']['successful_models']}/{results['summary']['total_models']}")
        logger.info(f"📊 Precisión promedio: {results['summary']['avg_test_accuracy']:.3f}")
        
        return results
    
    def save_models(self, symbol: str, timeframe: str):
        """Guardar modelos entrenados"""
        try:
            base_name = f"{symbol}_{timeframe}"
            
            for model_type in self.model_configs.keys():
                model_key = f"{base_name}_{model_type}"
                
                if model_key in self.trained_models:
                    self._save_single_model(symbol, timeframe, model_type)
                    
        except Exception as e:
            logger.error(f"❌ Error guardando modelos: {e}")
    
    def _save_single_model(self, symbol: str, timeframe: str, model_type: str):
        """Guardar un modelo específico"""
        try:
            model_key = f"{symbol}_{timeframe}_{model_type}"
            
            if model_key in self.trained_models:
                model_file = self.models_path / f"{model_key}.pkl"
                
                # Guardar modelo
                with open(model_file, 'wb') as f:
                    pickle.dump(self.trained_models[model_key], f)
                
                logger.info(f"💾 Modelo guardado: {model_file}")
                
                # Guardar métricas
                metrics_file = self.models_path / f"{model_key}_metrics.pkl"
                with open(metrics_file, 'wb') as f:
                    pickle.dump(self.model_metrics[model_key], f)
                
        except Exception as e:
            logger.error(f"❌ Error guardando modelo {model_type}: {e}")
    
    def print_training_summary(self, results: Dict):
        """Imprimir resumen detallado del entrenamiento"""
        print(f"\\n{'='*100}")
        print(f"🤖 RESUMEN DE ENTRENAMIENTO: {results['symbol']}_{results['timeframe']}")
        print(f"{'='*100}")
        
        summary = results['summary']
        print(f"📊 Modelos entrenados: {summary['successful_models']}/{summary['total_models']}")
        print(f"🎯 Tasa de éxito: {summary['success_rate']:.1%}")
        print(f"📈 Precisión promedio: {summary['avg_test_accuracy']:.3f}")
        print(f"⚠️ Modelos con overfitting: {summary['models_with_overfitting']}")
        
        print(f"\\n📋 DETALLE POR MODELO:")
        for model_type, result in results['models'].items():
            if result.get('success', False):
                config = self.model_configs[model_type]
                summary_data = result['summary']
                
                status = "⚠️ OVERFITTING" if summary_data['is_overfitting'] else "✅ OK"
                
                print(f"\\n{config['description']}")
                print(f"  Status: {status}")
                print(f"  Test Accuracy: {summary_data['best_test_score']:.3f}")
                print(f"  Overfitting Gap: {summary_data['avg_overfitting_gap']:.3f}")
                print(f"  Features: {len(result['selected_features'])}")
                print(f"  Validations: {summary_data['n_validations']}")
            else:
                print(f"\\n❌ {model_type.upper()}: {result.get('error', 'Unknown error')}")
        
        print(f"{'='*100}")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Model Trainer NvBot3 - Sistema Anti-Overfitting')
    parser.add_argument('--symbol', type=str, help='Símbolo específico (ej: BTCUSDT)')
    parser.add_argument('--timeframe', type=str, help='Timeframe específico (ej: 5m)')
    parser.add_argument('--model', type=str, choices=['momentum', 'rebound', 'regime', 'momentum_advanced'], 
                       help='Entrenar modelo específico')
    parser.add_argument('--quick-test', action='store_true', help='Entrenamiento rápido con pocos datos')
    
    args = parser.parse_args()
    
    # Crear trainer
    trainer = ModelTrainer()
    
    if args.symbol and args.timeframe:
        if args.model:
            # Entrenar modelo específico
            df = trainer.load_training_data(args.symbol, args.timeframe)
            if df is not None:
                result = trainer.train_single_model(df, args.model, args.symbol, args.timeframe)
                if result['success']:
                    print(f"\\n✅ {args.model.upper()} entrenado exitosamente")
                else:
                    print(f"\\n❌ Error entrenando {args.model}")
        else:
            # Entrenar todos los modelos
            results = trainer.train_all_models(args.symbol, args.timeframe)
            trainer.print_training_summary(results)
    else:
        print("❌ Especifica --symbol y --timeframe")
        print("\\nEjemplos:")
        print("  python src/models/model_trainer.py --symbol BTCUSDT --timeframe 5m")
        print("  python src/models/model_trainer.py --symbol BTCUSDT --timeframe 5m --model momentum")


if __name__ == "__main__":
    print("🤖 === MODEL TRAINER NVBOT3 - SISTEMA ANTI-OVERFITTING ===")
    print("🎯 Entrenando modelos especializados para trading con validación robusta")
    print("="*80)
    
    main()
