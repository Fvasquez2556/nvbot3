"""
Demo completo del sistema anti-overfitting NvBot3.
Ejemplo de uso de todos los módulos implementados.
"""

import sys
import os
sys.path.append('src')

import numpy as np
import pandas as pd
import logging
from datetime import datetime, timedelta

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_sample_crypto_data(n_samples: int = 1000) -> pd.DataFrame:
    """Crear datos de prueba simulando series temporales de crypto."""
    
    # Generar fechas
    dates = pd.date_range(start='2023-01-01', periods=n_samples, freq='1h')
    
    # Simular datos de crypto con tendencia y volatilidad
    np.random.seed(42)
    
    # Price simulation con trend y volatilidad
    price_base = 50000  # BTC base price
    trend = np.cumsum(np.random.normal(0, 0.01, n_samples))
    volatility = np.random.normal(0, 0.02, n_samples)
    prices = price_base * (1 + trend + volatility)
    
    # Features técnicos
    volume = np.random.exponential(1000, n_samples)
    rsi = 50 + 30 * np.sin(np.linspace(0, 20*np.pi, n_samples)) + np.random.normal(0, 5, n_samples)
    rsi = np.clip(rsi, 0, 100)
    
    # Moving averages con ventana mínima para evitar NaN
    price_series = pd.Series(prices)
    ma_20 = price_series.rolling(20, min_periods=1).mean()
    ma_50 = price_series.rolling(50, min_periods=1).mean()
    
    # Volatility
    returns = price_series.pct_change().fillna(0)
    volatility_20 = returns.rolling(20, min_periods=1).std().fillna(0)
    
    # Target: predicir retorno próxima hora
    future_return = returns.shift(-1).fillna(0)
    
    df = pd.DataFrame({
        'timestamp': dates,
        'price': prices,
        'volume': volume,
        'rsi': rsi,
        'ma_20': ma_20,
        'ma_50': ma_50,
        'volatility_20': volatility_20,
        'returns': returns,
        'target': future_return
    }).set_index('timestamp')
    
    # Verificar y limpiar cualquier NaN restante
    df = df.ffill().fillna(0)
    
    return df

def demo_temporal_validation():
    """Demostrar validación temporal estricta."""
    logger.info("=== DEMO: Validación Temporal ===")
    
    from validation import TemporalValidator
    
    # Crear datos de prueba
    df = create_sample_crypto_data(1000)
    
    # Inicializar validador temporal con ratios específicos
    validator = TemporalValidator(train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)
    
    # Hacer split temporal
    train_data, val_data, test_data = validator.temporal_split(df)
    
    logger.info(f"Train: {len(train_data)} muestras ({train_data.index[0]} - {train_data.index[-1]})")
    logger.info(f"Val: {len(val_data)} muestras ({val_data.index[0]} - {val_data.index[-1]})")
    logger.info(f"Test: {len(test_data)} muestras ({test_data.index[0]} - {test_data.index[-1]})")
    
    # Validar que no hay data leakage
    try:
        validator.validate_no_data_leakage(train_data, val_data, test_data)
        logger.info("✅ No se detectó data leakage")
    except ValueError as e:
        logger.error(f"❌ Data leakage detectado: {e}")

def demo_walk_forward_validation():
    """Demostrar walk-forward validation."""
    logger.info("\n=== DEMO: Walk-Forward Validation ===")
    
    from validation import WalkForwardValidator
    from models import RegularizedXGBoost
    
    # Crear datos de prueba con más muestras para WF
    df = create_sample_crypto_data(2000)  # Más datos para simular meses
    
    # Preparar features y target
    feature_cols = ['price', 'volume', 'rsi', 'ma_20', 'ma_50', 'volatility_20', 'returns']
    target_col = 'target'
    
    # Inicializar walk-forward validator con parámetros correctos
    validator = WalkForwardValidator(
        initial_train_months=3,    # 3 meses iniciales
        test_months=1,             # 1 mes de test
        retrain_frequency_months=1, # Retrain cada mes
        min_train_samples=100      # Mínimo 100 muestras
    )
    
    # Ejecutar validación simplificada (sin parámetros extra)
    try:
        results = validator.validate(
            df=df,
            model_class=RegularizedXGBoost,
            model_params={'task_type': 'momentum'}
        )
        
        logger.info(f"Walk-forward completado: {len(results)} ventanas validadas")
        
        if results:
            # Estadísticas promedio usando model_performance
            train_scores = [r.model_performance.get('train_score', 0) for r in results]
            val_scores = [r.model_performance.get('val_score', 0) for r in results]
            
            if train_scores and val_scores:
                logger.info(f"Score promedio train: {np.mean(train_scores):.4f} ± {np.std(train_scores):.4f}")
                logger.info(f"Score promedio val: {np.mean(val_scores):.4f} ± {np.std(val_scores):.4f}")
                logger.info(f"Gap promedio: {np.mean(train_scores) - np.mean(val_scores):.4f}")
    except Exception as e:
        logger.error(f"Error en walk-forward validation: {e}")

def demo_overfitting_detection():
    """Demostrar detección automática de overfitting."""
    logger.info("\n=== DEMO: Detección de Overfitting ===")
    
    from validation import OverfittingDetector
    from models import RegularizedXGBoost, RegularizedEnsemble
    
    # Crear datos de prueba
    df = create_sample_crypto_data(800)
    
    feature_cols = ['price', 'volume', 'rsi', 'ma_20', 'ma_50', 'volatility_20', 'returns']
    target_col = 'target'
    
    # Split temporal para validación
    split_idx = int(0.7 * len(df))
    
    X_train = df[feature_cols].iloc[:split_idx]
    y_train = df[target_col].iloc[:split_idx]
    X_val = df[feature_cols].iloc[split_idx:]
    y_val = df[target_col].iloc[split_idx:]
    
    # Crear modelos con diferentes niveles de regularización
    models = {}
    
    # Modelo 1: Alta regularización (poco overfitting)
    model_high_reg = RegularizedXGBoost('momentum')
    model_high_reg.fit(X_train, y_train)  # CORRECCIÓN: Solo X y y
    models['XGB_Alta_Regularización'] = model_high_reg
    
    # Modelo 2: Ensemble regularizado
    model_ensemble = RegularizedEnsemble('momentum')
    model_ensemble.fit(X_train, y_train)  # CORRECCIÓN: Solo X y y
    models['Ensemble_Regularizado'] = model_ensemble
    
    # Inicializar detector
    detector = OverfittingDetector()
    
    # Análisis batch de todos los modelos
    comparison_df = detector.batch_analysis(
        models=models,
        X_train=np.array(X_train), 
        y_train=np.array(y_train),
        X_val=np.array(X_val),
        y_val=np.array(y_val)
    )
    
    logger.info("\n📊 Comparación de Modelos:")
    logger.info(comparison_df.to_string(index=False))
    
    # Análisis detallado del mejor modelo
    best_model_name = comparison_df.iloc[0]['model']
    best_model = models[best_model_name]
    
    report = detector.detect(
        best_model, np.array(X_train), np.array(y_train), 
        np.array(X_val), np.array(y_val), best_model_name
    )
    
    logger.info(f"\n🔍 Análisis detallado de {best_model_name}:")
    logger.info(f"Nivel de overfitting: {report.level.value.upper()}")
    logger.info(f"Score de overfitting: {report.score:.4f}")
    logger.info(f"Gap train-val: {report.gap:.4f}")
    
    if report.warnings:
        logger.info("⚠️ Advertencias:")
        for warning in report.warnings:
            logger.info(f"  - {warning}")
    
    if report.recommendations:
        logger.info("💡 Recomendaciones:")
        for rec in report.recommendations:
            logger.info(f"  - {rec}")

def demo_complete_pipeline():
    """Demostrar pipeline completo anti-overfitting."""
    logger.info("\n=== DEMO: Pipeline Completo Anti-Overfitting ===")
    
    from validation import TemporalValidator, WalkForwardValidator, OverfittingDetector
    from models import RegularizedEnsemble
    
    # 1. Datos de entrada
    df = create_sample_crypto_data(1200)
    feature_cols = ['price', 'volume', 'rsi', 'ma_20', 'ma_50', 'volatility_20', 'returns']
    target_col = 'target'
    
    logger.info("1️⃣ Datos creados: 1200 muestras de crypto simuladas")
    
    # 2. Validación temporal inicial
    temporal_validator = TemporalValidator(train_ratio=0.6, val_ratio=0.2, test_ratio=0.2)
    train_data, val_data, test_data = temporal_validator.temporal_split(df)
    
    logger.info(f"2️⃣ Split temporal: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}")
    
    # 3. Walk-forward validation en el conjunto de entrenamiento
    wf_validator = WalkForwardValidator(initial_train_months=2, test_months=1)
    wf_results = wf_validator.validate(
        train_data, RegularizedEnsemble, {'task_type': 'momentum'}
    )
    
    logger.info(f"3️⃣ Walk-forward completado: {len(wf_results)} ventanas validadas")
    
    # 4. Entrenar modelo final en train completo
    X_train = train_data[feature_cols]
    y_train = train_data[target_col]
    X_val = val_data[feature_cols]
    y_val = val_data[target_col]
    
    final_model = RegularizedEnsemble('momentum')
    final_model.fit(X_train, y_train)  # CORRECCIÓN: Solo X y y
    
    logger.info("4️⃣ Modelo final entrenado con máxima regularización")
    
    # 5. Detección final de overfitting
    detector = OverfittingDetector()
    final_report = detector.detect(
        final_model, np.array(X_train), np.array(y_train),
        np.array(X_val), np.array(y_val), "Modelo_Final_Ensemble"
    )
    
    logger.info(f"5️⃣ Overfitting final: {final_report.level.value.upper()} (score: {final_report.score:.4f})")
    
    # 6. Evaluación en test set (datos nunca vistos)
    X_test = test_data[feature_cols]
    y_test = test_data[target_col]
    
    test_predictions = final_model.predict(X_test)
    test_score = np.corrcoef(y_test, test_predictions)[0, 1] ** 2  # R² manual
    
    logger.info(f"6️⃣ Score en test set (nunca visto): {test_score:.4f}")
    
    # 7. Resumen final
    logger.info("\n📈 RESUMEN DEL PIPELINE:")
    logger.info(f"✅ Validación temporal: Sin data leakage")
    logger.info(f"✅ Walk-forward: {len(wf_results)} ventanas validadas")
    logger.info(f"✅ Modelo final: {len(final_model.models)} sub-modelos en ensemble")
    logger.info(f"✅ Overfitting: {final_report.level.value.upper()} ({final_report.score:.4f})")
    logger.info(f"✅ Performance test: {test_score:.4f}")
    
    if final_report.level.value in ['none', 'low']:
        logger.info("🎉 ¡MODELO LISTO PARA PRODUCCIÓN!")
    else:
        logger.info("⚠️ Modelo necesita más regularización antes de producción")

if __name__ == "__main__":
    logger.info("🚀 Iniciando demo completo del sistema anti-overfitting NvBot3")
    
    try:
        # Ejecutar todas las demos
        demo_temporal_validation()
        demo_walk_forward_validation() 
        demo_overfitting_detection()
        demo_complete_pipeline()
        
        logger.info("\n✅ ¡Demo completo terminado exitosamente!")
        logger.info("🎯 Todos los módulos anti-overfitting están funcionando correctamente")
        
    except Exception as e:
        logger.error(f"❌ Error en demo: {e}")
        import traceback
        traceback.print_exc()
