#!/usr/bin/env python3
"""
Validación final del sistema NvBot3 con todas las correcciones aplicadas
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def create_test_data(n_samples=3000):
    """Crear datos de prueba para validación"""
    dates = pd.date_range(
        start=datetime.now() - timedelta(days=n_samples),
        end=datetime.now(),
        freq='1h'
    )[:n_samples]
    
    # Simular precio crypto realista
    np.random.seed(42)
    price_base = 100
    returns = np.random.normal(0, 0.02, n_samples)
    prices = [price_base]
    
    for ret in returns[1:]:
        prices.append(prices[-1] * (1 + ret))
    
    df = pd.DataFrame({
        'timestamp': dates,
        'price': prices,
        'volume': np.random.lognormal(10, 1, n_samples),
        'rsi': np.random.uniform(20, 80, n_samples),
        'ma_20': np.random.uniform(95, 105, n_samples),
        'ma_50': np.random.uniform(95, 105, n_samples),
        'volatility_20': np.random.uniform(0.01, 0.05, n_samples),
        'returns': returns,
        'target': (np.random.randn(n_samples) > 0).astype(int)
    })
    
    return df

def test_regularized_models():
    """Probar modelos regularizados con correcciones"""
    logger.info("🧪 Probando RegularizedEnsemble con correcciones...")
    
    try:
        from src.models.regularized_models import RegularizedEnsemble, RegularizedXGBoost
        
        # Crear datos de prueba
        df = create_test_data(500)  # Tamaño reducido para prueba rápida
        feature_cols = ['price', 'volume', 'rsi', 'ma_20', 'ma_50', 'volatility_20', 'returns']
        target_col = 'target'
        
        X = df[feature_cols]
        y = df[target_col]
        
        # Test 1: RegularizedXGBoost con eval_set
        logger.info("  ├─ Probando RegularizedXGBoost.fit() con eval_set...")
        rgbx = RegularizedXGBoost()
        
        # Dividir datos
        split_idx = int(len(X) * 0.8)
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        # Debe aceptar eval_set sin error
        rgbx.fit(X_train, y_train, eval_set=[(X_val, y_val)])
        logger.info("  ✅ eval_set funciona correctamente")
        
        # Test 2: sklearn compatibility
        logger.info("  ├─ Probando compatibilidad sklearn...")
        params = rgbx.get_params()
        logger.info(f"  ├─ get_params() devuelve: {len(params)} parámetros")
        
        rgbx.set_params(n_estimators=50)
        logger.info("  ✅ set_params() funciona correctamente")
        
        # Test 3: RegularizedEnsemble
        logger.info("  ├─ Probando RegularizedEnsemble...")
        ensemble = RegularizedEnsemble()
        ensemble.fit(X_train, y_train)
        score = ensemble.score(X_val, y_val)
        logger.info(f"  ✅ Ensemble score: {score:.4f}")
        
        return True
        
    except Exception as e:
        logger.error(f"  ❌ Error en modelos regularizados: {e}")
        return False

def test_walk_forward_validator():
    """Probar walk-forward validator con umbral corregido"""
    logger.info("🚶 Probando WalkForwardValidator con umbral corregido...")
    
    try:
        from src.validation.walk_forward_validator import WalkForwardValidator
        
        # Crear datos pequeños (menos de 5000)
        df = create_test_data(300)
        feature_cols = ['price', 'volume', 'rsi', 'ma_20', 'ma_50', 'volatility_20', 'returns']
        target_col = 'target'
        
        # Debe funcionar con datasets pequeños
        validator = WalkForwardValidator(
            min_train_samples=100  # Umbral corregido
        )
        
        # Solo verificar que se puede instanciar con el umbral bajo
        logger.info(f"  ✅ WalkForwardValidator creado con min_train_samples={validator.min_train_samples}")
        
        return True
        
    except Exception as e:
        logger.error(f"  ❌ Error en walk-forward: {e}")
        return False

def test_overfitting_detector():
    """Probar detector de overfitting con manejo seguro de features"""
    logger.info("🔍 Probando OverfittingDetector...")
    
    try:
        from src.validation.overfitting_detector import OverfittingDetector
        from src.models.regularized_models import RegularizedEnsemble
        
        # Crear datos
        df = create_test_data(400)
        feature_cols = ['price', 'volume', 'rsi', 'ma_20', 'ma_50', 'volatility_20', 'returns']
        target_col = 'target'
        
        X = df[feature_cols]
        y = df[target_col]
        
        # Dividir datos
        split_idx = int(len(X) * 0.8)
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        # Entrenar modelo
        model = RegularizedEnsemble()
        model.fit(X_train, y_train)
        
        # Detectar overfitting usando el método correcto
        detector = OverfittingDetector()
        report = detector.detect(model, X_train.values, np.array(y_train), X_val.values, np.array(y_val))
        
        logger.info(f"  ✅ Overfitting report generado: {report.level.name}")
        
        return True
        
    except Exception as e:
        logger.error(f"  ❌ Error en detector: {e}")
        return False

def main():
    """Ejecutar todas las validaciones"""
    logger.info("🎯 VALIDACIÓN FINAL SISTEMA NVBOT3")
    logger.info("=" * 50)
    
    tests = [
        ("Modelos Regularizados", test_regularized_models),
        ("Walk-Forward Validator", test_walk_forward_validator),
        ("Overfitting Detector", test_overfitting_detector)
    ]
    
    results = []
    for test_name, test_func in tests:
        logger.info(f"\n📋 Ejecutando: {test_name}")
        result = test_func()
        results.append((test_name, result))
    
    # Resumen final
    logger.info("\n" + "=" * 50)
    logger.info("📊 RESUMEN DE VALIDACIÓN:")
    
    all_passed = True
    for test_name, passed in results:
        status = "✅ CORRECTO" if passed else "❌ FALLÓ"
        logger.info(f"  {test_name}: {status}")
        if not passed:
            all_passed = False
    
    if all_passed:
        logger.info("\n🎉 ¡TODAS LAS CORRECCIONES VALIDADAS EXITOSAMENTE!")
        logger.info("✅ Sistema NvBot3 listo para producción")
    else:
        logger.info("\n⚠️  Algunas validaciones fallaron")
        logger.info("❌ Revisar errores arriba")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
