#!/usr/bin/env python3
"""
Test de correcciones del sistema anti-overfitting
Ejecutar después de aplicar las correcciones
"""

import sys
import numpy as np
import warnings
warnings.filterwarnings('ignore')

def test_ensemble_score():
    """Test crítico: Ensemble debe tener método score."""
    try:
        sys.path.append('src')
        from models.regularized_models import RegularizedEnsemble
        import pandas as pd
        
        # Datos de prueba como DataFrames
        X = pd.DataFrame(np.random.randn(100, 5), columns=[f'feature_{i}' for i in range(5)])
        y = pd.Series(np.random.randn(100))
        
        # Crear ensemble
        ensemble = RegularizedEnsemble('momentum')
        ensemble.fit(X, y)
        
        # TEST CRÍTICO: Debe tener método score
        score = ensemble.score(X, y)
        print(f"✅ Ensemble score funciona: {score:.4f}")
        return True
        
    except Exception as e:
        print(f"❌ Error en ensemble score: {e}")
        return False

def test_walk_forward_adjustment():
    """Test: Walk-forward debe ajustarse automáticamente."""
    try:
        sys.path.append('src')
        from validation.walk_forward_validator import WalkForwardValidator
        import pandas as pd
        
        # Datos pequeños intencionalmente
        dates = pd.date_range('2023-01-01', periods=150, freq='5min')
        df = pd.DataFrame({
            'price': np.random.randn(150) + 100,
            'target': np.random.randn(150)
        }, index=dates)
        
        validator = WalkForwardValidator(
            initial_train_months=6,  # Muchos meses para pocos datos
            test_months=2,
            retrain_frequency_months=1
        )
        
        # Debe ajustarse automáticamente
        results = validator.validate(df, 'DummyModel', {})
        print(f"✅ Walk-forward se ejecutó sin errores")
        return True
        
    except Exception as e:
        print(f"❌ Error en walk-forward: {e}")
        return False

def test_temporal_validation():
    """Test: Validación temporal debe funcionar."""
    try:
        sys.path.append('src')
        from validation.temporal_validator import TemporalValidator
        import pandas as pd
        
        # Crear datos temporales
        dates = pd.date_range('2023-01-01', periods=200, freq='1h')
        df = pd.DataFrame({
            'price': np.random.randn(200),
            'target': np.random.randn(200)
        }, index=dates)
        
        validator = TemporalValidator(0.6, 0.2, 0.2)
        train, val, test = validator.temporal_split(df)
        
        print(f"✅ Split temporal exitoso: {len(train)}/{len(val)}/{len(test)}")
        return True
        
    except Exception as e:
        print(f"❌ Error en temporal validation: {e}")
        return False

def test_overfitting_detector():
    """Test: Detector de overfitting debe funcionar."""
    try:
        sys.path.append('src')
        from validation.overfitting_detector import OverfittingDetector
        from sklearn.ensemble import RandomForestRegressor
        
        # Datos de prueba
        X_train = np.random.randn(100, 5)
        y_train = np.random.randn(100)
        X_val = np.random.randn(50, 5)
        y_val = np.random.randn(50)
        
        model = RandomForestRegressor(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)
        
        detector = OverfittingDetector()
        report = detector.detect(model, X_train, y_train, X_val, y_val)
        
        print(f"✅ Detector funcionó: Nivel {report.level.value}")
        return True
        
    except Exception as e:
        print(f"❌ Error en detector: {e}")
        return False

def main():
    """Función principal de testing."""
    print("🧪 NvBot3 - Test de Correcciones Anti-Overfitting")
    print("=" * 55)
    
    # Verificar entorno
    if 'nvbot3_env' not in sys.executable:
        print("⚠️ Entorno virtual no activo")
        return False
    
    # Ejecutar tests críticos
    test1 = test_ensemble_score()
    test2 = test_walk_forward_adjustment() 
    test3 = test_temporal_validation()
    test4 = test_overfitting_detector()
    
    if test1 and test2 and test3 and test4:
        print("\n🎉 ¡Todas las correcciones funcionan correctamente!")
        print("✅ Sistema listo para datos reales")
        return True
    else:
        print("\n❌ Algunas correcciones aún fallan")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
