"""
Demo simplificado y funcional del sistema anti-overfitting NvBot3.
Versión corregida que usa las interfaces reales de los módulos.
"""

import numpy as np
import pandas as pd
import logging
from pathlib import Path
import sys
from typing import Optional, Dict, Any, List
import warnings

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suprimir warnings innecesarios
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# Agregar src al path
src_path = Path(__file__).parent.parent / 'src'
if str(src_path) not in sys.path:
    sys.path.append(str(src_path))

def create_simple_test_data(n_samples: int = 1000) -> pd.DataFrame:
    """
    Crear datos simples para testing sin errores de typing.
    
    Args:
        n_samples: Número de muestras a generar
        
    Returns:
        pd.DataFrame: Dataset simple con features y target
    """
    try:
        np.random.seed(42)
        logger.info(f"Generando {n_samples} muestras de datos simples")
        
        # Crear datos básicos con tipos explícitos
        timestamps = pd.date_range(start='2023-01-01', periods=n_samples, freq='5T')
        
        # Precio base simple sin volatilidad compleja
        price_trend = np.linspace(50000, 55000, n_samples)
        noise = np.random.normal(0, 500, n_samples)
        price = price_trend + noise
        
        # Crear DataFrame con tipos explícitos
        data = pd.DataFrame({
            'timestamp': timestamps,
            'price': price.astype(float),
            'volume': np.random.exponential(1000, n_samples).astype(float),
        })
        
        # Features técnicos simples usando .astype() explícito
        data['ma_5'] = data['price'].rolling(window=5, min_periods=1).mean().astype(float)
        data['ma_20'] = data['price'].rolling(window=20, min_periods=1).mean().astype(float)
        
        # Returns simple
        data['returns'] = data['price'].pct_change().fillna(0).astype(float)
        
        # Volatilidad simple
        data['volatility'] = data['price'].rolling(window=10, min_periods=1).std().fillna(0).astype(float)
        
        # RSI simplificado usando numpy arrays directamente
        delta = data['price'].diff().fillna(0)
        delta_np = np.array(delta)  # Conversión explícita a numpy
        gain = np.where(delta_np > 0, delta_np, 0)
        loss = np.where(delta_np < 0, -delta_np, 0)
        
        avg_gain = pd.Series(gain).rolling(window=14, min_periods=1).mean()
        avg_loss = pd.Series(loss).rolling(window=14, min_periods=1).mean()
        
        rs = avg_gain / (avg_loss + 1e-8)
        data['rsi'] = (100 - (100 / (1 + rs))).astype(float)
        
        # Target simple: 1 si precio sube en próximo período, 0 si baja
        future_price = data['price'].shift(-1)
        data['target'] = (future_price > data['price']).astype(int)
        
        # Rellenar NaNs usando método moderno
        data = data.ffill().fillna(0)
        
        # Eliminar última fila con target NaN
        data = data.dropna(subset=['target'])
        
        # Establecer índice de timestamp
        data.set_index('timestamp', inplace=True)
        
        logger.info(f"✅ Dataset creado: {data.shape}")
        logger.info(f"Columnas: {list(data.columns)}")
        logger.info(f"Target distribution: {data['target'].value_counts().to_dict()}")
        
        return data
        
    except Exception as e:
        logger.error(f"Error creando dataset: {e}")
        raise

def test_walk_forward_validator(data: pd.DataFrame) -> bool:
    """
    Probar WalkForwardValidator con la interfaz real.
    
    Args:
        data: DataFrame con timestamp como índice
        
    Returns:
        bool: True si el test fue exitoso
    """
    try:
        logger.info("\n🔍 Probando WalkForwardValidator...")
        
        from validation.walk_forward_validator import WalkForwardValidator
        from models.regularized_models import RegularizedXGBoost
        
        # Crear validador con parámetros conservadores
        validator = WalkForwardValidator(
            initial_train_months=1,  # Solo 1 mes inicial
            test_months=1,           # 1 mes de test
            retrain_frequency_months=1,  # Reentrenar cada mes
            min_train_samples=50     # Mínimo 50 samples
        )
        
        # Crear modelo XGBoost
        model_class = RegularizedXGBoost
        model_params = {'task_type': 'momentum'}
        
        # Ejecutar validación con interfaz real
        results = validator.validate(
            df=data,
            model_class=model_class,
            model_params=model_params
        )
        
        if results and len(results) > 0:
            logger.info(f"✅ WalkForwardValidator: {len(results)} períodos completados")
            for i, result in enumerate(results[:3]):  # Mostrar primeros 3
                logger.info(f"   Período {i+1}: train={result.train_size}, test={result.test_size}")
            return True
        else:
            logger.warning("⚠️ WalkForwardValidator: Sin resultados")
            return False
            
    except Exception as e:
        logger.error(f"❌ Error en WalkForwardValidator: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False

def test_regularized_models(data: pd.DataFrame) -> bool:
    """
    Probar modelos regularizados con datos simples.
    
    Args:
        data: DataFrame con features y target
        
    Returns:
        bool: True si el test fue exitoso
    """
    try:
        logger.info("\n🔍 Probando Modelos Regularizados...")
        
        from models.regularized_models import RegularizedXGBoost
        
        # Preparar datos
        feature_cols = [col for col in data.columns if col not in ['target']]
        X = data[feature_cols].values
        y = data['target'].values
        
        logger.info(f"Features: {feature_cols}")
        logger.info(f"Datos: X={X.shape}, y={y.shape}")
        
        # Split simple train/test
        split_idx = int(len(X) * 0.7)
        X_train = X[:split_idx]
        X_test = X[split_idx:]
        y_train = y[:split_idx]
        y_test = y[split_idx:]
        
        # Crear y entrenar modelo
        model = RegularizedXGBoost(task_type='momentum')
        
        # Fit con eval_set
        model.fit(
            X_train=X_train,
            y_train=y_train,
            eval_set=[(X_test, y_test)]
        )
        
        # Hacer predicciones
        predictions = model.predict(X_test)
        score = model.score(X_test, y_test)
        
        logger.info(f"✅ RegularizedXGBoost entrenado exitosamente")
        logger.info(f"   Score: {score:.4f}")
        logger.info(f"   Predicciones shape: {predictions.shape}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error en modelos regularizados: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False

def test_overfitting_detector(data: pd.DataFrame) -> bool:
    """
    Probar detector de overfitting con datos simples.
    
    Args:
        data: DataFrame con features y target
        
    Returns:
        bool: True si el test fue exitoso
    """
    try:
        logger.info("\n🔍 Probando OverfittingDetector...")
        
        from validation.overfitting_detector import OverfittingDetector
        from models.regularized_models import RegularizedXGBoost
        
        # Preparar datos
        feature_cols = [col for col in data.columns if col not in ['target']]
        X = data[feature_cols]  # Mantener como DataFrame
        y = data['target']      # Mantener como Series
        
        # Split simple
        split_idx = int(len(X) * 0.7)
        X_train = X.iloc[:split_idx]
        X_test = X.iloc[split_idx:]
        y_train = y.iloc[:split_idx]
        y_test = y.iloc[split_idx:]
        
        # Entrenar modelo
        model = RegularizedXGBoost(task_type='momentum')
        model.fit(X_train.values, y_train.values)
        
        # Crear detector
        detector = OverfittingDetector()
        
        # Analizar overfitting
        report = detector.detect(
            model=model,
            X_train=X_train,
            y_train=y_train,
            X_val=X_test,
            y_val=y_test,
            model_name="Test_XGBoost"
        )
        
        logger.info(f"✅ OverfittingDetector completado:")
        logger.info(f"   Nivel: {report.level.value.upper()}")
        logger.info(f"   Score overfitting: {report.score:.3f}")
        logger.info(f"   Gap train-val: {report.gap:.3f}")
        logger.info(f"   Advertencias: {len(report.warnings)}")
        
        if report.warnings:
            logger.info("   Primeras advertencias:")
            for i, warning in enumerate(report.warnings[:2], 1):
                logger.info(f"     {i}. {warning}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error en OverfittingDetector: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False

def run_simplified_demo() -> bool:
    """
    Ejecutar demo simplificado que realmente funciona.
    
    Returns:
        bool: True si la demo fue exitosa
    """
    try:
        logger.info("="*70)
        logger.info("🚀 DEMO SIMPLIFICADO SISTEMA ANTI-OVERFITTING NVBOT3")
        logger.info("="*70)
        
        # 1. Crear datos simples
        logger.info("\n1️⃣ Creando datos de prueba simples...")
        data = create_simple_test_data(n_samples=1000)
        
        if data.empty:
            raise ValueError("No se pudieron crear datos de prueba")
        
        logger.info(f"✅ Datos creados: {data.shape}")
        
        # 2. Test individual de módulos
        success_count = 0
        total_tests = 3
        
        # Test WalkForwardValidator
        if test_walk_forward_validator(data):
            success_count += 1
        
        # Test RegularizedModels
        if test_regularized_models(data):
            success_count += 1
        
        # Test OverfittingDetector
        if test_overfitting_detector(data):
            success_count += 1
        
        # 3. Resumen final
        logger.info("\n📊 RESUMEN FINAL:")
        logger.info("="*50)
        logger.info(f"Tests exitosos: {success_count}/{total_tests}")
        
        if success_count == total_tests:
            logger.info("✅ WalkForwardValidator funcionando")
            logger.info("✅ RegularizedModels operativos")
            logger.info("✅ OverfittingDetector funcional")
            logger.info("="*50)
            logger.info("🎉 DEMO COMPLETADA EXITOSAMENTE")
            logger.info("Sistema anti-overfitting NvBot3 operativo!")
            return True
        else:
            logger.warning(f"⚠️ Solo {success_count}/{total_tests} tests exitosos")
            logger.warning("Sistema parcialmente funcional")
            return False
        
    except Exception as e:
        logger.error(f"❌ Error crítico en demo: {e}")
        import traceback
        logger.error(f"Traceback completo: {traceback.format_exc()}")
        return False

def main():
    """
    Función principal para ejecutar la demo.
    """
    try:
        # Verificar entorno virtual
        if not hasattr(sys, 'real_prefix') and not (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
            logger.warning("⚠️ No se detectó entorno virtual activo")
            logger.warning("   Recomendado: activar nvbot3_env antes de ejecutar")
        else:
            logger.info("✅ Entorno virtual detectado")
        
        # Ejecutar demo simplificado
        success = run_simplified_demo()
        
        if success:
            logger.info("\n🚀 Demo ejecutada exitosamente!")
            logger.info("El sistema anti-overfitting está funcionando.")
            return 0
        else:
            logger.error("\n❌ Demo con errores - revisar logs para detalles")
            return 1
            
    except KeyboardInterrupt:
        logger.info("\n⏹️ Demo interrumpida por usuario")
        return 1
    except Exception as e:
        logger.error(f"\n💥 Error fatal en main: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
