#!/usr/bin/env python3
"""
🎯 Script para optimizar LSTM Regime: Overfitting 0.295 → 0.18
Implementa todas las optimizaciones anti-overfitting para el modelo LSTM
"""

import sys
import os
import pandas as pd
import numpy as np
import logging
from datetime import datetime

# Agregar path del proyecto
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_lstm_regime_optimization():
    """Test específico para LSTM Regime con parámetros optimizados."""
    logger.info("🎯 OPTIMIZACIÓN LSTM REGIME: Overfitting 0.295 → 0.18")
    logger.info("🔧 Implementando todas las optimizaciones anti-overfitting")
    logger.info("="*70)
    
    try:
        # Importar después de configurar el path
        from models.model_trainer import ModelTrainer
        
        # Crear trainer optimizado
        trainer = ModelTrainer()
        
        # Entrenar solo el modelo regime con optimizaciones
        logger.info("🧪 Entrenando LSTM Regime con optimizaciones anti-overfitting...")
        
        # Cargar datos BTCUSDT 5m
        df = trainer.load_training_data('BTCUSDT', '5m')
        if df is None:
            logger.error("❌ No se pudieron cargar los datos de entrenamiento")
            return False
        
        # Entrenar modelo regime optimizado
        result = trainer.train_single_model(df, 'regime', 'BTCUSDT', '5m')
        
        if result['success']:
            summary = result['summary']
            overfitting_gap = summary['avg_overfitting_gap']
            test_accuracy = summary['best_test_score']
            
            logger.info(f"\n📊 RESULTADOS LSTM REGIME OPTIMIZADO:")
            logger.info(f"  Test Accuracy: {test_accuracy:.3f}")
            logger.info(f"  Overfitting Gap: {overfitting_gap:.3f}")
            logger.info(f"  Features: {len(result['selected_features'])}")
            logger.info(f"  Validaciones: {summary['n_validations']}")
            
            # Verificación target
            target_min, target_max = 0.15, 0.20
            target_value = 0.18
            previous_gap = 0.295
            
            # Análisis de resultados
            if target_min <= overfitting_gap <= target_max:
                if abs(overfitting_gap - target_value) <= 0.02:
                    status = "🎯 TARGET PERFECTO ALCANZADO"
                    logger.info(f"  ✅ {status} (Gap: {overfitting_gap:.3f} ≈ {target_value})")
                else:
                    status = "✅ RANGO ÓPTIMO"
                    logger.info(f"  ✅ {status} (Gap: {overfitting_gap:.3f} en [0.15-0.20])")
            elif overfitting_gap > target_max:
                status = "❌ OVERFITTING AÚN ALTO"
                logger.warning(f"  ❌ {status} (Gap: {overfitting_gap:.3f} > 0.20)")
                logger.warning("  🔧 Requiere mayor regularización")
            else:
                status = "⚠️ POSIBLE UNDERFITTING"
                logger.warning(f"  ⚠️ {status} (Gap: {overfitting_gap:.3f} < 0.15)")
            
            # Comparación con resultado anterior
            improvement = previous_gap - overfitting_gap
            improvement_pct = (improvement / previous_gap) * 100
            
            logger.info(f"\n📈 COMPARACIÓN CON MODELO ANTERIOR:")
            logger.info(f"  Overfitting anterior: {previous_gap:.3f}")
            logger.info(f"  Overfitting actual: {overfitting_gap:.3f}")
            logger.info(f"  Mejora: {improvement:.3f} ({improvement_pct:.1f}%)")
            
            # Análisis de optimizaciones aplicadas
            logger.info(f"\n🔧 OPTIMIZACIONES APLICADAS:")
            logger.info(f"  🔽 LSTM units: 64→24→12 (reducción drástica)")
            logger.info(f"  🔼 Dropout: 0.8/0.6 (máximo anti-overfitting)")
            logger.info(f"  🔼 L2 regularization: 0.1/0.05 (alto)")
            logger.info(f"  🔽 Learning rate: 0.0003 (conservador)")
            logger.info(f"  🔽 Sequence length: 20→12 (secuencias cortas)")
            logger.info(f"  🔽 Epochs: 50→30 (entrenamiento corto)")
            logger.info(f"  🔽 Batch size: 32→12 (lotes pequeños)")
            logger.info(f"  🔽 Early stopping: patience 10→5 (agresivo)")
            logger.info(f"  🔽 Features: ~50→25 (selección super agresiva)")
            
            if overfitting_gap <= 0.20:
                logger.info("\n🎉 OPTIMIZACIÓN LSTM EXITOSA")
                logger.info("✅ Modelo Regime optimizado para reducir overfitting")
                return True
            else:
                logger.warning("\n⚠️ OPTIMIZACIÓN PARCIAL")
                logger.warning("🔧 Requiere ajustes adicionales más agresivos")
                return False
                
        else:
            logger.error("❌ Error en el entrenamiento del modelo")
            return False
            
    except Exception as e:
        logger.error(f"❌ Error en optimización LSTM: {e}")
        import traceback
        traceback.print_exc()
        return False

def suggest_further_optimizations():
    """Sugerir optimizaciones adicionales si es necesario."""
    logger.info("\n🔧 OPTIMIZACIONES ADICIONALES DISPONIBLES:")
    logger.info("  Si overfitting > 0.20, aplicar:")
    logger.info("  🔽 LSTM units: 24→16→8 (aún más pequeño)")
    logger.info("  🔼 Dropout: 0.9 (máximo absoluto)")
    logger.info("  🔽 Epochs: 30→20 (súper corto)")
    logger.info("  🔽 Features: 25→15 (ultra agresivo)")
    logger.info("  🔽 Sequence length: 12→8 (secuencias mínimas)")
    logger.info("  🔼 L2 regularization: 0.1→0.2 (máximo)")

if __name__ == "__main__":
    logger.info("🚀 Iniciando optimización LSTM Regime...")
    
    success = test_lstm_regime_optimization()
    
    if success:
        logger.info("\n✅ OPTIMIZACIÓN COMPLETADA EXITOSAMENTE")
        logger.info("🎯 LSTM Regime optimizado para target overfitting ≤0.18")
    else:
        logger.warning("\n⚠️ OPTIMIZACIÓN REQUIERE AJUSTES ADICIONALES")
        suggest_further_optimizations()
        
    logger.info(f"\n📋 Próximos pasos:")
    logger.info(f"  1. Verificar modelo guardado en data/models/")
    logger.info(f"  2. Entrenar otros símbolos si la optimización fue exitosa")
    logger.info(f"  3. Integrar con trading bot si todo funciona bien")
