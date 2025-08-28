"""
🧪 Test Single Model Training - NvBot3
======================================

Script de prueba para entrenar modelos con una sola moneda y timeframe.
Útil para validar que todo funciona antes del entrenamiento masivo.

🔥 Momentum Model (XGBoost): Detectar movimientos alcistas ≥5%
⚡ Rebound Model (Random Forest): Predecir rebotes 1-3%
📊 Regime Model (LSTM): Clasificar tendencia de mercado
🎯 Advanced Momentum (Ensemble): Momentum con filtros de volumen
"""

import sys
import os
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import logging
import time
import traceback
from typing import Dict, Optional
import warnings
warnings.filterwarnings('ignore')

# Importar el ModelTrainer existente
from src.models.model_trainer import ModelTrainer

# Crear directorio de logs si no existe
Path('logs').mkdir(exist_ok=True)

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/test_single_model.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class SingleModelTester:
    """Probador de entrenamiento con una sola moneda"""
    
    def __init__(self):
        self.data_dir = Path('data/processed')
        self.models_dir = Path('data/models')
        self.models_dir.mkdir(exist_ok=True)
        
        # Inicializar ModelTrainer
        self.trainer = ModelTrainer()
        
        # Configuración de modelos (sincronizada con ModelTrainer)
        self.model_configs = {
            'momentum': {
                'target_column': 'momentum_target',
                'model_type': 'xgb',
                'max_features': 40,
                'description': '🔥 Momentum Model: Detectar movimientos ≥5%'
            },
            'rebound': {
                'target_column': 'rebound_target', 
                'model_type': 'rf',
                'max_features': 35,
                'description': '⚡ Rebound Model: Predecir rebotes 1-3%'
            },
            'regime': {
                'target_column': 'regime_target',
                'model_type': 'lstm',
                'max_features': 45,
                'description': '📊 Regime Model: Clasificar tendencia de mercado'
            },
            'momentum_advanced': {
                'target_column': 'momentum_advanced_target',
                'model_type': 'ensemble',
                'max_features': 50,
                'description': '🎯 Advanced Momentum: Momentum con filtros'
            }
        }
    
    def load_single_data_file(self, symbol: str, timeframe: str) -> Optional[pd.DataFrame]:
        """Cargar un archivo específico de datos procesados"""
        filename = f"{symbol}_{timeframe}_with_targets.csv"
        filepath = self.data_dir / filename
        
        if not filepath.exists():
            logger.error(f"❌ Archivo no encontrado: {filepath}")
            return None
        
        try:
            df = pd.read_csv(filepath)
            
            # Verificar que tiene todos los targets
            required_targets = ['momentum_target', 'rebound_target', 'regime_target', 'momentum_advanced_target']
            missing_targets = [target for target in required_targets if target not in df.columns]
            
            if missing_targets:
                logger.error(f"❌ Targets faltantes: {missing_targets}")
                return None
            
            logger.info(f"✅ Datos cargados: {len(df)} registros de {symbol}_{timeframe}")
            
            # Estadísticas de targets
            for target in required_targets:
                target_stats = df[target].value_counts()
                logger.info(f"   📊 {target}: {dict(target_stats)}")
            
            return df
            
        except Exception as e:
            logger.error(f"❌ Error cargando {filepath}: {e}")
            return None
    
    def test_single_model(self, symbol: str, timeframe: str, model_name: str) -> Dict:
        """Probar entrenamiento de un modelo específico"""
        logger.info(f"🧪 === PROBANDO {model_name.upper()} MODEL: {symbol}_{timeframe} ===")
        
        # Cargar datos
        df = self.load_single_data_file(symbol, timeframe)
        if df is None:
            return {'success': False, 'error': 'No se pudieron cargar los datos'}
        
        # Obtener configuración del modelo
        if model_name not in self.model_configs:
            return {'success': False, 'error': f'Modelo {model_name} no reconocido'}
        
        config = self.model_configs[model_name]
        logger.info(f"📋 {config['description']}")
        
        try:
            start_time = time.time()
            
            # Verificar target
            target_column = config['target_column']
            if target_column not in df.columns:
                return {
                    'success': False,
                    'error': f"Target {target_column} no encontrado"
                }
            
            # Estadísticas del target
            target_stats = df[target_column].value_counts()
            logger.info(f"📊 Target {target_column}: {dict(target_stats)}")
            
            # Verificar distribución mínima
            if len(target_stats) < 2:
                return {
                    'success': False,
                    'error': f"Target {target_column} no tiene suficiente varianza"
                }
            
            # Para targets binarios, verificar ratio de positivos
            if len(target_stats) == 2:
                positive_ratio = target_stats.iloc[1] / len(df)
                logger.info(f"   📈 Ratio de positivos: {positive_ratio:.3f}")
                if positive_ratio < 0.01:
                    logger.warning(f"⚠️ Ratio de positivos muy bajo: {positive_ratio:.3f}")
            
            # Preparar datos usando el ModelTrainer
            X, y = self.trainer.prepare_features_targets(df, model_name)
            
            if len(X) == 0:
                return {
                    'success': False,
                    'error': "No hay features válidas"
                }
            
            logger.info(f"🧮 Features preparadas: {len(X.columns)} columnas")
            logger.info(f"📏 Datos para entrenamiento: {len(X)} registros")
            
            # Entrenar modelo usando el ModelTrainer
            result = self.trainer.train_single_model(
                df=df,
                model_type=model_name,
                symbol=symbol,
                timeframe=timeframe
            )
            
            # Calcular tiempo
            training_time = time.time() - start_time
            
            # Agregar información adicional
            result['training_time_seconds'] = training_time
            result['symbol'] = symbol
            result['timeframe'] = timeframe
            result['model_name'] = model_name
            result['data_records'] = len(df)
            result['features_used'] = len(X.columns)
            
            if result.get('success', False):
                summary = result.get('summary', {})
                best_score = summary.get('best_test_score', 0)
                overfitting = summary.get('is_overfitting', False)
                
                logger.info(f"✅ Entrenamiento exitoso en {training_time:.1f}s")
                logger.info(f"   🎯 Mejor accuracy: {best_score:.3f}")
                logger.info(f"   ⚠️ Overfitting: {'Sí' if overfitting else 'No'}")
            else:
                logger.error(f"❌ Entrenamiento falló: {result.get('error', 'Error desconocido')}")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Error entrenando {model_name}: {e}")
            logger.error(f"Traceback:\n{traceback.format_exc()}")
            return {
                'success': False,
                'error': str(e),
                'model_name': model_name
            }
    
    def test_all_models_single_symbol(self, symbol: str = 'BTCUSDT', timeframe: str = '5m') -> Dict:
        """Probar todos los modelos con una sola moneda y timeframe"""
        logger.info("🧪 INICIANDO PRUEBA DE MODELOS CON UNA SOLA MONEDA")
        logger.info("=" * 70)
        logger.info(f"📊 Símbolo: {symbol}")
        logger.info(f"⏰ Timeframe: {timeframe}")
        logger.info("=" * 70)
        
        start_time = time.time()
        results = {}
        
        # Probar cada modelo
        for model_name in self.model_configs.keys():
            logger.info(f"\n🔄 Probando modelo: {model_name}")
            result = self.test_single_model(symbol, timeframe, model_name)
            results[model_name] = result
            
            # Pausa breve entre modelos
            time.sleep(1)
        
        # Calcular tiempo total
        total_time = time.time() - start_time
        
        # Generar reporte
        self._generate_test_report(results, symbol, timeframe, total_time)
        
        return {
            'success': True,
            'symbol': symbol,
            'timeframe': timeframe,
            'results': results,
            'total_time_seconds': total_time
        }
    
    def _generate_test_report(self, results: Dict, symbol: str, timeframe: str, total_time: float):
        """Generar reporte de prueba"""
        logger.info("\n" + "=" * 70)
        logger.info("📊 REPORTE DE PRUEBA COMPLETADO")
        logger.info("=" * 70)
        
        logger.info(f"📊 Símbolo probado: {symbol}")
        logger.info(f"⏰ Timeframe: {timeframe}")
        logger.info(f"⏱️ Tiempo total: {total_time:.1f} segundos")
        
        successful = 0
        failed = 0
        
        logger.info(f"\n🤖 RESULTADOS POR MODELO:")
        for model_name, result in results.items():
            if result.get('success', False):
                successful += 1
                summary = result.get('summary', {})
                best_score = summary.get('best_test_score', 0)
                overfitting = summary.get('is_overfitting', False)
                training_time = result.get('training_time_seconds', 0)
                
                status = "✅ EXITOSO"
                if overfitting:
                    status += " ⚠️ (Overfitting detectado)"
                
                logger.info(f"   {model_name}: {status}")
                logger.info(f"      🎯 Accuracy: {best_score:.3f}")
                logger.info(f"      ⏱️ Tiempo: {training_time:.1f}s")
            else:
                failed += 1
                error = result.get('error', 'Error desconocido')
                logger.info(f"   {model_name}: ❌ FALLÓ - {error}")
        
        logger.info(f"\n📈 RESUMEN GENERAL:")
        logger.info(f"   ✅ Modelos exitosos: {successful}/4")
        logger.info(f"   ❌ Modelos fallidos: {failed}/4")
        logger.info(f"   📊 Tasa de éxito: {(successful/4)*100:.1f}%")
        
        if successful > 0:
            logger.info(f"\n🎯 PRÓXIMOS PASOS:")
            logger.info(f"   1. Si todo está bien, ejecutar: python scripts/train_all_models.py")
            logger.info(f"   2. Para validar modelos: python scripts/validate_trained_models.py")
            logger.info(f"   3. Para backtest: python scripts/backtest_models.py")
        else:
            logger.info(f"\n⚠️ ACCIÓN REQUERIDA:")
            logger.info(f"   Revisar errores antes de proceder con entrenamiento masivo")
        
        logger.info("=" * 70)

def main():
    """Función principal"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test Model Training - Una sola moneda')
    parser.add_argument('--symbol', type=str, default='BTCUSDT', help='Símbolo a probar (default: BTCUSDT)')
    parser.add_argument('--timeframe', type=str, default='5m', help='Timeframe a probar (default: 5m)')
    parser.add_argument('--model', type=str, choices=['momentum', 'rebound', 'regime', 'momentum_advanced'], 
                       help='Probar solo un modelo específico')
    
    args = parser.parse_args()
    
    logger.info("🧪 NvBot3 - Test de Entrenamiento con Una Moneda")
    
    try:
        # Verificar entorno virtual
        if 'nvbot3_env' not in sys.executable:
            logger.error("❌ Entorno virtual nvbot3_env no está activo!")
            logger.error("   Ejecutar: nvbot3_env\\Scripts\\activate")
            return 1
        
        logger.info("✅ Entorno virtual nvbot3_env activo")
        
        # Crear probador
        tester = SingleModelTester()
        
        if args.model:
            # Probar solo un modelo
            logger.info(f"🎯 Probando solo modelo: {args.model}")
            result = tester.test_single_model(args.symbol, args.timeframe, args.model)
            
            if result['success']:
                logger.info("🎉 ¡Prueba de modelo individual exitosa!")
                return 0
            else:
                logger.error("❌ Prueba de modelo individual falló")
                return 1
        else:
            # Probar todos los modelos
            final_result = tester.test_all_models_single_symbol(args.symbol, args.timeframe)
            
            if final_result['success']:
                logger.info("🎉 ¡Prueba completa exitosa!")
                return 0
            else:
                logger.error("❌ Prueba completa falló")
                return 1
                
    except Exception as e:
        logger.error(f"💥 Error crítico: {e}")
        logger.error(f"Traceback:\n{traceback.format_exc()}")
        return 1

if __name__ == "__main__":
    exit(main())
