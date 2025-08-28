"""
🚀 Train All Models - NvBot3
============================

Script principal para entrenar los 4 modelos con TODOS los datos procesados:
🔥 Momentum Model (XGBoost): Detectar movimientos alcistas ≥5%
⚡ Rebound Model (Random Forest): Predecir rebotes 1-3%
📊 Regime Model (LSTM): Clasificar tendencia de mercado
🎯 Advanced Momentum (Ensemble): Momentum con filtros de volumen

Características:
- Utiliza TODOS los datos procesados (3M+ registros)
- Sistema anti-overfitting completo
- Walk-forward validation en múltiples símbolos
- Entrenamiento paralelo por timeframe
- Reportes detallados de performance
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
import json
from typing import Dict, List, Optional
import yaml
from concurrent.futures import ThreadPoolExecutor, as_completed
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
        logging.FileHandler('logs/training_all_models.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ComprehensiveModelTrainer:
    """Entrenador completo de todos los modelos usando todos los datos procesados"""
    
    def __init__(self):
        self.data_dir = Path('data/processed')
        self.models_dir = Path('data/models')
        self.models_dir.mkdir(exist_ok=True)
        
        # Crear directorio de checkpoints
        self.checkpoint_dir = Path('data/checkpoints')
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        # Inicializar ModelTrainer
        self.trainer = ModelTrainer()
        
        # Configuración de modelos (sincronizada con ModelTrainer)
        self.model_configs = {
            'momentum': {
                'target_column': 'momentum_target',
                'model_type': 'xgb',
                'max_features': 40,
                'description': '🔥 Momentum Model: Detectar movimientos ≥5%',
                'priority': 1
            },
            'rebound': {
                'target_column': 'rebound_target', 
                'model_type': 'rf',
                'max_features': 35,
                'description': '⚡ Rebound Model: Predecir rebotes 1-3%',
                'priority': 2
            },
            'regime': {
                'target_column': 'regime_target',
                'model_type': 'lstm',
                'max_features': 45,
                'description': '📊 Regime Model: Clasificar tendencia de mercado',
                'priority': 3
            },
            'momentum_advanced': {
                'target_column': 'momentum_advanced_target',
                'model_type': 'ensemble',
                'max_features': 50,
                'description': '🎯 Advanced Momentum: Momentum con filtros',
                'priority': 4
            }
        }
        
        # Estadísticas de entrenamiento
        self.training_stats = {
            'total_records_processed': 0,
            'total_features_used': 0,
            'models_trained': 0,
            'successful_trainings': 0,
            'failed_trainings': 0,
            'overfitting_detected': 0,
            'best_performances': {},
            'session_start_time': time.time(),
            'last_checkpoint': None
        }
        
        # Cargar checkpoint si existe
        self.load_checkpoint()
        
    def detect_missing_models(self) -> List[Dict]:
        """Detectar qué modelos faltan por entrenar"""
        logger.info("🔍 Detectando modelos faltantes...")
        
        # Obtener timeframes disponibles
        target_files = list(self.data_dir.glob('*_with_targets.csv'))
        timeframes = set()
        
        for file_path in target_files:
            filename = file_path.name
            symbol_tf = filename.replace('_with_targets.csv', '')
            parts = symbol_tf.split('_')
            if len(parts) >= 2:
                timeframe = parts[-1]
                timeframes.add(timeframe)
        
        # Verificar qué modelos existen
        existing_models = set()
        model_files = list(self.models_dir.glob('ALL_SYMBOLS_*.pkl'))
        
        for model_path in model_files:
            if not model_path.name.endswith('_metrics.pkl'):
                # Extraer modelo de: ALL_SYMBOLS_1h_momentum.pkl
                filename = model_path.name.replace('.pkl', '')
                parts = filename.split('_')
                if len(parts) >= 4:
                    timeframe = parts[2]
                    model_type = '_'.join(parts[3:])
                    existing_models.add(f"{model_type}_{timeframe}")
        
        # Determinar modelos faltantes
        missing_jobs = []
        for timeframe in sorted(timeframes):
            for model_name, config in self.model_configs.items():
                model_key = f"{model_name}_{timeframe}"
                if model_key not in existing_models:
                    missing_jobs.append({
                        'timeframe': timeframe,
                        'model_name': model_name,
                        'config': config,
                        'priority': config['priority'],
                        'model_key': model_key
                    })
        
        logger.info(f"✅ Modelos existentes: {len(existing_models)}")
        logger.info(f"❌ Modelos faltantes: {len(missing_jobs)}")
        
        # Mostrar resumen
        if missing_jobs:
            logger.info("📋 MODELOS PENDIENTES:")
            for job in missing_jobs:
                logger.info(f"   ❌ {job['model_key']} - {job['config']['description']}")
        else:
            logger.info("🎉 ¡TODOS LOS MODELOS YA ESTÁN ENTRENADOS!")
        
        return missing_jobs
    
    def save_checkpoint(self, completed_jobs: List[str]):
        """Guardar checkpoint con progreso actual"""
        checkpoint_data = {
            'completed_jobs': completed_jobs,
            'training_stats': self.training_stats,
            'timestamp': time.time(),
            'session_id': getattr(self, 'session_id', f"session_{int(time.time())}")
        }
        
        checkpoint_file = self.checkpoint_dir / 'training_checkpoint.json'
        
        try:
            import json
            with open(checkpoint_file, 'w') as f:
                json.dump(checkpoint_data, f, indent=2)
            
            self.training_stats['last_checkpoint'] = time.time()
            logger.info(f"💾 Checkpoint guardado: {len(completed_jobs)} trabajos completados")
            
        except Exception as e:
            logger.error(f"❌ Error guardando checkpoint: {e}")
    
    def load_checkpoint(self):
        """Cargar checkpoint si existe"""
        checkpoint_file = self.checkpoint_dir / 'training_checkpoint.json'
        
        if checkpoint_file.exists():
            try:
                import json
                with open(checkpoint_file, 'r') as f:
                    checkpoint_data = json.load(f)
                
                # Restaurar estadísticas
                if 'training_stats' in checkpoint_data:
                    saved_stats = checkpoint_data['training_stats']
                    for key, value in saved_stats.items():
                        if key in self.training_stats:
                            self.training_stats[key] = value
                
                logger.info(f"🔄 Checkpoint cargado: {checkpoint_data['timestamp']}")
                return checkpoint_data.get('completed_jobs', [])
                
            except Exception as e:
                logger.error(f"❌ Error cargando checkpoint: {e}")
        
        return []
    
    def clear_checkpoint(self):
        """Limpiar checkpoint al completar entrenamiento"""
        checkpoint_file = self.checkpoint_dir / 'training_checkpoint.json'
        
        if checkpoint_file.exists():
            try:
                checkpoint_file.unlink()
                logger.info("🗑️ Checkpoint limpiado")
            except Exception as e:
                logger.error(f"❌ Error limpiando checkpoint: {e}")
        
    def load_all_processed_data(self) -> Dict[str, pd.DataFrame]:
        """Cargar todos los archivos procesados y consolidar por timeframe"""
        logger.info("🔍 Cargando todos los datos procesados...")
        
        # Buscar archivos *_with_targets.csv
        target_files = list(self.data_dir.glob('*_with_targets.csv'))
        
        if not target_files:
            raise FileNotFoundError("❌ No se encontraron archivos *_with_targets.csv")
        
        logger.info(f"📁 Encontrados {len(target_files)} archivos procesados")
        
        # Organizar por timeframe
        timeframe_data = {}
        
        for file_path in target_files:
            try:
                # Extraer símbolo y timeframe del nombre
                filename = file_path.name
                # Formato: SYMBOL_TIMEFRAME_with_targets.csv
                symbol_tf = filename.replace('_with_targets.csv', '')
                parts = symbol_tf.split('_')
                
                if len(parts) >= 2:
                    symbol = '_'.join(parts[:-1])
                    timeframe = parts[-1]
                    
                    # Cargar datos
                    df = pd.read_csv(file_path)
                    
                    # Validar que tiene targets
                    required_targets = ['momentum_target', 'rebound_target', 'regime_target', 'momentum_advanced_target']
                    if not all(target in df.columns for target in required_targets):
                        logger.warning(f"⚠️ {filename}: Faltan targets, omitiendo...")
                        continue
                    
                    # Agregar información del símbolo
                    df['symbol'] = symbol
                    df['timeframe'] = timeframe
                    
                    # Consolidar por timeframe
                    if timeframe not in timeframe_data:
                        timeframe_data[timeframe] = []
                    
                    timeframe_data[timeframe].append(df)
                    
                    logger.info(f"✅ {symbol}_{timeframe}: {len(df)} registros cargados")
                    
            except Exception as e:
                logger.error(f"❌ Error cargando {file_path}: {e}")
                continue
        
        # Concatenar datos por timeframe
        consolidated_data = {}
        for timeframe, dfs in timeframe_data.items():
            if dfs:
                combined_df = pd.concat(dfs, ignore_index=True)
                consolidated_data[timeframe] = combined_df
                
                total_records = len(combined_df)
                symbols_count = combined_df['symbol'].nunique()
                self.training_stats['total_records_processed'] += total_records
                
                logger.info(f"📊 {timeframe}: {total_records:,} registros de {symbols_count} símbolos")
        
        logger.info(f"🎯 Total consolidado: {self.training_stats['total_records_processed']:,} registros")
        return consolidated_data
    
    def train_model_for_timeframe(self, timeframe: str, df: pd.DataFrame, model_name: str, config: Dict) -> Dict:
        """Entrenar un modelo específico para un timeframe"""
        logger.info(f"🤖 Entrenando {config['description']} para {timeframe}")
        
        try:
            # Verificar que el target existe
            target_column = config['target_column']
            if target_column not in df.columns:
                return {
                    'success': False,
                    'error': f"Target {target_column} no encontrado",
                    'model_name': model_name,
                    'timeframe': timeframe
                }
            
            # Estadísticas del target
            target_stats = df[target_column].value_counts()
            logger.info(f"📊 Target {target_column}: {dict(target_stats)}")
            
            # Verificar distribución mínima
            if len(target_stats) < 2:
                return {
                    'success': False,
                    'error': f"Target {target_column} no tiene suficiente varianza",
                    'model_name': model_name,
                    'timeframe': timeframe
                }
            
            # Para targets binarios, verificar que hay al menos 1% de positivos
            if len(target_stats) == 2:
                positive_ratio = target_stats.iloc[1] / len(df)
                if positive_ratio < 0.01:
                    logger.warning(f"⚠️ Ratio de positivos muy bajo: {positive_ratio:.3f}")
            
            # Preparar datos usando el ModelTrainer
            X, y = self.trainer.prepare_features_targets(df, model_name)
            
            if len(X) == 0:
                return {
                    'success': False,
                    'error': "No hay features válidas",
                    'model_name': model_name,
                    'timeframe': timeframe
                }
            
            self.training_stats['total_features_used'] = len(X.columns)
            
            # Entrenar modelo usando walk-forward validation
            result = self.trainer.train_single_model(
                df=df,
                model_type=model_name,
                symbol='ALL_SYMBOLS',  # Múltiples símbolos
                timeframe=timeframe
            )
            
            # Actualizar estadísticas
            if result.get('success', False):
                self.training_stats['successful_trainings'] += 1
                
                # Verificar overfitting
                summary = result.get('summary', {})
                if summary.get('is_overfitting', False):
                    self.training_stats['overfitting_detected'] += 1
                    logger.warning(f"⚠️ Overfitting detectado en {model_name}_{timeframe}")
                
                # Guardar mejor performance
                best_score = summary.get('best_test_score', 0)
                key = f"{model_name}_{timeframe}"
                if key not in self.training_stats['best_performances'] or best_score > self.training_stats['best_performances'][key]:
                    self.training_stats['best_performances'][key] = best_score
                
                logger.info(f"✅ {config['description']} {timeframe}: Accuracy {best_score:.3f}")
            else:
                self.training_stats['failed_trainings'] += 1
                logger.error(f"❌ Fallo entrenando {model_name}_{timeframe}")
            
            self.training_stats['models_trained'] += 1
            return result
            
        except Exception as e:
            logger.error(f"❌ Error entrenando {model_name}_{timeframe}: {e}")
            self.training_stats['failed_trainings'] += 1
            return {
                'success': False,
                'error': str(e),
                'model_name': model_name,
                'timeframe': timeframe
            }
    
    def train_all_models_parallel(self, max_workers: int = 2, resume_mode: bool = True) -> Dict:
        """Entrenar todos los modelos en paralelo con límite de workers y recuperación automática"""
        logger.info("🚀 INICIANDO ENTRENAMIENTO MASIVO DE MODELOS")
        logger.info("=" * 70)
        
        start_time = time.time()
        self.session_id = f"session_{int(start_time)}"
        
        # Detectar trabajos faltantes
        if resume_mode:
            training_jobs = self.detect_missing_models()
            if not training_jobs:
                logger.info("🎉 ¡TODOS LOS MODELOS YA ESTÁN COMPLETADOS!")
                return {'success': True, 'message': 'All models already trained'}
        else:
            # Modo completo: entrenar todo desde cero
            logger.info("🔄 Modo completo: entrenando todos los modelos desde cero")
            timeframe_data = self.load_all_processed_data()
            training_jobs = []
            for timeframe, df in timeframe_data.items():
                for model_name, config in self.model_configs.items():
                    training_jobs.append({
                        'timeframe': timeframe,
                        'df': df,
                        'model_name': model_name,
                        'config': config,
                        'priority': config['priority'],
                        'model_key': f"{model_name}_{timeframe}"
                    })
        
        if not training_jobs:
            logger.error("❌ No hay trabajos de entrenamiento")
            return {'success': False, 'error': 'No training jobs'}
        
        # Cargar datos una sola vez para todos los trabajos
        logger.info("🔍 Cargando datos consolidados...")
        timeframe_data = self.load_all_processed_data()
        
        # Asignar dataframes a los trabajos
        for job in training_jobs:
            job['df'] = timeframe_data.get(job['timeframe'])
            if job['df'] is None:
                logger.error(f"❌ No hay datos para {job['timeframe']}")
                continue
        
        # Filtrar trabajos válidos
        valid_jobs = [job for job in training_jobs if job.get('df') is not None]
        
        # Ordenar por prioridad
        valid_jobs.sort(key=lambda x: x['priority'])
        
        logger.info(f"📋 {len(valid_jobs)} trabajos de entrenamiento programados")
        logger.info(f"💻 Usando {max_workers} workers paralelos")
        
        results = []
        completed_jobs = []
        
        try:
            # Ejecutar en paralelo con límite
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Enviar trabajos
                future_to_job = {}
                for job in valid_jobs:
                    future = executor.submit(
                        self.train_model_for_timeframe,
                        job['timeframe'],
                        job['df'],
                        job['model_name'],
                        job['config']
                    )
                    future_to_job[future] = job
                
                # Procesar resultados conforme completan
                for future in as_completed(future_to_job):
                    job = future_to_job[future]
                    try:
                        result = future.result()
                        results.append(result)
                        
                        # Marcar como completado si fue exitoso
                        if result.get('success', False):
                            completed_jobs.append(job['model_key'])
                        
                        # Guardar checkpoint cada 2 modelos completados
                        if len(completed_jobs) % 2 == 0:
                            self.save_checkpoint(completed_jobs)
                        
                        # Log progreso
                        completed = len(results)
                        total = len(valid_jobs)
                        progress = (completed / total) * 100
                        
                        logger.info(f"📈 Progreso: {completed}/{total} ({progress:.1f}%)")
                        
                    except Exception as e:
                        logger.error(f"❌ Error en {job['model_name']}_{job['timeframe']}: {e}")
                        results.append({
                            'success': False,
                            'error': str(e),
                            'model_name': job['model_name'],
                            'timeframe': job['timeframe']
                        })
        
        except KeyboardInterrupt:
            logger.warning("⚠️ Entrenamiento interrumpido por usuario")
            self.save_checkpoint(completed_jobs)
            return {
                'success': False, 
                'interrupted': True, 
                'completed_jobs': completed_jobs,
                'message': 'Training interrupted, checkpoint saved'
            }
        
        except Exception as e:
            logger.error(f"💥 Error crítico durante entrenamiento: {e}")
            self.save_checkpoint(completed_jobs)
            raise
        
        # Calcular tiempo total
        total_time = time.time() - start_time
        
        # Guardar checkpoint final
        self.save_checkpoint(completed_jobs)
        
        # Generar reporte final
        self._generate_training_report(results, total_time)
        
        # Limpiar checkpoint si todo se completó exitosamente
        if len(completed_jobs) == len(valid_jobs):
            self.clear_checkpoint()
            logger.info("🎉 ¡ENTRENAMIENTO COMPLETADO! Checkpoint limpiado.")
        
        return {
            'success': True,
            'results': results,
            'stats': self.training_stats,
            'total_time_seconds': total_time,
            'completed_jobs': completed_jobs
        }
    
    def _generate_training_report(self, results: List[Dict], total_time: float):
        """Generar reporte comprensivo del entrenamiento"""
        logger.info("\n" + "=" * 70)
        logger.info("📊 REPORTE FINAL DE ENTRENAMIENTO")
        logger.info("=" * 70)
        
        # Estadísticas generales
        stats = self.training_stats
        logger.info(f"⏱️  Tiempo total: {total_time/60:.1f} minutos")
        logger.info(f"📈 Registros procesados: {stats['total_records_processed']:,}")
        logger.info(f"🧮 Features utilizadas: {stats['total_features_used']}")
        logger.info(f"🤖 Modelos entrenados: {stats['models_trained']}")
        logger.info(f"✅ Entrenamientos exitosos: {stats['successful_trainings']}")
        logger.info(f"❌ Entrenamientos fallidos: {stats['failed_trainings']}")
        logger.info(f"⚠️ Overfitting detectado: {stats['overfitting_detected']}")
        
        # Mejores performances
        logger.info(f"\n🏆 MEJORES PERFORMANCES:")
        for model_tf, score in stats['best_performances'].items():
            logger.info(f"   {model_tf}: {score:.3f}")
        
        # Resumen por modelo
        logger.info(f"\n📋 RESUMEN POR MODELO:")
        model_summary = {}
        for result in results:
            if result.get('success', False):
                model_name = result['model_name']
                if model_name not in model_summary:
                    model_summary[model_name] = {'success': 0, 'failed': 0}
                model_summary[model_name]['success'] += 1
            else:
                model_name = result.get('model_name', 'unknown')
                if model_name not in model_summary:
                    model_summary[model_name] = {'success': 0, 'failed': 0}
                model_summary[model_name]['failed'] += 1
        
        for model_name, counts in model_summary.items():
            total = counts['success'] + counts['failed']
            success_rate = (counts['success'] / total) * 100 if total > 0 else 0
            logger.info(f"   {model_name}: {counts['success']}/{total} exitosos ({success_rate:.1f}%)")
        
        # Timeframes procesados
        timeframes = set()
        for result in results:
            if 'timeframe' in result:
                timeframes.add(result['timeframe'])
        
        logger.info(f"\n⏰ TIMEFRAMES PROCESADOS: {', '.join(sorted(timeframes))}")
        
        # Modelos guardados
        models_saved = len(list(self.models_dir.glob('*.pkl')))
        logger.info(f"💾 Modelos guardados: {models_saved}")
        
        logger.info("\n🎯 PRÓXIMOS PASOS:")
        logger.info("   1. Validar modelos: python scripts/validate_trained_models.py")
        logger.info("   2. Backtest señales: python scripts/backtest_models.py")
        logger.info("   3. Monitoreo en vivo: python scripts/live_monitoring.py")
        
        logger.info("=" * 70)

def main():
    """Función principal"""
    logger.info("🤖 NvBot3 - Entrenamiento Masivo de Modelos")
    logger.info("Procesando TODOS los datos de las 30 monedas")
    
    try:
        # Verificar entorno virtual
        if 'nvbot3_env' not in sys.executable:
            logger.error("❌ Entorno virtual nvbot3_env no está activo!")
            logger.error("   Ejecutar: nvbot3_env\\Scripts\\activate")
            return 1
        
        logger.info("✅ Entorno virtual nvbot3_env activo")
        
        # Crear entrenador
        trainer = ComprehensiveModelTrainer()
        
        # Verificar argumentos de línea de comandos
        import argparse
        parser = argparse.ArgumentParser(description='NvBot3 Model Training')
        parser.add_argument('--full', action='store_true', 
                          help='Entrenar todos los modelos desde cero')
        parser.add_argument('--resume', action='store_true', default=True,
                          help='Solo entrenar modelos faltantes (default)')
        parser.add_argument('--workers', type=int, default=2,
                          help='Número de workers paralelos (default: 2)')
        
        args = parser.parse_args()
        
        # Determinar modo
        resume_mode = not args.full
        
        if resume_mode:
            logger.info("🔄 MODO RECUPERACIÓN: Solo entrenando modelos faltantes")
        else:
            logger.info("🚀 MODO COMPLETO: Entrenando todos los modelos desde cero")
        
        # Entrenar modelos
        final_result = trainer.train_all_models_parallel(
            max_workers=args.workers, 
            resume_mode=resume_mode
        )
        
        if final_result['success']:
            if final_result.get('message') == 'All models already trained':
                logger.info("✅ Todos los modelos ya estaban entrenados!")
            else:
                logger.info("🎉 ¡ENTRENAMIENTO MASIVO COMPLETADO EXITOSAMENTE!")
            return 0
        elif final_result.get('interrupted'):
            logger.warning("⚠️ Entrenamiento interrumpido, pero checkpoint guardado")
            logger.info("💡 Para continuar: python scripts/train_all_models.py --resume")
            return 2
        else:
            logger.error("❌ Entrenamiento masivo falló")
            return 1
            
    except KeyboardInterrupt:
        logger.warning("⚠️ Entrenamiento interrumpido por usuario (Ctrl+C)")
        logger.info("💾 Los modelos completados han sido guardados")
        logger.info("💡 Para continuar: python scripts/train_all_models.py --resume")
        return 2
        
    except Exception as e:
        logger.error(f"💥 Error crítico: {e}")
        import traceback
        logger.error(f"Traceback:\n{traceback.format_exc()}")
        return 1

if __name__ == "__main__":
    exit(main())
