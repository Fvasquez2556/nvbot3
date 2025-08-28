"""
🚀 Procesador Completo de Datos - NvBot3
Pipeline completo de procesamiento de datos para entrenamiento

Ejecuta en orden:
1. Validación de datos (data_validator.py)
2. Cálculo de features (feature_calculator.py)  
3. Creación de targets (target_creator.py)
4. Resumen y estadísticas finales

Uso:
    python scripts/process_all_data.py --symbol BTCUSDT --timeframe 5m
    python scripts/process_all_data.py --all-symbols
    python scripts/process_all_data.py --quick-test  # Solo algunos símbolos para prueba
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import numpy as np
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import time
from datetime import datetime

# Importar nuestros módulos
from src.data.data_validator import DataValidator
from src.data.feature_calculator import FeatureCalculator
from src.data.target_creator import TargetCreator

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

class DataProcessor:
    """Procesador completo de datos para NvBot3"""
    
    def __init__(self):
        self.validator = DataValidator(data_path="data/raw")
        self.feature_calculator = FeatureCalculator(data_path="data/raw", output_path="data/processed")
        self.target_creator = TargetCreator(data_path="data/processed", output_path="data/processed")
        
        self.results = {}
        
    def process_single_symbol(self, symbol: str, timeframe: str) -> Dict:
        """Procesar un símbolo específico a través de todo el pipeline"""
        logger.info(f"🚀 === PROCESANDO {symbol}_{timeframe} ===")
        start_time = time.time()
        
        result = {
            'symbol': symbol,
            'timeframe': timeframe,
            'success': False,
            'steps_completed': [],
            'errors': [],
            'processing_time': 0,
            'final_stats': {}
        }
        
        try:
            # === PASO 1: Validación de datos ===
            logger.info("🔍 Paso 1/3: Validando datos...")
            df_raw = self.validator.load_data(symbol, timeframe)
            
            if df_raw is None:
                result['errors'].append("No se pudieron cargar datos raw")
                return result
                
            result['steps_completed'].append("data_validation")
            logger.info(f"✅ Datos validados: {len(df_raw)} registros")
            
            # === PASO 2: Cálculo de features ===
            logger.info("🧮 Paso 2/3: Calculando features técnicos...")
            df_features = self.feature_calculator.calculate_all_features(symbol, timeframe)
            
            if df_features is None:
                result['errors'].append("No se pudieron calcular features")
                return result
                
            # Guardar features
            self.feature_calculator.save_features(df_features, symbol, timeframe)
            result['steps_completed'].append("feature_calculation")
            
            features_added = len(df_features.columns) - len(df_raw.columns)
            logger.info(f"✅ Features calculadas: {features_added} indicadores añadidos")
            
            # === PASO 3: Creación de targets ===
            logger.info("🎯 Paso 3/3: Creando targets de entrenamiento...")
            
            # Determinar lookforward periods basado en timeframe
            if timeframe == '5m':
                momentum_periods = 48  # 4 horas
                rebound_periods = 24   # 2 horas
                regime_periods = 96    # 8 horas
            elif timeframe == '1h':
                momentum_periods = 4   # 4 horas
                rebound_periods = 2    # 2 horas
                regime_periods = 8     # 8 horas
            elif timeframe == '15m':
                momentum_periods = 16  # 4 horas
                rebound_periods = 8    # 2 horas
                regime_periods = 32    # 8 horas
            elif timeframe == '4h':
                momentum_periods = 1   # 4 horas
                rebound_periods = 1    # 4 horas (mismo período)
                regime_periods = 2     # 8 horas
            elif timeframe == '1d':
                momentum_periods = 1   # 1 día
                rebound_periods = 1    # 1 día
                regime_periods = 1     # 1 día
            else:
                # Default para otros timeframes
                momentum_periods = 24
                rebound_periods = 12
                regime_periods = 48
            
            # Crear targets
            momentum_target = self.target_creator.create_momentum_target(df_features, momentum_periods)
            rebound_target = self.target_creator.create_rebound_target(df_features, rebound_periods)
            regime_target = self.target_creator.create_regime_target(df_features, regime_periods)
            
            # Añadir targets al DataFrame
            df_final = df_features.copy()
            df_final['momentum_target'] = momentum_target
            df_final['rebound_target'] = rebound_target
            df_final['regime_target'] = regime_target
            
            # Crear advanced momentum target también
            advanced_momentum_target = self.target_creator.create_advanced_momentum_target(df_features, momentum_periods)
            df_final['momentum_advanced_target'] = advanced_momentum_target
            
            # Guardar dataset final con targets
            output_file = Path("data/processed") / f"{symbol}_{timeframe}_with_targets.csv"
            df_final.to_csv(output_file)
            
            result['steps_completed'].append("target_creation")
            logger.info(f"✅ Targets creados y guardados en: {output_file}")
            
            # === ESTADÍSTICAS FINALES ===
            processing_time = time.time() - start_time
            result['processing_time'] = processing_time
            result['success'] = True
            
            # Calcular estadísticas
            momentum_signals = momentum_target.sum()
            rebound_signals = rebound_target.sum()
            regime_signals = regime_target.value_counts().to_dict()
            advanced_momentum_signals = advanced_momentum_target.sum()
            
            momentum_ratio = momentum_target.mean()
            rebound_ratio = rebound_target.mean()
            regime_ratios = regime_target.value_counts(normalize=True).to_dict()
            advanced_momentum_ratio = advanced_momentum_target.mean()
            
            result['final_stats'] = {
                'total_records': len(df_final),
                'total_features': len(df_final.columns),
                'features_added': features_added,
                'momentum_signals': int(momentum_signals),
                'momentum_ratio': float(momentum_ratio),
                'rebound_signals': int(rebound_signals),
                'rebound_ratio': float(rebound_ratio),
                'regime_signals': {int(k): int(v) for k, v in regime_signals.items()},
                'regime_ratios': {int(k): float(v) for k, v in regime_ratios.items()},
                'advanced_momentum_signals': int(advanced_momentum_signals),
                'advanced_momentum_ratio': float(advanced_momentum_ratio),
                'data_quality': {
                    'missing_values': int(df_final.isnull().sum().sum()),
                    'missing_percentage': float((df_final.isnull().sum().sum() / (len(df_final) * len(df_final.columns))) * 100)
                }
            }
            
            # Crear resumen de targets
            summary_file = Path("data/processed") / f"{symbol}_{timeframe}_targets_summary.txt"
            with open(summary_file, 'w', encoding='utf-8') as f:
                f.write(f"Targets creados para {symbol}_{timeframe}\n")
                f.write(f"Total registros: {len(df_final)}\n")
                f.write(f"Total features: {len(df_final.columns)}\n")
                f.write(f"Rango de fechas: {df_final.index.min()} - {df_final.index.max()}\n\n")
                f.write(f"TARGETS:\n")
                f.write(f"🔥 Momentum (≥5%): {momentum_signals:,} señales ({momentum_ratio:.2%})\n")
                f.write(f"⚡ Rebotes (1-3%): {rebound_signals:,} señales ({rebound_ratio:.2%})\n")
                f.write(f"📊 Régimen: {regime_signals} distribución\n")
                f.write(f"🎯 Advanced Momentum: {advanced_momentum_signals:,} señales ({advanced_momentum_ratio:.2%})\n\n")
                f.write(f"CALIDAD DE DATOS:\n")
                f.write(f"Valores faltantes: {result['final_stats']['data_quality']['missing_percentage']:.3f}%\n")
                f.write(f"Tiempo de procesamiento: {processing_time:.1f} segundos\n")
            
            logger.info(f"📊 Resumen guardado en: {summary_file}")
            
            # Mostrar resumen en consola
            self.print_processing_summary(result)
            
        except Exception as e:
            result['errors'].append(str(e))
            logger.error(f"❌ Error procesando {symbol}_{timeframe}: {e}")
            
        return result
    
    def print_processing_summary(self, result: Dict):
        """Imprimir resumen del procesamiento"""
        symbol = result['symbol']
        timeframe = result['timeframe']
        
        print(f"\n{'='*80}")
        print(f"📊 RESUMEN DE PROCESAMIENTO: {symbol}_{timeframe}")
        print(f"{'='*80}")
        
        if result['success']:
            stats = result['final_stats']
            print(f"✅ Procesamiento EXITOSO")
            print(f"⏱️  Tiempo: {result['processing_time']:.1f} segundos")
            print(f"📈 Registros: {stats['total_records']:,}")
            print(f"🧮 Features: {stats['features_added']} añadidas ({stats['total_features']} total)")
            print(f"🎯 Calidad: {stats['data_quality']['missing_percentage']:.3f}% missing")
            
            print(f"\n🎯 TARGETS CREADOS:")
            print(f"  🔥 Momentum: {stats['momentum_signals']:,} señales ({stats['momentum_ratio']:.2%})")
            print(f"  ⚡ Rebotes:  {stats['rebound_signals']:,} señales ({stats['rebound_ratio']:.2%})")
            regime_summary = ", ".join([f"{k}:{v}" for k, v in stats['regime_signals'].items()])
            print(f"  📊 Régimen:  {regime_summary}")
            print(f"  🎯 Advanced: {stats['advanced_momentum_signals']:,} señales ({stats['advanced_momentum_ratio']:.2%})")
            
            print(f"\n✅ Pasos completados: {', '.join(result['steps_completed'])}")
        else:
            print(f"❌ Procesamiento FALLIDO")
            print(f"🔍 Pasos completados: {', '.join(result['steps_completed'])}")
            print(f"⚠️  Errores: {', '.join(result['errors'])}")
            
        print(f"{'='*80}")
    
    def process_multiple_symbols(self, symbols_timeframes: List[Tuple[str, str]]) -> Dict:
        """Procesar múltiples símbolos"""
        logger.info(f"🚀 === PROCESAMIENTO MASIVO: {len(symbols_timeframes)} símbolos ===")
        
        all_results = {}
        successful = 0
        total_time = 0
        
        for i, (symbol, timeframe) in enumerate(symbols_timeframes, 1):
            logger.info(f"\n📊 Procesando {i}/{len(symbols_timeframes)}: {symbol}_{timeframe}")
            
            result = self.process_single_symbol(symbol, timeframe)
            all_results[f"{symbol}_{timeframe}"] = result
            
            if result['success']:
                successful += 1
                total_time += result['processing_time']
        
        # Resumen general
        self.print_batch_summary(all_results, successful, len(symbols_timeframes), total_time)
        
        return all_results
    
    def print_batch_summary(self, results: Dict, successful: int, total: int, total_time: float):
        """Imprimir resumen de procesamiento en lote"""
        print(f"\n{'='*100}")
        print(f"📋 RESUMEN GENERAL DE PROCESAMIENTO EN LOTE")
        print(f"{'='*100}")
        
        print(f"📊 Símbolos procesados: {successful}/{total}")
        print(f"⏱️  Tiempo total: {total_time:.1f} segundos")
        
        if successful > 0:
            avg_time = total_time / successful
            print(f"⏱️  Tiempo promedio: {avg_time:.1f} segundos por símbolo")
            
            # Estadísticas agregadas
            total_records = sum(r['final_stats'].get('total_records', 0) for r in results.values() if r['success'])
            total_momentum_signals = sum(r['final_stats'].get('momentum_signals', 0) for r in results.values() if r['success'])
            total_rebound_signals = sum(r['final_stats'].get('rebound_signals', 0) for r in results.values() if r['success'])
            total_advanced_momentum_signals = sum(r['final_stats'].get('advanced_momentum_signals', 0) for r in results.values() if r['success'])
            
            print(f"\n📈 ESTADÍSTICAS AGREGADAS:")
            print(f"  📊 Total registros procesados: {total_records:,}")
            print(f"  🔥 Total señales momentum: {total_momentum_signals:,}")
            print(f"  ⚡ Total señales rebote: {total_rebound_signals:,}")
            print(f"  🎯 Total señales advanced: {total_advanced_momentum_signals:,}")
            
            print(f"\n✅ PROCESADOS EXITOSAMENTE:")
            for symbol, result in results.items():
                if result['success']:
                    stats = result['final_stats']
                    print(f"  • {symbol}: {stats['total_records']:,} registros, {stats['features_added']} features")
        
        failed = [symbol for symbol, result in results.items() if not result['success']]
        if failed:
            print(f"\n❌ ERRORES EN:")
            for symbol in failed:
                errors = ', '.join(results[symbol]['errors'])
                print(f"  • {symbol}: {errors}")
        
        print(f"{'='*100}")
    
    def get_available_symbols_timeframes(self) -> List[Tuple[str, str]]:
        """Obtener lista de símbolos y timeframes disponibles"""
        data_path = Path("data/raw")
        csv_files = list(data_path.glob("*.csv"))
        
        symbols_timeframes = []
        for csv_file in csv_files:
            filename = csv_file.stem
            parts = filename.split('_')
            
            if len(parts) >= 2:
                symbol = parts[0]
                timeframe = '_'.join(parts[1:])
                symbols_timeframes.append((symbol, timeframe))
        
        return symbols_timeframes


def main():
    parser = argparse.ArgumentParser(description='Procesador Completo de Datos NvBot3')
    parser.add_argument('--symbol', type=str, help='Símbolo específico (ej: BTCUSDT)')
    parser.add_argument('--timeframe', type=str, help='Timeframe específico (ej: 5m)')
    parser.add_argument('--all-symbols', action='store_true', help='Procesar todos los símbolos disponibles')
    parser.add_argument('--quick-test', action='store_true', help='Procesamiento rápido con pocos símbolos')
    
    args = parser.parse_args()
    
    # Crear procesador
    processor = DataProcessor()
    
    if args.all_symbols:
        # Procesar todos los símbolos
        symbols_timeframes = processor.get_available_symbols_timeframes()
        logger.info(f"🎯 Símbolos encontrados: {len(symbols_timeframes)}")
        
        if len(symbols_timeframes) > 20:
            response = input(f"⚠️  Se van a procesar {len(symbols_timeframes)} símbolos. ¿Continuar? (y/N): ")
            if response.lower() != 'y':
                print("❌ Procesamiento cancelado")
                return
        
        processor.process_multiple_symbols(symbols_timeframes)
        
    elif args.quick_test:
        # Procesamiento rápido con algunos símbolos
        test_symbols = [
            ('BTCUSDT', '5m'),
            ('ETHUSDT', '5m'),
            ('ADAUSDT', '5m'),
            ('BTCUSDT', '1h'),
            ('ETHUSDT', '1h')
        ]
        
        logger.info(f"🧪 Test rápido con {len(test_symbols)} símbolos")
        processor.process_multiple_symbols(test_symbols)
        
    elif args.symbol and args.timeframe:
        # Procesar símbolo específico
        processor.process_single_symbol(args.symbol, args.timeframe)
        
    else:
        print("❌ Especifica --symbol y --timeframe, o usa --all-symbols o --quick-test")
        print("\nEjemplos:")
        print("  python scripts/process_all_data.py --symbol BTCUSDT --timeframe 5m")
        print("  python scripts/process_all_data.py --quick-test")
        print("  python scripts/process_all_data.py --all-symbols")


if __name__ == "__main__":
    print("🚀 === PROCESADOR COMPLETO DE DATOS NVBOT3 ===")
    print(f"📅 Iniciado: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    
    main()
    
    print("\n🎯 === PROCESAMIENTO FINALIZADO ===")
    print(f"📅 Terminado: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
