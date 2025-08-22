#!/usr/bin/env python3
"""
Analizador de Problemas de Validación para NvBot3

Analiza los errores de validación de datos históricos para determinar
si son problemas reales o volatilidad extrema normal en crypto.
"""

import os
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Tuple
import yaml

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ValidationAnalyzer:
    def __init__(self, data_dir: str = "data/raw", config_path: str = "config/training_config.yaml"):
        self.data_dir = Path(data_dir)
        self.config_path = Path(config_path)
        
        # Cargar configuración
        with open(self.config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        # Monedas de entrenamiento
        self.training_symbols = []
        for tier in ['training_tier_1', 'training_tier_2', 'training_tier_3']:
            if tier in self.config['models']['training_symbols']:
                self.training_symbols.extend(self.config['models']['training_symbols'][tier])
        
        # Timeframes
        self.timeframes = self.config['data']['timeframes']
        
        logger.info(f"Analizando {len(self.training_symbols)} símbolos en {len(self.timeframes)} timeframes")

    def analyze_price_extremes(self, df: pd.DataFrame, symbol: str, timeframe: str) -> Dict:
        """Analiza cambios de precio extremos en los datos"""
        if df.empty or len(df) < 2:
            return {"error": "DataFrame vacío o insuficiente"}
        
        # Calcular cambios porcentuales
        df['pct_change'] = df['close'].pct_change()
        
        # Cambios extremos (>50% en una vela)
        extreme_changes = df[abs(df['pct_change']) > 0.5]
        
        # Análisis estadístico
        stats = {
            "symbol": symbol,
            "timeframe": timeframe,
            "total_records": len(df),
            "extreme_changes": len(extreme_changes),
            "extreme_percentage": len(extreme_changes) / len(df) * 100,
            "max_up_change": df['pct_change'].max() * 100 if not df['pct_change'].isna().all() else 0,
            "max_down_change": df['pct_change'].min() * 100 if not df['pct_change'].isna().all() else 0,
            "volatility_std": df['pct_change'].std() * 100 if not df['pct_change'].isna().all() else 0,
            "date_range": f"{df['timestamp'].min()} to {df['timestamp'].max()}"
        }
        
        # Identificar fechas de cambios extremos
        if len(extreme_changes) > 0:
            stats["extreme_dates"] = extreme_changes[['timestamp', 'pct_change']].to_dict('records')[:5]  # Top 5
        
        return stats

    def check_data_quality(self, df: pd.DataFrame, symbol: str, timeframe: str) -> Dict:
        """Verifica la calidad general de los datos"""
        quality = {
            "symbol": symbol,
            "timeframe": timeframe,
            "has_nulls": df.isnull().any().any(),
            "duplicate_timestamps": df['timestamp'].duplicated().sum(),
            "zero_volume": (df['volume'] == 0).sum(),
            "negative_prices": (df[['open', 'high', 'low', 'close']] <= 0).any().any(),
            "price_consistency": True,  # high >= low, etc.
            "gaps_in_time": False
        }
        
        # Verificar consistencia de precios OHLC
        if not df.empty:
            invalid_ohlc = (
                (df['high'] < df['low']) |
                (df['high'] < df['open']) |
                (df['high'] < df['close']) |
                (df['low'] > df['open']) |
                (df['low'] > df['close'])
            ).sum()
            quality["price_consistency"] = invalid_ohlc == 0
            quality["invalid_ohlc_count"] = invalid_ohlc
        
        return quality

    def analyze_file(self, file_path: Path) -> Dict:
        """Analiza un archivo de datos específico"""
        try:
            # Extraer símbolo y timeframe del nombre del archivo
            filename = file_path.stem  # sin extensión
            parts = filename.split('_')
            if len(parts) >= 2:
                symbol = parts[0]
                timeframe = parts[1]
            else:
                logger.warning(f"No se pudo extraer símbolo/timeframe de {filename}")
                return {
                    "file_path": str(file_path),
                    "error": f"No se pudo extraer símbolo/timeframe de {filename}"
                }
            
            # Leer datos
            df = pd.read_csv(file_path)
            
            # Convertir timestamp si es necesario
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            elif 'date' in df.columns:
                df['timestamp'] = pd.to_datetime(df['date'])
                
            # Análisis de extremos y calidad
            extremes_analysis = self.analyze_price_extremes(df, symbol, timeframe)
            quality_analysis = self.check_data_quality(df, symbol, timeframe)
            
            return {
                "file_path": str(file_path),
                "extremes": extremes_analysis,
                "quality": quality_analysis,
                "file_size_mb": file_path.stat().st_size / (1024 * 1024)
            }
            
        except Exception as e:
            logger.error(f"Error analizando {file_path}: {e}")
            return {
                "file_path": str(file_path),
                "error": str(e)
            }

    def analyze_all_files(self) -> Dict:
        """Analiza todos los archivos de datos disponibles"""
        results = {
            "analysis_summary": {
                "total_files_analyzed": 0,
                "files_with_errors": 0,
                "files_with_extremes": 0,
                "symbols_analyzed": set(),
                "timeframes_analyzed": set()
            },
            "detailed_results": []
        }
        
        # Buscar todos los archivos CSV
        csv_files = list(self.data_dir.glob("*.csv"))
        logger.info(f"Encontrados {len(csv_files)} archivos CSV")
        
        for file_path in csv_files:
            analysis = self.analyze_file(file_path)
            if analysis:
                results["detailed_results"].append(analysis)
                results["analysis_summary"]["total_files_analyzed"] += 1
                
                if "error" in analysis:
                    results["analysis_summary"]["files_with_errors"] += 1
                else:
                    if analysis["extremes"]["extreme_changes"] > 0:
                        results["analysis_summary"]["files_with_extremes"] += 1
                    
                    results["analysis_summary"]["symbols_analyzed"].add(analysis["extremes"]["symbol"])
                    results["analysis_summary"]["timeframes_analyzed"].add(analysis["extremes"]["timeframe"])
        
        # Convertir sets a listas para serialización
        results["analysis_summary"]["symbols_analyzed"] = sorted(list(results["analysis_summary"]["symbols_analyzed"]))
        results["analysis_summary"]["timeframes_analyzed"] = sorted(list(results["analysis_summary"]["timeframes_analyzed"]))
        
        return results

    def generate_recommendations(self, analysis_results: Dict) -> List[str]:
        """Genera recomendaciones basadas en el análisis"""
        recommendations = []
        
        # Resumen
        total = analysis_results["analysis_summary"]["total_files_analyzed"]
        with_extremes = analysis_results["analysis_summary"]["files_with_extremes"]
        
        if total > 0:
            extreme_percentage = (with_extremes / total) * 100
            
            if extreme_percentage > 50:
                recommendations.append(
                    "⚠️ CRÍTICO: Más del 50% de archivos tienen cambios de precio extremos. "
                    "Considerar ajustar parámetros de validación."
                )
            elif extreme_percentage > 25:
                recommendations.append(
                    "⚠️ ADVERTENCIA: 25-50% de archivos tienen cambios extremos. "
                    "Revisar configuración de validación."
                )
            else:
                recommendations.append(
                    "✅ NORMAL: Menos del 25% de archivos con cambios extremos. "
                    "Parámetros de validación apropiados."
                )
        
        # Análisis detallado de los archivos con más problemas
        problematic_files = [
            r for r in analysis_results["detailed_results"] 
            if "extremes" in r and r["extremes"]["extreme_changes"] > 5
        ]
        
        if problematic_files:
            recommendations.append(
                f"🔍 REVISAR: {len(problematic_files)} archivos con >5 cambios extremos:"
            )
            for f in sorted(problematic_files, key=lambda x: x["extremes"]["extreme_changes"], reverse=True)[:5]:
                recommendations.append(
                    f"   • {f['extremes']['symbol']}_{f['extremes']['timeframe']}: "
                    f"{f['extremes']['extreme_changes']} cambios extremos"
                )
        
        # Recomendaciones específicas
        high_volatility_symbols = set()
        for result in analysis_results["detailed_results"]:
            if "extremes" in result and result["extremes"]["volatility_std"] > 10:
                high_volatility_symbols.add(result["extremes"]["symbol"])
        
        if high_volatility_symbols:
            recommendations.append(
                f"🎯 SÍMBOLOS ALTA VOLATILIDAD: {', '.join(sorted(high_volatility_symbols))}"
            )
            recommendations.append(
                "💡 SUGERENCIA: Ajustar threshold de validación para estos símbolos o "
                "usar parámetros de validación más flexibles."
            )
        
        return recommendations

    def save_analysis_report(self, analysis_results: Dict, output_path: str = "validation_analysis_report.txt"):
        """Guarda el reporte de análisis en un archivo"""
        recommendations = self.generate_recommendations(analysis_results)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("REPORTE DE ANÁLISIS DE VALIDACIÓN - NvBot3\n")
            f.write("=" * 80 + "\n\n")
            
            # Resumen
            f.write("📊 RESUMEN EJECUTIVO:\n")
            summary = analysis_results["analysis_summary"]
            f.write(f"   • Archivos analizados: {summary['total_files_analyzed']}\n")
            f.write(f"   • Archivos con errores: {summary['files_with_errors']}\n")
            f.write(f"   • Archivos con extremos: {summary['files_with_extremes']}\n")
            f.write(f"   • Símbolos únicos: {len(summary['symbols_analyzed'])}\n")
            f.write(f"   • Timeframes únicos: {len(summary['timeframes_analyzed'])}\n\n")
            
            # Recomendaciones
            f.write("🎯 RECOMENDACIONES:\n")
            for i, rec in enumerate(recommendations, 1):
                f.write(f"{i}. {rec}\n")
            f.write("\n")
            
            # Detalle de archivos problemáticos
            problematic = [
                r for r in analysis_results["detailed_results"]
                if "extremes" in r and r["extremes"]["extreme_changes"] > 0
            ]
            
            if problematic:
                f.write("🔍 ARCHIVOS CON CAMBIOS EXTREMOS:\n")
                f.write("-" * 60 + "\n")
                for result in sorted(problematic, key=lambda x: x["extremes"]["extreme_changes"], reverse=True):
                    extremes = result["extremes"]
                    f.write(f"Archivo: {extremes['symbol']}_{extremes['timeframe']}.csv\n")
                    f.write(f"   Cambios extremos: {extremes['extreme_changes']}\n")
                    f.write(f"   Volatilidad std: {extremes['volatility_std']:.2f}%\n")
                    f.write(f"   Mayor subida: {extremes['max_up_change']:.2f}%\n")
                    f.write(f"   Mayor bajada: {extremes['max_down_change']:.2f}%\n")
                    f.write(f"   Rango de fechas: {extremes['date_range']}\n")
                    f.write("-" * 40 + "\n")
        
        logger.info(f"Reporte guardado en: {output_path}")

def main():
    """Función principal"""
    logger.info("🔍 Iniciando análisis de validación de datos...")
    
    analyzer = ValidationAnalyzer()
    analysis_results = analyzer.analyze_all_files()
    
    # Generar reporte
    analyzer.save_analysis_report(analysis_results)
    
    # Mostrar recomendaciones en consola
    recommendations = analyzer.generate_recommendations(analysis_results)
    
    print("\n" + "=" * 80)
    print("🎯 RECOMENDACIONES PARA VALIDACIÓN:")
    print("=" * 80)
    for i, rec in enumerate(recommendations, 1):
        print(f"{i}. {rec}")
    
    print(f"\n📋 Reporte completo guardado en: validation_analysis_report.txt")
    
    # Estadísticas rápidas
    summary = analysis_results["analysis_summary"]
    print(f"\n📊 ESTADÍSTICAS RÁPIDAS:")
    print(f"   • Total archivos: {summary['total_files_analyzed']}")
    print(f"   • Con extremos: {summary['files_with_extremes']} ({summary['files_with_extremes']/summary['total_files_analyzed']*100:.1f}%)")
    print(f"   • Con errores: {summary['files_with_errors']}")
    
    logger.info("✅ Análisis completado")

if __name__ == "__main__":
    main()
