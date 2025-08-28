#!/usr/bin/env python3
"""
🔍 NvBot3 - Verificación Estado de Modelos
Verifica qué modelos están completos y cuáles faltan
"""

import os
import glob
from datetime import datetime
from pathlib import Path

def analyze_training_status():
    """Analizar el estado actual del entrenamiento"""
    print("🎯 ANÁLISIS ESTADO DE ENTRENAMIENTO NVBOT3")
    print("=" * 60)
    
    # Verificar modelos guardados
    model_files = glob.glob('data/models/ALL_SYMBOLS_*.pkl')
    model_files = [f for f in model_files if not f.endswith('_metrics.pkl')]
    
    existing_models = {}
    for model_path in model_files:
        filename = os.path.basename(model_path)
        parts = filename.replace('.pkl', '').split('_')
        if len(parts) >= 4:
            timeframe = parts[2]
            model_type = '_'.join(parts[3:])
            
            # Obtener info del archivo
            stat = os.stat(model_path)
            size_mb = stat.st_size / (1024 * 1024)
            modified = datetime.fromtimestamp(stat.st_mtime)
            
            existing_models[f"{model_type}_{timeframe}"] = {
                'path': model_path,
                'size_mb': size_mb,
                'modified': modified
            }
    
    # Organizar por tipo
    timeframes = ['5m', '15m', '1h', '4h', '1d']
    model_types = ['momentum', 'rebound', 'regime', 'momentum_advanced']
    
    print(f"📊 RESUMEN GENERAL:")
    print(f"   Total modelos encontrados: {len(existing_models)}")
    print(f"   Modelos esperados: {len(timeframes) * len(model_types)} (20)")
    print(f"   Progreso: {len(existing_models)}/20 ({len(existing_models)/20*100:.1f}%)")
    print()
    
    # Mostrar estado por tipo
    for model_type in model_types:
        print(f"🔧 {model_type.upper()} MODELS:")
        completed = 0
        for tf in timeframes:
            model_key = f"{model_type}_{tf}"
            if model_key in existing_models:
                model_info = existing_models[model_key]
                print(f"   ✅ {tf}: {model_info['size_mb']:.1f}MB - {model_info['modified'].strftime('%Y-%m-%d %H:%M')}")
                completed += 1
            else:
                print(f"   ❌ {tf}: PENDIENTE")
        print(f"   📈 Progreso: {completed}/{len(timeframes)} ({completed/len(timeframes)*100:.0f}%)")
        print()
    
    # Modelos pendientes
    all_expected = set()
    for model_type in model_types:
        for tf in timeframes:
            all_expected.add(f"{model_type}_{tf}")
    
    pending = all_expected - set(existing_models.keys())
    
    if pending:
        print("⏳ MODELOS PENDIENTES:")
        for model in sorted(pending):
            model_type, tf = model.rsplit('_', 1)
            priority = "🔥 ALTA" if model_type == 'regime' else "📊 MEDIA"
            print(f"   ❌ {model} - Prioridad: {priority}")
        print()
    
    # Recomendaciones
    print("💡 RECOMENDACIONES:")
    if len(existing_models) >= 10:
        print("   ✅ Ya tienes suficientes modelos para trading básico")
        print("   ✅ Puedes usar los modelos actuales mientras completas los faltantes")
    
    if 'regime_1h' not in existing_models:
        print("   🔥 Prioridad: Completar modelos REGIME (usan LSTM, más lentos)")
    
    if any('momentum_advanced' in model for model in pending):
        print("   🚀 Los modelos ADVANCED son Ensemble (combinan múltiples modelos)")
    
    print()
    print("🔄 PARA CONTINUAR:")
    print("   1. Ejecutar: python scripts/resume_training.py")
    print("   2. O entrenar modelos específicos individualmente")
    print("   3. Los modelos existentes ya son funcionales para trading")

if __name__ == "__main__":
    # Cambiar al directorio raíz del proyecto
    os.chdir(Path(__file__).parent.parent)
    analyze_training_status()
