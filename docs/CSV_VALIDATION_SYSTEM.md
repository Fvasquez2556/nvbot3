# 📊 Sistema de Validación CSV - NvBot3

## 🎯 Objetivo
Validar la compatibilidad de todos los archivos CSV con los componentes de entrenamiento del sistema NvBot3.

## 🔧 Componente Principal
**Archivo:** `scripts/validate_csv_format.py`

### 📋 Funcionalidades

#### 1. Validación Estructural
- ✅ Columnas OHLCV obligatorias
- ✅ Tipos de datos correctos
- ✅ Formato de timestamps
- ✅ Valores numéricos válidos

#### 2. Validación Temporal
- ✅ Continuidad temporal
- ✅ Detección de gaps
- ✅ Intervalos correctos por timeframe
- ✅ Orden cronológico

#### 3. Validación de Calidad
- ✅ Suficiencia de datos (200+ períodos para features)
- ✅ Datos recomendados (1000+ para entrenamiento óptimo)
- ✅ Detección de duplicados
- ✅ Valores extremos

#### 4. Compatibilidad de Componentes
- ✅ **Feature Calculator** - Verificación de períodos lookback
- ✅ **Target Creator** - Verificación de períodos lookforward
- ✅ **Model Trainer** - Verificación de suficiencia de datos

## 📊 Estado Actual del Sistema

### Resumen de Validación (2025-08-21)
```
📁 Total archivos: 150
✅ Archivos válidos: 150 (100%)
⚠️ Archivos con advertencias: 150
🚨 Errores críticos: 0
📈 Tasa de éxito: 100.0%
```

### Compatibilidad por Componente
```
🔧 Feature Calculator listos: 150/150
🎯 Target Creator listos: 150/150  
🚀 Listos para entrenamiento: 150/150
```

### Advertencias Comunes Identificadas
1. **extra_columns** (150 archivos): Columnas adicionales presentes pero no críticas
2. **temporal_gaps** (60 archivos): Gaps menores (<0.1%) en series temporales
3. **limited_training_data** (21 archivos): Archivos 1d con <1000 períodos
4. **duplicate_timestamps** (4 archivos): Timestamps duplicados mínimos

## 🔍 Criterios de Validación

### Períodos Mínimos por Timeframe
```python
LOOKFORWARD_PERIODS = {
    "5m": 48,   # 4 horas lookforward
    "15m": 16,  # 4 horas lookforward  
    "1h": 4,    # 4 horas lookforward
    "4h": 1,    # 4 horas lookforward
    "1d": 1     # 1 día lookforward
}
```

### Umbrales de Calidad
```python
MIN_RECORDS_FOR_FEATURES = 200    # Mínimo para calcular features
MIN_RECORDS_FOR_TRAINING = 1000   # Recomendado para entrenamiento
MAX_NULL_PERCENTAGE = 5.0         # Máximo % de valores nulos
MAX_EXTREME_CHANGE_PERCENTAGE = 1.0  # Máximo % cambios extremos
MAX_GAP_PERCENTAGE = 2.0          # Máximo % de gaps temporales
```

## 🚀 Uso del Sistema

### Validación Completa
```bash
python scripts/validate_csv_format.py
```

### Reportes Generados
1. **Log en tiempo real** - Progreso y alertas
2. **Reporte detallado JSON** - `logs/csv_validation_detailed_report.json`
3. **Estadísticas finales** - Resumen en consola

## 📁 Estructura de Archivos Validados

### Símbolos de Entrenamiento (30)
```
1INCHUSDT, AAVEUSDT, ADAUSDT, ALPHAUSDT, AVAXUSDT, BATUSDT, 
BNBUSDT, BTCUSDT, CHZUSDT, COMPUSDT, CRVUSDT, DOTUSDT, 
ENJUSDT, ETHUSDT, FETUSDT, IOTAUSDT, LINKUSDT, MANAUSDT, 
MATICUSDT, MKRUSDT, OCEANUSDT, SANDUSDT, SNXUSDT, SOLUSDT,
STORJUSDT, SUSHIUSDT, UNIUSDT, XRPUSDT, YFIUSDT, ZRXUSDT
```

### Timeframes por Símbolo (5)
```
5m, 15m, 1h, 4h, 1d
```

### Total de Archivos
```
30 símbolos × 5 timeframes = 150 archivos CSV
```

## ✅ Verificaciones Realizadas

### 1. Validación Estructural ✅
- [x] Columnas obligatorias presentes
- [x] Tipos de datos correctos
- [x] Formato temporal válido
- [x] Valores numéricos consistentes

### 2. Validación Temporal ✅
- [x] Continuidad temporal verificada
- [x] Intervalos correctos por timeframe
- [x] Orden cronológico confirmado
- [x] Gaps menores detectados y documentados

### 3. Validación de Calidad ✅
- [x] Suficiencia de datos confirmada
- [x] Duplicados mínimos identificados
- [x] Valores extremos dentro de rangos
- [x] Calidad general excelente

### 4. Compatibilidad de Componentes ✅
- [x] Feature Calculator: Compatible 150/150
- [x] Target Creator: Compatible 150/150
- [x] Model Trainer: Listo 150/150

## 🎯 Conclusiones

### ✅ Sistema Listo para Entrenamiento
1. **Todos los 150 archivos** están validados y listos
2. **100% de compatibilidad** con componentes de entrenamiento
3. **Advertencias menores** identificadas pero no bloquean el entrenamiento
4. **Calidad de datos excelente** para modelos de machine learning

### 📈 Próximos Pasos Recomendados
1. Proceder con **Feature Calculator**
2. Ejecutar **Target Creator** 
3. Iniciar **Model Training**
4. Monitorear performance en **Walk Forward Validation**

### 🔧 Mantenimiento del Sistema
- Ejecutar validación tras nuevas descargas de datos
- Monitorear advertencias de gaps temporales
- Actualizar umbrales según necesidades del modelo
- Revisar compatibilidad con nuevos componentes

---

**📅 Última Validación:** 2025-08-21 19:13:23  
**🎯 Estado:** ✅ SISTEMA COMPLETAMENTE VALIDADO  
**🚀 Acción:** LISTO PARA ENTRENAMIENTO
