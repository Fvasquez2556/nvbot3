# 🤖 NvBot3 - Historical Data Downloader

Bot de trading avanzado para criptomonedas que utiliza Inteligencia Artificial para detectar oportunidades de trading específicas.

## 🚀 Inicio Rápido

### 1. Setup Inicial

```bash
# 1. Ejecutar setup automático
python setup_project.py

# 2. Activar entorno virtual
nvbot3_env\Scripts\activate  # Windows
# source nvbot3_env/bin/activate  # Linux/Mac

# 3. Validar instalación
python scripts/validate_setup.py
```

### 2. Configuración

```bash
# Copiar y editar archivo de configuración
copy .env.example .env
# Editar .env con tus valores (API keys opcionales para datos históricos)
```

### 3. Descargar Datos Históricos

```bash
# Asegúrate de que el entorno virtual está activo
nvbot3_env\Scripts\activate

# Ejecutar descarga completa
python scripts/download_historical_data.py
```

## 📊 Características del Downloader

### ✅ Funcionalidades Implementadas

- **✅ Descarga por chunks** - Manejo eficiente de grandes volúmenes de datos
- **✅ Rate limiting** - Respeta límites de API de Binance (1200 req/min)
- **✅ Progress tracking** - Barras de progreso con `tqdm`
- **✅ Retry automático** - Backoff exponencial para errores
- **✅ Resume capability** - Continúa descarga desde última fecha
- **✅ Data validation** - Verificación de calidad de datos
- **✅ Memory efficiency** - Procesamiento optimizado para laptop
- **✅ Extensive logging** - Logging detallado de todas las operaciones

### 📋 Especificaciones Técnicas

- **Símbolos**: BTCUSDT, ETHUSDT, BNBUSDT, ADAUSDT, SOLUSDT
- **Timeframes**: 5m, 15m, 1h, 4h, 1d
- **Período**: Desde enero 2022 hasta presente
- **Formato**: CSV en `data/raw/{symbol}_{timeframe}.csv`
- **Rate limiting**: Máximo 1200 requests/minuto

### 🔧 Estructura de Datos

```python
columns = [
    'timestamp', 'open', 'high', 'low', 'close', 'volume',
    'close_time', 'quote_asset_volume', 'number_of_trades',
    'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume'
]
```

## 📁 Estructura del Proyecto

```
nvbot3/
├── src/
│   ├── data/                    # Gestión de datos
│   ├── models/                  # Modelos ML
│   ├── analysis/                # Análisis técnico
│   └── utils/                   # Utilidades
├── config/
│   └── training_config.yaml     # Configuración principal
├── data/
│   ├── raw/                     # Datos descargados
│   ├── processed/               # Datos procesados
│   └── models/                  # Modelos entrenados
├── scripts/
│   ├── download_historical_data.py  # ⭐ Módulo principal
│   └── validate_setup.py        # Validador de setup
└── tests/                       # Tests unitarios
```

## ⚙️ Configuración Avanzada

### Variables de Entorno (.env)

```bash
# Binance API (opcional para datos históricos)
BINANCE_API_KEY=your_api_key_here
BINANCE_SECRET_KEY=your_secret_key_here

# Sistema
LOG_LEVEL=INFO
MAX_THREADS=4
MAX_PROCESSES=2
```

### Configuración YAML (config/training_config.yaml)

```yaml
data:
  symbols: ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'ADAUSDT', 'SOLUSDT']
  timeframes: ['5m', '15m', '1h', '4h', '1d']
  start_date: '2022-01-01'

api:
  binance:
    rate_limit_per_minute: 1200
    chunk_size_days: 30
    max_retries: 3
```

## 🔍 Validaciones Implementadas

### Validación de Datos
- ✅ Sin gaps > 2 períodos consecutivos
- ✅ No cambios de precio > 50% en un período
- ✅ Volumen > 0 en 95%+ de registros
- ✅ Detección de outliers extremos
- ✅ Consistencia temporal de timestamps

### Validación de Setup
- ✅ Entorno virtual activo
- ✅ Todas las dependencias instaladas
- ✅ Estructura de directorios correcta
- ✅ Archivos de configuración válidos

## 🚦 Estados y Logging

### Mensajes de Estado
```
🔄 Descargando BTCUSDT 5m: 45 chunks
✅ BTCUSDT 5m: 12,450 registros guardados
⚠️  Datos con problemas para ETHUSDT 1h: ['3 gaps grandes en timestamps']
❌ Error descargando BNBUSDT 4h: Rate limit exceeded
```

### Archivos de Log
- `data/download_log.txt` - Log detallado de todas las operaciones
- Console output - Progreso en tiempo real con `tqdm`

## 🎯 Uso del Módulo

### Uso Básico
```python
from scripts.download_historical_data import HistoricalDataDownloader

# Crear instancia
downloader = HistoricalDataDownloader()

# Descargar símbolo específico
success = downloader.download_symbol_timeframe('BTCUSDT', '5m', '2022-01-01')

# Descargar todo según configuración
results = downloader.download_all_data()
```

### Uso Avanzado
```python
# Configuración personalizada
downloader = HistoricalDataDownloader('config/custom_config.yaml')

# Descarga con fechas específicas
success = downloader.download_symbol_timeframe(
    'ETHUSDT', '1h', '2023-01-01', '2023-12-31'
)
```

## 🔧 Troubleshooting

### Problema: Entorno virtual no activo
```bash
❌ ERROR: Entorno virtual nvbot3_env no está activo!
```
**Solución:**
```bash
nvbot3_env\Scripts\activate  # Windows
source nvbot3_env/bin/activate  # Linux/Mac
```

### Problema: Dependencias faltantes
```bash
❌ Import "ccxt" could not be resolved
```
**Solución:**
```bash
pip install -r requirements.txt
```

### Problema: Rate limiting
```bash
❌ Error descargando BTCUSDT 5m: Rate limit exceeded
```
**Solución:** El módulo maneja automáticamente rate limiting con backoff exponencial.

### Problema: Gaps en datos
```bash
⚠️ Datos con problemas: ['5 gaps grandes en timestamps']
```
**Solución:** El módulo detecta y reporta gaps, pero continúa la descarga.

## 📈 Próximos Módulos

1. **Data Validator** (`src/data/data_validator.py`)
2. **Feature Calculator** (`src/data/feature_calculator.py`) 
3. **Target Creator** (`src/data/target_creator.py`)
4. **Model Trainer** (`scripts/train_models.py`)

## 🤝 Contribución

Este módulo sigue las especificaciones del documento "Instrucciones para GitHub Copilot - Training Pipeline NvBot3" y está diseñado para ser:

- ✅ **Robusto** - Manejo extensivo de errores
- ✅ **Eficiente** - Optimizado para recursos de laptop
- ✅ **Transparente** - Logging detallado y progress tracking
- ✅ **Resumible** - Capacidad de continuar descargas interrumpidas
- ✅ **Configurable** - Todo parametrizable via YAML

---

**⚠️ IMPORTANTE**: Siempre activar el entorno virtual antes de cualquier operación:
```bash
nvbot3_env\Scripts\activate
```
