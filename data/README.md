# 📊 Directorio de Datos - NvBot3

## 📁 Estructura de Directorios

```
data/
├── raw/           # Datos históricos descargados de Binance (NO incluidos en Git)
├── processed/     # Datos procesados con features (NO incluidos en Git)
├── models/        # Modelos entrenados guardados (NO incluidos en Git)
└── README.md      # Este archivo (SÍ incluido en Git)
```

## ⚠️ **Archivos NO incluidos en el repositorio**

Los archivos de datos **NO están incluidos en el repositorio Git** por las siguientes razones:

### 🚫 Por qué NO subimos los datos:
- **Tamaño**: Los archivos pueden ser de varios GB (ej: 45k registros = ~6MB cada uno)
- **Regenerable**: Los datos se pueden descargar nuevamente usando los scripts
- **Actualización**: Los datos cambian constantemente (nuevos datos cada día)
- **Bandwidth**: Evitar consumir ancho de banda innecesario
- **Performance**: Clonar el repo sería muy lento con archivos grandes

## 🔄 **Cómo regenerar los datos**

### 1. Configurar claves API
```bash
# Copia el archivo de ejemplo
cp .env.example .env

# Edita .env con tus claves reales de Binance
# BINANCE_API_KEY=tu_clave_aqui
# BINANCE_SECRET_KEY=tu_secret_aqui
```

### 2. Descargar datos históricos
```bash
# Activar entorno virtual
nvbot3_env\Scripts\activate  # Windows

# Descargar un símbolo específico
python scripts/download_historical_data.py --symbol BTCUSDT --timeframe 5m

# Descargar todos los símbolos y timeframes
python scripts/download_historical_data.py --download-all

# Validar datos existentes
python scripts/download_historical_data.py --validate-only
```

## 📈 **Datos Disponibles**

### Símbolos configurados:
- **BTCUSDT** - Bitcoin/Tether
- **ETHUSDT** - Ethereum/Tether  
- **BNBUSDT** - Binance Coin/Tether
- **ADAUSDT** - Cardano/Tether
- **SOLUSDT** - Solana/Tether

### Timeframes disponibles:
- **5m** - 5 minutos (~240k registros, ~30MB)
- **15m** - 15 minutos (~80k registros, ~10MB)
- **1h** - 1 hora (~32k registros, ~4MB)
- **4h** - 4 horas (~8k registros, ~1MB)
- **1d** - 1 día (~1.3k registros, ~200KB)

### Período de datos:
- **Desde**: 1 enero 2022
- **Hasta**: Presente (se actualiza automáticamente)
- **Total**: ~3.5+ años de datos históricos

## 📊 **Formato de Datos**

Cada archivo CSV contiene las siguientes columnas:
```
timestamp                    - Marca de tiempo
open                        - Precio de apertura
high                        - Precio máximo
low                         - Precio mínimo
close                       - Precio de cierre
volume                      - Volumen base
close_time                  - Tiempo de cierre
quote_asset_volume          - Volumen quote
number_of_trades            - Número de trades
taker_buy_base_asset_volume - Volumen base comprador
taker_buy_quote_asset_volume- Volumen quote comprador
```

## 🔍 **Validación de Datos**

Los datos descargados incluyen validaciones automáticas:

- ✅ **Consistencia temporal**: Sin gaps en timestamps
- ✅ **Integridad OHLC**: Open ≤ High, Low ≤ Close, etc.
- ✅ **Volumen válido**: Volumen > 0
- ✅ **Duplicados**: Sin registros duplicados
- ✅ **Completitud**: Verificación de registros esperados

## 🚀 **Uso en Desarrollo**

```python
# Cargar datos en pandas
import pandas as pd

# Cargar datos de 5 minutos de Bitcoin
df = pd.read_csv('data/raw/BTCUSDT_5m.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])
df.set_index('timestamp', inplace=True)

print(f"Registros: {len(df)}")
print(f"Período: {df.index.min()} a {df.index.max()}")
```

## 📝 **Notas Importantes**

1. **Primer setup**: Ejecuta `python scripts/download_historical_data.py --download-all` después de clonar
2. **Actualizaciones**: Los scripts pueden resumir descargas interrumpidas
3. **Rate limits**: Los scripts respetan los límites de Binance (1200 req/min)
4. **Tamaño total**: ~50-100MB por símbolo completo (todos los timeframes)
5. **Tiempo descarga**: ~2-5 minutos por símbolo completo

## 🛠️ **Troubleshooting**

### Error "Claves API no configuradas"
- Verifica que `.env` existe y tiene las claves correctas
- Asegúrate de estar en el directorio raíz del proyecto

### Error "Rate limit exceeded"
- Los scripts incluyen rate limiting automático
- Espera unos minutos y vuelve a intentar

### Archivos corruptos o incompletos
```bash
# Re-descargar archivo específico
python scripts/download_historical_data.py --symbol BTCUSDT --timeframe 5m --force

# Validar todos los archivos
python scripts/download_historical_data.py --validate-only
```

---

💡 **Tip**: Mantén los datos locales actualizados ejecutando las descargas periódicamente para obtener los datos más recientes.
