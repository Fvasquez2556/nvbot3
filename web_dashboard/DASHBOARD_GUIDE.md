# 📊 NvBot3 Dashboard - Guía de Usuario

## 🎯 Dashboard Unificado

El **NvBot3 Dashboard** es un sistema completo que combina análisis en tiempo real con seguimiento histórico, diseñado para traders de criptomonedas.

---

## 🚀 Características Principales

### ⚡ Panel de Tiempo Real
- **Monitoreo en Vivo**: Actualizaciones automáticas cada 30 segundos vía WebSocket
- **Categorización Inteligente**: Señales organizadas por Momentum, Rebotes y Tendencias
- **Top Coins**: Las 6 criptomonedas con mayor confianza de subida
- **Estadísticas en Vivo**: Contadores dinámicos de señales activas

### 📈 Panel de Análisis Histórico
- **Sistema de Feedback**: Evalúa el rendimiento de predicciones pasadas
- **Métricas de Rendimiento**: Tasa de éxito, confianza promedio
- **Seguimiento de Señales**: Monitoreo del ciclo de vida completo
- **Aprendizaje Automatizado**: Mejora continua basada en resultados reales

---

## 🎨 Interfaz de Usuario

### Navegación por Pestañas
```
📈 Tiempo Real        📊 Análisis Histórico
     ↓                       ↓
Señales Activas        Señales con Feedback
Top Confianza         Métricas Performance
Categorización        Sistema Aprendizaje
```

### Código de Colores
- 🟢 **Verde**: Señales de Momentum (alta velocidad)
- 🔵 **Azul**: Señales de Rebound (rebote técnico)
- 🟡 **Amarillo**: Señales de Trending (tendencia establecida)
- 🔴 **Rojo**: Alertas o señales fallidas

---

## 🔧 Integración con NvBot3

### Para Tiempo Real (Requerido)
```python
# En tu archivo de escaneo principal
from web_dashboard.database.signal_tracker import SignalTracker

# Inicializar tracker
tracker = SignalTracker()

# Al generar una señal
signal_data = {
    'symbol': 'BTCUSDT',
    'signal_type': 'momentum',
    'confidence_score': 0.85,
    'predicted_change': 2.5,
    'entry_price': 45000.0,
    'timeframe': '5m'
}

# Registrar señal
tracker.add_signal(**signal_data)
```

### Para Análisis Histórico (Opcional)
```python
# En tu función de validación de resultados
def validate_signal_result(signal_id, actual_result):
    feedback_type = 'success' if actual_result > 0 else 'failed'
    tracker.add_feedback(signal_id, feedback_type, 'Auto-validation')
```

---

## 📱 Uso del Dashboard

### 1. Iniciar el Dashboard
```bash
cd web_dashboard
python app.py
```

### 2. Acceder al Dashboard
- **URL Principal**: http://localhost:5000
- **Dashboard**: Navegación automática entre pestañas
- **API Endpoints**: Disponibles para integración externa

### 3. Pestañas Disponibles

#### 🚀 Tiempo Real
- Visualiza señales **activas** generadas por tu NvBot3
- Monitoreo de los **60 símbolos** configurados
- **Actualizaciones automáticas** cada 30 segundos
- **Top 6 monedas** con mayor potencial de subida

#### 📊 Análisis Histórico  
- Revisa **señales pasadas** que necesitan evaluación
- Proporciona **feedback manual** (Exitosa/Parcial/Fallida)
- Analiza **métricas de rendimiento** del bot
- **Mejora el algoritmo** con retroalimentación

---

## 🎯 Flujo de Trabajo Recomendado

### Mañana (Revisión Estratégica)
1. Abrir **Panel Análisis Histórico**
2. Revisar señales del día anterior
3. Proporcionar feedback a señales resueltas
4. Analizar métricas de rendimiento

### Durante el Día (Monitoreo Activo)
1. Mantener abierto **Panel Tiempo Real**
2. Monitorear Top Coins con alta confianza
3. Observar nuevas señales por categoría
4. Tomar decisiones basadas en categorización

### Noche (Análisis y Configuración)
1. Revisar estadísticas del día completo
2. Analizar patrones en categorías exitosas
3. Ajustar configuraciones si es necesario
4. Preparar para siguiente sesión

---

## 📊 Métricas Importantes

### Indicadores de Confianza
- **🟢 Alta (>80%)**: Señales muy confiables
- **🟡 Media (60-80%)**: Señales moderadas  
- **🔴 Baja (<60%)**: Señales de riesgo

### Categorías de Señales
- **Momentum**: Cambios rápidos de precio (scalping)
- **Rebound**: Rebotes técnicos en soportes/resistencias
- **Trending**: Movimientos direccionales sostenidos

### Estadísticas Clave
- **Total Señales**: Volumen de actividad del bot
- **Tasa Éxito**: Porcentaje de predicciones correctas
- **Confianza Promedio**: Nivel de certeza del algoritmo
- **Señales Activas**: Predicciones en monitoreo

---

## 🔗 URLs y Endpoints

### Dashboard Principal
- **Interfaz**: http://localhost:5000
- **WebSocket**: Conexión automática para tiempo real

### API Endpoints
- **Señales Categorizadas**: `/api/signals/categorized`
- **Top Confianza**: `/api/signals/top-confidence`  
- **Estadísticas**: `/api/realtime/stats`
- **Feedback**: `/api/feedback` (POST)

### Archivos Estáticos
- **CSS**: `/static/css/dashboard.css`
- **JavaScript**: `/static/js/dashboard.js`
- **Temas**: `/static/js/theme-manager.js`

---

## 🛠️ Solución de Problemas

### Dashboard No Carga
```bash
# Verificar puerto disponible
netstat -an | findstr 5000

# Reiniciar con puerto diferente
python app.py --port 5001
```

### Sin Señales en Tiempo Real
```python
# Verificar integración en tu código principal
from web_dashboard.database.signal_tracker import SignalTracker
tracker = SignalTracker()

# Test manual de señal
tracker.add_signal(
    symbol='TESTUSDT',
    signal_type='momentum', 
    confidence_score=0.75,
    predicted_change=1.5,
    entry_price=100.0
)
```

### WebSocket No Conecta
1. Verificar que Flask-SocketIO esté instalado
2. Comprobar firewall/antivirus
3. Usar http://127.0.0.1:5000 en lugar de localhost

---

## 📈 Optimización y Performance

### Configuración Recomendada
- **Máximo 100 señales activas** simultáneamente
- **Limpieza automática** de señales antigas (>24h)
- **Cache inteligente** para mejorar velocidad
- **Compresión WebSocket** para datos grandes

### Integración Avanzada
```python
# Configuración avanzada del tracker
tracker = SignalTracker(
    max_active_signals=100,
    auto_cleanup_hours=24,
    confidence_threshold=0.6
)

# Filtrado por timeframe
signals_5m = tracker.get_signals_by_timeframe('5m')
signals_1h = tracker.get_signals_by_timeframe('1h')
```

---

## 🎉 Próximas Características

### Versión 2.0 (Planificado)
- 📱 **App Móvil**: Dashboard responsive para móviles
- 🔔 **Notificaciones Push**: Alertas inmediatas de señales
- 📊 **Gráficos Avanzados**: Chart.js con análisis técnico
- 🤖 **AI Insights**: Comentarios automáticos del bot
- 📈 **Backtesting**: Pruebas históricas de estrategias

### Integraciones Futuras
- 🔗 **Telegram Bot**: Alertas vía Telegram
- 📧 **Email Reports**: Reportes diarios automáticos
- 🐳 **Whale Tracking**: Seguimiento de ballenas
- 📱 **Trading View**: Integración con gráficos profesionales

---

*© 2025 NvBot3 Dashboard - Sistema de Trading Inteligente*
