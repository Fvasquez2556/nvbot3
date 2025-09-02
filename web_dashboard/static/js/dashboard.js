// NvBot3 Dashboard JavaScript

// Configuración de WebSocket
const socket = io();

// Estado de conexión
socket.on('connect', function() {
    document.getElementById('connection-status').textContent = 'Conectado';
    document.querySelector('.status-dot').style.background = 'var(--success-color)';
});

socket.on('disconnect', function() {
    document.getElementById('connection-status').textContent = 'Desconectado';
    document.querySelector('.status-dot').style.background = 'var(--danger-color)';
});

// Recibir datos iniciales
socket.on('initial_data', function(data) {
    console.log('Datos iniciales recibidos:', data);
    updateDashboard(data);
});

// Recibir actualizaciones periódicas
socket.on('periodic_update', function(data) {
    console.log('Actualización periódica:', data);
    updateDashboard(data);
    updateLastUpdateTime();
});

// Recibir nuevas señales individuales
socket.on('new_signal', function(data) {
    console.log('Nueva señal:', data);
    showNewSignalNotification(data.signal);
});

// Función para actualizar el dashboard
function updateDashboard(data) {
    const { categorized_signals, top_confidence_coins, stats } = data;
    
    // Actualizar estadísticas
    if (stats) {
        document.getElementById('total-signals').textContent = stats.total_signals || 0;
    }
    
    // Actualizar contadores por categoría
    if (categorized_signals) {
        document.getElementById('momentum-count').textContent = categorized_signals.momentum?.length || 0;
        document.getElementById('rebound-count').textContent = categorized_signals.rebounds?.length || 0;
        document.getElementById('trend-count').textContent = categorized_signals.trending?.length || 0;
        document.getElementById('scanned-count').textContent = categorized_signals.scanned_symbols?.length || 0;
        
        // Actualizar secciones de señales
        updateSignalSection('momentum-signals', categorized_signals.momentum, 'momentum');
        updateSignalSection('rebound-signals', categorized_signals.rebounds, 'rebound');
        updateSignalSection('trend-signals', categorized_signals.trending, 'trend');
    }
    
    // Actualizar top coins
    if (top_confidence_coins) {
        updateTopCoins(top_confidence_coins);
    }
}

// Actualizar sección de señales
function updateSignalSection(containerId, signals, type) {
    const container = document.getElementById(containerId);
    if (!container) return;
    
    if (!signals || signals.length === 0) {
        container.innerHTML = '<div class="no-data">No hay señales disponibles</div>';
        return;
    }
    
    container.innerHTML = signals.map(signal => {
        const confidenceClass = getConfidenceClass(signal.confidence);
        
        return `
            <div class="signal-item">
                <div class="signal-info">
                    <div class="signal-symbol">${signal.symbol}</div>
                    <div class="signal-details">
                        ${signal.signal_type || type} • ${signal.predicted_change ? '+' + signal.predicted_change.toFixed(2) + '%' : ''}
                    </div>
                </div>
                <div class="signal-metrics">
                    <div class="confidence-bar">
                        <div class="confidence-fill ${confidenceClass}" data-confidence="${signal.confidence}"></div>
                    </div>
                    <div>${(signal.confidence * 100).toFixed(1)}%</div>
                </div>
            </div>
        `;
    }).join('');
    
    // Aplicar anchos de las barras de confianza sin inline styles
    updateConfidenceBars();
}

// Actualizar top coins
function updateTopCoins(coins) {
    const container = document.getElementById('top-coins-container');
    if (!container) return;
    
    if (!coins || coins.length === 0) {
        container.innerHTML = '<div class="no-data">No hay señales de alta confianza disponibles</div>';
        return;
    }
    
    container.innerHTML = coins.map(coin => `
        <div class="coin-card">
            <div class="coin-symbol">${coin.symbol}</div>
            <div class="confidence-badge">${(coin.confidence * 100).toFixed(1)}%</div>
            <div class="mt-2">
                <small>+${coin.predicted_change.toFixed(2)}%</small><br>
                <small>$${coin.current_price.toFixed(6)}</small>
            </div>
        </div>
    `).join('');
}

// Obtener clase de confianza
function getConfidenceClass(confidence) {
    if (confidence > 0.8) return 'confidence-high';
    if (confidence > 0.6) return 'confidence-medium';
    return 'confidence-low';
}

// Actualizar barras de confianza sin estilos inline
function updateConfidenceBars() {
    const confidenceBars = document.querySelectorAll('.confidence-fill[data-confidence]');
    
    confidenceBars.forEach(bar => {
        const confidence = parseFloat(bar.dataset.confidence);
        bar.style.width = `${confidence * 100}%`;
    });
}

// Mostrar notificación de nueva señal
function showNewSignalNotification(signal) {
    // Crear notificación temporal
    const notification = document.createElement('div');
    notification.className = 'alert alert-success position-fixed';
    notification.style.cssText = 'top: 20px; right: 20px; z-index: 2000; min-width: 300px;';
    notification.innerHTML = `
        <strong>🚨 Nueva señal!</strong><br>
        ${signal.symbol} - Confianza: ${(signal.confidence * 100).toFixed(1)}%
    `;
    
    document.body.appendChild(notification);
    
    // Remover después de 5 segundos
    setTimeout(() => {
        if (notification.parentNode) {
            notification.parentNode.removeChild(notification);
        }
    }, 5000);
}

// Actualizar tiempo de última actualización
function updateLastUpdateTime() {
    const now = new Date();
    document.getElementById('last-update').textContent = 
        `Última actualización: ${now.toLocaleTimeString()}`;
}

// Solicitar actualización manual cada 60 segundos
setInterval(() => {
    socket.emit('request_update');
}, 60000);

// Actualizar tiempo cada segundo
setInterval(updateLastUpdateTime, 1000);

// Inicializar cuando el DOM esté listo
document.addEventListener('DOMContentLoaded', function() {
    // Aplicar anchos iniciales a las barras de confianza
    updateConfidenceBars();
    
    // Actualizar tiempo inicial
    updateLastUpdateTime();
});
