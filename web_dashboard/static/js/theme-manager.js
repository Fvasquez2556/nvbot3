// Gestor de temas para NvBot3 Dashboard

class ThemeManager {
    constructor() {
        this.currentTheme = localStorage.getItem('nvbot3-theme') || 'dark';
        this.init();
    }

    init() {
        this.applyTheme(this.currentTheme);
        this.createThemeToggle();
        this.addEventListeners();
    }

    applyTheme(theme) {
        document.documentElement.setAttribute('data-theme', theme);
        this.currentTheme = theme;
        localStorage.setItem('nvbot3-theme', theme);
        
        // Actualizar icono del botón
        const toggleBtn = document.getElementById('theme-toggle');
        if (toggleBtn) {
            toggleBtn.innerHTML = theme === 'dark' ? 
                '<i class="fas fa-sun"></i>' : 
                '<i class="fas fa-moon"></i>';
        }
    }

    toggleTheme() {
        const newTheme = this.currentTheme === 'dark' ? 'light' : 'dark';
        this.applyTheme(newTheme);
        
        // Animación suave de transición
        document.body.style.transition = 'all 0.3s ease';
        setTimeout(() => {
            document.body.style.transition = '';
        }, 300);
    }

    createThemeToggle() {
        const existingToggle = document.getElementById('theme-toggle');
        if (existingToggle) return;

        const toggle = document.createElement('button');
        toggle.id = 'theme-toggle';
        toggle.className = 'theme-toggle';
        toggle.title = 'Cambiar tema';
        toggle.innerHTML = this.currentTheme === 'dark' ? 
            '<i class="fas fa-sun"></i>' : 
            '<i class="fas fa-moon"></i>';
        
        document.body.appendChild(toggle);
    }

    addEventListeners() {
        document.addEventListener('click', (e) => {
            if (e.target.closest('#theme-toggle')) {
                this.toggleTheme();
            }
        });

        // Detectar cambios de preferencia del sistema
        if (window.matchMedia) {
            window.matchMedia('(prefers-color-scheme: dark)').addListener((e) => {
                if (!localStorage.getItem('nvbot3-theme')) {
                    this.applyTheme(e.matches ? 'dark' : 'light');
                }
            });
        }
    }

    // Método para obtener el tema actual
    getCurrentTheme() {
        return this.currentTheme;
    }

    // Método para forzar un tema específico
    setTheme(theme) {
        if (['dark', 'light'].includes(theme)) {
            this.applyTheme(theme);
        }
    }
}

// Utilidades adicionales para el dashboard
class DashboardUtils {
    static formatNumber(num) {
        if (num >= 1000000) {
            return (num / 1000000).toFixed(1) + 'M';
        }
        if (num >= 1000) {
            return (num / 1000).toFixed(1) + 'K';
        }
        return num.toString();
    }

    static formatPercentage(value, decimals = 1) {
        return (value * 100).toFixed(decimals) + '%';
    }

    static formatCurrency(value, symbol = '$') {
        return symbol + value.toFixed(6);
    }

    static getTimeAgo(timestamp) {
        const now = new Date();
        const time = new Date(timestamp);
        const diffInSeconds = Math.floor((now - time) / 1000);

        if (diffInSeconds < 60) return 'hace ' + diffInSeconds + 's';
        if (diffInSeconds < 3600) return 'hace ' + Math.floor(diffInSeconds / 60) + 'm';
        if (diffInSeconds < 86400) return 'hace ' + Math.floor(diffInSeconds / 3600) + 'h';
        return 'hace ' + Math.floor(diffInSeconds / 86400) + 'd';
    }

    static showNotification(message, type = 'info', duration = 5000) {
        const notification = document.createElement('div');
        notification.className = `alert alert-${type} position-fixed notification-toast`;
        notification.style.cssText = `
            top: 80px; 
            right: 20px; 
            z-index: 2000; 
            min-width: 300px;
            animation: slideInRight 0.3s ease;
        `;
        notification.innerHTML = `
            <div class="d-flex justify-content-between align-items-center">
                <span>${message}</span>
                <button type="button" class="btn-close btn-close-white" aria-label="Close"></button>
            </div>
        `;

        document.body.appendChild(notification);

        // Auto-remove
        const timer = setTimeout(() => {
            if (notification.parentNode) {
                notification.style.animation = 'slideOutRight 0.3s ease';
                setTimeout(() => {
                    if (notification.parentNode) {
                        notification.parentNode.removeChild(notification);
                    }
                }, 300);
            }
        }, duration);

        // Manual close
        notification.querySelector('.btn-close').addEventListener('click', () => {
            clearTimeout(timer);
            notification.style.animation = 'slideOutRight 0.3s ease';
            setTimeout(() => {
                if (notification.parentNode) {
                    notification.parentNode.removeChild(notification);
                }
            }, 300);
        });
    }

    static addAnimationClass(element, className) {
        element.classList.add(className);
        element.addEventListener('animationend', () => {
            element.classList.remove(className);
        }, { once: true });
    }
}

// CSS para animaciones de notificaciones
const notificationStyles = `
    @keyframes slideInRight {
        from { transform: translateX(100%); opacity: 0; }
        to { transform: translateX(0); opacity: 1; }
    }
    
    @keyframes slideOutRight {
        from { transform: translateX(0); opacity: 1; }
        to { transform: translateX(100%); opacity: 0; }
    }
    
    .notification-toast {
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
`;

// Inyectar estilos de notificaciones
const styleSheet = document.createElement('style');
styleSheet.textContent = notificationStyles;
document.head.appendChild(styleSheet);

// Inicializar gestor de temas cuando el DOM esté listo
document.addEventListener('DOMContentLoaded', () => {
    window.themeManager = new ThemeManager();
    window.dashboardUtils = DashboardUtils;
});

// Exportar para uso global
window.ThemeManager = ThemeManager;
window.DashboardUtils = DashboardUtils;
