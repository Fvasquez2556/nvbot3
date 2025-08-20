# 🔧 Configuración Permanente - GitHub Copilot Agent

## ⚠️ INSTRUCCIÓN CRÍTICA PARA EL AGENTE

**ESTAS INSTRUCCIONES DEBEN APLICARSE AUTOMÁTICAMENTE EN CADA SESIÓN DE TRABAJO**

---

## 🐍 Activación Automática del Entorno Virtual

### Regla #1: SIEMPRE Activar Entorno Virtual
**El agente DEBE activar automáticamente el entorno virtual `nvbot3_env` en CADA terminal que abra, exclusiones si en Checked background terminal output aparece que esta activado el entorno virtual o si el usuario lo ha activado manualmente, omitirias la activación automática.**

### Para Windows:
```bash
# COMANDO OBLIGATORIO antes de cualquier operación Python
nvbot3_env\Scripts\activate

# Verificar que está activo (debe mostrar (nvbot3_env) en el prompt)
echo $VIRTUAL_ENV  # Debe mostrar la ruta del entorno
```

### Para Linux/Mac:
```bash
# COMANDO OBLIGATORIO antes de cualquier operación Python
source nvbot3_env/bin/activate

# Verificar que está activo
echo $VIRTUAL_ENV  # Debe mostrar la ruta del entorno
```

---

## 🔄 Secuencia Obligatoria para CADA Terminal

### Paso 1: Activación Automática
```bash
# Windows
nvbot3_env\Scripts\activate

# Linux/Mac  
source nvbot3_env/bin/activate
```

### Paso 2: Verificación Obligatoria
```bash
# Verificar que Python usa el entorno correcto
where python  # Windows
which python  # Linux/Mac

# Resultado esperado: debe mostrar ruta que contiene "nvbot3_env"
# Ejemplo Windows: C:\path\to\project\nvbot3_env\Scripts\python.exe
# Ejemplo Linux: /path/to/project/nvbot3_env/bin/python
```

### Paso 3: Validación de Dependencias
```bash
# Verificar que las dependencias están disponibles
python -c "import numpy, pandas, sklearn; print('✅ Entorno listo')"

# Si falla, instalar dependencias:
pip install -r requirements.txt
```

---

## 🚨 Detección y Solución de Problemas

### Problema: Entorno No Activo
**Síntomas:**
- El prompt NO muestra `(nvbot3_env)` al inicio
- `python -c "import pandas"` falla con ImportError
- `where python` muestra Python del sistema, no del entorno

**Solución Automática:**
```bash
# 1. Activar entorno inmediatamente
nvbot3_env\Scripts\activate  # Windows
source nvbot3_env/bin/activate  # Linux/Mac

# 2. Verificar activación exitosa
python -c "print('Entorno:', __import__('sys').executable)"

# 3. Si aún falla, recrear entorno
python -m venv nvbot3_env --clear
nvbot3_env\Scripts\activate  # Windows
pip install -r requirements.txt
```

---

## 📋 Checklist Pre-Ejecución (AUTOMÁTICO)

El agente DEBE ejecutar este checklist antes de cualquier comando Python:

```bash
# ✅ 1. Verificar que entorno virtual está activo
if [[ "$VIRTUAL_ENV" == *"nvbot3_env"* ]]; then
    echo "✅ Entorno virtual activo"
else
    echo "❌ ACTIVANDO ENTORNO VIRTUAL..."
    nvbot3_env\Scripts\activate  # Windows
    # source nvbot3_env/bin/activate  # Linux/Mac
fi

# ✅ 2. Verificar dependencias críticas
python -c "
try:
    import numpy, pandas, sklearn, xgboost, ccxt
    print('✅ Dependencias OK')
except ImportError as e:
    print(f'❌ Dependencia faltante: {e}')
    print('Ejecutando: pip install -r requirements.txt')
"

# ✅ 3. Verificar estructura de directorios
if [ -d "src" ] && [ -d "data" ] && [ -d "scripts" ]; then
    echo "✅ Estructura de proyecto OK"
else
    echo "⚠️ Verificar estructura de directorios"
fi
```

---

## 🔄 Scripts de Auto-Configuración

### Archivo: `activate_env.bat` (Windows)
```batch
@echo off
echo 🔄 Activando entorno virtual NvBot3...
call nvbot3_env\Scripts\activate.bat

echo ✅ Entorno activo: %VIRTUAL_ENV%
echo 📊 Python ubicación: 
where python

echo 🧪 Probando dependencias...
python -c "import numpy, pandas; print('✅ Dependencias básicas OK')"

echo.
echo 🚀 NvBot3 entorno listo para usar!
echo Prompt debería mostrar: (nvbot3_env)
```

### Archivo: `activate_env.sh` (Linux/Mac)
```bash
#!/bin/bash
echo "🔄 Activando entorno virtual NvBot3..."
source nvbot3_env/bin/activate

echo "✅ Entorno activo: $VIRTUAL_ENV"
echo "📊 Python ubicación:"
which python

echo "🧪 Probando dependencias..."
python -c "import numpy, pandas; print('✅ Dependencias básicas OK')"

echo ""
echo "🚀 NvBot3 entorno listo para usar!"
echo "Prompt debería mostrar: (nvbot3_env)"
```

---

## 🎯 Comandos de Trabajo Diario

### Iniciar Sesión de Trabajo
```bash
# 1. Activar entorno (SIEMPRE PRIMERO)
nvbot3_env\Scripts\activate  # Windows

# 2. Verificar estado del proyecto
python scripts/validate_setup.py

# 3. Ejecutar tareas específicas
python scripts/download_historical_data.py
python scripts/train_models.py
```

### Comandos de Desarrollo Frecuentes
```bash
# Ejecutar descarga de datos
python scripts/download_historical_data.py --symbol BTCUSDT --timeframe 5m

# Entrenar modelo específico  
python scripts/train_models.py --model momentum --symbol BTCUSDT

# Validar datos descargados
python src/data/data_validator.py --check-all

# Ejecutar tests
python -m pytest tests/ -v

# Calcular features para símbolo
python src/data/feature_calculator.py --symbol BTCUSDT --save
```

---

## ⚡ Configuración de VS Code (Recomendado)

### Archivo: `.vscode/settings.json`
```json
{
    "python.defaultInterpreterPath": "./nvbot3_env/Scripts/python.exe",
    "python.terminal.activateEnvironment": true,
    "python.terminal.activateEnvInCurrentTerminal": true,
    "terminal.integrated.defaultProfile.windows": "Command Prompt",
    "terminal.integrated.profiles.windows": {
        "NvBot3 Environment": {
            "path": "cmd.exe",
            "args": ["/k", "nvbot3_env\\Scripts\\activate.bat"]
        }
    }
}
```

### Archivo: `.vscode/tasks.json`
```json
{
    "version": "2.0.0",
    "tasks": [
        {
            "label": "Activate NvBot3 Environment",
            "type": "shell",
            "command": "nvbot3_env\\Scripts\\activate.bat",
            "group": "build",
            "presentation": {
                "echo": true,
                "reveal": "always",
                "focus": false,
                "panel": "shared"
            },
            "problemMatcher": []
        }
    ]
}
```

---

## 🚀 Automatización Avanzada

### PowerShell Profile (Windows Avanzado)
```powershell
# Agregar al perfil de PowerShell
function Start-NvBot3 {
    Write-Host "🤖 Iniciando entorno NvBot3..." -ForegroundColor Green
    & "nvbot3_env\Scripts\Activate.ps1"
    Write-Host "✅ Entorno activado: $env:VIRTUAL_ENV" -ForegroundColor Green
    python -c "import sys; print(f'Python: {sys.executable}')"
}

# Alias para activación rápida
Set-Alias -Name "nvbot" -Value Start-NvBot3
```

### Bash Profile (Linux/Mac Avanzado)  
```bash
# Agregar al ~/.bashrc o ~/.zshrc
nvbot() {
    echo "🤖 Iniciando entorno NvBot3..."
    source nvbot3_env/bin/activate
    echo "✅ Entorno activado: $VIRTUAL_ENV"
    python -c "import sys; print(f'Python: {sys.executable}')"
}

# Auto-activación al entrar al directorio
cd() {
    builtin cd "$@"
    if [[ -f "nvbot3_env/bin/activate" ]]; then
        echo "🔄 Auto-activando NvBot3..."
        source nvbot3_env/bin/activate
    fi
}
```

---

## 📝 Resumen para el Agente

### REGLAS ABSOLUTAS:
1. **NUNCA ejecutar comandos Python sin activar el entorno virtual primero**
2. **SIEMPRE verificar que `(nvbot3_env)` aparece en el prompt**
3. **AUTOMÁTICAMENTE activar entorno al abrir cualquier terminal**
4. **VALIDAR dependencias antes de ejecutar scripts principales**
5. **USAR los scripts de auto-configuración cuando sea posible**

### FLUJO ESTÁNDAR:
```
Abrir Terminal → Activar Entorno → Verificar → Ejecutar Comandos
```

### EN CASO DE ERROR:
1. Re-activar entorno virtual
2. Verificar ubicación de Python
3. Reinstalar dependencias si es necesario
4. Reportar problema específico

---

**🎯 OBJETIVO:** Que el agente NUNCA tenga que preguntarse si el entorno está activo. Debe ser automático e invisible al usuario.**