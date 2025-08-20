# 🔒 SEGURIDAD - NvBot3

## ⚠️ INFORMACIÓN IMPORTANTE DE SEGURIDAD

### 🚨 ARCHIVOS QUE NUNCA DEBEN SUBIRSE A GITHUB

1. **`.env`** - Contiene claves API reales
2. **`nvbot3_env/`** - Entorno virtual (muy pesado)
3. **Cualquier archivo con claves API reales**
4. **Datos de trading en vivo**
5. **Logs con información sensible**

### ✅ CONFIGURACIÓN SEGURA

1. **Usa `.env.example`** como plantilla
2. **Copia `.env.example` → `.env`**
3. **Rellena `.env` con tus claves reales**
4. **`.env` está en `.gitignore` automáticamente**

### 🔑 PERMISOS API DE BINANCE

Para este bot **SOLO** necesitas:

- ✅ **Enable Reading** (Lectura)
- ❌ **Enable Spot & Margin Trading** (NO habilitar)
- ❌ **Enable Futures** (NO habilitar)
- ❌ **Enable Withdrawals** (NO habilitar)

### 🛡️ BUENAS PRÁCTICAS

1. **Nunca** hagas commit del archivo `.env`
2. **Siempre** usa claves API con permisos mínimos
3. **Revisa** qué archivos subes antes de hacer push
4. **Usa** `git status` antes de cada commit
5. **Ten** un respaldo seguro de tus configuraciones

### 🚨 SI SUBISTE CLAVES POR ERROR

1. **Inmediatamente** ve a Binance y desactiva/elimina esas claves
2. **Genera** nuevas claves API
3. **Actualiza** tu archivo `.env` local
4. **Revisa** el historial de commits si es necesario

### 📝 COMANDOS ÚTILES

```bash
# Verificar qué archivos están a punto de subirse
git status

# Ver diferencias antes de commit
git diff --cached

# Eliminar archivo del repositorio (mantener local)
git rm --cached archivo_sensible.txt
```

---

## 🎯 Recuerda: La seguridad es responsabilidad de TODOS
