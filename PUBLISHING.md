# Checklist de Publicación — T-Tensor

---

## FASE 0 — Preparación local (antes de todo)

- [ ] El proyecto compila limpio: `python -m build` sin errores
- [ ] `import ttensor` funciona después de `pip install .\dist\*.whl`
- [ ] `python -m twine check dist/*` → **PASSED** (sin warnings)
- [ ] `LICENSE` existe (MIT) ✅
- [ ] `readme.md` / `README.md` está actualizado ✅
- [ ] `setup.py` tiene metadata completa (author, email, url, license, classifiers) ✅
- [ ] `.gitignore` configurado ✅

---

## FASE 1 — GitHub

### 1.1 Instalar Git (si no está)
```powershell
# Verificar
git --version

# Si no está: descargar de https://git-scm.com/download/win
```

### 1.2 Crear repositorio en GitHub
1. Ir a **github.com → New repository**
2. Nombre: `t-tensor`
3. Descripción: `Mini deep learning framework with CUDA/C++ backend and Python bindings`
4. Visibilidad: **Public**
5. **NO** marcar "Add a README" (ya tenemos uno)
6. **NO** marcar .gitignore ni License (ya los tenemos)
7. Clic en **Create repository**

### 1.3 Inicializar y subir desde local
```powershell
cd "C:\Users\PC-R0O7\Desktop\T-Tensor"

git init
git add .
git commit -m "feat: initial release v0.1.0"

# Reemplaza TU_USUARIO con tu nombre de GitHub
git remote add origin https://github.com/TU_USUARIO/t-tensor.git
git branch -M main
git push -u origin main
```

### 1.4 Verificar qué se subió
```powershell
git status     # debe estar limpio
git log --oneline -5
```

### 1.5 Crear Release v0.1.0 en GitHub
1. En tu repositorio → **Releases → Create a new release**
2. Tag: `v0.1.0`  |  Target: `main`
3. Title: `T-Tensor v0.1.0 — Initial release`
4. Adjuntar el archivo: `dist\ttensor-0.1.0-cp312-cp312-win_amd64.whl`
5. Clic en **Publish release**

> **Nota:** Subir la `.whl` precompilada aquí permite que usuarios con Windows + CUDA 12.8 + Python 3.12
> la descarguen directamente sin necesitar compilar desde cero.

---

## FASE 2 — TestPyPI (prueba antes de publicar en PyPI real)

### 2.1 Crear cuenta en TestPyPI
1. Ir a **test.pypi.org → Register**
2. Verificar email
3. Activar **2FA** (requerido para subir paquetes)
4. Ir a **Account settings → API tokens → Add API token**
5. Scope: "Entire account"
6. Copiar el token (empieza con `pypi-`)

### 2.2 Configurar credenciales localmente
```powershell
# Crear archivo de configuración (no subir esto a Git)
notepad "$env:USERPROFILE\.pypirc"
```

Contenido del `.pypirc`:
```ini
[distutils]
index-servers =
    pypi
    testpypi

[testpypi]
repository = https://test.pypi.org/legacy/
username = __token__
password = pypi-XXXXXXXXXXXXXXXXXX   # tu token de TestPyPI

[pypi]
username = __token__
password = pypi-XXXXXXXXXXXXXXXXXX   # tu token de PyPI (para después)
```

### 2.3 Instalar twine
```powershell
.\venv\Scripts\Activate.ps1
pip install twine
```

### 2.4 Verificar el paquete antes de subir
```powershell
python -m twine check dist/*
# Debe mostrar: PASSED
```

### 2.5 Subir a TestPyPI
```powershell
python -m twine upload --repository testpypi dist/*
```

### 2.6 Probar instalación desde TestPyPI
```powershell
# En una nueva terminal (fuera del venv del proyecto)
pip install -i https://test.pypi.org/simple/ ttensor

# Si importa bien:
python -c "import ttensor; print('TestPyPI OK')"
```

- [ ] `twine check` → PASSED
- [ ] Paquete visible en `test.pypi.org/project/ttensor`
- [ ] `pip install` desde TestPyPI funciona

---

## FASE 3 — PyPI oficial

### 3.1 Crear cuenta en PyPI
1. Ir a **pypi.org → Register**
2. Verificar email
3. Activar **2FA** (obligatorio)
4. **Account settings → API tokens → Add API token**
5. Agregar el token en `~/.pypirc` bajo `[pypi]`

### 3.2 Subir a PyPI
```powershell
python -m twine upload dist/*
```

> Primera vez: si el nombre `ttensor` ya está tomado, cambia `name="ttensor"` en `setup.py`
> por algo como `name="t-tensor"` o `name="ttensor-cuda"` y regenera el paquete.

### 3.3 Verificar
```powershell
pip install ttensor
python -c "import ttensor; print('PyPI OK')"
```

- [ ] Paquete visible en `pypi.org/project/ttensor`
- [ ] `pip install ttensor` funciona desde cualquier máquina

---

## FASE 4 — Actualizar URL del proyecto en setup.py

Antes de publicar en PyPI, actualizar `url` en `setup.py`:

```python
url="https://github.com/TU_USUARIO/t-tensor",
```

Y regenerar:
```powershell
Remove-Item -Recurse -Force dist, ttensor.egg-info -ErrorAction SilentlyContinue
python -m build
```

---

## Comandos de referencia rápida

| Tarea | Comando |
|---|---|
| Regenerar paquete limpio | `Remove-Item -Recurse -Force dist, ttensor.egg-info; python -m build` |
| Verificar paquete | `python -m twine check dist/*` |
| Subir a TestPyPI | `python -m twine upload --repository testpypi dist/*` |
| Subir a PyPI | `python -m twine upload dist/*` |
| Instalar localmente | `pip install .\dist\ttensor-0.1.0-cp312-cp312-win_amd64.whl --force-reinstall` |
| Ver metadata instalada | `pip show ttensor` |
| Ver estado Git | `git status` |
| Nuevo commit | `git add . && git commit -m "mensaje"` |
| Subir cambios | `git push` |

---

## Notas importantes para CUDA wheels

> Los usuarios necesitan tener instalado **CUDA 12.8** y **Python 3.12** para usar la `.whl` precompilada.
> Usuarios con otras versiones deberán compilar desde fuente (`pip install` desde el `.tar.gz`).

Considera en el futuro:
- GitHub Actions para compilar wheels automáticamente en cada release (`cibuildwheel`)
- Soporte para Python 3.11 y 3.13
- Wheel para CUDA 12.x genérico con `-gencode arch=compute_XX`
