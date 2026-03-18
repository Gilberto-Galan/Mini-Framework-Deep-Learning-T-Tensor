# Contribuir a T-Tensor

¡Gracias por querer aportar a T-Tensor!  
Nos emociona construir este mini-framework de Deep Learning con CUDA junto a la comunidad.

Si te gusta aprender cómo funciona un framework por dentro, este repo es un gran lugar para colaborar.

## ¿Cómo puedes ayudar?

Puedes contribuir de muchas formas, incluso si no eres experto en CUDA.

- Reportando bugs con pasos claros para reproducir.
- Mejorando la documentación y ejemplos de uso en Python.
- Proponiendo o implementando nuevas operaciones de tensor.
- Optimizando kernels CUDA existentes.
- Añadiendo capas, funciones de pérdida u optimizadores.
- Mejorando manejo de memoria y rendimiento en GPU.
- Mejorando mensajes de error y experiencia de desarrollador.
- Ayudando a probar en distintos entornos de Windows/CUDA.

## Herramientas necesarias

Para contribuir localmente, recomendamos este entorno:

- Git
- Python 3.12+
- CMake 3.18+
- CUDA Toolkit (idealmente 12.x)
- Visual Studio Build Tools o Visual Studio con compilador MSVC C++
- PowerShell (Windows)

Entorno recomendado de compilacion nativa en este repo:

- Visual Studio 18 2026 (x64)

Dependencias Python principales del proyecto:

- pybind11
- setuptools
- wheel

## Configuración rápida del entorno (Windows)

1. Haz fork del repositorio en GitHub.
2. Clona tu fork y entra a la carpeta del proyecto.
3. Crea y activa un entorno virtual.
4. Instala el proyecto desde fuente.

Ejemplo:

```powershell
git clone <URL_DE_TU_FORK>
cd T-Tensor
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install .
```

## Verificación básica

Después de instalar, valida que el módulo cargue correctamente:

```powershell
python -c "import ttensor; print('OK')"
```

Si esto imprime `OK`, ya estás listo para contribuir.

## Verificación de compilación nativa (Visual Studio 18 2026)

Desde la raíz del proyecto:

```powershell
cmake -S . -B build -G "Visual Studio 18 2026" -A x64 -DCMAKE_CUDA_FLAGS="--allow-unsupported-compiler" -Dpybind11_DIR="C:/Users/PC-R0O7/Desktop/T-Tensor/venv/Lib/site-packages/pybind11/share/cmake/pybind11"
cmake --build build --config Release
```

Pruebas rápidas después del build:

```powershell
python -c "import ttensor; print('OK')"
python examples/xor_cpu.py
python -m unittest tests/test_import_smoke.py
```

## Flujo recomendado para contribuir

1. Crea una rama para tu cambio.
2. Haz cambios pequeños y claros.
3. Verifica que el proyecto siga compilando/instalando.
4. Abre un Pull Request con una descripción concreta.

Ejemplo de nombres de rama:

- feat/nueva-op-matmul
- fix/cuda-memory-leak
- docs/mejorar-readme

## Buenas prácticas para Pull Requests

- Explica el problema y la solución.
- Incluye pasos para probar el cambio.
- Si aplica, agrega ejemplo mínimo en Python.
- Mantén el alcance del PR enfocado (mejor varios PR pequeños que uno gigante).
- Sé respetuoso y abierto al feedback durante la revisión.

## Ideas de contribución para empezar

Si quieres una tarea de inicio, estas suelen ser muy útiles:

- Corregir tipos y mejorar claridad del readme.
- Añadir ejemplos de entrenamiento con modelos simples.
- Mejorar mensajes de error al fallar la inicialización de GPU.
- Agregar validaciones extra en operaciones de tensor.
- Documentar limitaciones actuales y próximos pasos.

## ¿Encontraste un problema?

Abre un Issue con esta información:

- Qué esperabas que pasara.
- Qué pasó realmente.
- Pasos para reproducir.
- Tu entorno (Python, CUDA, GPU, Windows).
- Log o error completo si es posible.

## Código de convivencia

Queremos una comunidad amable, técnica y colaborativa.  
Toda contribución con buena intención es bienvenida.

---

¡Gracias por sumarte! Tu aporte ayuda a que T-Tensor sea más sólido, más rápido y más útil para todos.
