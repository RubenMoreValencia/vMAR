# Guía de instalación: C23, Python, Google Colab

Esta distribución conserva las ecuaciones y algoritmos en C23. Python enlaza
las mismas fuentes C23; no sustituye los integradores por una copia Python.

## 1. Contenido

- `include/vmar/`: contratos C23 y encabezados públicos instalables.
- `src/`: implementación numérica C23.
- `vmar/`: interfaz Python; compila la extensión nativa `vmar._native`.
- `python/`: pruebas de contratos Python.
- `examples/`: consumidor Python y consumidor C23 independiente.
- `reference/`: generadores de referencias científicas R.
- `docs/`: contratos, límites, matriz de correspondencia e historial.

## 2. CLion o terminal macOS/Linux (C23)

Abrir la carpeta `vMAR` que contiene `CMakeLists.txt`. Se requiere CMake
3.21+ y compilador con las características C23 utilizadas por el proyecto.

```sh
cmake -S . -B build-release -DCMAKE_BUILD_TYPE=Release \
  -DVMAR_BUILD_TESTS=OFF \
  -DVMAR_BUILD_EXPERIMENTAL_CURVATURE=OFF \
  -DCMAKE_INSTALL_PREFIX="$PWD/vmar-install"
cmake --build build-release
cmake --install build-release
cmake -S examples/external_consumer -B build-consumer \
  -DCMAKE_PREFIX_PATH="$PWD/vmar-install"
cmake --build build-consumer
ctest --test-dir build-consumer --output-on-failure
```

Para ejecutar las pruebas disponibles del propio núcleo, configurar otra
carpeta con `-DVMAR_BUILD_TESTS=ON`; las referencias R opcionales requieren
generación independiente y no se presuponen incluidas.

## 3. Python en macOS/Linux

Desde el directorio **padre** de `vMAR`:

```sh
python3 -m pip install ./vMAR
cd /tmp
python3 -c "import vmar; from importlib.metadata import version; print(version('vmar-dynamics'), vmar.__file__)"
python3 /ruta/real/vMAR/examples/python_public_consumer.py
python3 -m unittest discover -s /ruta/real/vMAR/python -p 'test_interop_extended.py' -v
```

Sustituir `/ruta/real/vMAR` por la ruta descomprimida en el equipo. El
directorio `/tmp` evita que una carpeta de código fuente llamada `vmar`
oculte accidentalmente el paquete instalado. Los comandos `import` se
ejecutan dentro de Python; `%pip` funciona en notebooks, no en `zsh`.

## 4. Google Colab sin repositorio actualizado

Subir el ZIP de distribución (el que contiene la carpeta `vMAR`):

```python
from google.colab import files
files.upload()
```

```python
import zipfile
with zipfile.ZipFile('/content/vMAR_distribucion_GitHub.zip') as z:
    z.extractall('/content')
```

```python
%pip install /content/vMAR
```

```python
from importlib.metadata import version
import vmar
print(version('vmar-dynamics'), vmar.__file__)
%run /content/vMAR/examples/python_public_consumer.py
```

```python
import subprocess
subprocess.run(['python', '-m', 'unittest', 'discover',
                '-s', '/content/vMAR/python',
                '-p', 'test_interop_extended.py', '-v'], check=True)
```

**Importante:** en Colab es necesario utilizar un runtime capaz de compilar
la extensión C23. La instalación genera un wheel para ese runtime, no un
binario universal válido en todos los sistemas.

## 5. Posteriormente, desde GitHub

Solo **después** de actualizar y revisar el repositorio oficial:

```sh
python3 -m pip install 'git+https://github.com/RubenMoreValencia/vMAR.git'
```

## 6. Correspondencia con R

`vMAR.Core` es una distribución R independiente. Sus referencias matemáticas
permiten generar los CSV empleados por los contratos C23. Instalar Python
no instala `vMAR.Core` ni ejecuta automáticamente las 14 referencias R
opcionales. Véase `docs/REPRODUCIBILITY.md`.

