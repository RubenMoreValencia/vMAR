# vMAR — Núcleo computacional diferencial en C23

**Carpeta preparada para revisión de publicación (27-11-2026).** Contiene la
biblioteca C23 y el consumidor Python nativo. No representa todavía una
publicación oficial ni compromete una ABI estable para los contratos
experimentales.

**Guías iniciales:** [instalación y Colab](docs/INSTALACION_ES.md) ·
[inventario de API](docs/PUBLIC_DISTRIBUTION_INVENTORY.md) ·
[evidencias y requisitos de publicación](docs/PUBLICATION_EVIDENCE.md).

La distribución Python se instala con el nombre `vmar-dynamics==0.1.16a2` y
se importa como `vmar`. El proyecto C23 declara `0.1.16`; las diferencias
de sufijo reflejan el estado preliminar del empaquetado Python.

> **Candidata G0.9-P.3:** la API pública C23 (`include/vmar/*.h`) y el
> destino `vmar::core` son la interfaz independiente del lenguaje para
> cualquier consumidor. La nueva capa Python incorpora redes, gradientes,
> memoria, control, modos M0–M3 y trazabilidad, además de Euler/RK4.
> No se modifican ni duplican los algoritmos numéricos consolidados.
> Consulta la [matriz C23–R–Python](docs/PYTHON_INTEROP_MATRIX.md) y la
> [recuperación del puente Colab](docs/COLAB_BRIDGE_RECOVERY.md).

```python
from vmar import Network, NetworkMechanism, memory_ema

red = Network((2, 4, 1))
print(red.predict([[1., 0.]]))
print(memory_ema((0., 0.), (1., 2.), .8))
print(NetworkMechanism(red, [[1., 0.]], [1.], mode="observe").step())
```

Para ejecutar un caso completo desde Python utiliza
`python examples/python_public_consumer.py` después de instalar el paquete.
Las activaciones personalizadas y los regímenes adaptativos especializados
siguen siendo contratos C23 experimentales que requieren adaptadores propios.

## Uso directo desde Python (experimental)

**Antes de actualizar GitHub**, instala la carpeta descargada con
`python3 -m pip install ./vMAR` desde su directorio padre. El comando
de GitHub siguiente corresponde a la versión que se encuentre publicada en
el repositorio, que podría diferir de esta candidata.

En Linux o macOS, con un compilador compatible con las características C23
requeridas, puede instalarse desde GitHub:

```sh
python -m pip install "git+https://github.com/RubenMoreValencia/vMAR.git"
```

```python
from vmar import Differential

modelo = Differential(field=lambda t, y: (-y[0],), initial_state=[1.0])
resultado = modelo.integrate(method="rk4", step=0.1, steps=10)
print(resultado.state)  # aproximadamente (0.3678797744124984,)
```

El nombre del paquete distribuido es `vmar-dynamics`; la importación es
`vmar`. La instalación compila el **núcleo C23 original**; no reproduce sus
integradores en Python. La API inicial G0.9-P.2 contenía solo integración;
esta candidata la amplía con capacidades generales aprobadas. Véase la
[matriz de correspondencia](docs/PYTHON_INTEROP_MATRIX.md).

**Estado:** versión de desarrollo `0.1.16-alpha`. Esta denominación expresa
el estado preliminar; las macros públicas y CMake identifican la versión
numérica `0.1.16`. [Technical README in English](README.md).

**Documentos de publicación:** [API pública](docs/PUBLIC_API.md) ·
[reproducibilidad](docs/REPRODUCIBILITY.md) ·
[auditoría G0.9-A.3](docs/G0.9-A3_RELEASE_READINESS.md) ·
[declaración de procedencia](docs/PROVENANCE_DECLARATION_ES.md).

vMAR investiga el aprendizaje neuronal como evolución dinámica observable,
con memoria explícita y control retroalimentado. Forma parte de la línea
científica **DSNeuralRNAS** —Dinámica de Sistemas; Regresores Neuronales y
Arquitecturas Simples— desarrollada por Rubén Alexander More Valencia.

No es una conversión de todo el ecosistema DSNeuralRNAS ni pretende
reemplazar a `DSNeuralRNAS.FormalSpec` o a la implementación matemática
de referencia `vMAR.Core` en R.

## Capacidades actuales

- Integración numérica Euler y RK4 de campos diferenciales explícitos.
- Neuronas y redes multicapa densas de salida escalar, con gradientes
  analíticos y funciones de activación configurables.
- Estados aumentados que distinguen parámetros neuronales y memoria.
- M0: referencia sin evolución de memoria; M1: memoria observable
  (modo predeterminado); M2 y M3: acoplamiento y control experimentales,
  seleccionados explícitamente.
- Registros de decisiones, pruebas de contratos y referencias numéricas R–C23.

El diagnóstico de curvatura C01 es **experimental** y se distribuye
separado del núcleo instalable. Euler y RK4 son integradores, mientras
que Adam se mantiene como comparación de optimización discreta; no son
conceptos intercambiables.

## Compilar e instalar

Se necesita un compilador con el soporte C23 requerido por el proyecto y
CMake 3.21 o posterior. Desde el directorio raíz `vMAR`:

```sh
cmake -S . -B build-release -DVMAR_BUILD_TESTS=OFF \
  -DVMAR_BUILD_EXPERIMENTAL_CURVATURE=OFF \
  -DCMAKE_INSTALL_PREFIX="$PWD/vmar-install"
cmake --build build-release
cmake --install build-release
```

Para comprobar que otro programa puede reutilizar la biblioteca:

```sh
cmake -S examples/external_consumer -B build-consumer \
  -DCMAKE_PREFIX_PATH="$PWD/vmar-install"
cmake --build build-consumer
ctest --test-dir build-consumer --output-on-failure
```

En el programa consumidor se incluye `<vmar/vmar.h>`; su archivo
`CMakeLists.txt` utiliza `find_package(vMAR CONFIG REQUIRED)` y enlaza
`vmar::core`. El ejemplo está en `examples/external_consumer/`.

## Validación e interpretación

El modo de instalación puede verificarse sin ejecutar R. La auditoría
científica estricta es diferente: exige las 14 referencias históricas
R–C23 y se activa mediante `VMAR_STRICT_PARITY_FIXTURES=ON`.
Los resultados de pruebas anteriores no demuestran superioridad universal
de una política de aprendizaje ni equivalencia completa con FormalSpec.

## Reutilización, autoría y citación

**Autor:** Rubén Alexander More Valencia.

Hay metadatos preliminares de citación en `CITATION.cff`. El autor ha
ratificado la procedencia y la identidad de la publicación; por ello, la
distribución incorpora ahora la [licencia MIT](LICENSE). La ratificación
del autor está documentada, pero no sustituye una auditoría independiente
de toda la historia de contribuciones. Aún no se ha publicado una versión
oficial en GitHub ni asignado una URL permanente o DOI.

Consultar también el [registro G0.9-A.1](docs/G0.9-A1_PUBLIC_API_INSTALL_VERSION_AUDIT.md),
el [historial de versiones](CHANGELOG.md) y la documentación en inglés.
