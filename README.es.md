# vMAR — Núcleo computacional diferencial en C23

## Uso directo desde Python (experimental)

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
integradores en Python. Por ahora la API Python expone únicamente integración
diferencial. Véase el [contrato G0.9-P.2](docs/G0.9-P2_PYTHON_API.md).

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
