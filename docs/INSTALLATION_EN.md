# Installing vMAR: C23, Python, and Google Colab

The source ZIP contains `vMAR/` with the C23 numerical implementation,
public headers, Python native bindings, reference scripts and reproducible
consumer examples. Python builds the original C23 sources as `vmar._native`.

## C23 applications (CLion or terminal)

Open the `vMAR/` directory in CLion. Use CMake 3.21+ and a compiler with
the C23 features required by this project. In a shell within that directory:

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

External applications use `find_package(vMAR CONFIG REQUIRED)` and link
`vmar::core`. No R or Python runtime is required by the C23 consumer.

## Python installed from a downloaded folder

From the **parent** directory of `vMAR`:

```sh
python3 -m pip install ./vMAR
cd /tmp
python3 -c "import vmar; from importlib.metadata import version; print(version('vmar-dynamics'), vmar.__file__)"
python3 /actual/path/vMAR/examples/python_public_consumer.py
python3 -m unittest discover -s /actual/path/vMAR/python -p test_interop_extended.py -v
```

Replace `/actual/path/vMAR` with the unpacked directory. Running tests
outside the source tree avoids Python import shadowing. Use Python syntax
inside a Python interpreter, not directly at the shell prompt.

## Colab before GitHub is updated

1. Upload `vMAR_distribucion_GitHub.zip` with `google.colab.files.upload()`.
2. Extract its `vMAR/` folder into `/content`.
3. Run `%pip install /content/vMAR` in a notebook cell.
4. Run `%run /content/vMAR/examples/python_public_consumer.py`.
5. Execute `python -m unittest discover -s /content/vMAR/python -p test_interop_extended.py -v` through `subprocess.run(..., check=True)`.

The full copy/paste notebook cells are in [Spanish installation](INSTALACION_ES.md).

**After GitHub publication**, users may install the reviewed revision using
`python3 -m pip install 'git+https://github.com/RubenMoreValencia/vMAR.git'`.
The old repository revision may not contain these expanded Python bindings.

## Scientific reproducibility

The 14 optional R-produced fixtures are not silently regenerated or claimed
present. See [reproducibility](REPRODUCIBILITY.md),
[interoperability matrix](PYTHON_INTEROP_MATRIX.md), and
[evidence](PUBLICATION_EVIDENCE.md).

