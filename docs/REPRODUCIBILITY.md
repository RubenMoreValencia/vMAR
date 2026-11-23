# Reproducibility and validation — vMAR C23

## A. Installation and independent consumer (no R required)

From the root of the C23 project:

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

Passing the consumer proves install/link/run functionality in the chosen
environment; it does not certify the complete mathematical API.

## B. Strict R–C23 reference gate

Historical fixture generators live in `reference/`; the user must load
the independently installed/developed `vMAR.Core` R package and then run
`reference/recover_historical_fixtures.R` from this C project root.
That script records the availability/checksums of 14 named fixture CSVs.

```sh
cmake -S . -B cmake-build-scientific \
  -DCMAKE_BUILD_TYPE=Debug \
  -DVMAR_BUILD_TESTS=ON \
  -DVMAR_STRICT_PARITY_FIXTURES=ON \
  -DVMAR_BUILD_EXPERIMENTAL_CURVATURE=ON
cmake --build cmake-build-scientific
ctest --test-dir cmake-build-scientific --output-on-failure
```

**G0.9-A.2 macOS evidence (researcher-provided, 2026-11-22):**
AppleClang 17 configured and installed `libvmar_core.a`; its downstream
consumer passed 1/1 CTest and printed `0.904837500000`; strict mode
reported 14/14 historical fixtures and passed 39/39 registered CTest
tests. The experimental R–C23 curvature **parity fixture/test did not
appear in that 39-test run**, so it must not be described as 43/43.

## C. Minimum evidence for an externally reproducible publication

For each result, record:

- exact source revision (`git rev-parse HEAD` after Git initialization)
  or source ZIP SHA-256;
- C23 compiler identification/version and CMake version;
- OS/architecture and build options;
- `ctest` summary and full output on failure;
- R/R package versions and input fixture checksums;
- numerical tolerances and sample/problem definitions;
- distinction between contract assertions, test executables and fixture
  coordinate comparisons.

The project should not silently regenerate scientific fixtures from C
outputs. R reference generation and C consumption remain separate.
