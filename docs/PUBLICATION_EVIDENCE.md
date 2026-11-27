# Publication evidence and review checklist

## Evidence available before repository update

- **Current prepared-folder verification (Linux/Python 3.12):** wheel
  `vmar-dynamics 0.1.16a2` built successfully with native C23 sources,
  installed into an isolated target, and 8/8 Python extended contracts
  passed. This check is not a substitute for CMake/CTest.

- Researcher-reported **macOS AppleClang 17** C23/CMake build and CTest
  suite: 25/25 tests passed for the interoperability candidate.
- Researcher-reported Python 3.13/macOS wheel installation:
  `vmar-dynamics 0.1.16a2`, imported from installed `site-packages`.
- Python interoperability unit contracts: previous macOS run reported
  8/8 passes; a later terminal excerpt shows five named tests passing but
  omits the final summary.
- Researcher-reported Google Colab/Linux Python 3.13 wheel built and
  installed; a `unittest` subprocess returned exit status 0.
- The numerical reference generators for R–C23 studies are included, but
  **0/14 available historical fixtures in some clean configurations must
  not be misreported as 14/14 strict parity**.

These results verify installation and the exercised contracts. They are
not evidence of universal superiority, cross-platform ABI stability or
full C23–R–Python coordinate-level parity for every public mechanism.

## Maintainer publication gate

- [ ] Review source diff against the currently published GitHub `vMAR`.
- [ ] Confirm license, copyright ownership and external contributions.
- [ ] Run `cmake --build`, `ctest` and the independent C23 consumer on the
      final exact uploaded revision.
- [ ] Re-run Python examples and tests after the exact final installation.
- [ ] Record release SHA-256 or Git commit and actual platform versions.
- [ ] Generate all independent R fixtures before asserting strict parity.
- [ ] Confirm version and metadata, tag/NEWS/README consistency.
- [ ] Publish to GitHub only after the maintainer approves these changes.

## Planned dependent releases

1. `vMAR.MLP`: configurable neural network library, C23 + R + Python;
   reuse `vMAR` instead of copying numerical mechanisms.
2. `vMAR.MLP.RL`: episodic transition/replay/target-policy extension,
   C23 + R + Python; distinguish experience replay from vMAR dynamic memory.
3. Future `vMAR.ML`: meta-learning, as a separate formal research stage.

Negative RL outcomes in prior consumer experiments remain part of the
scientific record and must not be replaced by only successful examples.
