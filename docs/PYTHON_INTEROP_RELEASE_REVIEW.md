# G0.9-P.3 interoperability review candidate

## Implemented and locally checked

- C23 numerical sources unchanged; `vmar._native` rebuild uses the exact
  existing source list and the compiler's native shared-extension loader.
- Python wheel `vmar-dynamics==0.1.16a2` built using GCC-compatible C23
  compiler (`CC=cc`) and installed in an independent local Python target.
- 8 new tests passed on scalar-output network, C23 Euler/RK4 step,
  activations, memory, control, four M0/M1/M2/M3 modes, state layout and
  confirmed trace.
- 4 previous direct Python API tests passed unchanged.
- Native public C23 network remains scalar-output; multi-output architecture
  belongs to `vMAR.MLP`.

## Pending platform and publication gates

1. Run `python3 -m unittest discover -s python -p 'test_*api*.py'` and
   `python3 -m unittest discover -s python -p 'test_interop_extended.py'`
   on the investigator's AppleClang/macOS and Google Colab setup.
2. Execute the CMake/CTest suite on macOS against generated independent R
   parity fixtures. Historical results must not be inferred from 0/14
   optional fixtures present in a different build.
3. Add safe foreign-language adapters for experimental custom activation,
   regime and adaptive control only after their callback/data contracts are
   separately approved. `curvature.h` C01 remains out of public install.
4. Review Python ABI, package version, CITATION/NEWS, and update GitHub only
   after the owner approves the proposed source diff.

**This is an interoperability candidate, not a claim of universal coverage
or stable ABI for all experimental mechanisms.**
