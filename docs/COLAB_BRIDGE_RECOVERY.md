# Scientific recovery of `vmar_colab_kit.zip`

The investigator supplied `bridge.c`, `vmar_lab.py` and `CMakeLists.txt`.
These call the original vMAR C23 mathematical engine for neural training,
controlled memory, ecological regimes, Lotka–Volterra, pharmacokinetics and
Hill kinetics. This establishes historical practical feasibility in Colab.

**General and reusable mathematical capabilities:** model layout, forward
evaluation, analytic MSE gradient, differential updates, memory, feedback,
regimes, and model state traces. These should be accessed through C23 public
headers and safe language adapters rather than hardcoded application bridges.

**Application-specific methods:** Lotka–Volterra, pharmacokinetics, Hill
kinetics, ecological observations and dataset-specific training experiments.
Keep those in independent consumers/examples; they are not universal vMAR
public API functions.

The historical `vmar_lab.py` assumes a Colab-specific absolute `.so` path,
requires NumPy and calls `np.random.seed(seed)` globally. It wraps only a
subset of symbols available in `bridge.c`. Its buffer/shape validation and
installation are too narrowly application-specific for direct promotion.

The new `vmar.interop` module uses the existing source-built C23 native
extension (`vmar._native`) with caller-owned buffers and no extra Python
runtime dependencies. No mathematical algorithms are copied from the bridge.
Numerical modeling applications may still use the historical kit as a
separate research consumer if its dependency/environment assumptions hold.
