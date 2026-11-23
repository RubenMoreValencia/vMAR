# vMAR — Independent differential core (C23)

**Languages:** English (canonical technical documentation) · [Español](README.es.md).

**Research release documentation:** [public API](docs/PUBLIC_API.md) ·
[reproducibility](docs/REPRODUCIBILITY.md) ·
[G0.9-A.3 release review](docs/G0.9-A3_RELEASE_READINESS.md) ·
[source provenance](docs/G0.9-A2_SOURCE_PROVENANCE_REVIEW.md) ·
[contributing](CONTRIBUTING.md) · [citation](CITATION.cff).

## Use vMAR from another C23 project

The installable **stable-core candidate** is `vmar_core`; experimental
curvature and reference optimizers are not exported as part of its API.
The following example uses a private installation prefix (no administrator
access required):

```sh
cmake -S . -B build-release -DVMAR_BUILD_TESTS=OFF \
  -DVMAR_BUILD_EXPERIMENTAL_CURVATURE=OFF \
  -DCMAKE_INSTALL_PREFIX="$PWD/vmar-install"
cmake --build build-release
cmake --install build-release

cmake -S examples/external_consumer -B build-consumer \
  -DCMAKE_PREFIX_PATH="$PWD/vmar-install"
cmake --build build-consumer
ctest --test-dir build-consumer --output-on-failure
```

The independent consumer uses `find_package(vMAR CONFIG REQUIRED)` and
links the imported target `vmar::core`. See
[G0.9-A.1 package audit](docs/G0.9-A1_PUBLIC_API_INSTALL_VERSION_AUDIT.md)
and [version history](CHANGELOG.md). The research author has ratified
provenance and publication identity, and the source is now distributed
under the [MIT License](LICENSE). This is a pre-release API, not a 1.0.0
promise. An official GitHub release and permanent repository URL remain
to be established; the author's declaration is not an independent
source-history or third-party rights audit.

The scientific strict-parity gate requires the original 14 R-produced
fixtures and is deliberately separate from the installation smoke test.

**Project status:** version 0.1.16-alpha (as declared by CMake and API headers).
**G0.8-D mathematical consolidation:** the C01 local Hessian diagnostic is
technically validated as a **separate experimental opt-in** target, not a
part of the installed stable core or a curvature-controlled learning mode.
See [G0.8-D decision](docs/G0.8-D_CONSOLIDATED_MATHEMATICAL_CORE.md),
[mathematical mechanism register](docs/G0.8-D_MATHEMATICAL_MECHANISMS.csv),
and [G0.8-C.1 implementation contract](docs/G0.8-C1_CURVATURE_IMPLEMENTATION_AUDIT.md).
The researcher reports 43/43 strict macOS CTest passes, including 14/14
historical fixtures and 972/972 C01 Hessian-coordinate R–C23 comparisons.
This verifies a bounded implementation, **not** a general stability theorem.
**G0.7-C consolidated mechanism selection:** see
[core decision and explainability contracts](docs/G0.7-C_CONSOLIDATED_CORE.md)
and [selection matrix](docs/G0.7-C_MECHANISM_SELECTION.csv).
**G0.7-B.2 contribution experiment:** see
[experimental protocol and evidence gates](docs/G0.7-B2_EFFECTIVE_CONTRIBUTION.md).
**Coupled multilayer milestone G0.7-B.1:** see
[coupled-state mathematical contract and recovery gate](docs/G0.7-B1_COUPLED_MULTILAYER.md).
**Current classification:** see [G0.6-A core inventory](docs/G0.6-A_CORE_INVENTORY.md).
**Unified state boundary:** see [G0.6-B API contracts](docs/G0.6-B_UNIFIED_API.md).
**Formal contract alignment:** see [G0.6-C FormalSpec obligations](docs/G0.6-C_FORMALSPEC_ALIGNMENT.md).
**Trace interoperability:** see [G0.6-C.2](docs/G0.6-C2_TRACE_INTEROPERABILITY.md).
**Trace R–C23 parity:** see [G0.6-C.3](docs/G0.6-C3_TRACE_PARITY.md).
**Core consolidation and stable-candidate decisions:** see
[G0.6-D](docs/G0.6-D_CONSOLIDATION.md) and
[decision matrix](docs/G0.6-D_CORE_DECISIONS.csv).
**G0.7-A multilayer mathematical architecture:** see
[specification](docs/G0.7-A_MULTILAYER_ARCHITECTURE.md) and
[source mapping](docs/G0.7-A_SOURCE_MAPPING.csv). The R reference functions
belong to `vMAR.Core` 0.0.13. **G0.7-B** introduces an initial public C23
multilayer network API; see [implementation and parity protocol](docs/G0.7-B_MULTILAYER_C23.md).
This is an *independent C23*
numerical core, not a conversion of the entire DSNeuralRNAS ecosystem.
Implementation comments, interfaces, and test identifiers use English.

The historical G0.5-E milestone paragraphs below preserve the experimental
record at the time they were written. They are **not** a current release claim
that every legacy fixture ships with this archive. In G0.6-A, the user-reported
17/17 CTest success relied on locally generated R–C23 fixture CSV files;
CMake enables such suites conditionally when those fixtures are present.

## Core scope

- Explicit vector differential-field callback `vmar_field_fn`.
- Euler and classical RK4, selected independently of the mathematical field.
- No hidden allocation per step: caller supplies workspace (`n` doubles for
  Euler, `5*n` doubles for RK4).
- Transactional step: failed numeric/callback validation leaves output
  unchanged, provided callers honor the documented non-overlap restrictions.
- Explicit numerical status values and deterministic unit tests.
- Analytical solution and independently generated R references as test oracles.
- A separate natural-learning single-neuron module with `tanh`/`sigmoid`,
  MSE, analytic gradient and constant-rate gradient-flow field. Euler
  reproduces its original update algebra; RK4 integrates the **same field**
  as an alternative reference scheme.
- A fully-connected multilayer regression model in `vmar/network.h` with
  layer-specific built-in activations, row-major parameter packing, forward
  evaluation, analytical backpropagation, and a differential learning field.

The exact solution is intentionally **not** exposed as a generic integrator:
it exists only for particular equations and belongs in reference tests.

## CLion setup (Intel macOS, Apple Clang 17)

1. Unzip the archive and open the **vMAR** directory in CLion (the directory
   containing `CMakeLists.txt`).
2. Select Xcode/Apple Clang as the C compiler, system CMake 3.21.2 or newer,
   and the detected Ninja or Make build tool.
3. Reload CMake and build `vmar_test_core`.
4. Run `vmar_test_core`, then use CLion's CTest runner or terminal `ctest`.

The project requests `C_STANDARD 23`, `C_STANDARD_REQUIRED ON` and
`C_EXTENSIONS OFF`. The test executable independently checks for
`__STDC_VERSION__ >= 202311L`.

## Terminal commands (from vMAR directory)

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build
ctest --test-dir build --output-on-failure
```

## Numerical evidence and interpretation

The C tests include independent analytic references to `y'=-y`, comparison
to the R G0.5-D.1 fixture (`Euler=0.3486784401`,
`RK4=0.36787977441250` for 10 steps with `h=0.1`), field evaluation counts,
large-step stability examples, multidimensional fields, and rollback.

G0.7-B's multilayer module extends the previously validated neuronal, memory,
control, and trace foundations. This does **not** establish automatic
multilayer-memory/controller coupling or a superior optimizer. No R runtime
is required by this C library; R is used only to generate scientific oracles.

## Public API contract

`include/vmar/vmar.h` and the module headers under `include/vmar/` define
the public boundary. `vmar_workspace_required()`
calculates capacity in **doubles**, not bytes. An integrator must not mutate
the input state until success; caller-owned workspaces enable deterministic
allocation strategies. A field callback must not mutate its `state` input.

Current limits: CPU scalar `double` only, no GPU, no adaptive integrator,
no threading contract, no serialization, scalar-output MLP only and no promise
of numerical identity across different floating-point compiler settings.

## G0.5-E.1 reproducible R-C23 parity

The project includes eight frozen scalar fixtures taken directly from the
user's executed vMAR.Core R references (G0.5-D.1). A separate CTest case,
`vmar_r_parity`, reads `reference/vmar_parity.csv` and checks output states
and field-evaluation counts against the reference records.

To extend the fixture to vector, shifted-equilibrium and time-dependent
fields, load vMAR.Core in **R/RStudio**, set the working directory to this
`vMAR` C project root, then execute:

```r
library(vMAR.Core)
source("reference/generate_R_fixtures.R")
```

If the R package is loaded using `devtools::load_all()` rather than installed,
load it first, set the working directory to the C project root, and run the
same script. The generator overwrites the fixture with **14 R-computed cases**.

Rebuild and run `ctest --test-dir cmake-build-debug --output-on-failure`.
Two distinct CTest suites must pass: `vmar_core_contracts` and `vmar_r_parity`.

Each fixture uses a declared `5e-11` absolute + relative numerical tolerance.
The R output is an external reference; the C implementation does not link R.

## Contract audit

See `docs/G0.5-E_CONTRACTS.md` for the source-to-C traceability matrix,
reference values, and environment acceptance gates. The first preparation
compile with GCC 14 `-std=c23` passed 47 contract assertions and UBSan;
GCC in this environment reports draft `202000L`, so final C23-language
conformance must be checked with Apple Clang on the user's Mac.

The expanded R–C23 numerical-equivalence contract is in
`docs/G0.5-E1_PARITY.md`. The initial parity checker has passed eight
previously executed R references locally; the 14-case cross-language fixture
generation and macOS validation remain explicit next gates.

## G0.5-E.2 natural-learning neuron

The public neural API lives in `include/vmar/neuron.h` and is documented
in `docs/G0.5-E2_NEURON.md`. This adds a third default CTest suite,
`vmar_neuron_contracts` (58 local assertions), without altering Euler/RK4.
Run the R reference generator `reference/generate_R_neuron_fixtures.R`
to create **16 R neuronal fixtures**; after reconfiguring CMake a fourth
suite, `vmar_r_neuron_parity`, is enabled. This latter R–C gate is
**not considered passed** until RStudio and CLion execute it successfully.

## G0.5-E.3 learning-memory state

`include/vmar/memory.h` introduces discrete EMA, exact constant-signal
relaxation, and an augmented state `[theta,memory]` with a gradient-observation
relaxation field. `coupling=0` preserves baseline neural trajectories; nonzero
coupling is an unvalidated research intervention. See
`docs/G0.5-E3_MEMORY.md` for mathematical limits and the activation-roadmap
gate. The `vmar_memory_contracts` suite adds 37 local assertions. Generate
the 16 R fixtures with `reference/generate_R_memory_fixtures.R` to activate
`vmar_r_memory_parity` after a CMake reconfigure.

## G0.5-E.4 feedback controller

`include/vmar/control.h` adds a transactional, deterministic backtracking
controller over the augmented state `[theta,memory]`. Memory gating observes
only confirmed memory. Loss acceptance controls atomic commit, independently
of whether Euler or RK4 computes candidate trajectories. See
`docs/G0.5-E4_CONTROL.md` for contracts and limits. The local controller
suite has 35 passing assertions. Generate 16 R reference cases with
`reference/generate_R_control_fixtures.R`; the additional
`vmar_r_control_parity` suite appears after CMake reconfiguration.

## G0.5-E.4.1 stability and feedback audit

`reference/run_R_stability_control_audit.R` generates a numerical study
with **84 cases**, separating fixed-time evolution, controlled proposals,
and equal field-evaluation budgets. The new `vmar_stability_audit_invariants`
suite tests independence of unlinked observational memory (176 local
assertions), while `vmar_r_stability_audit` compares the 84 external R
references after regeneration and CMake configuration. The hypotheses,
computational cost definitions and interpretation constraints are in
`docs/G0.5-E41_STABILITY_AUDIT.md`. No additional public API is introduced.

## G0.5-E.4.2 demanding-regime robustness

`tests/test_robustness.c` exercises 648 configurations across activation,
integrator, initial perturbation, memory response time, coupling, initial
step and maximum retry attempts. Local C tests passed with 589 accepted,
59 exhausted, 59 recovered and 170 rejected attempts. Independently generate
the matching R fixtures with `reference/run_R_robustness_audit.R`; this
enables `vmar_r_robustness_parity` on reconfiguring CMake. See
`docs/G0.5-E42_ROBUSTNESS.md` for the precise scientific scope.

## G0.5-E.5 activation catalog

R `vMAR.Core 0.0.9` is expanded first with a centralized activation/
derivative contract, used by both reference neuron and MLP. C23 extends
the same built-in catalog (`tanh`, `sigmoid`, `identity`, `softplus`, `relu`,
`leaky_relu`) without changing the differential integrators or previously
validated activation identifiers. `vmar_activation_contracts` passed 142
local assertions. Generate 48 R references with
`reference/generate_R_activation_fixtures.R`, then reconfigure CMake to
enable `vmar_r_activation_parity`. See `docs/G0.5-E5_ACTIVATIONS.md` for
the kink convention and research limits.

## G0.5-E.5.1 custom and parametric activations

Version **0.1.8-alpha** introduces `include/vmar/custom_activation.h`:
a user-supplied callback defines `phi(z; alpha)`, its input derivative,
and parameter derivatives. Activation parameters can remain fixed or become
part of the neuron differential state, without changing Euler/RK4 or the
previously validated built-in activation ABI. `vmar_custom_activation_contracts`
passed 103 local C assertions. Generate 40 independent R references with
`reference/generate_R_custom_activation_fixtures.R` and reconfigure CMake to
enable `vmar_r_custom_activation_parity`. See
`docs/G0.5-E51_CUSTOM_ACTIVATIONS.md` for mathematical contracts and limits.

## G0.5-E.5.2 adaptive activations with memory and control

Version **0.1.9-alpha** adds `include/vmar/adaptive_control.h` for an
augmented `[theta, memory]` state. Trainable activation parameters are inside
theta and participate in the same differential field, memory relaxation and
transactional feedback policy as neural weights. The separate extension does
not alter the previously validated built-in APIs. The new C contract suite
has 51 passing local assertions. Generate 16 R references with
`reference/generate_R_adaptive_fixtures.R` to activate the conditional parity
CTest. See `docs/G0.5-E52_ADAPTIVE_CONTROL.md` for mathematical assumptions,
transaction rules and the experimental limits.

## G0.5-E.5.3 scientific coupling and feedback audit

The validated adaptive API remains frozen. The R audit generates 48
equal-final-time cases, 24 equal-field-evaluation-budget comparisons and 24
controlled sequences with explicit acceptance, time and computational cost.
Run `reference/run_R_coupling_audit.R` after loading vMAR.Core; regenerate
CMake to enable `vmar_r_coupling_parity` for 48 independent C23 comparisons.
See `docs/G0.5-E53_COUPLING_AUDIT.md` for the numerical reference, invariants
and research limits.

## G0.5-E.5.4 dynamic regime diagnostics

Experimental API version **0.1.10-alpha** adds
`include/vmar/regime.h`: a geometric gradient-memory diagnostic and a bounded
candidate coupling, evaluated once from confirmed state and frozen across
control retries and RK4 stages. The C contracts have 20 local assertions
passing. Generate 32 R references and 96 comparative policy trajectories via
`reference/run_R_regime_study.R`; reconfigure CMake to include optional
`vmar_r_regime_parity`. See `docs/G0.5-E54_REGIME_ADAPTATION.md` for limits.

## G0.5-E.5.4.1 effective intervention audit

The frozen regime API is evaluated in a new research-only study. Run
`reference/run_R_intervention_audit.R` with the validated R package loaded to
produce 24 controlled trajectories, their complete intervention trace, and
24 matched final-time / 40-gradient-evaluation comparisons. An optional
`vmar_r_intervention_parity` CTest checks all 24 equal-budget cases, including
actual intervention rates and relative memory contributions. See
`docs/G0.5-E541_INTERVENTION_UTILITY.md` for the accounting and limits.

## G0.5-E.5.4.2 transfer decision gate

`reference/run_R_transfer_audit.R` provides a deliberately bounded 72-case
study across tanh, trainable PReLU and trainable Swish, with 24 paired
adaptive-vs-baseline contrasts and 24 independent R–C23 parity fixtures.
At `t=1`, all policies use 40 gradient evaluations inclusive of a common
diagnostic cost. Optional `vmar_r_transfer_parity` becomes available after
generating the R fixture and reconfiguring CMake. See
`docs/G0.5-E542_TRANSFER_GATE.md` for research limits and stopping criteria.
# G0.6-C.2 trace interoperability

The C23 core includes `vmar/trace.h` for recording confirmed state snapshots
and a separate proposal-attempt log in caller-owned buffers. The reference
FormalSpec R adapter is `reference/vmar_formalspec_adapter.R`; its executable
audit is `reference/run_R_formalspec_interop_audit.R`. See
`docs/G0.6-C2_TRACE_INTEROPERABILITY.md` for the semantic contract and
limitations. This feature does not implement FormalSpec Ψ/Γ/Φ.

