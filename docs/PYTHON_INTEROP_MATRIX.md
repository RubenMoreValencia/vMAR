# C23–R–Python interoperability contract (vMAR G0.9-P.3 candidate)

This change extends Python **without changing** any established numerical C23
algorithm or stable C23 symbol. The language-neutral contract remains
`vmar::core` and its installed `include/vmar/*.h`; Python is one consumer.

| C23 public contract | R mathematical reference | Python consumer | Status |
| --- | --- | --- | --- |
| `vmar/vmar.h`: Euler/RK4 | `vMAR.Core` integrators | `Differential` | Implemented, old+new tests |
| `vmar/neuron.h`: built-in activations | `vMAR.Core` activation functions | `activation` | Implemented, reference tests |
| `vmar/neuron.h`: neuron gradient/field | `vMAR.Core` neuron functions | `ControlledNeuron` (native C field) | Implemented for controlled dynamics |
| `vmar/network.h`: scalar-output multilayer MLP | `vMAR.Core` MLP functions | `Network` (forward, analytic gradient, RK4/Euler) | Implemented |
| `vmar/memory.h`: EMA/exact relaxation | `vMAR.Core` memory functions | `memory_ema`, `memory_relaxation` | Implemented |
| `vmar/memory.h` + `control.h` | R learning/control reference | `ControlledNeuron` | Implemented |
| `vmar/network_system.h` and `network_mechanism.h`: M0–M3 | R system reference | `NetworkMechanism` | Implemented (mode selection, transaction report) |
| `vmar/state_layout.h` | R state description | `StateLayout` | Implemented |
| `vmar/trace.h`: confirmed trace | R trajectory reference | `ConfirmedTrace` | Implemented |
| `vmar/trace.h`: attempt log | R rejection accounting | C23 header available | Python convenience adapter pending |
| `vmar/custom_activation.h` | R custom activation reference | C23 header available | Specialized callback/derivative ABI pending |
| `vmar/adaptive_control.h` | R adaptive control reference | C23 header available | Specialized adapter pending |
| `vmar/regime.h` | R regime diagnostic | C23 header available | Specialized adapter pending |
| `vmar/curvature.h` C01 | experimental R oracle | **not installed/public** | Deliberately excluded |

**Interpretation of "universal API":** Any C23-compatible consumer can use
the documented library headers and `find_package(vMAR CONFIG REQUIRED)` /
`vmar::core`. Python additionally has high-level data-owning wrappers for
the general validated families listed above. This is not yet a promise that
every experimental callback and C struct has a stable foreign-language ABI.

**Important boundaries:** vMAR's core neural network is scalar-output,
whereas `vMAR.MLP` is the independently configured multi-output neural
library. The future `vMAR.MLP` Python adapter should build on these C23
contracts; RL replay/experience logic belongs in `vMAR.MLP.RL`, not vMAR.

The C23, R and Python numerical oracles must compare parameter packing,
MSE normalization, native Euler/RK4 time, memory, control acceptance and
confirmed-vs-rejected trace semantics. Numeric tolerance is declared per
test, not assumed to be bitwise identical across compilers.
