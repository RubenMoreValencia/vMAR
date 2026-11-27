# Public distribution inventory (review candidate)

| Domain | C23 library | Python consumer | R reference |
| --- | --- | --- | --- |
| Euler/RK4 integration | `vmar/vmar.h` | `Differential` | `vMAR.Core` |
| Built-in activations | `vmar/neuron.h` | `activation` | `vMAR.Core` |
| Network, gradient, learning field | `vmar/network.h` | `Network` | `vMAR.Core` |
| Explicit memory | `vmar/memory.h` | `memory_ema`, `memory_relaxation` | `vMAR.Core` |
| Feedback controller | `vmar/control.h` | `ControlledNeuron` | `vMAR.Core` |
| Multilayer mechanism M0–M3 | `vmar/network_mechanism.h` | `NetworkMechanism` | `vMAR.Core` |
| State partition | `vmar/state_layout.h` | `StateLayout` | `vMAR.Core` |
| Confirmed trace | `vmar/trace.h` | `ConfirmedTrace` | `vMAR.Core` |
| Attempt/rejection trace | `vmar/trace.h` | Not wrapped yet | `vMAR.Core` |
| Custom activation callbacks | `vmar/custom_activation.h` | Not wrapped yet | `vMAR.Core` |
| Adaptive control and regimes | `vmar/adaptive_control.h`, `vmar/regime.h` | Not wrapped yet | `vMAR.Core` |
| C01 curvature observer | Experimental non-installed contract | Not part of release API | Experimental |

The C23 headers/CMake package define the *language-neutral* public
interface. Python offers higher-level bindings for the checked subset;
the table does **not** assert that every R and C23 operation is numerically
identical for arbitrary inputs or that a stable foreign ABI exists for
experimental callbacks. The C23 network's output is scalar; the general
multilayer/multi-output library is `vMAR.MLP`.

Consumers can inspect `include/vmar/`, `docs/PUBLIC_API.md`,
`docs/PYTHON_INTEROP_MATRIX.md`, and the tests to determine the precise
accepted input domains, status values and numerical tolerances.

