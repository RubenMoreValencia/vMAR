# Public API guide — vMAR C23 (0.1.16-alpha)

## Audience and package contract

`vmar_core` is an independent C23 static library. After installation,
consumers use `find_package(vMAR CONFIG REQUIRED)` and link `vmar::core`.
The package is not a complete port of DSNeuralRNAS, and R is **not** a
runtime dependency of the installed library.

## Installed headers and API families

| Header | Public capability and boundary |
|---|---|
| `vmar/vmar.h` | Status codes, generic vector-field callback, Euler/RK4 step, workspace query, numeric API version macros. |
| `vmar/neuron.h` | Scalar-output neuron, activation and gradient-flow reference. |
| `vmar/network.h` | Fully connected arbitrary-depth **scalar-output regression**, layer layouts, row-major parameters, forward pass, analytic MSE gradient and field. |
| `vmar/memory.h` | EMA, exact constant-signal relaxation, neural augmented-state field. |
| `vmar/network_system.h` | Coupled `[theta, memory]` field and loss-based controlled steps. |
| `vmar/network_mechanism.h` | M0/M1/M2/M3 selection and decision report; M1 observational memory is the default. |
| `vmar/control.h` | Transactional policy and control diagnostic types. |
| `vmar/adaptive_control.h` | Experimental opt-in adaptive activation/memory/control policy. |
| `vmar/custom_activation.h` | Parameterized/custom activation functionality and its contracts. |
| `vmar/state_layout.h` | Explicit canonical state layout. |
| `vmar/regime.h` | Diagnostic regime identification; does not imply proof of global stability. |
| `vmar/trace.h` | Separate confirmed-state and attempted-proposal logs. |

The header `vmar/curvature.h` is intentionally **not** installed: C01
remains an independently built experimental observer. Reference Adam and
second-moment comparison targets are likewise excluded from public install.

## Basic usage pattern

1. Declare the field/problem and dimension.
2. Ask the relevant API for required workspace in `double` elements.
3. Provide caller-owned, disjoint buffers as specified by the header.
4. Select Euler or RK4 separately from the learning/control model.
5. Inspect the returned `vmar_status` before using results.
6. Record confirmed states separately from accepted/rejected proposals.

The executable working example is
[`examples/external_consumer/`](../examples/external_consumer/).

## Important scientific/engineering limitations

- Only CPU scalar `double` arithmetic; no GPU, tensor engine or threading
  safety guarantee.
- Scalar-output multilayer regression; no bundled general multi-class
  classifier, multi-output network, minibatch trainer or model checkpoint API.
- No stable ABI pledge for major version `0`; changes require explicit
  version and contract review.
- R–C23 parity is numeric within declared tolerances, not bitwise identity
  across toolchains.
- Euler/RK4 are integration methods, not versions of Adam.
- M2/M3 interventions and C01 are research mechanisms; no general
  predictive or stability advantage has been established.
- FormalSpec Ψ/Γ/Φ are delegated to DSNeuralRNAS.FormalSpec and are not
  recreated by the C23 trace API.

## Version/provenance

`CMakeLists.txt` and `vmar/vmar.h` contain numeric version `0.1.16`.
`alpha` is an explicit release-stage label, not a separate binary API
macro. Record the source commit or archive SHA-256 with every study.
