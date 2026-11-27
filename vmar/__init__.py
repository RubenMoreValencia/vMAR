"""vMAR's simple, scientific Python interface backed by the C23 core.

    from vmar import Differential

Interoperability v1 additionally exposes selected validated public C23
families without replacing the numerical implementation.
"""

from .differential import Differential, Trajectory, VmarError
from .interop import (ACTIVATIONS, INTEGRATORS, VmarInteropError,
                      activation, memory_ema, memory_relaxation,
                      Network, ControlledNeuron, ConfirmedTrace, TracePoint,
                      StateLayout, NetworkMechanism, NETWORK_MODES)

__all__ = ["Differential", "Trajectory", "VmarError", "VmarInteropError",
           "ACTIVATIONS", "INTEGRATORS", "activation", "memory_ema",
           "memory_relaxation", "Network", "ControlledNeuron",
           "ConfirmedTrace", "TracePoint", "StateLayout",
           "NetworkMechanism", "NETWORK_MODES"]
