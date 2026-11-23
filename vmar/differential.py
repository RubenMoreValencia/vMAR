"""Direct Python API to the vMAR C23 differential integrators."""

from __future__ import annotations

import ctypes as ct
from dataclasses import dataclass
import math
from typing import Callable, Sequence

from . import _native


_POINTER = ct.POINTER(ct.c_double)
_FIELD = ct.CFUNCTYPE(ct.c_int, ct.c_double, _POINTER, ct.c_size_t, _POINTER, ct.c_void_p)
_METHODS = {"euler": 1, "rk4": 2}
_CORE = ct.CDLL(_native.__file__)
_CORE.vmar_workspace_required.argtypes = [ct.c_int, ct.c_size_t, ct.POINTER(ct.c_size_t)]
_CORE.vmar_workspace_required.restype = ct.c_int
_CORE.vmar_step.argtypes = [
    ct.c_int, ct.c_double, ct.c_double, _FIELD, ct.c_void_p,
    _POINTER, ct.c_size_t, _POINTER, ct.c_size_t, _POINTER,
]
_CORE.vmar_step.restype = ct.c_int
_CORE.vmar_status_string.argtypes = [ct.c_int]
_CORE.vmar_status_string.restype = ct.c_char_p


class VmarError(RuntimeError):
    """C23 returned a failure status or the Python field callback failed."""


def _check(status: int) -> None:
    if status != 0:
        detail = _CORE.vmar_status_string(status).decode("utf-8")
        raise VmarError(f"vMAR C23 status {status}: {detail}")


def _vector(values: Sequence[float], name: str) -> tuple[float, ...]:
    try:
        result = tuple(float(value) for value in values)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite numeric sequence") from exc
    if not result or not all(math.isfinite(value) for value in result):
        raise ValueError(f"{name} must be nonempty and finite")
    return result


@dataclass(frozen=True)
class Trajectory:
    """Confirmed numerical states at times including the initial time."""

    times: tuple[float, ...]
    states: tuple[tuple[float, ...], ...]

    @property
    def state(self) -> tuple[float, ...]:
        """Final state, preserving vector dimension."""
        return self.states[-1]


class Differential:
    """Integrate a user-supplied field using the vMAR C23 implementation.

    The Python field returns a finite vector with the same dimension as the
    state. Euler and RK4 are integration methods, not learning algorithms.
    """

    def __init__(
        self,
        field: Callable[[float, tuple[float, ...]], Sequence[float]],
        initial_state: Sequence[float],
    ):
        if not callable(field):
            raise TypeError("field must be callable")
        self.field = field
        self.initial_state = _vector(initial_state, "initial_state")

    def integrate(
        self,
        *,
        method: str = "rk4",
        step: float = 0.1,
        steps: int = 1,
        time: float = 0.0,
    ) -> Trajectory:
        if method not in _METHODS:
            raise ValueError("method must be 'euler' or 'rk4'")
        if isinstance(steps, bool) or not isinstance(steps, int) or steps < 0:
            raise ValueError("steps must be a nonnegative integer")
        if not math.isfinite(step) or step <= 0 or not math.isfinite(time):
            raise ValueError("step must be positive and finite; time must be finite")
        method_code = _METHODS[method]
        n = len(self.initial_state)
        workspace_size = ct.c_size_t()
        _check(_CORE.vmar_workspace_required(method_code, n, ct.byref(workspace_size)))
        workspace = (ct.c_double * workspace_size.value)()
        state = (ct.c_double * n)(*self.initial_state)
        candidate = (ct.c_double * n)()
        times = [float(time)]
        states = [self.initial_state]
        failures: list[Exception] = []

        @_FIELD
        def callback(t, values, dimension, derivative, _context):
            try:
                current = tuple(values[i] for i in range(dimension))
                result = _vector(self.field(t, current), "field derivative")
                if len(result) != dimension:
                    raise ValueError("field derivative has incorrect dimension")
                for i, value in enumerate(result):
                    derivative[i] = value
                return 0
            except Exception as exc:
                failures.append(exc)
                return 5

        for index in range(steps):
            current_time = time + index * step
            if not math.isfinite(current_time + step):
                raise VmarError("time overflow")
            status = _CORE.vmar_step(
                method_code, current_time, step, callback, None,
                state, n, workspace, workspace_size.value, candidate,
            )
            if failures:
                raise VmarError("field callback failed") from failures[0]
            _check(status)
            state[:] = candidate[:]
            times.append(current_time + step)
            states.append(tuple(state))
        return Trajectory(tuple(times), tuple(states))
