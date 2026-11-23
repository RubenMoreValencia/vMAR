"""Experimental Python/ctypes access to the vMAR C23 differential integrator.

G0.9-P.1: deliberately restricted to the public vmar_step API.
No independent mathematical integrator is implemented in Python.
"""

from __future__ import annotations

import ctypes as ct
import math
from pathlib import Path
from typing import Callable, Sequence


_DOUBLE_PTR = ct.POINTER(ct.c_double)
_FIELD = ct.CFUNCTYPE(
    ct.c_int, ct.c_double, _DOUBLE_PTR, ct.c_size_t, _DOUBLE_PTR, ct.c_void_p
)
_METHODS = {"euler": 1, "rk4": 2}


class VmarError(RuntimeError):
    """An argument, numerical, callback, or C23 status error."""


class VmarDifferential:
    """Load libvmar_ffi and invoke the stable-core candidate's integration API."""

    def __init__(self, library_path: str | Path):
        self._library = ct.CDLL(str(Path(library_path).resolve()))
        self._library.vmar_workspace_required.argtypes = [
            ct.c_int, ct.c_size_t, ct.POINTER(ct.c_size_t)
        ]
        self._library.vmar_workspace_required.restype = ct.c_int
        self._library.vmar_step.argtypes = [
            ct.c_int, ct.c_double, ct.c_double, _FIELD, ct.c_void_p,
            _DOUBLE_PTR, ct.c_size_t, _DOUBLE_PTR, ct.c_size_t, _DOUBLE_PTR
        ]
        self._library.vmar_step.restype = ct.c_int
        self._library.vmar_status_string.argtypes = [ct.c_int]
        self._library.vmar_status_string.restype = ct.c_char_p

    def step(
        self,
        field: Callable[[float, tuple[float, ...]], Sequence[float]],
        state: Sequence[float],
        *,
        time: float = 0.0,
        step: float = 0.1,
        method: str = "rk4",
    ) -> tuple[float, ...]:
        if method not in _METHODS:
            raise ValueError("method must be 'euler' or 'rk4'")
        values = tuple(float(x) for x in state)
        if not values or not all(math.isfinite(x) for x in values):
            raise ValueError("state must be nonempty and finite")
        if not math.isfinite(time) or not math.isfinite(step) or step <= 0:
            raise ValueError("time must be finite and step positive and finite")
        dimension = len(values)
        workspace_size = ct.c_size_t()
        method_id = _METHODS[method]
        status = self._library.vmar_workspace_required(
            method_id, dimension, ct.byref(workspace_size)
        )
        self._check(status)
        input_buffer = (ct.c_double * dimension)(*values)
        output_buffer = (ct.c_double * dimension)()
        workspace = (ct.c_double * workspace_size.value)()
        callback_error: list[Exception] = []

        @_FIELD
        def callback(t, state_ptr, n, derivative_ptr, _context):
            try:
                current = tuple(state_ptr[i] for i in range(n))
                derivative = tuple(float(x) for x in field(t, current))
                if len(derivative) != n or not all(math.isfinite(x) for x in derivative):
                    raise ValueError("field must return finite derivatives of matching dimension")
                for i, value in enumerate(derivative):
                    derivative_ptr[i] = value
                return 0
            except Exception as error:
                # ctypes callback exceptions must never escape into C.
                callback_error.append(error)
                return 5  # Nonzero; core converts callback failures to VMAR_ERR_FIELD.

        status = self._library.vmar_step(
            method_id, time, step, callback, None, input_buffer, dimension,
            workspace, workspace_size.value, output_buffer
        )
        if callback_error:
            raise VmarError("Python field callback failed") from callback_error[0]
        self._check(status)
        return tuple(output_buffer)

    def _check(self, status: int) -> None:
        if status:
            message = self._library.vmar_status_string(status).decode("utf-8")
            raise VmarError(f"vMAR C23 returned {status}: {message}")
