"""Consumer-facing Python access to validated public vMAR C23 families.

The original C23 code owns every numerical algorithm. Python binds only
public headers and owns all buffers used for each native operation.
"""
from __future__ import annotations

import ctypes as ct
from dataclasses import dataclass
import math
from typing import Sequence

from . import _native

_lib = ct.CDLL(_native.__file__)
_D = ct.POINTER(ct.c_double)
_S = ct.POINTER(ct.c_size_t)
_I = ct.POINTER(ct.c_int)
_FIELD = ct.CFUNCTYPE(ct.c_int, ct.c_double, _D, ct.c_size_t, _D, ct.c_void_p)
ACTIVATIONS = {"tanh": 1, "sigmoid": 2, "identity": 3, "softplus": 4,
               "relu": 5, "leaky_relu": 6}
INTEGRATORS = {"euler": 1, "rk4": 2}


def _bind(name, args, result=ct.c_int):
    fn = getattr(_lib, name)
    fn.argtypes = args
    fn.restype = result
    return fn


_status_string = _bind("vmar_status_string", [ct.c_int], ct.c_char_p)


class VmarInteropError(RuntimeError):
    """Nonzero status returned by the canonical vMAR C23 implementation."""


def _ok(status):
    if status:
        raise VmarInteropError(f"vMAR C23 ({status}): {_status_string(status).decode()}")


def _vec(values, width=None, label="values"):
    try:
        row = tuple(float(x) for x in values)
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError(f"{label} must contain finite numbers") from exc
    if not row or (width is not None and len(row) != width) or not all(map(math.isfinite, row)):
        raise ValueError(f"{label} has invalid length or nonfinite elements")
    return (ct.c_double * len(row))(*row)


def _flatten(rows, width, label):
    if not rows:
        raise ValueError(f"{label} cannot be empty")
    values = [x for row in rows for x in _vec(row, width, label)]
    return (ct.c_double * len(values))(*values)


def _code(item, mapping):
    if item not in mapping:
        raise ValueError(f"Unsupported choice: {item}")
    return mapping[item]


_activation = _bind("vmar_activation_evaluate", [ct.c_int, ct.c_double, _D, _D])
_ema = _bind("vmar_memory_ema", [_D, _D, ct.c_size_t, ct.c_double, _D])
_relax = _bind("vmar_memory_relaxation", [_D, _D, ct.c_size_t, ct.c_double, ct.c_double, _D])
_workspace_required = _bind("vmar_workspace_required", [ct.c_int, ct.c_size_t, _S])
_step = _bind("vmar_step", [ct.c_int, ct.c_double, ct.c_double,
                           _FIELD, ct.c_void_p, _D, ct.c_size_t,
                           _D, ct.c_size_t, _D])


def activation(name: str, z: float) -> tuple[float, float]:
    """Return activation value and C23 reference derivative (including kinks)."""
    y, dy = ct.c_double(), ct.c_double()
    _ok(_activation(_code(name, ACTIVATIONS), float(z), ct.byref(y), ct.byref(dy)))
    return y.value, dy.value


def memory_ema(memory: Sequence[float], observation: Sequence[float], decay: float):
    """Update observable EMA memory; independent of RL experience replay."""
    m = _vec(memory, label="memory")
    o = _vec(observation, len(m), "observation")
    out = (ct.c_double * len(m))()
    _ok(_ema(m, o, len(m), decay, out))
    return tuple(out)


def memory_relaxation(memory, observation, step: float, tau: float):
    """Exact constant-signal neuronal memory relaxation on a positive interval."""
    m = _vec(memory, label="memory")
    o = _vec(observation, len(m), "observation")
    out = (ct.c_double * len(m))()
    _ok(_relax(m, o, len(m), step, tau, out))
    return tuple(out)


class _Layout(ct.Structure):
    _fields_ = [("widths", _S), ("activations", _I), ("offsets", _S),
                ("layer_count", ct.c_size_t), ("parameters", ct.c_size_t),
                ("sum_widths", ct.c_size_t), ("max_width", ct.c_size_t)]


_layout_make = _bind("vmar_network_layout_make", [_S, ct.c_size_t, _I, _S,
                                                   ct.c_size_t, ct.POINTER(_Layout)])
_network_ws = _bind("vmar_network_workspace_required", [ct.POINTER(_Layout), _S])
_forward = _bind("vmar_network_forward", [ct.POINTER(_Layout), _D, ct.c_size_t,
                                         _D, _D, ct.c_size_t, _D])
_network_grad = _bind("vmar_network_loss_gradient", [ct.POINTER(_Layout), _D, _D,
                                                     ct.c_size_t, _D, _D,
                                                     ct.c_size_t, _D, _D, _D])


class _NetworkProblem(ct.Structure):
    _fields_ = [("layout", ct.POINTER(_Layout)), ("x", _D), ("target", _D),
                ("samples", ct.c_size_t), ("eta", ct.c_double),
                ("workspace", _D), ("workspace_doubles", ct.c_size_t)]


class Network:
    """Public vMAR C23 scalar-output MLP (not the multi-output vMAR.MLP library).

    Caller supplies flat row-major theta with weights then biases per layer.
    The object retains descriptor-array ownership and never duplicates math.
    """

    def __init__(self, widths, activations=None, parameters=None):
        self.widths = tuple(int(w) for w in widths)
        if len(self.widths) < 2 or any(w < 1 for w in self.widths) or self.widths[-1] != 1:
            raise ValueError("vMAR core network requires widths ending in 1")
        if activations is None:
            activations = ("tanh",) * (len(self.widths) - 2) + ("identity",)
        if len(activations) != len(self.widths) - 1:
            raise ValueError("Exactly one activation per layer is required")
        self.activations = tuple(activations)
        self._widths = (ct.c_size_t * len(self.widths))(*self.widths)
        self._acts = (ct.c_int * len(activations))(*(_code(a, ACTIVATIONS) for a in activations))
        self._offsets = (ct.c_size_t * len(activations))()
        self._layout = _Layout()
        _ok(_layout_make(self._widths, len(self.widths), self._acts,
                         self._offsets, len(activations), ct.byref(self._layout)))
        self.n_parameters = self._layout.parameters
        self._parameters = _vec(parameters if parameters is not None else
                                [0.] * self.n_parameters, self.n_parameters, "parameters")
        self._ws_size = ct.c_size_t()
        _ok(_network_ws(ct.byref(self._layout), ct.byref(self._ws_size)))

    @property
    def parameters(self):
        return tuple(self._parameters)

    @parameters.setter
    def parameters(self, values):
        values = _vec(values, self.n_parameters, "parameters")
        self._parameters[:] = values[:]

    def predict(self, observations):
        rows = list(observations)
        x = _flatten(rows, self.widths[0], "observations")
        ws = (ct.c_double * self._ws_size.value)()
        out = (ct.c_double * len(rows))()
        _ok(_forward(ct.byref(self._layout), x, len(rows), self._parameters,
                     ws, len(ws), out))
        return tuple(out)

    def loss_gradient(self, observations, targets):
        rows = list(observations)
        target_values = tuple(float(t) for t in targets)
        if len(rows) != len(target_values):
            raise ValueError("Target count differs from number of observations")
        x = _flatten(rows, self.widths[0], "observations")
        y = _vec(target_values, len(rows), "targets")
        ws = (ct.c_double * self._ws_size.value)()
        loss = ct.c_double()
        gradient = (ct.c_double * self.n_parameters)()
        predictions = (ct.c_double * len(rows))()
        _ok(_network_grad(ct.byref(self._layout), x, y, len(rows), self._parameters,
                          ws, len(ws), ct.byref(loss), gradient, predictions))
        return loss.value, tuple(gradient), tuple(predictions)

    def step(self, observations, targets, *, method="rk4", time=0., step=.1, eta=1.):
        """Integrate theta'=-eta*grad(MSE) with original C23 vector field."""
        rows = list(observations)
        if len(rows) != len(targets):
            raise ValueError("Target count mismatch")
        x = _flatten(rows, self.widths[0], "observations")
        y = _vec(targets, len(rows), "targets")
        ws = (ct.c_double * self._ws_size.value)()
        problem = _NetworkProblem(ct.pointer(self._layout), x, y, len(rows),
                                  eta, ws, len(ws))
        method_code = _code(method, INTEGRATORS)
        required = ct.c_size_t()
        _ok(_workspace_required(method_code, self.n_parameters, ct.byref(required)))
        solver_ws = (ct.c_double * required.value)()
        candidate = (ct.c_double * self.n_parameters)()
        field = ct.cast(_lib.vmar_network_field, _FIELD)
        _ok(_step(method_code, time, step, field, ct.byref(problem),
                  self._parameters, self.n_parameters, solver_ws,
                  len(solver_ws), candidate))
        self._parameters[:] = candidate[:]
        return self.parameters


class _NeuronProblem(ct.Structure):
    _fields_ = [("x", _D), ("target", _D), ("samples", ct.c_size_t),
                ("features", ct.c_size_t), ("activation", ct.c_int),
                ("eta", ct.c_double)]


class _LearningSystem(ct.Structure):
    _fields_ = [("neuron", _NeuronProblem), ("tau", ct.c_double),
                ("coupling", ct.c_double)]


class _ControlPolicy(ct.Structure):
    _fields_ = [("initial_step", ct.c_double), ("reduction", ct.c_double),
                ("max_attempts", ct.c_size_t), ("loss_tolerance", ct.c_double),
                ("memory_threshold", ct.c_double)]


class _ControlReport(ct.Structure):
    _fields_ = [("accepted", ct.c_int), ("attempts", ct.c_size_t),
                ("rejected", ct.c_size_t), ("field_evaluations", ct.c_size_t),
                ("old_loss", ct.c_double), ("candidate_loss", ct.c_double),
                ("effective_step", ct.c_double), ("final_time", ct.c_double)]


_control_ws = _bind("vmar_control_workspace_required", [ct.c_int, ct.c_size_t, _S])
_control_step = _bind("vmar_control_step", [ct.c_int, ct.POINTER(_LearningSystem),
                                           ct.POINTER(_ControlPolicy), _D, ct.c_size_t,
                                           _D, _S, _D, ct.c_size_t,
                                           ct.POINTER(_ControlReport)])


class ControlledNeuron:
    """Observable C23 single-neuron memory/control with transactional updates."""

    def __init__(self, x, targets, theta, *, activation="tanh", eta=1.,
                 tau=.25, coupling=0., memory=None):
        rows = list(x)
        if not rows:
            raise ValueError("At least one training observation is required")
        self.features = len(rows[0])
        self._x = _flatten(rows, self.features, "x")
        self._y = _vec(targets, len(rows), "targets")
        self._theta = _vec(theta, self.features + 1, "theta")
        self._memory = _vec(memory if memory is not None else [0.] * len(self._theta),
                            len(self._theta), "memory")
        self._state = (ct.c_double * (2 * len(self._theta)))(*self._theta, *self._memory)
        self._sys = _LearningSystem(_NeuronProblem(self._x, self._y, len(rows),
                              self.features, _code(activation, ACTIVATIONS), eta),
                              tau, coupling)
        self.time, self.iteration = ct.c_double(0.), ct.c_size_t(0)

    @property
    def state(self):
        return tuple(self._state)

    def step(self, *, method="rk4", initial_step=.1, reduction=.5,
             max_attempts=8, loss_tolerance=0., memory_threshold=0.):
        method_code = _code(method, INTEGRATORS)
        required = ct.c_size_t()
        _ok(_control_ws(method_code, len(self._state), ct.byref(required)))
        workspace = (ct.c_double * required.value)()
        policy = _ControlPolicy(initial_step, reduction, max_attempts,
                                loss_tolerance, memory_threshold)
        report = _ControlReport()
        _ok(_control_step(method_code, ct.byref(self._sys), ct.byref(policy),
                          self._state, len(self._state), ct.byref(self.time),
                          ct.byref(self.iteration), workspace, len(workspace),
                          ct.byref(report)))
        return {"accepted": bool(report.accepted), "attempts": report.attempts,
                "rejected": report.rejected, "field_evaluations": report.field_evaluations,
                "old_loss": report.old_loss, "candidate_loss": report.candidate_loss,
                "effective_step": report.effective_step, "time": self.time.value,
                "state": self.state}


class _TraceRecord(ct.Structure):
    _fields_ = [("iter", ct.c_size_t), ("time", ct.c_double),
                ("loss", ct.c_double), ("step", ct.c_double)]


class _Trace(ct.Structure):
    _fields_ = [("dimension", ct.c_size_t), ("capacity", ct.c_size_t),
                ("length", ct.c_size_t), ("records", ct.POINTER(_TraceRecord)),
                ("states", _D)]


_trace_init = _bind("vmar_trace_init", [ct.POINTER(_Trace), ct.c_size_t,
                                       ct.c_size_t, ct.POINTER(_TraceRecord), _D])
_trace_append = _bind("vmar_trace_append", [ct.POINTER(_Trace), ct.c_size_t,
                                           ct.c_double, ct.c_double, ct.c_double, _D])


@dataclass(frozen=True)
class TracePoint:
    iteration: int
    time: float
    loss: float
    step: float
    state: tuple[float, ...]


class ConfirmedTrace:
    """Native-backed confirmed state trace, not a rejected-attempt log."""

    def __init__(self, dimension: int, capacity: int):
        if not isinstance(dimension, int) or dimension < 1 or not isinstance(capacity, int) or capacity < 1:
            raise ValueError("Trace dimensions must be positive integers")
        self.dimension, self.capacity = dimension, capacity
        self._records = (_TraceRecord * capacity)()
        self._states = (ct.c_double * (capacity * dimension))()
        self._trace = _Trace()
        _ok(_trace_init(ct.byref(self._trace), dimension, capacity,
                        self._records, self._states))

    def append(self, iteration: int, time: float, loss: float, step: float, state):
        values = _vec(state, self.dimension, "state")
        _ok(_trace_append(ct.byref(self._trace), iteration, time, loss, step, values))

    @property
    def records(self):
        return tuple(TracePoint(self._records[i].iter, self._records[i].time,
                                self._records[i].loss, self._records[i].step,
                                tuple(self._states[i*self.dimension:(i+1)*self.dimension]))
                     for i in range(self._trace.length))


class _StateLayout(ct.Structure):
    _fields_ = [("kind", ct.c_int), ("parameters", ct.c_size_t),
                ("theta_offset", ct.c_size_t), ("first_offset", ct.c_size_t),
                ("second_offset", ct.c_size_t), ("total_doubles", ct.c_size_t)]


_state_layout = _bind("vmar_state_layout_make", [ct.c_size_t, ct.c_int,
                                                 ct.POINTER(_StateLayout)])
_state_validate = _bind("vmar_state_layout_validate", [ct.POINTER(_StateLayout),
                                                       _D, ct.c_size_t])
STATE_KINDS = {"parameters": 1, "first_memory": 2, "second_moment": 3}


@dataclass(frozen=True)
class StateLayout:
    kind: str
    parameters: int
    theta_offset: int
    first_offset: int
    second_offset: int
    total_doubles: int

    @classmethod
    def make(cls, parameters: int, kind="first_memory"):
        descriptor = _StateLayout()
        _ok(_state_layout(parameters, _code(kind, STATE_KINDS), ct.byref(descriptor)))
        return cls(kind, parameters, descriptor.theta_offset, descriptor.first_offset,
                   descriptor.second_offset, descriptor.total_doubles)

    def validate(self, values):
        descriptor = _StateLayout()
        _ok(_state_layout(self.parameters, _code(self.kind, STATE_KINDS), ct.byref(descriptor)))
        data = _vec(values, self.total_doubles, "state")
        _ok(_state_validate(ct.byref(descriptor), data, len(data)))
        return True


class _NetworkSystem(ct.Structure):
    _fields_ = [("network", _NetworkProblem), ("tau", ct.c_double),
                ("coupling", ct.c_double)]


class _Mechanism(ct.Structure):
    _fields_ = [("mode", ct.c_int), ("system", _NetworkSystem),
                ("policy", _ControlPolicy)]


class _NetworkDecision(ct.Structure):
    _fields_ = [("mode", ct.c_int), ("accepted", ct.c_int),
                ("attempts", ct.c_size_t), ("rejected", ct.c_size_t),
                ("field_evaluations", ct.c_size_t),
                ("loss_before", ct.c_double), ("loss_after", ct.c_double),
                ("gradient_norm_before", ct.c_double),
                ("memory_norm_before", ct.c_double),
                ("coupling_term_norm_before", ct.c_double),
                ("requested_step", ct.c_double), ("effective_step", ct.c_double),
                ("time_before", ct.c_double), ("time_after", ct.c_double)]


_network_control_ws = _bind("vmar_network_control_workspace_required",
                            [ct.c_int, ct.c_size_t, _S])
_mechanism_step = _bind("vmar_network_mechanism_step", [ct.c_int, ct.POINTER(_Mechanism),
                                                        ct.c_double, _D, ct.c_size_t,
                                                        _D, _S, _D, ct.c_size_t,
                                                        ct.POINTER(_NetworkDecision)])
NETWORK_MODES = {"baseline": 1, "observe": 2, "coupled": 3, "controlled": 4}


class NetworkMechanism:
    """Native M0/M1/M2/M3 network/memory modes, with explicit intervention."""

    def __init__(self, network: Network, observations, targets, *, mode="observe",
                 eta=1., tau=.25, coupling=0., memory=None):
        self.network, self.mode = network, mode
        rows = list(observations)
        self._x = _flatten(rows, network.widths[0], "observations")
        self._y = _vec(targets, len(rows), "targets")
        self._ws = (ct.c_double * network._ws_size.value)()
        self._problem = _NetworkProblem(ct.pointer(network._layout), self._x,
                                        self._y, len(rows), eta, self._ws, len(self._ws))
        p = network.n_parameters
        self._initial_memory = _vec(memory if memory is not None else [0.] * p, p, "memory")
        self._state = (ct.c_double * (2 * p))(*network.parameters, *self._initial_memory)
        self._system = _NetworkSystem(self._problem, tau, coupling)
        self._time, self._iter = ct.c_double(0.), ct.c_size_t(0)
        _code(mode, NETWORK_MODES)

    @property
    def state(self):
        return tuple(self._state)

    def step(self, *, method="rk4", step=.1, reduction=.5,
             max_attempts=8, loss_tolerance=0., memory_threshold=0.):
        method_code = _code(method, INTEGRATORS)
        size = ct.c_size_t()
        _ok(_network_control_ws(method_code, self.network.n_parameters, ct.byref(size)))
        workspace = (ct.c_double * size.value)()
        descriptor = _Mechanism(_code(self.mode, NETWORK_MODES), self._system,
                                _ControlPolicy(step, reduction, max_attempts,
                                               loss_tolerance, memory_threshold))
        report = _NetworkDecision()
        _ok(_mechanism_step(method_code, ct.byref(descriptor), step,
                            self._state, len(self._state), ct.byref(self._time),
                            ct.byref(self._iter), workspace, len(workspace),
                            ct.byref(report)))
        if report.accepted:
            self.network.parameters = self._state[:self.network.n_parameters]
        return dict(mode=self.mode, accepted=bool(report.accepted),
                    attempts=report.attempts, rejected=report.rejected,
                    field_evaluations=report.field_evaluations,
                    loss_before=report.loss_before, loss_after=report.loss_after,
                    gradient_norm=report.gradient_norm_before,
                    memory_norm=report.memory_norm_before,
                    coupling_norm=report.coupling_term_norm_before,
                    effective_step=report.effective_step, time=self._time.value,
                    state=self.state)
