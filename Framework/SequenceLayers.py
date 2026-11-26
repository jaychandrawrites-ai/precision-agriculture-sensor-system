"""Experimental sequence layers with explicit forward caches and adjoints.

Sequence tensors are batch by time by channel. These layers are independent
of the default derivative-feature forecasting architecture.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

from .Activations import Sigmoid
from .Initializers import Initialization, initialize
from .Layers import Conv2D, Dense, Layer, Parameter


class LayerNorm(Layer):
    """Normalize the last axis, with independent statistics for each position."""

    def __init__(self, backend, channels, epsilon=1e-5):
        if channels < 1 or not math.isfinite(epsilon) or epsilon <= 0:
            raise ValueError("Invalid LayerNorm configuration")
        self.backend = backend
        self.channels = channels
        self.epsilon = epsilon
        self.gamma = Parameter(backend.ones((channels,)), backend.zeros((channels,)), "gamma", False)
        self.beta = Parameter(backend.zeros((channels,)), backend.zeros((channels,)), "beta", False)

    def parameters(self):
        return [self.gamma, self.beta]

    def forward(self, x, training=True):
        if x.ndim < 1 or x.shape[-1] != self.channels:
            raise ValueError("LayerNorm expects channels on the last axis")
        with self.backend.scope():
            work = x.astype("float64" if x.dtype.itemsize == 8 else "float32")
            centered = work - work.mean(axis=-1, keepdims=True)
            inverse = (self.epsilon + (centered**2).mean(axis=-1, keepdims=True))**(-0.5)
            normalized = centered * inverse
            output = normalized * self.gamma.value + self.beta.value
            return self.backend.validate(output.astype(x.dtype)), (normalized, inverse)

    def backward(self, dy, cache):
        normalized, inverse = cache
        if dy.shape != normalized.shape:
            raise ValueError("LayerNorm adjoint shape mismatch")
        with self.backend.scope():
            axes = tuple(range(dy.ndim - 1))
            self.gamma.grad += (dy * normalized).sum(axis=axes)
            self.beta.grad += dy.sum(axis=axes)
            g = dy * self.gamma.value
            dx = g - g.mean(axis=-1, keepdims=True)
            dx -= normalized * (g * normalized).mean(axis=-1, keepdims=True)
            return (dx * inverse).astype(dy.dtype)


class Conv1D(Layer):
    """Temporal convolution, optionally left-padded for causal forecasting."""

    def __init__(self, backend, inputs, outputs, kernel, *, stride=1, dilation=1, causal=True,
                 initializer=None):
        if any(not isinstance(value, int) or value < 1 for value in (kernel, stride, dilation)):
            raise ValueError("Kernel size, stride and dilation must be positive integers")
        self.backend = backend
        self.padding = (kernel - 1) * dilation if causal else 0
        self.conv = Conv2D(backend, inputs, outputs, (1, kernel), stride=(1, stride),
                           dilation=(1, dilation), initializer=initializer)

    def parameters(self):
        return self.conv.parameters()

    def forward(self, x, training=True):
        if x.ndim != 3:
            raise ValueError("Conv1D expects batch by time by channel")
        with self.backend.scope():
            signal = x.transpose(0, 2, 1)[:, :, None, :]
            if self.padding:
                signal = self.backend.xp.pad(signal, ((0, 0), (0, 0), (0, 0), (self.padding, 0)))
            result, cache = self.conv.forward(signal, training)
            return result[:, :, 0, :].transpose(0, 2, 1), cache

    def backward(self, dy, cache):
        with self.backend.scope():
            dx = self.conv.backward(dy.transpose(0, 2, 1)[:, :, None, :], cache)
            return dx[:, :, 0, self.padding:].transpose(0, 2, 1)


class SinusoidalPosition(Layer):
    """Add deterministic fixed positional encoding without trainable weights."""

    def __init__(self, backend, offset=0, base=10000.0):
        if not isinstance(offset, int) or offset < 0 or not math.isfinite(base) or base <= 1:
            raise ValueError("Invalid sinusoidal encoding configuration")
        self.backend = backend
        self.offset = offset
        self.base = base

    def forward(self, x, training=True):
        if x.ndim != 3 or min(x.shape[1:]) < 1:
            raise ValueError("Positional encoding expects batch by time by channel")
        with self.backend.scope():
            xp = self.backend.xp
            steps, channels = x.shape[1:]
            positions = xp.arange(self.offset, self.offset + steps, dtype="float64")[:, None]
            dimensions = xp.arange(channels)
            frequency = self.base**(-2.0 * (dimensions // 2) / channels)
            angles = positions * frequency
            encoded = xp.where(dimensions % 2 == 0, xp.sin(angles), xp.cos(angles))
            return x + encoded.astype(x.dtype), x.shape

    def backward(self, dy, cache):
        if dy.shape != cache:
            raise ValueError("Positional encoding adjoint shape mismatch")
        return dy


@dataclass(frozen=True)
class AttentionCache:
    shape: tuple
    query: object
    key: object
    value: object
    weights: object
    valid_queries: object
    projections: tuple


class MultiHeadAttention(Layer):
    """Self-attention with causal and optional sequence-validity masks."""

    def __init__(self, backend, channels, heads, *, causal=True, initializer=None):
        if channels < 1 or heads < 1 or channels % heads != 0:
            raise ValueError("Attention channels must be a positive multiple of heads")
        self.backend = backend
        self.channels = channels
        self.heads = heads
        self.causal = causal
        self.scale = (channels // heads)**(-0.5)
        init = initializer or Initialization("xavier_uniform")
        self.query = Dense(backend, channels, channels, init)
        self.key = Dense(backend, channels, channels, init)
        self.value = Dense(backend, channels, channels, init)
        self.output = Dense(backend, channels, channels, init)

    def parameters(self):
        return self.query.parameters() + self.key.parameters() + self.value.parameters() + self.output.parameters()

    def _heads(self, array, batch, steps):
        return array.reshape(batch, steps, self.heads, -1).transpose(0, 2, 1, 3)

    def _merge(self, array):
        return array.transpose(0, 2, 1, 3).reshape(-1, self.channels)

    def forward(self, x, training=True, *, valid=None):
        if x.ndim != 3 or x.shape[-1] != self.channels or min(x.shape[:2]) < 1:
            raise ValueError("Attention expects a nonempty batch by time by channel")
        batch, steps, _ = x.shape
        with self.backend.scope():
            xp = self.backend.xp
            keep = xp.ones((batch, steps), dtype=bool) if valid is None else self.backend.array(valid, dtype=bool)
            if keep.shape != (batch, steps):
                raise ValueError("Attention validity mask must be batch by time")
            allowed = keep[:, None, :, None] & keep[:, None, None, :]
            if self.causal:
                allowed = allowed & xp.tril(xp.ones((steps, steps), dtype=bool))
            flat = x.reshape(-1, self.channels)
            q, cq = self.query.forward(flat, training)
            k, ck = self.key.forward(flat, training)
            v, cv = self.value.forward(flat, training)
            dtype = "float64" if x.dtype.itemsize == 8 else "float32"
            q, k, v = (self._heads(item.astype(dtype), batch, steps) for item in (q, k, v))
            scores = (q @ k.swapaxes(-1, -2)) * self.scale
            masked = xp.where(allowed, scores, -xp.inf)
            valid_queries = allowed.any(axis=-1, keepdims=True)
            maximum = xp.where(valid_queries, masked.max(axis=-1, keepdims=True), 0)
            exponent = xp.exp(masked - maximum)
            denominator = xp.where(valid_queries, exponent.sum(axis=-1, keepdims=True), 1)
            weights = exponent / denominator
            context = self._merge(weights @ v).astype(x.dtype)
            projected, co = self.output.forward(context, training)
            output = projected.reshape(x.shape) * keep[..., None]
            cache = AttentionCache(x.shape, q, k, v, weights, keep, (cq, ck, cv, co))
            return output, cache

    def backward(self, dy, cache):
        if dy.shape != cache.shape:
            raise ValueError("Attention adjoint shape mismatch")
        with self.backend.scope():
            q, k, v, weights = cache.query, cache.key, cache.value, cache.weights
            cq, ck, cv, co = cache.projections
            dy = dy * cache.valid_queries[..., None]
            gradient = self.output.backward(dy.reshape(-1, self.channels), co)
            gradient = self._heads(gradient, *dy.shape[:2]).astype(q.dtype)
            dv = weights.swapaxes(-1, -2) @ gradient
            dw = gradient @ v.swapaxes(-1, -2)
            scores_grad = weights * (dw - (dw * weights).sum(axis=-1, keepdims=True)) * self.scale
            dq = scores_grad @ k
            dk = scores_grad.swapaxes(-1, -2) @ q
            dx = (self.query.backward(self._merge(dq).astype(dy.dtype), cq)
                  + self.key.backward(self._merge(dk).astype(dy.dtype), ck)
                  + self.value.backward(self._merge(dv).astype(dy.dtype), cv))
            return dx.reshape(dy.shape)


@dataclass(frozen=True)
class GRUStep:
    previous: object
    update: object
    reset: object
    candidate: object


@dataclass(frozen=True)
class GRUCache:
    sequence: object
    validity: object
    steps: tuple[GRUStep, ...]


class GRU(Layer):
    """One-layer gated recurrent unit with optional initial state and validity mask.

    Invalid time steps carry their previous state unchanged. The state
    initializer and its adjoint are available for windowed training.
    """

    def __init__(self, backend, inputs, hidden, initializer=None):
        if min(inputs, hidden) < 1:
            raise ValueError("GRU input and hidden sizes must be positive")
        self.backend = backend
        self.inputs = inputs
        self.hidden = hidden
        init = initializer or Initialization("xavier_uniform")
        self.input_weights = self._parameter((inputs, 3 * hidden), "input_weights", init)
        self.gate_weights = self._parameter((hidden, 2 * hidden), "gate_weights", init)
        self.candidate_weights = self._parameter((hidden, hidden), "candidate_weights", init)
        self.bias = Parameter(backend.zeros((3 * hidden,)), backend.zeros((3 * hidden,)), "bias", False)
        self.sigmoid = Sigmoid(backend)

    def _parameter(self, shape, name, init):
        return Parameter(initialize(shape, self.backend, init), self.backend.zeros(shape), name)

    def parameters(self):
        return [self.input_weights, self.gate_weights, self.candidate_weights, self.bias]

    def forward(self, x, training=True, *, initial=None, valid=None):
        if x.ndim != 3 or x.shape[-1] != self.inputs or min(x.shape[:2]) < 1:
            raise ValueError("GRU expects a nonempty batch by time by input")
        batch, length, _ = x.shape
        expected = (batch, self.hidden)
        with self.backend.scope():
            xp = self.backend.xp
            state = xp.zeros(expected, dtype=x.dtype) if initial is None else self.backend.array(initial, dtype=x.dtype)
            mask = xp.ones((batch, length), dtype=bool) if valid is None else self.backend.array(valid, dtype=bool)
            if state.shape != expected or mask.shape != (batch, length):
                raise ValueError("GRU initial state or validity mask has the wrong shape")
            self.backend.require_finite(state, "initial GRU state")
            self.backend.validate(x, "GRU input")
            # Each input projection shares one matrix multiplication.
            terms = x @ self.input_weights.value + self.bias.value
            outputs, cache = [], []
            h = self.hidden
            for t in range(length):
                gates = self.sigmoid.evaluate(terms[:, t, :2*h] + state @ self.gate_weights.value)
                update, reset = gates[:, :h], gates[:, h:]
                candidate = xp.tanh(terms[:, t, 2*h:] + (reset * state) @ self.candidate_weights.value)
                cache.append(GRUStep(state, update, reset, candidate))
                state = xp.where(mask[:, t, None], state + update * (candidate - state), state)
                outputs.append(state)
            sequence = xp.stack(outputs, axis=1)
            return self.backend.validate(sequence), GRUCache(x, mask, tuple(cache))

    def backward(self, dy, cache):
        return self.backward_with_state(dy, cache)[0]

    def backward_with_state(self, dy, cache, *, final_gradient=None):
        """Return input and initial-state adjoints, accumulating parameter gradients."""
        x = cache.sequence
        if dy.shape != (*x.shape[:2], self.hidden):
            raise ValueError("GRU adjoint shape mismatch")
        with self.backend.scope():
            xp = self.backend.xp
            state_grad = (xp.zeros_like(dy[:, 0]) if final_gradient is None
                          else self.backend.array(final_gradient, dtype=dy.dtype))
            if state_grad.shape != dy[:, 0].shape:
                raise ValueError("Final-state adjoint shape mismatch")
            terms = xp.empty((*dy.shape[:2], 3 * self.hidden), dtype=dy.dtype)
            for t in reversed(range(dy.shape[1])):
                step = cache.steps[t]
                state, update, reset, candidate = step.previous, step.update, step.reset, step.candidate
                total = state_grad + dy[:, t]
                local = xp.where(cache.validity[:, t, None], total, 0)
                dc = local * update * (1 - candidate**2)
                dz = local * (candidate - state) * update * (1 - update)
                dreset_state = dc @ self.candidate_weights.value.T
                dr = dreset_state * state * reset * (1 - reset)
                gates = xp.concatenate((dz, dr), axis=-1)
                self.gate_weights.grad += state.T @ gates
                self.candidate_weights.grad += (reset * state).T @ dc
                terms[:, t] = xp.concatenate((gates, dc), axis=-1)
                state_grad = (total - local * update + dreset_state * reset
                              + gates @ self.gate_weights.value.T)
            flat = terms.reshape(-1, terms.shape[-1])
            self.input_weights.grad += x.reshape(-1, self.inputs).T @ flat
            self.bias.grad += flat.sum(axis=0)
            return terms @ self.input_weights.value.T, state_grad


@dataclass(frozen=True)
class LSTMStep:
    hidden: object
    cell: object
    write: object
    keep: object
    show: object
    candidate: object
    cell_output: object


class LSTM(Layer):
    """Causal LSTM with explicit hidden and cell adjoints across time."""

    def __init__(self, backend, inputs, hidden, initializer=None):
        if min(inputs, hidden) < 1:
            raise ValueError("LSTM input and hidden sizes must be positive")
        self.backend, self.inputs, self.hidden = backend, inputs, hidden
        init = initializer or Initialization("xavier_uniform")
        self.input_weights = Parameter(initialize((inputs, 4 * hidden), backend, init),
                                       backend.zeros((inputs, 4 * hidden)), "input_weights")
        self.state_weights = Parameter(initialize((hidden, 4 * hidden), backend, init),
                                       backend.zeros((hidden, 4 * hidden)), "state_weights")
        self.bias = Parameter(backend.zeros((4 * hidden,)), backend.zeros((4 * hidden,)), "bias", False)
        with backend.scope():
            self.bias.value[hidden:2 * hidden] = 1
        self.sigmoid = Sigmoid(backend)

    def parameters(self):
        return [self.input_weights, self.state_weights, self.bias]

    def forward(self, x, training=True):
        if x.ndim != 3 or x.shape[-1] != self.inputs or min(x.shape[:2]) < 1:
            raise ValueError("LSTM input must be batch by time by features")
        with self.backend.scope():
            xp = self.backend.xp
            self.backend.validate(x, "LSTM input")
            hidden = xp.zeros((len(x), self.hidden), dtype=x.dtype)
            memory = xp.zeros_like(hidden)
            terms = x @ self.input_weights.value + self.bias.value
            states, steps = [], []
            for position in range(x.shape[1]):
                i, f, o, g = xp.split(terms[:, position] + hidden @ self.state_weights.value, 4, axis=-1)
                write, keep, show = (self.sigmoid.evaluate(gate) for gate in (i, f, o))
                candidate = xp.tanh(g)
                next_memory = keep * memory + write * candidate
                displayed = xp.tanh(next_memory)
                steps.append(LSTMStep(hidden, memory, write, keep, show, candidate, displayed))
                hidden, memory = show * displayed, next_memory
                states.append(hidden)
            return self.backend.validate(xp.stack(states, axis=1)), (x, tuple(steps))

    def backward(self, dy, cache):
        x, steps = cache
        if dy.shape != (*x.shape[:2], self.hidden):
            raise ValueError("LSTM gradient shape mismatch")
        with self.backend.scope():
            xp = self.backend.xp
            dh = xp.zeros_like(dy[:, 0])
            dm = xp.zeros_like(dh)
            terms = xp.empty((*dy.shape[:2], 4 * self.hidden), dtype=dy.dtype)
            for position in reversed(range(dy.shape[1])):
                step = steps[position]
                h_grad = dy[:, position] + dh
                m_grad = dm + h_grad * step.show * (1 - step.cell_output**2)
                di = m_grad * step.candidate * step.write * (1 - step.write)
                df = m_grad * step.cell * step.keep * (1 - step.keep)
                do = h_grad * step.cell_output * step.show * (1 - step.show)
                dg = m_grad * step.write * (1 - step.candidate**2)
                local = xp.concatenate((di, df, do, dg), axis=-1)
                terms[:, position] = local
                self.state_weights.grad += step.hidden.T @ local
                dh = local @ self.state_weights.value.T
                dm = m_grad * step.keep
            flat = terms.reshape(-1, terms.shape[-1])
            self.input_weights.grad += x.reshape(-1, self.inputs).T @ flat
            self.bias.grad += flat.sum(axis=0)
            return terms @ self.input_weights.value.T
