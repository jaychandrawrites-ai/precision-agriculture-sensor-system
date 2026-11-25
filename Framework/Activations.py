"""Manual activation functions and vector-Jacobian products."""
from __future__ import annotations

from dataclasses import dataclass
import math
from .Math_Utilities import logsumexp


@dataclass
class ActivationCache:
    input: object
    output: object


class Activation:
    def __init__(self, backend):
        self.backend = backend

    def evaluate(self, x):
        raise NotImplementedError

    def derivative(self, cache):
        raise NotImplementedError

    def forward(self, x):
        with self.backend.scope():
            y = self.evaluate(x)
            self.backend.validate(y, type(self).__name__)
            return y, ActivationCache(x, y)

    def backward(self, gradient, cache):
        if gradient.shape != cache.output.shape:
            raise ValueError("Activation gradient shape mismatch")
        with self.backend.scope():
            return gradient * self.derivative(cache)


class Linear(Activation):
    def evaluate(self, x):
        return x

    def derivative(self, cache):
        return self.backend.xp.ones_like(cache.input)


class ReLU(Activation):
    def evaluate(self, x):
        return self.backend.xp.maximum(x, 0)

    def derivative(self, cache):
        return (cache.input > 0).astype(cache.input.dtype)


class LeakyReLU(Activation):
    def __init__(self, backend, slope=0.01):
        super().__init__(backend)
        if not 0 <= slope <= 1:
            raise ValueError("Invalid negative slope")
        self.slope = slope

    def evaluate(self, x):
        return self.backend.xp.where(x > 0, x, self.slope*x)

    def derivative(self, cache):
        return self.backend.xp.where(cache.input > 0, 1.0, self.slope)


class Sigmoid(Activation):
    def evaluate(self, x):
        xp = self.backend.xp
        magnitude = xp.exp(-xp.abs(x))
        return xp.where(x >= 0, 1/(1+magnitude), magnitude/(1+magnitude))

    def derivative(self, cache):
        y = cache.output
        return y*(1-y)


class Tanh(Activation):
    def evaluate(self, x):
        return self.backend.xp.tanh(x)

    def derivative(self, cache):
        return 1-cache.output**2


class Softplus(Activation):
    def evaluate(self, x):
        xp = self.backend.xp
        return xp.maximum(x, 0)+xp.log1p(xp.exp(-xp.abs(x)))

    def derivative(self, cache):
        return Sigmoid(self.backend).evaluate(cache.input)


class ELU(Activation):
    def __init__(self, backend, alpha=1.0):
        super().__init__(backend)
        if alpha <= 0:
            raise ValueError("Alpha must be positive")
        self.alpha = alpha

    def evaluate(self, x):
        xp = self.backend.xp
        return xp.where(x >= 0, x, self.alpha*xp.expm1(xp.minimum(x, 0)))

    def derivative(self, cache):
        return self.backend.xp.where(cache.input >= 0, 1, cache.output+self.alpha)


class GELU(Activation):
    """Tanh approximation to Gaussian error linear unit."""
    def evaluate(self, x):
        xp = self.backend.xp
        z = math.sqrt(2/math.pi)*(x+0.044715*x**3)
        return 0.5*x*(1+xp.tanh(z))

    def derivative(self, cache):
        xp = self.backend.xp
        x = cache.input
        k = math.sqrt(2/math.pi)
        t = xp.tanh(k*(x+0.044715*x**3))
        return 0.5*(1+t)+0.5*x*(1-t*t)*k*(1+3*0.044715*x*x)


class SiLU(Activation):
    def evaluate(self, x):
        return x*Sigmoid(self.backend).evaluate(x)

    def derivative(self, cache):
        x = cache.input
        s = Sigmoid(self.backend).evaluate(x)
        return s+x*s*(1-s)


class Softmax(Activation):
    """Coupled activation: backward computes the full Jacobian product."""
    def __init__(self, backend, axis=-1):
        super().__init__(backend)
        self.axis = axis

    def evaluate(self, x):
        return self.backend.stable_softmax(x, axis=self.axis)

    def backward(self, gradient, cache):
        if gradient.shape != cache.output.shape:
            raise ValueError("Softmax gradient shape mismatch")
        with self.backend.scope():
            y = cache.output
            projection = (gradient*y).sum(axis=self.axis, keepdims=True)
            return y*(gradient-projection)


class LogSoftmax(Softmax):
    def evaluate(self, x):
        return x-logsumexp(x, self.backend, axis=self.axis, keepdims=True)

    def backward(self, gradient, cache):
        if gradient.shape != cache.output.shape:
            raise ValueError("LogSoftmax gradient shape mismatch")
        with self.backend.scope():
            probability = self.backend.xp.exp(cache.output)
            return gradient-probability*gradient.sum(axis=self.axis, keepdims=True)


ACTIVATIONS = {
    "linear": Linear,
    "relu": ReLU,
    "leaky_relu": LeakyReLU,
    "sigmoid": Sigmoid,
    "tanh": Tanh,
    "softplus": Softplus,
    "elu": ELU,
    "gelu": GELU,
    "silu": SiLU,
    "softmax": Softmax,
    "log_softmax": LogSoftmax,
}


def make_activation(name, backend, **kwargs):
    try:
        cls = ACTIVATIONS[name.lower()]
    except KeyError as exc:
        raise ValueError(f"Unknown activation: {name}") from exc
    return cls(backend, **kwargs)
