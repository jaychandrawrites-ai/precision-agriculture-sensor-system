"""Differentiable penalties and reproducible training-time noise."""
from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class Penalty:
    l1: float = 0.0
    l2: float = 0.0

    def __post_init__(self):
        if any(not math.isfinite(x) or x < 0 for x in (self.l1, self.l2)):
            raise ValueError("Penalty coefficients must be finite and nonnegative")

    def evaluate(self, parameters, backend):
        loss = 0.0
        gradients = []
        with backend.scope():
            xp = backend.xp
            for value in parameters:
                work = value.astype("float64")
                loss += self.l1*float(backend.scalar(xp.abs(work).sum()))
                loss += 0.5*self.l2*float(backend.scalar((work*work).sum()))
                gradient = self.l1*xp.sign(work)+self.l2*work
                gradients.append(gradient.astype(value.dtype))
        return loss, gradients


@dataclass
class DropoutCache:
    mask: object | None
    shape: tuple


class Dropout:
    def __init__(self, backend, probability=0.1):
        if not 0 <= probability < 1:
            raise ValueError("Dropout probability must be in [0, 1)")
        self.backend = backend
        self.probability = probability

    def forward(self, x, training=True):
        if not training or self.probability == 0:
            return x, DropoutCache(None, tuple(x.shape))
        keep = 1-self.probability
        with self.backend.scope():
            mask = self.backend.bernoulli(x.shape, keep).astype(x.dtype)/keep
            return x*mask, DropoutCache(mask, tuple(x.shape))

    def backward(self, gradient, cache):
        if tuple(gradient.shape) != cache.shape:
            raise ValueError("Dropout gradient shape mismatch")
        with self.backend.scope():
            return gradient if cache.mask is None else gradient*cache.mask


class GaussianNoise:
    def __init__(self, backend, standard_deviation=0.01):
        if not math.isfinite(standard_deviation) or standard_deviation < 0:
            raise ValueError("Invalid noise scale")
        self.backend = backend
        self.standard_deviation = standard_deviation

    def forward(self, x, training=True):
        if not training or self.standard_deviation == 0:
            return x
        with self.backend.scope():
            return x+self.backend.normal(x.shape, std=self.standard_deviation)

    def backward(self, gradient):
        return gradient


def clip_by_value(gradients, backend, bound):
    if not math.isfinite(bound) or bound <= 0:
        raise ValueError("Invalid clipping bound")
    with backend.scope():
        for gradient in gradients:
            backend.xp.clip(gradient, -bound, bound, out=gradient)


def total_variation_squared(field, backend, dx=1.0, dy=1.0):
    """Return half squared edge-gradient energy and its exact adjoint."""
    if field.ndim < 2 or min(dx, dy) <= 0:
        raise ValueError("Invalid spatial field or grid spacing")
    with backend.scope():
        horizontal = (field[..., :, 1:]-field[..., :, :-1])/dx
        vertical = (field[..., 1:, :]-field[..., :-1, :])/dy
        energy = 0.5*((horizontal**2).sum()+(vertical**2).sum())
        gradient = backend.xp.zeros_like(field)
        gradient[..., :, 1:] += horizontal/dx
        gradient[..., :, :-1] -= horizontal/dx
        gradient[..., 1:, :] += vertical/dy
        gradient[..., :-1, :] -= vertical/dy
        return energy, gradient


def curvature_penalty(field, backend, dx=1.0, dy=1.0):
    from .Math_Utilities import laplacian_neumann
    with backend.scope():
        lap = laplacian_neumann(field, backend, dy=dy, dx=dx)
        energy = 0.5*(lap*lap).sum()
        gradient = laplacian_neumann(lap, backend, dy=dy, dx=dx)
        return energy, gradient
