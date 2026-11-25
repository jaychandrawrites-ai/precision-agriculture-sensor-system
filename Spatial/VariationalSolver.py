"""Projected descent with a spectral stability bound and residual diagnostics."""
from dataclasses import dataclass
import math


@dataclass
class VariationalResult:
    field: object
    energies: list[float]
    iterations: int
    converged: bool
    gradient_rms: float
    step: float


def minimize(energy, initial, *, iterations=500, tolerance=1e-6, step=None):
    if iterations < 1 or not math.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("Invalid solver budget or tolerance")
    backend = energy.backend
    with backend.scope():
        field = energy.project(backend.array(initial, copy=True))
        backend.require_finite(field, "initial spatial field")
        free = ~energy.fixed
        count = int(backend.scalar(free.sum()))
        if not count:
            components, _ = energy.value_gradient(field)
            return VariationalResult(field, [sum(components)], 0, True, 0.0, 0.0)
        if energy.lipschitz <= 0:
            raise ValueError("Objective has no active terms")
        step = 0.9 / energy.lipschitz if step is None else step
        if not 0 < step < 2 / energy.lipschitz:
            raise ValueError("Explicit step exceeds stability bound")
        energies = []
        for index in range(iterations + 1):
            components, gradient = energy.value_gradient(field)
            energies.append(sum(components))
            gradient *= free
            rms = backend.global_norm([gradient]) / math.sqrt(count)
            if rms <= tolerance or index == iterations:
                return VariationalResult(field, energies, index, rms <= tolerance, rms, step)
            field = energy.project(field - step * gradient)
