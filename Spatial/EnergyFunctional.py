"""Quadratic classical field energy and its analytic derivative."""
from dataclasses import dataclass
import math

from Spatial.BoundaryConditions import laplacian, spectral_bound


@dataclass(frozen=True)
class EnergyWeights:
    gradient: float = 1.0
    curvature: float = 0.1

    def __post_init__(self):
        if not all(math.isfinite(v) and v >= 0 for v in (self.gradient, self.curvature)):
            raise ValueError("Energy weights must be finite and nonnegative")


class EnergyFunctional:
    def __init__(self, grid, constraints, backend, weights=EnergyWeights(), boundary="neumann"):
        self.grid, self.backend, self.weights, self.boundary = grid, backend, weights, boundary
        self.dx, self.dy = grid.spacing
        if boundary not in ("neumann", "periodic"):
            raise ValueError("Unsupported Laplacian boundary")
        with backend.scope():
            self.fidelity = backend.array(constraints.weights, copy=True)
            self.targets = backend.array(constraints.targets, copy=True)
            self.fixed = backend.array(constraints.fixed, dtype="bool", copy=True)
            self.fixed_values = backend.array(constraints.fixed_values, copy=True)
            for values in (self.fidelity, self.targets, self.fixed, self.fixed_values):
                if values.shape != grid.shape:
                    raise ValueError("Constraint arrays must match the grid")
                backend.require_finite(values)
            if backend.scalar(self.fidelity.min()) < 0:
                raise ValueError("Fidelity weights must be nonnegative")
            bound = spectral_bound(self.dx, self.dy)
            self.lipschitz = (weights.gradient * bound + weights.curvature * bound**2 +
                              backend.scalar(self.fidelity.max()))

    def project(self, field):
        with self.backend.scope():
            return self.backend.xp.where(self.fixed, self.fixed_values, field)

    def value_gradient(self, field):
        if field.shape != self.grid.shape:
            raise ValueError("Field must match the grid")
        with self.backend.scope():
            lap = laplacian(field, self.backend, self.dx, self.dy, self.boundary)
            residual = field - self.targets
            first = -0.5 * self.weights.gradient * (field * lap).sum()
            second = 0.5 * self.weights.curvature * (lap * lap).sum()
            fidelity = 0.5 * (self.fidelity * residual**2).sum()
            gradient = (-self.weights.gradient * lap +
                        self.weights.curvature * laplacian(lap, self.backend, self.dx,
                                                            self.dy, self.boundary) +
                        self.fidelity * residual)
            components = tuple(self.backend.scalar(v) for v in (first, second, fidelity))
            if not all(math.isfinite(v) for v in components):
                raise FloatingPointError("Nonfinite spatial energy")
            self.backend.require_finite(gradient, "energy gradient")
            return components, gradient
