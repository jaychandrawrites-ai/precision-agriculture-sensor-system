"""Quadratic classical field energy and its analytic derivative."""
from dataclasses import dataclass
import math

import numpy as np

from Spatial.BoundaryConditions import laplacian, spectral_bound


@dataclass(frozen=True)
class EnergyWeights:
    gradient: float = 1.0
    curvature: float = 0.1

    def __post_init__(self):
        if not all(math.isfinite(v) and v >= 0 for v in (self.gradient, self.curvature)):
            raise ValueError("Energy weights must be finite and nonnegative")


class EnergyFunctional:
    def __init__(self, grid, constraints, backend, weights=EnergyWeights(), boundary="neumann", *, domain=None):
        self.grid, self.backend, self.weights, self.boundary = grid, backend, weights, boundary
        self.dx, self.dy = grid.spacing
        if boundary not in ("neumann", "periodic"):
            raise ValueError("Unsupported Laplacian boundary")
        self.domain = domain
        self.east_faces = self.north_faces = None
        if domain is None:
            active = np.ones(grid.shape, dtype=bool)
        else:
            from Spatial.FieldDomain import regular_masks
            if boundary != "neumann":
                raise ValueError("Irregular fields use zero normal flux at excluded faces")
            active, east, north = regular_masks(grid, domain)
            if not active.any():
                raise ValueError("No grid nodes in the admissible field domain")
            self.east_faces = backend.array(east, dtype=bool)
            self.north_faces = backend.array(north, dtype=bool)
        self.active = backend.array(active, dtype=bool)
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
            if backend.scalar(((self.fixed | (self.fidelity > 0)) & ~self.active).any()):
                raise ValueError("Observation or boundary constraint snaps to an excluded grid node")
            self.fixed |= ~self.active
            self.fixed_values = backend.xp.where(self.active, self.fixed_values, 0)
            bound = spectral_bound(self.dx, self.dy)
            self.lipschitz = (weights.gradient * bound + weights.curvature * bound**2 +
                              backend.scalar(self.fidelity.max()))

    def project(self, field):
        if field.shape != self.grid.shape:
            raise ValueError("Field must match the grid")
        with self.backend.scope():
            return self.backend.xp.where(self.fixed, self.fixed_values, field)

    def laplacian(self, field):
        if self.domain is None:
            return laplacian(field, self.backend, self.dx, self.dy, self.boundary)
        with self.backend.scope():
            xp = self.backend.xp
            east = self.east_faces * (field[:, 1:] - field[:, :-1]) / self.dx**2
            north = self.north_faces * (field[1:] - field[:-1]) / self.dy**2
            divergence = xp.zeros_like(field)
            divergence[:, :-1] += east
            divergence[:, 1:] -= east
            divergence[:-1] += north
            divergence[1:] -= north
            return divergence

    def require_anchored(self):
        """Reject improper free components, treating excluded nodes as absent."""
        fixed = self.backend.to_host(self.fixed).ravel()
        fidelity = self.backend.to_host(self.fidelity).ravel()
        active = self.backend.to_host(self.active).ravel()
        anchors = fixed | (fidelity > 0)
        if self.weights.gradient == self.weights.curvature == 0:
            if np.any(active & ~anchors):
                raise ValueError("The free field is unanchored and has an improper sampling density")
            return
        if self.domain is None:
            if not anchors.any():
                raise ValueError("The free field is unanchored and has an improper sampling density")
            return
        from Spatial.AdaptiveVariational import _components
        indices = np.arange(len(active)).reshape(self.grid.shape)
        east, north = (self.backend.to_host(mask) for mask in (self.east_faces, self.north_faces))
        faces = list(zip(indices[:, :-1][east], indices[:, 1:][east]))
        faces += list(zip(indices[:-1][north], indices[1:][north]))
        labels = _components(len(active), faces)
        observed = labels[anchors & active]
        if not np.isin(labels[active], observed).all():
            raise ValueError("An active field component is unanchored and has an improper sampling density")

    def hessian_vector(self, direction):
        """Apply the linear Hessian without materializing a dense field matrix."""
        if direction.shape != self.grid.shape:
            raise ValueError("Direction must match the grid")
        with self.backend.scope():
            lap = self.laplacian(direction)
            return (-self.weights.gradient * lap +
                    self.weights.curvature * self.laplacian(lap) +
                    self.fidelity * direction)

    def value_gradient(self, field):
        if field.shape != self.grid.shape:
            raise ValueError("Field must match the grid")
        with self.backend.scope():
            lap = self.laplacian(field)
            residual = field - self.targets
            first = -0.5 * self.weights.gradient * (field * lap).sum()
            second = 0.5 * self.weights.curvature * (lap * lap).sum()
            fidelity = 0.5 * (self.fidelity * residual**2).sum()
            gradient = (-self.weights.gradient * lap +
                        self.weights.curvature * self.laplacian(lap) +
                        self.fidelity * residual)
            components = tuple(self.backend.scalar(v) for v in (first, second, fidelity))
            if not all(math.isfinite(v) for v in components):
                raise FloatingPointError("Nonfinite spatial energy")
            self.backend.require_finite(gradient, "energy gradient")
            return components, gradient
