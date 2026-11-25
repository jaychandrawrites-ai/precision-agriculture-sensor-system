"""Anisotropic local regularization with heteroscedastic robust observations."""
import numpy as np

from Numerical.Optimization import lbfgs


class RobustField:
    """Edge arrays control spatially varying east and north conductance."""

    def __init__(self, grid, east, north, targets, precision, *, curvature=.1, delta=1.):
        n, m = grid.shape
        self.east = np.broadcast_to(np.asarray(east, dtype=float), (n, m - 1)) / grid.spacing[0]**2
        self.north = np.broadcast_to(np.asarray(north, dtype=float), (n - 1, m)) / grid.spacing[1]**2
        self.targets = np.asarray(targets, dtype=float)
        self.precision = np.asarray(precision, dtype=float)
        if (self.targets.shape != grid.shape or self.precision.shape != grid.shape
                or not all(np.isfinite(array).all()
                           for array in (self.east, self.north, self.targets, self.precision))
                or any((array < 0).any() for array in (self.east, self.north, self.precision))
                or not np.isfinite(curvature) or curvature < 0 or not np.isfinite(delta) or delta <= 0):
            raise ValueError("Invalid field geometry, weights or robust fidelity")
        self.curvature, self.delta = curvature, delta

    def laplacian(self, field):
        if np.shape(field) != self.targets.shape:
            raise ValueError("Field must match the observation grid")
        east = self.east * np.diff(field, axis=1)
        north = self.north * np.diff(field, axis=0)
        result = np.zeros_like(field, dtype=float)
        result[:, :-1] -= east
        result[:, 1:] += east
        result[:-1] -= north
        result[1:] += north
        return result

    def value_gradient(self, field):
        lap = self.laplacian(field)
        residual = (field - self.targets) * np.sqrt(self.precision)
        magnitude = np.hypot(self.delta, residual)
        value = .5 * np.sum(field * lap + self.curvature * lap**2)
        value += np.sum(self.delta * (magnitude - self.delta))
        gradient = (lap + self.curvature * self.laplacian(lap) +
                    self.delta * residual / magnitude * np.sqrt(self.precision))
        return float(value), gradient

    def fit(self, initial, fixed=None, fixed_values=None, **options):
        field = np.asarray(initial, dtype=float).copy()
        if field.shape != self.targets.shape:
            raise ValueError("Initial field must match observation grid")
        mask = np.zeros(field.shape, bool) if fixed is None else np.asarray(fixed, dtype=bool)
        if mask.shape != field.shape:
            raise ValueError("Fixed cells must match observation grid")
        if fixed_values is not None:
            constraints = np.asarray(fixed_values, dtype=float)
            if constraints.shape != field.shape:
                raise ValueError("Fixed values must match observation grid")
            field[mask] = constraints[mask]
        if not np.isfinite(field).all() or mask.all():
            raise ValueError("Need a finite field with some free cells")

        def objective(point):
            field[~mask] = point
            value, gradient = self.value_gradient(field)
            return value, gradient[~mask]

        result = lbfgs(objective, field[~mask], **options)
        field[~mask] = result.point
        return field, result
