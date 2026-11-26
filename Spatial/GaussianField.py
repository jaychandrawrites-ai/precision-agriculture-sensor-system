"""Dense analytical reference for small quadratic field posteriors."""
from dataclasses import dataclass

import numpy as np

from Spatial.Hamiltonian import hessian_columns


def field_system(energy):
    """Return the sparse precision columns and Gaussian linear information."""
    host = energy.backend.to_host
    fixed = host(energy.fixed).ravel()
    anchor = host(energy.fixed_values).ravel()
    info = host(energy.fidelity).ravel() * host(energy.targets).ravel()
    return hessian_columns(energy), fixed, anchor, info


@dataclass(frozen=True)
class GaussianField:
    mean: np.ndarray
    covariance: np.ndarray
    free_cells: np.ndarray

    @property
    def variance(self):
        result = np.zeros(self.mean.size)
        result[self.free_cells] = np.diag(self.covariance)
        return result.reshape(self.mean.shape)


def exact_gaussian_field(energy, temperature=1.0, max_free=256):
    """Solve the fixed-cell conditional distribution in row-major order."""
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("Temperature must be positive")
    energy.require_anchored()
    columns, fixed, anchor, info = field_system(energy)
    free = np.flatnonzero(~fixed)
    if len(free) > max_free:
        raise ValueError("Dense analytical reference exceeds its free-cell budget")
    if not len(free):
        return GaussianField(anchor.reshape(energy.grid.shape), np.empty((0, 0)), free)
    lookup = np.full(len(fixed), -1, dtype=int)
    lookup[free] = np.arange(len(free))
    matrix = np.zeros((len(free), len(free)))
    right = info[free].copy()
    for node, center in enumerate(free):
        indices, weights, _ = columns[center]
        inside = ~fixed[indices]
        matrix[lookup[indices[inside]], node] = weights[inside]
        right[node] -= weights[~inside] @ anchor[indices[~inside]]
    try:
        factor = np.linalg.cholesky(matrix)
    except np.linalg.LinAlgError as error:
        raise ValueError("Field has an improper conditional distribution") from error
    solved = np.linalg.solve(factor, right)
    inverse = np.linalg.solve(factor, np.eye(len(free)))
    mean = anchor.copy()
    mean[free] = np.linalg.solve(factor.T, solved)
    return GaussianField(mean.reshape(energy.grid.shape), temperature * inverse.T @ inverse, free)
