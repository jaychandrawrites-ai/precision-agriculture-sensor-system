"""Map measured locations to grid cells and assemble observation constraints."""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Constraints:
    weights: np.ndarray
    targets: np.ndarray
    fixed: np.ndarray
    fixed_values: np.ndarray


def seed_constraints(grid, points, values, weight, hard=False, boundary=None,
                     boundary_weight=None):
    points, values = np.asarray(points, dtype=float), np.asarray(values, dtype=float)
    if (values.ndim != 1 or points.shape != (len(values), 2) or not len(values)
            or not np.isfinite(values).all() or not np.isfinite(points).all()
            or not np.isfinite(weight) or weight <= 0):
        raise ValueError("Expected finite measurements and positive seed weight")
    spacing = grid.spacing
    offsets = (points - [grid.east[0], grid.north[0]]) / spacing
    if ((offsets < -1e-10) | (offsets > np.array(grid.shape[::-1]) - 1 + 1e-10)).any():
        raise ValueError("All seed coordinates must be inside the grid")
    east, north = np.rint(offsets).astype(int).T
    counts, totals = np.zeros(grid.shape), np.zeros(grid.shape)
    np.add.at(counts, (north, east), 1)
    np.add.at(totals, (north, east), values)
    observed = counts > 0
    targets = np.divide(totals, counts, out=np.zeros_like(totals), where=observed)
    # Keep the original per-cell penalty even when multiple points share a cell.
    weights = np.zeros(grid.shape) if hard else np.where(observed, weight, 0.0)
    fixed = observed.copy() if hard else np.zeros_like(observed)
    fixed_values = targets.copy() if hard else np.zeros_like(targets)
    if boundary is not None:
        values_on_edge = np.asarray(boundary, dtype=float)
        if values_on_edge.shape != grid.shape or np.isinf(values_on_edge).any():
            raise ValueError("Boundary array must match the grid; use NaN for unspecified nodes")
        mask = np.isfinite(values_on_edge)
        if mask[1:-1, 1:-1].any():
            raise ValueError("Boundary conditions must be on the grid perimeter")
        if boundary_weight is None:
            if not np.allclose(fixed_values[fixed & mask], values_on_edge[fixed & mask]):
                raise ValueError("Hard boundary conflicts with a hard seed")
            fixed_values[mask] = values_on_edge[mask]
            fixed |= mask
        else:
            if not np.isfinite(boundary_weight) or boundary_weight <= 0:
                raise ValueError("Soft boundary weight must be positive")
            targets[mask] = ((weights[mask] * targets[mask] +
                             boundary_weight * values_on_edge[mask]) /
                             (weights[mask] + boundary_weight))
            weights[mask] += boundary_weight
    elif boundary_weight is not None:
        raise ValueError("Boundary weight requires boundary values")
    return Constraints(weights, targets, fixed, fixed_values)
