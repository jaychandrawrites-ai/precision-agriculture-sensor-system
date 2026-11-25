"""Reconstruct a field from real point values supplied by the caller."""
import numpy as np

from Spatial.Interpolation import idw
from Spatial.PDESolver import solve


def reconstruct(grid, points, values, backend, settings, hard_seeds=False):
    points, values = np.asarray(points, dtype=float), np.asarray(values, dtype=float)
    initial = idw(points, values, grid.points()).reshape(grid.shape)
    dx, dy = grid.east[1] - grid.east[0], grid.north[1] - grid.north[0]
    offsets = (points - [grid.east[0], grid.north[0]]) / [dx, dy]
    if ((offsets < 0) | (offsets > np.array(grid.shape[::-1]) - 1)).any():
        raise ValueError("All seed coordinates must be inside the grid")
    east, north = np.rint(offsets).astype(int).T
    counts = np.zeros(grid.shape, dtype=int)
    sums = np.zeros(grid.shape)
    np.add.at(counts, (north, east), 1)
    np.add.at(sums, (north, east), values)
    mask = counts > 0
    np.divide(sums, counts, out=sums, where=mask)
    return solve(initial, sums, mask, backend, kappa=settings.gradient_weight,
                 gamma=settings.curvature_weight, penalty=settings.seed_weight,
                 dx=dx, dy=dy, iterations=settings.iterations, hard_seeds=hard_seeds)
