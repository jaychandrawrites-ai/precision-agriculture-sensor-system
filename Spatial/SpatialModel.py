"""Reconstruct a field from real point values supplied by the caller."""
import numpy as np

from Spatial.Interpolation import idw
from Spatial.PDESolver import solve


def reconstruct(grid, points, values, backend, settings, hard_seeds=False):
    points, values = np.asarray(points, dtype=float), np.asarray(values, dtype=float)
    initial = idw(points, values, grid.points()).reshape(grid.shape)
    dx, dy = grid.spacing
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


def reconstruct_variational(grid, points, values, backend, settings, *, hard_seeds=False,
                            boundary_values=None, boundary_weight=None, boundary="neumann"):
    """Build a model energy and minimize it from an IDW initialization."""
    from Spatial.EnergyFunctional import EnergyFunctional, EnergyWeights
    from Spatial.SeedManager import seed_constraints
    from Spatial.VariationalSolver import minimize

    constraints = seed_constraints(grid, points, values, settings.seed_weight, hard_seeds,
                                   boundary_values, boundary_weight)
    weights = EnergyWeights(settings.gradient_weight, settings.curvature_weight)
    energy = EnergyFunctional(grid, constraints, backend, weights, boundary)
    initial = idw(points, values, grid.points()).reshape(grid.shape)
    return energy, minimize(energy, initial, iterations=settings.iterations)
