"""Masked heat-equation interpolation in local metres."""
import math

import numpy as np

from Framework.Backend import create_backend
from Spatial.Interpolation import idw
from Spatial.PDESolver import solve


def diffusion_map(grid, points, values, inside, *, diffusivity=1., iterations=500, tolerance=1e-6):
    """Advance explicit diffusion with pinned seed cells and no-flux edges."""
    points, values = np.asarray(points), np.asarray(values, dtype=float)
    inside = np.asarray(inside, dtype=bool)
    if (not math.isfinite(diffusivity) or diffusivity <= 0 or type(iterations) is not int
            or not 1 <= iterations <= 5000 or not math.isfinite(tolerance)
            or tolerance <= 0 or inside.shape != grid.shape or not inside.any()):
        raise ValueError("Invalid diffusion settings or domain")
    queries = grid.points()
    available = np.flatnonzero(inside.ravel())
    cells = [available[np.argmin(np.sum((queries[available] - point)**2, axis=1))]
             for point in points]
    seed = np.zeros(len(queries))
    count = np.bincount(cells, minlength=len(seed))
    totals = np.bincount(cells, weights=values, minlength=len(seed))
    np.divide(totals, count, out=seed, where=count > 0)
    initial = idw(points, values, queries).reshape(grid.shape)
    dx, dy = grid.spacing
    result = solve(initial, seed.reshape(grid.shape), (count > 0).reshape(grid.shape),
                   create_backend(dtype="float64"), kappa=diffusivity, gamma=0., dx=dx, dy=dy,
                   iterations=iterations, tolerance=tolerance, hard_seeds=True, domain=inside)
    support = np.sqrt(np.min(np.sum((queries[:, None] - points[None, :])**2, axis=-1), axis=1))
    return result.field, {"iterations": result.iterations, "converged": result.converged,
                          "initial_energy": result.energies[0], "final_energy": result.energies[-1],
                          "seed_cells": int((count > 0).sum()), "coincident_seed_rule": "mean",
                          "nearest_seed_m": np.where(inside, support.reshape(grid.shape), None).tolist(),
                          "boundary": "no flux", "time": "computational diffusion"}
