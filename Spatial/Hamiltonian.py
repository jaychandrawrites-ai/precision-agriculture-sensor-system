"""Local Hessian of the classical quadratic field model used by Metropolis."""
from collections import defaultdict

import numpy as np


def _laplacian_row(index, shape, dx, dy, boundary, east=None, north=None):
    height, width = shape
    row, col = divmod(index, width)
    entries = defaultdict(float)
    for r, c, weight in ((row, col - 1, 1 / dx**2), (row, col + 1, 1 / dx**2),
                         (row - 1, col, 1 / dy**2), (row + 1, col, 1 / dy**2)):
        if boundary == "periodic":
            r, c = r % height, c % width
        if 0 <= r < height and 0 <= c < width:
            if east is not None and not (east[row, min(c, col)] if r == row else north[min(r, row), col]):
                continue
            entries[r * width + c] += weight
            entries[index] -= weight
    return entries


def hessian_columns(energy):
    """Build symmetric sparse columns on CPU for O(stencil size) local proposals."""
    shape = energy.grid.shape
    count = int(np.prod(shape))
    east = north = None
    if energy.domain is not None:
        east = energy.backend.to_host(energy.east_faces)
        north = energy.backend.to_host(energy.north_faces)
    rows = [_laplacian_row(i, shape, energy.dx, energy.dy, energy.boundary, east, north)
            for i in range(count)]
    fidelity = energy.backend.to_host(energy.fidelity).ravel()
    result = []
    for i, row in enumerate(rows):
        column = defaultdict(float, {i: float(fidelity[i])})
        for neighbor, weight in row.items():
            column[neighbor] -= energy.weights.gradient * weight
            if energy.weights.curvature:
                for node, second in rows[neighbor].items():
                    column[node] += energy.weights.curvature * weight * second
        result.append((np.fromiter(column, dtype=int),
                       np.fromiter(column.values(), dtype=float), column[i]))
    return result
