"""Evaluate reconstructions on explicitly separated held-out measurement sites."""
import numpy as np

from Spatial.Interpolation import idw, ordinary_kriging
from Spatial.SpatialModel import reconstruct
from Spatial.SpatialLoss import held_out_metrics


def sample_grid(grid, field, queries):
    east = (queries[:, 0] - grid.east[0]) / (grid.east[1] - grid.east[0])
    north = (queries[:, 1] - grid.north[0]) / (grid.north[1] - grid.north[0])
    if ((east < 0) | (east > len(grid.east) - 1)
            | (north < 0) | (north > len(grid.north) - 1)).any():
        raise ValueError("Held-out sites must be inside the grid")
    x = np.minimum(east.astype(int), len(grid.east) - 2)
    y = np.minimum(north.astype(int), len(grid.north) - 2)
    wx, wy = east - x, north - y
    lower = (1 - wx) * field[y, x] + wx * field[y, x + 1]
    upper = (1 - wx) * field[y + 1, x] + wx * field[y + 1, x + 1]
    return (1 - wy) * lower + wy * upper


def evaluate_held_out(grid, points, values, queries, targets, backend, settings,
                      kriging_range_m):
    points, queries = np.asarray(points, dtype=float), np.asarray(queries, dtype=float)
    if (queries.ndim != 2 or queries.shape[1] != 2 or not np.isfinite(queries).all()
            or points.ndim != 2 or points.shape[1] != 2):
        raise ValueError("Expected finite held-out sites and two-column training sites")
    if (queries[:, None] == points[None, :]).all(axis=-1).any():
        raise ValueError("Held-out coordinates overlap training coordinates")
    truth = np.asarray(targets, dtype=float)
    if truth.shape != (len(queries),) or not len(truth) or not np.isfinite(truth).all():
        raise ValueError("Expected finite held-out values")
    field = reconstruct(grid, points, values, backend, settings).field
    predictions = {
        "variational": sample_grid(grid, backend.to_host(field), queries),
        "idw": idw(points, values, queries),
        "kriging": ordinary_kriging(points, values, queries, kriging_range_m),
    }
    return {name: held_out_metrics(prediction, truth)
            for name, prediction in predictions.items()}
