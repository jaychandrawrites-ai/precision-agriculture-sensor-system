"""Evaluate reconstructions on explicitly separated held-out measurement sites."""
import numpy as np

from Spatial.Interpolation import idw, ordinary_kriging
from Spatial.SpatialModel import reconstruct
from Spatial.SpatialLoss import held_out_metrics


def sample_grid(grid, field, queries):
    field, queries = np.asarray(field), np.asarray(queries, dtype=float)
    if (field.shape[-2:] != grid.shape or queries.ndim != 2 or queries.shape[1] != 2
            or not np.isfinite(queries).all()):
        raise ValueError("Expected a matching grid field and finite east/north queries")
    east = (queries[:, 0] - grid.east[0]) / (grid.east[1] - grid.east[0])
    north = (queries[:, 1] - grid.north[0]) / (grid.north[1] - grid.north[0])
    if ((east < 0) | (east > len(grid.east) - 1)
            | (north < 0) | (north > len(grid.north) - 1)).any():
        raise ValueError("Held-out sites must be inside the grid")
    x = np.minimum(east.astype(int), len(grid.east) - 2)
    y = np.minimum(north.astype(int), len(grid.north) - 2)
    wx, wy = east - x, north - y
    lower = (1 - wx) * field[..., y, x] + wx * field[..., y, x + 1]
    upper = (1 - wx) * field[..., y + 1, x] + wx * field[..., y + 1, x + 1]
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



def score_snapshot(grid, field, training_points, training_values, queries, truth,
                   kriging_range_m=None, sampling=None):
    """Score only independently located labels, optionally checking interval coverage."""
    training_points, queries = np.asarray(training_points), np.asarray(queries)
    truth = np.asarray(truth, dtype=float)
    if (queries[:, None] == training_points[None]).all(axis=-1).any():
        raise ValueError("Held-out coordinates overlap training coordinates")
    predictions = {"variational": sample_grid(grid, field, queries),
                   "idw": idw(training_points, training_values, queries)}
    if kriging_range_m is not None:
        predictions["kriging"] = ordinary_kriging(training_points, training_values,
                                                  queries, kriging_range_m)
    metrics = {}
    for name, predicted in predictions.items():
        metrics[name] = {key: float(value[0]) if np.isfinite(value[0]) else None
                         for key, value in held_out_metrics(predicted, truth).items()}
    if sampling is not None:
        site_samples = sample_grid(grid, sampling.fields, queries)
        lower, upper = np.quantile(site_samples, [0.025, 0.975], axis=(0, 1))
        metrics["conditional_interval"] = {
            "covered": int(((lower <= truth) & (truth <= upper)).sum()),
            "count": len(truth), "nominal_mass": 0.95, "mean_width": float((upper - lower).mean()),
        }
    return metrics
