"""Empirical semivariograms and explicit spatial holdout folds."""
import numpy as np

from Spatial.Interpolation import idw, ordinary_kriging, radial_basis
from Spatial.SpatialLoss import held_out_metrics


def _observations(points, values):
    points, values = np.asarray(points, dtype=float), np.asarray(values, dtype=float)
    if (values.ndim != 1 or len(values) < 3 or points.shape != (len(values), 2)
            or not np.isfinite(points).all() or not np.isfinite(values).all()):
        raise ValueError("Expected three or more finite spatial observations")
    return points, values


def variogram(points, values, edges):
    points, values = _observations(points, values)
    edges = np.asarray(edges, dtype=float)
    if (edges.ndim != 1 or len(edges) < 2 or edges[0] < 0 or not np.isfinite(edges).all()
            or (np.diff(edges) <= 0).any()):
        raise ValueError("Lag edges must be finite and increasing")
    a, b = np.triu_indices(len(points), k=1)
    distance = np.linalg.norm(points[a] - points[b], axis=1)
    semivariance = .5 * (values[a] - values[b])**2
    count, _ = np.histogram(distance, edges)
    total, _ = np.histogram(distance, edges, weights=semivariance)
    centers, _ = np.histogram(distance, edges, weights=distance)
    return {"edges": edges, "pairs": count,
            "distances": np.divide(centers, count, out=np.full(len(count), np.nan), where=count > 0),
            "semivariance": np.divide(total, count, out=np.full(len(count), np.nan), where=count > 0)}


def spatial_folds(points, block_size=None):
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
        raise ValueError("Expected finite east/north coordinates")
    if block_size is not None and (not np.isfinite(block_size) or block_size <= 0):
        raise ValueError("Block size must be positive meters")
    if not len(points):
        raise ValueError("No locations")
    groups = points if block_size is None else np.floor((points - points.min(axis=0)) / block_size)
    _, assignment = np.unique(groups, axis=0, return_inverse=True)
    if len(np.unique(assignment)) < 2:
        raise ValueError("Need two distinct location groups")
    for group in np.unique(assignment):
        yield np.flatnonzero(assignment != group), np.flatnonzero(assignment == group)


def compare_interpolators(points, values, range_m, block_size=None, extra=None):
    points, values = _observations(points, values)
    methods = {"idw": idw,
               "kriging": lambda p, v, q: ordinary_kriging(p, v, q, range_m),
               "radial_basis": lambda p, v, q: radial_basis(p, v, q, range_m)}
    methods.update(extra or {})
    predictions = {name: np.empty(len(values)) for name in methods}
    sizes = []
    for fit, test in spatial_folds(points, block_size):
        sizes.append(len(test))
        for name, predict in methods.items():
            predictions[name][test] = predict(points[fit], values[fit], points[test])
    return {"fold_sizes": sizes, "metrics": {name: held_out_metrics(result, values)
                                             for name, result in predictions.items()}}
