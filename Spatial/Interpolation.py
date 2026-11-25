"""Inverse distance weighting as a spatial baseline."""
import numpy as np


def idw(points, values, queries, power=2.0):
    points, values, queries = (np.asarray(array, dtype=float)
                               for array in (points, values, queries))
    if (values.ndim != 1 or points.shape != (len(values), 2) or not len(values)
            or queries.ndim != 2 or queries.shape[1] != 2 or not np.isfinite(power)
            or power <= 0 or not all(np.isfinite(a).all() for a in (points, values, queries))):
        raise ValueError("Invalid IDW samples or queries")
    distance = np.linalg.norm(queries[:, None, :] - points[None, :, :], axis=2)
    exact = distance == 0
    minimum = distance.min(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        weights = np.where(minimum == 0, exact, (minimum / distance) ** power)
    return weights @ values / weights.sum(axis=1)


def ordinary_kriging(points, values, queries, range_m, nugget=1e-6):
    """Ordinary kriging with a caller-supplied exponential covariance range."""
    points, values, queries = (np.asarray(array, dtype=float)
                               for array in (points, values, queries))
    if (values.ndim != 1 or not len(values) or points.shape != (len(values), 2)
            or queries.ndim != 2 or queries.shape[1] != 2
            or not np.isfinite(range_m) or range_m <= 0
            or not np.isfinite(nugget) or nugget <= 0
            or not all(np.isfinite(a).all() for a in (points, values, queries))):
        raise ValueError("Invalid kriging samples, queries, or covariance settings")
    pair_distances = np.linalg.norm(points[:, None] - points[None, :], axis=2)
    query_distances = np.linalg.norm(points[:, None] - queries[None, :], axis=2)
    system = np.ones((len(values) + 1, len(values) + 1))
    system[:-1, :-1] = np.exp(-pair_distances / range_m) + nugget * np.eye(len(values))
    system[-1, -1] = 0
    rhs = np.vstack((np.exp(-query_distances / range_m), np.ones(len(queries))))
    weights = np.linalg.solve(system, rhs)[:-1]
    return values @ weights
