"""Inverse distance weighting as a spatial baseline."""
import numpy as np


def idw(points, values, queries, power=2.0):
    points, values, queries = (np.asarray(array, dtype=float)
                               for array in (points, values, queries))
    if (points.shape != (len(values), 2) or values.ndim != 1 or not len(values)
            or queries.ndim != 2 or queries.shape[1] != 2 or not np.isfinite(power)
            or power <= 0 or not all(np.isfinite(a).all() for a in (points, values, queries))):
        raise ValueError("Invalid IDW samples or queries")
    distance = np.linalg.norm(queries[:, None, :] - points[None, :, :], axis=2)
    exact = distance == 0
    minimum = distance.min(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        weights = np.where(minimum == 0, exact, (minimum / distance) ** power)
    return weights @ values / weights.sum(axis=1)
