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


def ordinary_kriging(points, values, queries, range_m, nugget=1e-6, *, model="exponential"):
    """Ordinary kriging with a caller-supplied covariance model and range."""
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
    system[:-1, :-1] = covariance(pair_distances / range_m, model) + nugget * np.eye(len(values))
    system[-1, -1] = 0
    rhs = np.vstack((covariance(query_distances / range_m, model), np.ones(len(queries))))
    weights = np.linalg.solve(system, rhs)[:-1]
    return values @ weights



def covariance(distance_scaled, model):
    if model == "exponential":
        return np.exp(-distance_scaled)
    if model == "gaussian":
        return np.exp(-distance_scaled**2)
    if model == "spherical":
        distance = np.minimum(distance_scaled, 1.0)
        return 1.0 - 1.5 * distance + 0.5 * distance**3
    if model == "matern32":
        distance = np.sqrt(3) * distance_scaled
        return (1 + distance) * np.exp(-distance)
    raise ValueError("Unknown covariance model")


def radial_basis(points, values, queries, range_m, ridge=1e-6):
    """Regularized Gaussian radial basis regression with an unpenalized constant."""
    p, y, q = (np.asarray(item, dtype=float) for item in (points, values, queries))
    if (y.ndim != 1 or len(y) < 1 or p.shape != (len(y), 2) or q.ndim != 2
            or q.shape[1] != 2 or not all(np.isfinite(item).all() for item in (p, q, y))
            or not np.isfinite(range_m) or range_m <= 0 or not np.isfinite(ridge) or ridge <= 0):
        raise ValueError("Invalid radial basis inputs")
    distance = np.linalg.norm(p[:, None] - p[None], axis=-1)
    gram = covariance(distance / range_m, "gaussian")
    system = np.ones((len(y) + 1, len(y) + 1))
    system[:-1, :-1] = gram + ridge * np.eye(len(y))
    system[-1, -1] = 0
    weights = np.linalg.solve(system, np.r_[y, 0])
    query = np.linalg.norm(q[:, None] - p[None], axis=-1)
    return covariance(query / range_m, "gaussian") @ weights[:-1] + weights[-1]
