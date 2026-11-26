"""Gaussian linear geostatistics with explicit observation and latent variances."""
from dataclasses import dataclass
import math

import numpy as np

from Spatial.Interpolation import covariance


@dataclass(frozen=True)
class CovarianceModel:
    range_m: float
    sill: float = 1.0
    nugget: float = 0.0
    shape: str = "exponential"
    angle_deg: float = 0.0
    minor_ratio: float = 1.0

    def __post_init__(self):
        if (not all(map(math.isfinite, (self.range_m, self.sill, self.nugget,
                                       self.angle_deg, self.minor_ratio)))
                or min(self.range_m, self.sill) <= 0 or self.nugget < 0
                or not 0 < self.minor_ratio <= 1
                or self.shape not in {"exponential", "gaussian", "matern32", "matern52", "spherical"}):
            raise ValueError("Invalid spatial covariance parameters")

    def scaled_distance(self, left, right):
        displacement = left[:, None] - right[None, :]
        theta = math.radians(self.angle_deg)
        major = displacement[..., 0] * math.cos(theta) + displacement[..., 1] * math.sin(theta)
        minor = -displacement[..., 0] * math.sin(theta) + displacement[..., 1] * math.cos(theta)
        return np.hypot(major, minor / self.minor_ratio) / self.range_m

    def __call__(self, left, right):
        return self.sill * covariance(self.scaled_distance(left, right), self.shape)


@dataclass(frozen=True)
class KrigingPrediction:
    mean: np.ndarray
    latent_variance: np.ndarray
    conditioning_jitter: float


def coordinates(points):
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or not len(points) or not np.isfinite(points).all():
        raise ValueError("Expected nonempty finite east/north coordinates")
    return points


def nearest_neighbor(points, values, queries):
    points, queries = coordinates(points), coordinates(queries)
    values = np.asarray(values, dtype=float)
    if values.shape != (len(points),) or not np.isfinite(values).all():
        raise ValueError("Expected one finite value per site")
    delta = queries[:, None] - points
    distance = np.hypot(delta[..., 0], delta[..., 1])
    exact = distance == distance.min(axis=1, keepdims=True)
    return exact @ values / exact.sum(axis=1)


class KrigingModel:
    """Fit a constant or affine drift by generalized least squares.

    The nugget and optional per-site measurement variances enter only the
    observed covariance. Returned variances are for the underlying field.
    """

    def __init__(self, points, values, model, *, trend="constant", measurement_variance=None):
        self.points = coordinates(points).copy()
        values = np.asarray(values, dtype=float)
        if values.shape != (len(self.points),) or not np.isfinite(values).all():
            raise ValueError("Expected one finite value per site")
        if trend not in {"constant", "affine"}:
            raise ValueError("Choose constant or affine drift")
        noise = (np.zeros(len(values)) if measurement_variance is None
                 else np.asarray(measurement_variance, dtype=float))
        if noise.shape != values.shape or not np.isfinite(noise).all() or (noise < 0).any():
            raise ValueError("Measurement variances must be finite and nonnegative")
        self.model = model
        self.trend = trend
        self.center = self.points.mean(axis=0)
        self.scale = max(float(np.max(np.abs(self.points - self.center))), 1.0)
        design = self._design(self.points)
        if np.linalg.matrix_rank(design) != design.shape[1]:
            raise ValueError("Observation geometry cannot identify the selected drift")

        matrix = model(self.points, self.points)
        matrix.flat[::len(values)+1] += model.nugget + noise
        self.jitter = 0.0
        for attempt in range(7):
            try:
                self.factor = np.linalg.cholesky(matrix + self.jitter * np.eye(len(values)))
                break
            except np.linalg.LinAlgError:
                self.jitter = model.sill * 10.0**(-12+attempt)
        else:
            raise ValueError("Observation covariance is not positive definite")
        self.whitened_design = np.linalg.solve(self.factor, design)
        self.drift_factor = np.linalg.cholesky(self.whitened_design.T @ self.whitened_design)
        response = np.linalg.solve(self.factor, values)
        self.beta = self._drift_solve(self.whitened_design.T @ response)
        self.residual = response - self.whitened_design @ self.beta

    def _design(self, locations):
        constant = np.ones((len(locations), 1))
        if self.trend == "constant":
            return constant
        return np.column_stack((constant, (locations - self.center) / self.scale))

    def _drift_solve(self, rhs):
        return np.linalg.solve(self.drift_factor.T, np.linalg.solve(self.drift_factor, rhs))

    def predict(self, queries, *, batch_size=1024):
        queries = coordinates(queries)
        if not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("Invalid kriging query batch size")
        means, variances = [], []
        for start in range(0, len(queries), batch_size):
            block = queries[start:start+batch_size]
            design = self._design(block)
            kernel = np.linalg.solve(self.factor, self.model(self.points, block))
            drift = design.T - self.whitened_design.T @ kernel
            mean = design @ self.beta + kernel.T @ self.residual
            correction = np.linalg.solve(self.drift_factor, drift)
            variance = self.model.sill - (kernel**2).sum(axis=0) + (correction**2).sum(axis=0)
            if (variance < -1e-8 * self.model.sill).any() or not np.isfinite(variance).all():
                raise FloatingPointError("Invalid conditional kriging variance")
            means.append(mean)
            variances.append(np.maximum(variance, 0.0))
        return KrigingPrediction(np.concatenate(means), np.concatenate(variances), self.jitter)
