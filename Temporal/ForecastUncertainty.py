"""Validation-only marginal error intervals and ensemble diagnostics."""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ErrorIntervals:
    radius: np.ndarray
    observations: int
    nominal_mass: float

    def bounds(self, prediction):
        return prediction - self.radius, prediction + self.radius


def calibrate(predictions, observations, mass=.9):
    predictions, observations = np.asarray(predictions), np.asarray(observations)
    if (predictions.shape != observations.shape or predictions.ndim != 3 or not 0 < mass < 1
            or not np.isfinite(predictions).all() or not np.isfinite(observations).all()):
        raise ValueError("Expected aligned finite forecasts and a nominal probability")
    n = len(observations)
    rank = int(np.ceil((n + 1) * mass))
    if rank > n:
        raise ValueError("Too few calibration origins for this nominal probability")
    residual = np.abs(predictions - observations)
    return ErrorIntervals(np.partition(residual, rank - 1, axis=0)[rank - 1], n, mass)


def evaluate_intervals(intervals, prediction, actual):
    lower, upper = intervals.bounds(np.asarray(prediction))
    actual = np.asarray(actual)
    if lower.shape != actual.shape or not len(actual) or not np.isfinite(actual).all():
        raise ValueError("Need nonempty interval verification observations")
    return {"coverage": ((lower <= actual) & (actual <= upper)).mean(axis=0),
            "mean_width": (upper - lower).mean(axis=0),
            "calibration_origins": intervals.observations,
            "nominal_mass": intervals.nominal_mass}


def ensemble(members):
    members = np.asarray(members)
    if members.ndim != 4 or not len(members) or not np.isfinite(members).all():
        raise ValueError("Expected finite [member, origin, lead, feature] predictions")
    return members.mean(axis=0), members.std(axis=0)


def reliability(calibration_scores, test_scores, prediction, observed, *, quantile=.95):
    """Flag unusually large input scores relative to the calibration period."""
    calibration_scores, test_scores = np.asarray(calibration_scores), np.asarray(test_scores)
    prediction, observed = np.asarray(prediction), np.asarray(observed)
    if (calibration_scores.ndim != 1 or not len(calibration_scores) or test_scores.ndim != 1
            or prediction.shape != observed.shape or len(prediction) != len(test_scores)
            or not all(np.isfinite(a).all() for a in (calibration_scores, test_scores, prediction, observed))
            or not 0 < quantile < 1):
        raise ValueError("Expected finite, aligned reliability inputs")
    threshold = float(np.quantile(calibration_scores, quantile))
    ood = test_scores > threshold
    groups = {}
    for name, keep in (("flagged", ood), ("within_reference", ~ood)):
        error = prediction[keep] - observed[keep]
        groups[name] = {"origins": int(keep.sum()),
                        "rmse": np.sqrt((error**2).mean(axis=0)) if len(error) else None}
    return {"calibration_quantile": quantile, "threshold": threshold,
            "score": "largest absolute training-standardized direct input",
            "groups": groups}


def _simplex_projection(values):
    """Project the final axis onto the nonnegative unit simplex."""
    ordered = np.sort(values, axis=-1)[..., ::-1]
    count = np.arange(1, values.shape[-1] + 1)
    threshold = (np.cumsum(ordered, axis=-1) - 1) / count
    index = (ordered > threshold).sum(axis=-1, keepdims=True) - 1
    return np.maximum(values - np.take_along_axis(threshold, index, axis=-1), 0.)


@dataclass(frozen=True)
class ValidationBlend:
    """Convex weights indexed by lead, feature and member."""
    weights: np.ndarray
    fit_origins: int
    iterations: int

    @classmethod
    def fit(cls, validation_members, validation_targets, *, penalty=.02, max_iterations=1000):
        members, targets = map(lambda x: np.asarray(x, dtype=float), (validation_members, validation_targets))
        if (targets.ndim != 3 or not targets.size or members.ndim != 4 or not len(members)
                or members.shape[1:] != targets.shape or not np.isfinite(penalty) or penalty <= 0
                or type(max_iterations) is not int or max_iterations < 1
                or not all(np.isfinite(a).all() for a in (members, targets))):
            raise ValueError("Expected complete validation-only member and target trajectories")
        # With weights summing to one, modeling residuals removes the
        # physical offset while leaving the objective unchanged.
        residuals = (members - targets).transpose(2, 3, 1, 0)
        scale = np.maximum(np.sqrt(np.mean(residuals**2, axis=(2, 3))), 1e-9)
        residuals /= scale[..., None, None]
        gram = residuals.swapaxes(-1, -2) @ residuals / len(targets)
        step = 1. / (np.linalg.eigvalsh(gram)[..., -1, None] + penalty)
        prior = np.full(gram.shape[:-1], 1 / len(members))
        weights = prior.copy()
        for iteration in range(1, max_iterations + 1):
            gradient = (gram @ weights[..., None])[..., 0] + penalty * (weights - prior)
            update = _simplex_projection(weights - step * gradient)
            if np.max(np.abs(update - weights)) < 1e-8:
                weights = update
                break
            weights = update
        return cls(weights, len(targets), iteration)

    def predict(self, members, *, fallback=None):
        """Use a caller-declared causal fallback for unavailable members."""
        members = np.asarray(members, dtype=float)
        if (members.ndim != 4 or members.shape[1] == 0
                or members.transpose(1, 2, 3, 0).shape[1:] != self.weights.shape):
            raise ValueError("Member axis or forecast shape differs from validation")
        weights = self.weights.transpose(2, 0, 1)[:, None]
        valid = np.isfinite(members)
        total = (weights * valid).sum(axis=0)
        numerator = (np.where(valid, members, 0.) * weights).sum(axis=0)
        if fallback is None:
            if (total == 0).any():
                raise ValueError("At least one forecast slot requires a causal fallback")
            result = np.empty_like(numerator)
        else:
            result = np.array(fallback, dtype=float, copy=True)
            if result.shape != numerator.shape or not np.isfinite(result).all():
                raise ValueError("Fallback must use the same finite lead and feature axes")
        np.divide(numerator, total, out=result, where=total > 0)
        return result


@dataclass(frozen=True)
class TrajectoryResiduals:
    """Empirical residual vectors with joint axes and non-overlapping origins."""
    residuals: np.ndarray
    retained_origins: np.ndarray

    @classmethod
    def calibrate(cls, forecasts, observations, origin_hours, *, spacing_hours=None):
        pred, actual = map(lambda x: np.asarray(x, dtype=float), (forecasts, observations))
        hours = np.asarray(origin_hours)
        if (pred.ndim not in (3, 4) or pred.shape != actual.shape or not pred.size
                or hours.shape != (len(pred),) or hours.dtype.kind not in "iu"
                or np.any(hours[1:] <= hours[:-1])
                or not all(np.isfinite(x).all() for x in (pred, actual))):
            raise ValueError("Expected chronological calibration origins with aligned finite observations")
        leads = pred.shape[1]
        spacing_hours = leads if spacing_hours is None else spacing_hours
        if type(spacing_hours) is not int or spacing_hours < leads:
            raise ValueError("Calibration trajectory blocks must not overlap")
        keep = [0]
        for index in range(1, len(hours)):
            if int(hours[index]) - int(hours[keep[-1]]) >= spacing_hours:
                keep.append(index)
        if len(keep) < 2:
            raise ValueError("At least two non-overlapping calibration origins are required")
        return cls(actual[keep] - pred[keep], hours[keep].copy())

    def sample(self, point, draws, seed=42):
        point = np.asarray(point, dtype=float)
        if point.shape != self.residuals.shape[1:] or not np.isfinite(point).all() or type(draws) is not int or draws < 1:
            raise ValueError("Expected one complete forecast and a positive draw count")
        rng = np.random.default_rng(seed)
        chosen = rng.integers(len(self.residuals), size=draws)
        return point + self.residuals[chosen], self.retained_origins[chosen]
