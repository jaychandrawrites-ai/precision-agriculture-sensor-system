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
