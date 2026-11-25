"""Empirical multi-horizon intervals for spaced rolling-origin verification."""
from dataclasses import dataclass

import numpy as np

from Temporal.Seasonality import clock_hours


def separated_indices(timestamps, hours):
    """Greedy chronological spacing independent of outcomes and forecast errors."""
    clock = clock_hours(timestamps)
    if clock.ndim != 1 or (np.diff(clock) <= 0).any() or not isinstance(hours, int) or hours < 1:
        raise ValueError("Expected increasing times and a positive hour spacing")
    indices = []
    for index, time in enumerate(clock):
        if not indices or time - clock[indices[-1]] >= hours:
            indices.append(index)
    return np.asarray(indices, dtype=int)


def _quantile_rank(errors, mass):
    n = len(errors)
    rank = int(np.ceil((n + 1) * mass))
    if not 0 < mass < 1 or not 1 <= rank <= n:
        raise ValueError("Need more separated calibration origins or a lower nominal mass")
    return np.partition(errors, rank - 1, axis=0)[rank - 1]


@dataclass(frozen=True)
class BlockIntervals:
    marginal_radius: np.ndarray
    trajectory_radius: np.ndarray
    calibration_count: int
    spacing_hours: int
    mass: float

    @classmethod
    def fit(cls, predictions, targets, origins, mass=.9, spacing_hours=None):
        if (predictions.shape != targets.shape or targets.ndim != 3 or targets.shape[1] < 1
                or len(origins) != len(targets) or not np.isfinite(predictions).all()
                or not np.isfinite(targets).all()):
            raise ValueError("Expected aligned finite forecasts, targets and origins")
        lead_count = targets.shape[1]
        spacing = lead_count if spacing_hours is None else spacing_hours
        if spacing < lead_count:
            raise ValueError("Spacing must avoid overlapping target windows")
        sample = separated_indices(origins, spacing)
        absolute = np.abs(predictions[sample] - targets[sample])
        marginal = _quantile_rank(absolute, mass)
        # Max across lead times for each feature. The result is not joint across features.
        joint = _quantile_rank(absolute.max(axis=1), mass)
        return cls(marginal, joint, len(sample), spacing, mass)

    def score(self, predictions, targets, origins):
        if (predictions.shape != targets.shape or len(origins) != len(targets)
                or targets.shape[1:] != self.marginal_radius.shape
                or not np.isfinite(predictions).all() or not np.isfinite(targets).all()):
            raise ValueError("Need aligned finite verification forecasts, targets and origins")
        take = separated_indices(origins, self.spacing_hours)
        if not len(take):
            raise ValueError("No verification origins")
        error = np.abs(predictions[take] - targets[take])
        width = 2 * self.marginal_radius
        # Central interval score: width with a penalty for errors beyond either endpoint.
        score = width + 2 / (1 - self.mass) * np.maximum(0, error - self.marginal_radius)
        return {"calibration_origins": self.calibration_count, "verification_origins": len(take),
                "spacing_hours": self.spacing_hours, "nominal_mass": self.mass,
                "marginal_coverage": (error <= self.marginal_radius).mean(axis=0),
                "marginal_width": width, "mean_interval_score": score.mean(axis=0),
                "whole_trajectory_coverage_per_feature": (error <= self.trajectory_radius).all(axis=1).mean(axis=0),
                "whole_trajectory_width_per_feature": 2 * self.trajectory_radius}
