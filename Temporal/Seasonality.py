"""Causal seasonal lookup, train-only climatology and harmonic regression."""
from dataclasses import dataclass

import numpy as np

from Data.DataQuality import invalid_rows
from Physics.Constraints import project


HOUR = np.timedelta64(1, "h")


def clock_hours(timestamps):
    times = np.asarray(timestamps, dtype="datetime64[ns]")
    if np.isnat(times).any() or (times != times.astype("datetime64[h]")).any():
        raise ValueError("Seasonal methods require UTC timestamps aligned to whole hours")
    return times.astype("datetime64[h]").astype(np.int64)


def validate_series(values, timestamps):
    values = np.asarray(values, dtype=float)
    hours = clock_hours(timestamps)
    if hours.ndim != 1 or values.shape != (len(hours), 6) or not len(hours) or (np.diff(hours) <= 0).any():
        raise ValueError("Expected an increasing UTC time series with six physical features")
    return values, hours


class SeasonalLookup:
    """Use the latest observed slot from the preceding daily or weekly cycle."""
    def __init__(self, values, timestamps):
        self.values, self.hours = validate_series(values, timestamps)
        self.valid = ~invalid_rows(self.values)

    def predict(self, origin_times, histories, horizon, period_hours=24):
        origins = clock_hours(origin_times)
        histories = np.asarray(histories, dtype=float)
        if (origins.ndim != 1 or histories.shape != (len(origins), 3, 6)
                or not isinstance(horizon, int) or not isinstance(period_hours, int)
                or not 1 <= horizon or not 1 <= period_hours or not np.isfinite(histories).all()):
            raise ValueError("Invalid seasonal history, forecast horizon or period")
        # The modular offset is always in [-period + 1, 0], even at long leads.
        offsets = (np.arange(horizon) % period_hours) + 1 - period_hours
        requested = origins[:, None] + offsets
        index = np.searchsorted(self.hours, requested).clip(0, len(self.hours) - 1)
        found = (self.hours[index] == requested) & self.valid[index]
        result = np.where(found[..., None], self.values[index], histories[:, -1:, :])
        return result, ~found


@dataclass(frozen=True)
class DailyClimatology:
    means: np.ndarray
    counts: np.ndarray

    @classmethod
    def fit(cls, values, timestamps):
        values, hours = validate_series(values, timestamps)
        valid = ~invalid_rows(values)
        if not valid.any():
            raise ValueError("No complete climatology training rows")
        clock = hours[valid] % 24
        sums = np.zeros((24, 6))
        counts = np.bincount(clock, minlength=24)
        np.add.at(sums, clock, values[valid])
        means = np.broadcast_to(values[valid].mean(axis=0), sums.shape).copy()
        np.divide(sums, counts[:, None], out=means, where=counts[:, None] > 0)
        return cls(means, counts)

    def predict(self, timestamps):
        return self.means[clock_hours(timestamps) % 24]


def harmonic_design(hours, reference_hour, periods):
    parts = [np.ones(np.shape(hours), dtype=float)]
    for period, order in periods:
        for harmonic in range(1, order + 1):
            phase = 2 * np.pi * harmonic * ((hours - reference_hour) % period) / period
            parts.extend([np.cos(phase), np.sin(phase)])
    return np.stack(parts, axis=-1)


@dataclass(frozen=True)
class HarmonicRegression:
    reference_hour: int
    periods: tuple
    coefficients: np.ndarray
    condition: float

    @classmethod
    def fit(cls, values, timestamps, periods=((24, 3), (168, 2), (8765.82, 2)), ridge=.005):
        values, hours = validate_series(values, timestamps)
        if (not all(np.isfinite(period) and period >= 2 and isinstance(order, int) and order > 0
                    for period, order in periods)
                or not np.isfinite(ridge) or ridge <= 0):
            raise ValueError("Harmonic periods, orders and ridge must be positive")
        valid = ~invalid_rows(values)
        reference = int(hours[0])
        design = harmonic_design(hours[valid], reference, periods)
        if len(design) < design.shape[1]:
            raise ValueError("Insufficient training rows for the requested harmonics")
        # Penalize only the harmonic coefficients. Normalize the normal equations by n.
        regularizer = np.sqrt(ridge * len(design)) * np.eye(design.shape[1])
        regularizer[0, 0] = 0.
        matrix = np.vstack((design, regularizer))
        response = np.vstack((values[valid], np.zeros((design.shape[1], 6))))
        coefficients, _, _, singular = np.linalg.lstsq(matrix, response, rcond=None)
        return cls(reference, tuple(periods), coefficients, float(singular[0] / singular[-1]))

    def predict(self, timestamps):
        design = harmonic_design(clock_hours(timestamps), self.reference_hour, self.periods)
        return project(design @ self.coefficients)


@dataclass(frozen=True)
class ConvexCorrection:
    """Per lead/feature amendment to an anchor fitted only on selection targets."""
    weight: np.ndarray

    @classmethod
    def fit(cls, anchor, candidate, targets, shrinkage=.01):
        anchor, candidate, targets = (np.asarray(x, dtype=float) for x in (anchor, candidate, targets))
        if (anchor.shape != candidate.shape or anchor.shape != targets.shape or anchor.ndim != 3
                or not len(anchor) or not all(np.isfinite(x).all() for x in (anchor, candidate, targets))
                or not np.isfinite(shrinkage) or shrinkage < 0):
            raise ValueError("Expected finite, aligned validation predictions and labels")
        change = candidate - anchor
        baseline_error = targets - anchor
        denominator = (change**2).sum(axis=0) + shrinkage * (baseline_error**2).sum(axis=0)
        numerator = (baseline_error * change).sum(axis=0)
        weight = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 0)
        return cls(np.clip(weight, 0, 1))

    def predict(self, anchor, candidate):
        return project(anchor + self.weight * (candidate - anchor))
