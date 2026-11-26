"""Predeclared rolling-origin windows for station archives."""
from dataclasses import dataclass

import numpy as np

from Data.DataQuality import invalid_rows
from Temporal.Seasonality import validate_series


@dataclass(frozen=True)
class OriginFold:
    train_rows: slice
    test_boundary_hours: tuple[int, int]
    starts: np.ndarray
    potential: int


def rolling_folds(values, timestamps, *, initial_hours, test_hours, horizon, stride=None,
                  max_training_hours=None):
    """Fit only before each test block. Read observations at or before each forecast origin."""
    values, timeline = validate_series(values, timestamps)
    stride = test_hours if stride is None else stride
    if (not all(type(v) is int for v in (initial_hours, test_hours, horizon, stride))
            or initial_hours < 3 or test_hours < horizon or horizon < 1 or stride < test_hours
            or max_training_hours is not None and
            (type(max_training_hours) is not int or max_training_hours < 3)):
        raise ValueError("Require positive chronological spans and nonoverlapping test blocks")
    lower = int(timeline[0])
    upper = int(timeline[-1]) + 1
    bad = invalid_rows(values)
    offsets = np.arange(-3, horizon)
    for test_left in range(lower + initial_hours, upper - test_hours + 1, stride):
        test_right = test_left + test_hours
        train_left = lower if max_training_hours is None else max(lower, test_left - max_training_hours)
        start, cutoff, end = np.searchsorted(timeline, [train_left, test_left, test_right])
        candidates = np.arange(max(cutoff, 3), end - horizon + 1)
        positions = candidates[:, None] + offsets
        eligible = ((np.diff(timeline[positions], axis=1) == 1).all(axis=1) &
                    ~bad[positions].any(axis=1))
        yield OriginFold(slice(int(start), int(cutoff)), (test_left, test_right),
                         candidates[eligible], test_hours - horizon + 1)
