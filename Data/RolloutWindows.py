"""Common chronological windows for every forecasting method."""
from dataclasses import dataclass

import numpy as np

from Data.DataQuality import invalid_rows


@dataclass
class RolloutWindows:
    histories: np.ndarray
    targets: np.ndarray
    starts: np.ndarray
    potential: int


def rollout_windows(dataset, rows, values, timestamps, steps):
    return evaluation_windows(values, timestamps, steps, dataset.target_indices[rows])


def evaluation_windows(values, timestamps, steps, selected=None):
    """Complete rolling-origin windows without fitting a dataset or normalizer."""

    if steps < 1:
        raise ValueError("Window horizon must be positive")
    values = np.asarray(values)
    timestamps = np.asarray(timestamps, dtype="datetime64[ns]")
    if selected is None:
        selected = np.arange(3, len(values))
    else:
        selected = np.asarray(selected, dtype=int)
    if selected.ndim != 1 or (np.diff(selected) <= 0).any() or (
        len(selected) and (selected[0] < 3 or selected[-1] >= len(values))
    ):
        raise ValueError("Expected increasing target positions with three preceding observations")
    candidates = selected[:max(0, len(selected) - steps + 1)]
    offsets = np.arange(-3, steps)
    indices = candidates[:, None] + offsets
    available = np.zeros(len(values), dtype=bool)
    available[selected] = True
    valid = ~invalid_rows(values)
    contiguous = (np.diff(timestamps[indices], axis=1) == np.timedelta64(1, "h")).all(axis=1)
    keep = contiguous & valid[indices].all(axis=1) & available[indices[:, 3:]].all(axis=1)
    indices = indices[keep]
    return RolloutWindows(values[indices[:, :3]], values[indices[:, 3:]],
                          candidates[keep], len(candidates))
