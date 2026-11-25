"""Chronological examples with training-only normalization."""
from dataclasses import dataclass

import numpy as np

from Data.DataQuality import invalid_rows
from Data.Derivatives import supervised
from Data.Filters import clean_inputs
from Data.Normalization import Standardizer
from Data.TimeSeries import split_at, split_indices


@dataclass
class Dataset:
    inputs: np.ndarray
    targets: np.ndarray
    persistence: np.ndarray
    target_indices: np.ndarray
    weights: np.ndarray
    train: np.ndarray
    validation: np.ndarray
    test: np.ndarray
    input_scaler: Standardizer
    target_scaler: Standardizer


def build_dataset(values, timestamps, train_fraction=0.7, validation_fraction=0.15,
                  max_gap=0, detect_outliers=False, *, cutoffs=None):
    observed = np.array(values, dtype=float, copy=True)
    invalid = invalid_rows(observed)
    observed[invalid] = np.nan
    timestamps = np.asarray(timestamps, dtype="datetime64[ns]")
    if timestamps.shape != (len(observed),) or np.isnat(timestamps).any():
        raise ValueError("Invalid hourly timestamps")
    gaps = np.diff(timestamps)
    if (gaps <= np.timedelta64(0, "s")).any():
        raise ValueError("Timestamps must strictly increase")
    cleaned, filled = observed.copy(), np.zeros(observed.shape, dtype=bool)
    boundaries = np.r_[0, np.flatnonzero(gaps != np.timedelta64(1, "h")) + 1, len(observed)]
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        cleaned[start:end], filled[start:end] = clean_inputs(
            observed[start:end], max_gap, detect_outliers
        )
    inputs, _, indices = supervised(cleaned, timestamps)
    valid_targets = ~invalid[indices] & ~filled[indices].any(axis=1)
    inputs, indices = inputs[valid_targets], indices[valid_targets]
    targets = observed[indices]
    train, validation, test = (split_indices(indices, train_fraction, validation_fraction)
                               if cutoffs is None else split_at(timestamps, indices, cutoffs))
    imputed_rows = filled.any(axis=1)
    weights = 1 - (imputed_rows[indices-1].astype(float) +
                   imputed_rows[indices-2] + imputed_rows[indices-3]) / 4
    input_scaler = Standardizer().fit(inputs[train])
    target_scaler = Standardizer().fit(targets[train])
    return Dataset(input_scaler.transform(inputs), targets, inputs[:, :6], indices,
                   weights, train, validation, test, input_scaler, target_scaler)
