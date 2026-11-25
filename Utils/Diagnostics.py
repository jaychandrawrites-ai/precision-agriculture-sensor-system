"""Prediction and model-input diagnostics in physical or training-scale units."""
import numpy as np

from config import FEATURES
from Physics.Constraints import violations


def forecast_diagnostics(history, predictions, target_scaler):
    history, predictions = np.asarray(history), np.asarray(predictions)
    flags = violations(predictions)
    deltas = np.diff(np.concatenate((history[-1:], predictions)), axis=0)
    normalized = target_scaler.transform(history)
    return {
        "violation_counts": {name: int(mask.sum()) for name, mask in flags.items()},
        "last_history_z": dict(zip(FEATURES, normalized[-1].tolist())),
        "max_forecast_step": dict(zip(FEATURES, np.abs(deltas).max(axis=0).tolist())),
    }
