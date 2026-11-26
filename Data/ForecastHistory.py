"""Shared eligibility checks for the latest hourly forecast context."""
import numpy as np

from config import FEATURES
from Physics.Constraints import LOWER, UPPER


HISTORY_HOURS = 3


def inspect_history(values, timestamps):
    values = np.asarray(values, dtype=float)
    timestamps = np.asarray(timestamps, dtype="datetime64[ns]")
    if timestamps.ndim != 1 or values.shape != (len(timestamps), len(FEATURES)):
        raise ValueError("Expected aligned hourly timestamps and six physical features")
    history = values[-HISTORY_HOURS:]
    times = timestamps[-HISTORY_HOURS:]
    bad = ~np.isfinite(history) | (history < LOWER) | (history > UPPER)
    consecutive = (not np.isnat(times).any()
                   and (np.diff(times) == np.timedelta64(1, "h")).all())
    return {
        "ready": bool(len(history) == HISTORY_HOURS and consecutive and not bad.any()),
        "required_hours": HISTORY_HOURS,
        "available_hours": len(history),
        "consecutive": bool(len(history) == HISTORY_HOURS and consecutive),
        "invalid_features": [name for name, invalid in zip(FEATURES, bad.any(axis=0)) if invalid],
    }


def require_history(values, timestamps, message="Forecast requires three consecutive, physically valid hourly samples"):
    if not inspect_history(values, timestamps)["ready"]:
        raise ValueError(message)
    return np.asarray(values, dtype=float)[-HISTORY_HOURS:]
