"""Causal derivatives for irregular observations, in hours."""
import numpy as np


def derivatives(values, timestamps, *, maximum_gap_seconds):
    """Return velocity and acceleration, leaving duplicates and gaps unavailable."""
    values = np.asarray(values, dtype=float)
    times = np.asarray(timestamps, dtype="datetime64[ns]")
    if (values.ndim != 2 or times.shape != (len(values),) or np.isnat(times).any()
            or not np.isfinite(maximum_gap_seconds) or maximum_gap_seconds <= 0):
        raise ValueError("Expected aligned observations and a finite positive maximum gap")
    seconds = np.diff(times) / np.timedelta64(1, "s")
    if (seconds < 0).any():
        raise ValueError("Observations must be chronological")
    hours = seconds / 3600.
    valid = (seconds > 0) & (seconds <= maximum_gap_seconds)
    velocity = np.full_like(values, np.nan)
    acceleration = np.full_like(values, np.nan)
    np.divide(np.diff(values, axis=0), hours[:, None], out=velocity[1:], where=valid[:, None])
    np.divide(2 * np.diff(velocity[1:], axis=0), (hours[1:] + hours[:-1])[:, None],
              out=acceleration[2:], where=(valid[1:] & valid[:-1])[:, None])
    return velocity, acceleration
