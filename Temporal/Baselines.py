"""Persistence and fixed-period seasonal persistence."""
import numpy as np

from Data.DataQuality import invalid_rows


def baseline_forecasts(values, timestamps, windows, period_hours=24):
    """Use only observations at or before each origin, with persistence fallback."""
    if period_hours < 1:
        raise ValueError("Seasonal period must be positive")
    steps = windows.targets.shape[1]
    persistence = np.repeat(windows.histories[:, -1:, :], steps, axis=1)
    seasonal = persistence.copy()
    fallback = np.zeros(steps, dtype=int)
    timestamps = np.asarray(timestamps, dtype="datetime64[ns]")
    values = np.asarray(values)
    valid = ~invalid_rows(values)
    by_time = {timestamp: i for i, timestamp in enumerate(timestamps) if valid[i]}
    for row, start in enumerate(windows.starts):
        origin = timestamps[start - 1]
        for lead in range(1, steps + 1):
            lag = (lead - 1) % period_hours + 1 - period_hours
            index = by_time.get(origin + np.timedelta64(lag, "h"))
            if index is None:
                fallback[lead - 1] += 1
            else:
                seasonal[row, lead - 1] = values[index]
    return persistence, seasonal, fallback
