"""Read explicitly prepared hourly physical data."""
import pandas as pd

from config import FEATURES


def read_hourly(path):
    frame = pd.read_csv(path)
    timestamps = frame["timestamp"]
    if not timestamps.str.contains(r"(?:Z|[+-]\d{2}:\d{2})$", na=False).all():
        raise ValueError("Timestamp must carry a UTC offset")
    timestamps = pd.to_datetime(timestamps, format="ISO8601", utc=True)
    if not timestamps.is_monotonic_increasing or timestamps.duplicated().any():
        raise ValueError("Timestamps must strictly increase")
    return frame.loc[:, list(FEATURES)].to_numpy(dtype=float), timestamps.to_numpy(dtype="datetime64[ns]")
