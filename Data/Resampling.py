"""Hourly aggregation with explicit timestamp and coverage semantics."""
import pandas as pd

def hourly_mean(frame,timestamp_column,columns,timezone="America/Phoenix",minimum_count=1):
    if minimum_count<1:
        raise ValueError("Invalid minimum coverage")
    data=frame.copy()
    timestamps=pd.to_datetime(data[timestamp_column],errors="raise")
    if timestamps.dt.tz is None:
        timestamps=timestamps.dt.tz_localize(timezone,ambiguous="raise",nonexistent="raise")
    data.index=timestamps.dt.tz_convert("UTC")
    data=data.sort_index()
    data=data.loc[~data.index.duplicated(keep="last"),list(columns)]
    data=data.apply(pd.to_numeric,errors="coerce")
    bins=data.resample("1h",label="right",closed="right")
    counts=bins.count()
    means=bins.mean().where(counts>=minimum_count)
    return means,counts
