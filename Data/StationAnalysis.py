"""Seasonal and cross-domain diagnostics for independent station time series."""
import numpy as np


def seasonal_profile(features, *, timezone="America/Phoenix", heat_threshold=35.0):
    if not np.isfinite(heat_threshold) or features.index.tz is None:
        raise ValueError("Need a timezone-aware hourly index and a finite heat threshold")
    local_month = features.index.tz_convert(timezone).month
    records = {}
    for month in sorted(set(local_month)):
        period = features.loc[local_month == month]
        metrics = {}
        for column in period:
            values = period[column].to_numpy(dtype=float)
            finite = values[np.isfinite(values)]
            if not len(finite):
                metrics[column] = {"count": 0, "missing": len(values)}
                continue
            low, median, high = np.quantile(finite, [.05, .5, .95])
            metrics[column] = {
                "count": len(finite), "missing": len(values) - len(finite),
                "mean": float(finite.mean()), "p05": float(low), "median": float(median),
                "p95": float(high), "min": float(finite.min()), "max": float(finite.max()),
            }
        temperature = period["air_temperature_c"].to_numpy(dtype=float)
        records[str(month)] = {"observed_rows": len(period),
                               "air_temperature_hours_above_threshold": int((temperature > heat_threshold).sum()),
                               "features": metrics}
    return {"calendar_timezone": timezone, "heat_threshold_c": heat_threshold, "months": records}


def domain_shift(reference, field):
    """Compare marginal feature distributions without mixing sites or fitting on field data."""
    if tuple(reference.columns) != tuple(field.columns):
        raise ValueError("Features must share an identical ordered schema")
    report = {}
    for name in reference:
        a, b = (frame[name].to_numpy(dtype=float) for frame in (reference, field))
        a, b = a[np.isfinite(a)], b[np.isfinite(b)]
        record = {"reference_count": len(a), "field_count": len(b)}
        if len(a) and len(b):
            scale = a.std()
            lo, hi = np.quantile(a, [.05, .95])
            record.update({"mean_difference": float(b.mean() - a.mean()),
                           "mean_difference_reference_sd": float((b.mean() - a.mean()) / scale)
                           if scale > 0 else None,
                           "field_fraction_outside_reference_p05_p95": float(((b < lo) | (b > hi)).mean())})
        report[name] = record
    return report


def compare_reference_et(path, schema, features, source_column):
    """Compare proxy ET against an explicitly mapped station-estimated mm column."""
    from pathlib import Path

    import pandas as pd

    from Data.AZMET import _frame, _timestamps

    frame = _frame(Path(path).read_bytes(), schema.delimiter, schema.headerless_columns)
    if source_column not in frame:
        raise ValueError("Station ET comparison column is absent")
    time = _timestamps(frame, schema)
    values = frame[source_column].str.strip()
    numeric = pd.to_numeric(values.mask(values.isin(schema.missing)), errors="coerce")
    # Preserve the station's negative hourly estimates. They are not sensor labels.
    station = pd.Series(numeric.to_numpy(dtype=float), index=time).reindex(features.index)
    proxy = features["eto_mm_interval"]
    good = np.isfinite(station) & np.isfinite(proxy)
    delta = (proxy - station)[good]
    return {"station_column": source_column, "unit": "mm per interval",
            "method": "comparison of two ET estimates",
            "matched_hours": int(good.sum()), "negative_station_hours": int((station[good] < 0).sum()),
            "bias_mm": float(delta.mean()) if len(delta) else None,
            "rmse_mm": float(np.sqrt((delta**2).mean())) if len(delta) else None}
