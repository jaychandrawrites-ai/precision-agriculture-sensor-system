"""Season-matched distribution and missingness checks for hourly data."""
import numpy as np


SEASONS = {"DJF": (12, 1, 2), "MAM": (3, 4, 5),
           "JJA": (6, 7, 8), "SON": (9, 10, 11)}
QUANTILES = (.05, .25, .5, .75, .95)


def local_months(frame, timezone):
    if frame.index.tz is None:
        raise ValueError("Prepared records require timezone-aware interval ends")
    # Assign 00:00 interval ends to the preceding calendar day and month.
    return (frame.index - np.timedelta64(1, "ns")).tz_convert(timezone).month


def _valid(series):
    array = np.asarray(series, dtype=float)
    return array[np.isfinite(array)]


def _longest(mask):
    bounds = np.flatnonzero(np.diff(np.r_[False, mask, False].astype(np.int8)))
    return int(np.max(bounds[1::2] - bounds[::2])) if len(bounds) else 0


def distribution_profile(frame, *, timezone, hot_threshold_c=35.0):
    if not np.isfinite(hot_threshold_c):
        raise ValueError("Heat threshold must be finite")
    months = local_months(frame, timezone)
    distribution = {}
    for name, group in frame.groupby(months):
        stats = {}
        for feature in group:
            values = _valid(group[feature])
            row = {"finite_hours": len(values), "missing_hours": len(group) - len(values)}
            if len(values):
                row.update({"mean": float(values.mean()), "std": float(values.std()),
                            "min": float(values.min()), "max": float(values.max()),
                            "quantiles": dict(zip(map(str, QUANTILES), np.quantile(values, QUANTILES)))})
            stats[feature] = row
        monthly = group.air_temperature_c.to_numpy()
        distribution[str(name)] = {"features": stats, "hours": len(group),
                                   "hot_hours": int((monthly > hot_threshold_c).sum())}
    temperature = frame.air_temperature_c.to_numpy()
    return {"timezone": timezone, "interval_calendar": "one nanosecond before interval end",
            "hot_threshold_c": hot_threshold_c,
            "longest_consecutive_hot_hours": _longest(temperature > hot_threshold_c),
            "longest_incomplete_feature_run": _longest(~np.isfinite(frame.to_numpy()).all(axis=1)),
            "months": distribution}


def _cdf_distance(first, second):
    a, b = np.sort(first), np.sort(second)
    support = np.unique(np.concatenate([a, b]))
    return float(np.max(np.abs(np.searchsorted(a, support, side="right") / len(a) -
                               np.searchsorted(b, support, side="right") / len(b))))


def seasonal_drift(reference, target, *, reference_timezone, target_timezone, min_hours=24):
    """Describe comparable seasons without significance claims for autocorrelated hours."""
    if list(reference.columns) != list(target.columns) or min_hours < 1:
        raise ValueError("Seasonal drift requires a shared feature contract and positive minimum hours")
    a_month = local_months(reference, reference_timezone)
    b_month = local_months(target, target_timezone)
    result = {}
    for season, months in SEASONS.items():
        a_rows = reference[a_month.isin(months)]
        b_rows = target[b_month.isin(months)]
        features = {}
        for name in reference:
            a = _valid(a_rows[name])
            b = _valid(b_rows[name])
            row = {"reference_finite_hours": len(a), "target_finite_hours": len(b)}
            if min(len(a), len(b)) >= min_hours:
                qa = np.quantile(a, QUANTILES)
                qb = np.quantile(b, QUANTILES)
                row.update({"target_minus_reference_mean": float(b.mean() - a.mean()),
                            "target_minus_reference_quantiles": dict(zip(map(str, QUANTILES), qb - qa)),
                            "empirical_cdf_distance": _cdf_distance(a, b)})
            features[name] = row
        result[season] = {"reference_total_hours": len(a_rows), "target_total_hours": len(b_rows),
                          "features": features}
    return {"method": "descriptive marginal seasonal drift", "seasons": result,
            "minimum_finite_hours_per_archive": min_hours}
