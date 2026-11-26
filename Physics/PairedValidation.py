"""Align explicitly comparable modeled quantities with independent field observations."""
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from io import BytesIO

import numpy as np
import pandas as pd

from Acquisition.Calibration import LeafCalibration


FEATURES = {"lwi", "eto_mm_interval", "eta_mm_interval", "cotton_heat_stress"}
UNITS = {
    "lwi": {"fraction", "percent", "mV"},
    "eto_mm_interval": {"mm/hour", "mm/interval", "kg/m2/hour"},
    "eta_mm_interval": {"mm/hour", "mm/interval", "kg/m2/hour"},
    "cotton_heat_stress": {"index"},
}


@dataclass(frozen=True)
class ObservationContract:
    feature: str
    units: str
    timestamp_column: str
    value_column: str
    method: str
    provenance: str
    interval_hours: int = 1
    index_definition: str | None = None
    calibration: dict | None = None

    def __post_init__(self):
        if (self.feature not in FEATURES or self.units not in UNITS[self.feature]
                or not self.timestamp_column or not self.value_column or not self.method.strip()
                or not self.provenance.strip() or type(self.interval_hours) is not int or self.interval_hours != 1):
            raise ValueError("Use explicit observable, provenance, method and compatible hourly units")
        if self.feature == "cotton_heat_stress" and not self.index_definition:
            raise ValueError("Modeled cotton stress can be compared only to the same explicitly defined index")
        if self.units == "mV" and self.calibration is None:
            raise ValueError("A voltage measurement requires a documented leaf calibration")
        if self.units != "mV" and self.calibration is not None:
            raise ValueError("Do not supply voltage calibration for already converted observations")


def _time_axis(raw):
    strings = raw.astype("string")
    if not strings.str.contains(r"(?:Z|[+-]\d{2}:\d{2})$", na=False).all():
        raise ValueError("All interval-end timestamps need explicit UTC offsets")
    timestamps = pd.DatetimeIndex(pd.to_datetime(strings, utc=True, format="ISO8601"))
    if timestamps.isna().any() or timestamps.has_duplicates or (timestamps != timestamps.floor("h")).any():
        raise ValueError("Timestamps must be unique and aligned to exact hourly UTC boundaries")
    return timestamps


def _series(data, time, name):
    values = pd.to_numeric(data[name], errors="coerce").replace([np.inf, -np.inf], np.nan)
    return pd.Series(values.to_numpy(dtype=float), index=_time_axis(data[time])).sort_index()


def _converted(data, contract):
    series = _series(data, contract.timestamp_column, contract.value_column)
    if contract.units == "percent":
        series /= 100
    elif contract.units == "mV":
        calibration = LeafCalibration(**contract.calibration)
        valid = series.notna()
        # Out-of-calibration observations are unavailable for comparison. Keep the count explicit.
        lo, hi = sorted((calibration.wet_mv, calibration.dry_mv))
        inside = valid & series.between(lo, hi)
        series.loc[~inside] = np.nan
        series.loc[inside] = series.loc[inside].map(calibration.apply)
    if contract.feature == "lwi":
        series.loc[~series.between(0, 1)] = np.nan
    else:
        series.loc[series < 0] = np.nan
    return series


def _scores(estimate, measured):
    error = np.asarray(estimate) - np.asarray(measured)
    return {"count": len(error), "bias": float(error.mean()), "mae": float(abs(error).mean()),
            "rmse": float(np.sqrt(np.mean(error**2)))}


def paired_validation(model_path, observation_path, contract, *, validation_from,
                      model_index_definition=None, calibrate_bias=False, synthetic=False):
    """Compare fixed model output without fitting on the validation period."""
    if contract.feature == "cotton_heat_stress" and contract.index_definition != model_index_definition:
        raise ValueError("Crop index definitions differ; modeled heat stress is not measured crop damage")
    if calibrate_bias and contract.units == "mV" and not contract.calibration.get("source"):
        raise ValueError("Bias correction requires a documented sensor calibration source")
    threshold = _time_axis(pd.Series([validation_from]))[0]
    model_bytes, observation_bytes = [Path(path).read_bytes() for path in (model_path, observation_path)]
    if sha256(model_bytes).digest() == sha256(observation_bytes).digest():
        raise ValueError("Field observations must be independent of the modeled output file")
    modeled = _series(pd.read_csv(BytesIO(model_bytes)), "timestamp", contract.feature)
    observed = _converted(pd.read_csv(BytesIO(observation_bytes)), contract)
    joined = pd.DataFrame({"modeled": modeled, "observed": observed}).sort_index()
    joined.index.name = "timestamp"
    pairs = joined.dropna(subset=["modeled", "observed"])
    fitting = pairs.loc[pairs.index < threshold].copy()
    validation = pairs.loc[pairs.index >= threshold].copy()
    if validation.empty or (calibrate_bias and fitting.empty):
        raise ValueError("Need matched validation rows and, if fitting bias, separate calibration rows")
    correction = float((fitting.observed - fitting.modeled).mean()) if calibrate_bias else 0.0
    validation["bias_corrected_model"] = validation.modeled + correction
    output = {
        "feature": contract.feature, "unit": {"lwi": "fraction", "cotton_heat_stress": "index"}.get(
            contract.feature, "mm per hour"),
        "observation_method": contract.method, "observation_provenance": contract.provenance,
        "instrument_calibration": contract.calibration, "index_definition": contract.index_definition,
        "synthetic": bool(synthetic), "validation_from_utc": threshold.isoformat(),
        "source_hashes": {"modeled": sha256(model_bytes).hexdigest(), "observed": sha256(observation_bytes).hexdigest()},
        "alignment": {"union_rows": len(joined), "matched_count": len(pairs),
                      "unavailable_model": int(joined.modeled.isna().sum()),
                      "unavailable_or_out_of_range_observation": int(joined.observed.isna().sum())},
        "calibration": {"rows_before_holdout": len(fitting), "fitted_bias": correction if calibrate_bias else None},
        "validation": {"uncorrected": _scores(validation.modeled, validation.observed),
                       "bias_corrected": _scores(validation.bias_corrected_model, validation.observed)
                       if calibrate_bias else None},
    }
    validation["residual"] = validation.modeled - validation.observed
    return validation, output
