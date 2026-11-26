"""Calibration-only diagnostics and explicit forecasting abstention."""
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from config import FEATURES
from Data.ForecastHistory import inspect_history


def _hours(timestamps):
    times = np.asarray(timestamps, dtype="datetime64[ns]")
    if times.ndim != 1 or np.isnat(times).any() or (np.diff(times) <= np.timedelta64(0, "h")).any():
        raise ValueError("Expected ordered unique timestamps")
    return times


def _finite_positive(array, shape):
    return array.shape == shape and np.isfinite(array).all() and (array > 0).all()


def _season(timestamp):
    month = int(timestamp.astype("datetime64[M]").astype(int) % 12 + 1)
    return ("winter", "spring", "summer", "fall")[month // 3 % 4]


@dataclass(frozen=True)
class ReliabilityProfile:
    """Source distribution and thresholds fitted only on explicit calibration rows."""
    center: np.ndarray
    scale: np.ndarray
    feature_lower: np.ndarray
    feature_upper: np.ndarray
    joint_shift_limit: float
    maximum_disagreement: float | None
    supported_steps: int | None
    station_ids: tuple
    source_kind: str
    calibration_id: str | None
    seasons: tuple
    calibration_last_utc: str
    origins: int
    tail: float
    history_hours: int
    clock_hours: tuple = ()

    @classmethod
    def fit(cls, values, timestamps, *, history_hours=3, stride=25, tail=.05,
            station_ids=(), source_kind="historical", calibration_id=None,
            disagreements=None, supported_steps=None):
        times, values = _hours(timestamps), np.asarray(values, dtype=float)
        if (values.shape != (len(times), len(FEATURES)) or type(stride) is not int or stride < 1
                or type(history_hours) is not int or history_hours < 3 or not 0 < tail < .5
                or not isinstance(source_kind, str) or not source_kind
                or not all(isinstance(site, str) and site for site in station_ids)
                or calibration_id is not None and (not isinstance(calibration_id, str) or not calibration_id)
                or supported_steps is not None and (type(supported_steps) is not int or not 1 <= supported_steps <= 72)):
            raise ValueError("Invalid calibration series or prescribed context")
        origins = eligible_origins(values, times, history_hours=history_hours, stride=stride)
        minimum = int(np.ceil(1 / tail)) + 1
        if len(origins) < minimum:
            raise ValueError(f"Reliability calibration requires at least {minimum} eligible spaced origins")
        histories = values[origins[:, None] + np.arange(1 - history_hours, 1)]
        states = histories[:, -1]
        center = np.median(states, axis=0)
        lower, upper = np.quantile(states, [tail / 2, 1 - tail / 2], axis=0)
        # Physical-unit floors prevent division by zero on seasonally constant signals.
        floor = [.5, 2., .01, .01, .01, .1]
        scale = np.maximum((upper - lower) / 2, floor)
        scores = max_feature_shift(histories, center, scale)
        maximum_disagreement = None
        if disagreements is not None:
            raw = np.asarray(disagreements, dtype=float)
            if raw.shape != (len(origins),) or not np.isfinite(raw).all() or (raw < 0).any():
                raise ValueError("Disagreement needs one score per eligible calibration origin")
            if supported_steps is None:
                raise ValueError("Declare the calibrated horizon for disagreement")
            maximum_disagreement = float(np.quantile(raw, 1 - tail, method="higher"))
        last = pd.Timestamp(times[-1], tz="UTC").isoformat()
        return cls(center, scale, lower, upper, float(np.quantile(scores, 1 - tail, method="higher")),
                   maximum_disagreement, supported_steps, tuple(station_ids), source_kind, calibration_id,
                   tuple(sorted({_season(times[i]) for i in origins})), last, len(origins), tail, history_hours,
                   tuple(sorted({int(value) % 24 for value in times[origins].astype("datetime64[h]").astype(int)})))

    def to_record(self):
        return {"format": "jupiter-reliability-v1", **asdict(self)}

    @classmethod
    def from_record(cls, record):
        raw = dict(record)
        if raw.pop("format", None) != "jupiter-reliability-v1":
            raise ValueError("Unsupported reliability profile")
        for key in ("center", "scale", "feature_lower", "feature_upper"):
            raw[key] = np.asarray(raw[key], dtype=float)
        result = cls(**raw)
        shape = (len(FEATURES),)
        if (not _finite_positive(result.scale, shape) or any(a.shape != shape or not np.isfinite(a).all()
                 for a in (result.center, result.feature_lower, result.feature_upper))
                or (result.feature_lower > result.feature_upper).any()
                or not np.isfinite(result.joint_shift_limit) or result.joint_shift_limit < 0
                or result.maximum_disagreement is not None
                and (not np.isfinite(result.maximum_disagreement) or result.maximum_disagreement < 0)
                or not 0 < result.tail < .5 or type(result.history_hours) is not int or result.history_hours < 3):
            raise ValueError("Invalid reliability profile")
        return result

    def assess(self, values, timestamps, *, steps, model_max_steps=None, station_id=None,
               source_kind="historical", calibration_id=None, disagreement=None, enforce=True):
        """Assess structure first. Include context warnings even for an accepted forecast."""
        values = np.asarray(values, dtype=float)
        issues, diagnostics = [], {}
        def report(code, message, blocks):
            issues.append({"code": code, "message": message, "blocks_forecast": bool(blocks)})
        supplied = inspect_history(values, timestamps, hours=self.history_hours)
        diagnostics["history"] = supplied
        if not supplied["ready"]:
            explanation = "Provide consecutive, physically valid hourly values for the full model context."
            if supplied["invalid_features"]:
                explanation += " Unavailable or invalid: " + ", ".join(supplied["invalid_features"]) + "."
            report("missing_or_invalid_hourly_history", explanation, True)
        steps_valid = type(steps) is int and 1 <= steps <= 72
        if not steps_valid or (model_max_steps is not None and steps > model_max_steps):
            report("unsupported_horizon", "The requested lead exceeds this checkpoint or workflow's supported horizon.", True)
        if self.station_ids and station_id not in self.station_ids:
            report("station_transfer", "This station was outside the declared calibration stations.", False)
        if source_kind != self.source_kind:
            report("domain_transfer", "The data source differs from the reliability calibration source.", False)
        if self.calibration_id is not None and self.calibration_id != calibration_id:
            report("calibration_mismatch", "The input calibration ID does not match the reliability profile.", True)
        if supplied["ready"]:
            times = _hours(timestamps)
            season = _season(times[-1])
            diagnostics["season"] = season
            hour = int(times[-1].astype("datetime64[h]").astype(int)) % 24
            diagnostics["hour_utc"] = hour
            if self.clock_hours and hour not in self.clock_hours:
                report("diurnal_transfer", "This UTC hour was absent from reliability calibration.", False)
            if season not in self.seasons:
                report("seasonal_transfer", f"{season.capitalize()} was absent from reliability calibration.", False)
            history = values[-self.history_hours:]
            shifts = np.max(np.abs(history - self.center) / self.scale, axis=0)
            diagnostics["standardized_feature_shift"] = dict(zip(FEATURES, shifts.tolist()))
            diagnostics["joint_shift_limit"] = self.joint_shift_limit
            extreme = history[-1, 0] < self.feature_lower[0] or history[-1, 0] > self.feature_upper[0]
            if extreme:
                report("extreme_temperature", "Current air temperature is outside the calibrated central range.", False)
            if float(shifts.max()) > self.joint_shift_limit:
                report("distribution_shift", "Input feature shift exceeds the calibration-only threshold.", enforce)
        if disagreement is not None:
            if not np.isfinite(disagreement) or disagreement < 0:
                raise ValueError("Disagreement must be a finite nonnegative score")
            diagnostics["ensemble_disagreement"] = float(disagreement)
            if self.maximum_disagreement is None or steps != self.supported_steps:
                report("uncalibrated_ensemble", "No disagreement threshold was calibrated for this horizon.", False)
            elif disagreement > self.maximum_disagreement:
                report("ensemble_disagreement", "The ensemble spread exceeds its calibrated threshold.", enforce)
        return {"decision": "abstain" if any(issue["blocks_forecast"] for issue in issues) else "forecast",
                "issues": issues, "diagnostics": diagnostics,
                "meaning": "Empirical diagnostics calibrated on designated past observations. Coverage is not guaranteed under transfer."}


def max_feature_shift(histories, center, scale):
    return (np.abs(histories - center) / scale).max(axis=(1, 2))


def eligible_origins(values, times, *, history_hours, stride):
    indices = []
    for i in range(history_hours - 1, len(values)):
        if indices and times[i] - times[indices[-1]] < np.timedelta64(stride, "h"):
            continue
        start = i - history_hours + 1
        if inspect_history(values[start:i + 1], times[start:i + 1], hours=history_hours)["ready"]:
            indices.append(i)
    return np.asarray(indices, dtype=int)


def ensemble_disagreement(predictions, scale):
    """Dimensionless spread over equal-weight members at fixed model and horizon."""
    cube = np.asarray(predictions, dtype=float)
    scale = np.asarray(scale, dtype=float)
    if (cube.ndim != 3 or cube.shape[0] < 2 or cube.shape[1] < 1
            or cube.shape[2] != len(FEATURES) or not np.isfinite(cube).all()
            or not _finite_positive(scale, (len(FEATURES),))):
        raise ValueError("Need multiple complete aligned forecasts and a positive physical scale")
    return float((cube.std(axis=0) / scale).max())
