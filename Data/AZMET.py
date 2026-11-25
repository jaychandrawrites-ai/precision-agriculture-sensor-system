"""Explicit, station-file-specific mapping for hourly AZMET exports."""
from dataclasses import asdict, dataclass, field
from hashlib import sha256
from io import BytesIO
from pathlib import Path
import csv

import numpy as np
import pandas as pd

from assumptions import Assumptions
from Data.Imputation import forward_fill
from Data.WeatherPhysics import ReferenceET


# Canonical units: C, %, W/m², m/s, mm per hourly interval.
# Each conversion is (scale, offset); source units are supplied by the caller.
UNITS = {
    "air_temperature_c": {"C": (1, 0), "F": (5 / 9, -32 * 5 / 9)},
    "relative_humidity_pct": {"%": (1, 0), "fraction": (100, 0)},
    "shortwave_w_m2": {"W/m2": (1, 0), "MJ/m2/hour": (1e6 / 3600, 0),
                       "kJ/m2/hour": (1e3 / 3600, 0)},
    "wind_m_s": {"m/s": (1, 0), "km/h": (1 / 3.6, 0), "mph": (0.44704, 0)},
    "precipitation_mm": {"mm/hour": (1, 0), "in/hour": (25.4, 0)},
    "irrigation_mm": {"mm/hour": (1, 0), "in/hour": (25.4, 0)},
}
BOUNDS = {
    "air_temperature_c": (-60, 60),
    "relative_humidity_pct": (0, 100),
    "shortwave_w_m2": (0, np.inf),
    "wind_m_s": (0, np.inf),
    "precipitation_mm": (0, np.inf),
    "irrigation_mm": (0, np.inf),
}
REQUIRED = set(UNITS) - {"irrigation_mm"}


@dataclass(frozen=True)
class Column:
    source: str
    unit: str


@dataclass(frozen=True)
class AZMETSchema:
    timestamp: str
    columns: dict[str, Column]
    convention: str
    timezone: str | None = None
    timestamp_format: str = "ISO8601"
    delimiter: str = ","
    missing: tuple[str, ...] = ("",)
    max_state_gap_hours: int = 0

    def __post_init__(self):
        if (self.convention not in ("start", "end") or len(self.delimiter) != 1
                or not isinstance(self.max_state_gap_hours, int) or self.max_state_gap_hours < 0):
            raise ValueError("Invalid timestamp convention, delimiter, or gap limit")
        if not REQUIRED <= self.columns.keys() <= UNITS.keys():
            raise ValueError(f"Expected mappings for {sorted(REQUIRED)}")
        sources = [self.timestamp]
        for name, field_spec in self.columns.items():
            if field_spec.unit not in UNITS[name]:
                raise ValueError(f"Unsupported unit {field_spec.unit!r} for {name}")
            sources.append(field_spec.source)
        if len(sources) != len(set(sources)) or not all(sources):
            raise ValueError("Source headers must be nonempty and mapped at most once")

    @classmethod
    def from_dict(cls, record):
        return cls(**{**record, "columns": {
            name: Column(**column) for name, column in record["columns"].items()
        }})


@dataclass
class WeatherData:
    values: pd.DataFrame
    flags: pd.DataFrame
    provenance: dict = field(default_factory=dict)


def _header(payload, delimiter):
    text = payload.decode("utf-8-sig")
    headers = next(csv.reader(iter(text.splitlines()), delimiter=delimiter), [])
    if not headers or any(not name for name in headers) or len(headers) != len(set(headers)):
        raise ValueError("CSV must have unique nonempty column headers")
    return headers


def inspect_schema(path, delimiter=","):
    """Return the actual header, row count and per-column numeric coverage."""
    payload = Path(path).read_bytes()
    headers = _header(payload, delimiter)
    frame = pd.read_csv(BytesIO(payload), sep=delimiter, dtype=str, keep_default_na=False)
    return {
        "sha256": sha256(payload).hexdigest(),
        "row_count": len(frame),
        "columns": {
            name: {"nonempty": int(frame[name].str.strip().ne("").sum()),
                   "numeric": int(pd.to_numeric(frame[name], errors="coerce").notna().sum())}
            for name in headers
        },
    }


def _timestamps(raw, schema):
    if schema.timezone is None:
        if not raw.str.contains(r"(?:Z|[+-]\d{2}:?\d{2})$", na=False).all():
            raise ValueError("Supply offset timestamps or an explicit timezone for naive timestamps")
        index = pd.DatetimeIndex(pd.to_datetime(raw, format=schema.timestamp_format, utc=True))
    else:
        parsed = pd.DatetimeIndex(pd.to_datetime(raw, format=schema.timestamp_format))
        if parsed.tz is not None:
            raise ValueError("Use the timezone setting only for naive timestamps")
        index = parsed.tz_localize(schema.timezone, ambiguous="raise", nonexistent="raise").tz_convert("UTC")
    if index.hasnans:
        raise ValueError("Missing source timestamps")
    if schema.convention == "start":
        index += pd.Timedelta(hours=1)
    if not (index == index.floor("h")).all():
        raise ValueError("Expected hourly records aligned to UTC hour boundaries")
    if index.has_duplicates:
        raise ValueError("Duplicate source timestamps")
    return index.rename("timestamp")


def read_azmet(path, schema, *, source_label, synthetic=False):
    if not source_label.strip():
        raise ValueError("Supply a source label identifying the station/export or fixture")
    payload = Path(path).read_bytes()
    headers = _header(payload, schema.delimiter)
    required = {schema.timestamp, *(spec.source for spec in schema.columns.values())}
    if missing := required - set(headers):
        raise ValueError(f"Missing mapped CSV headers: {sorted(missing)}")
    raw = pd.read_csv(BytesIO(payload), sep=schema.delimiter, dtype=str, keep_default_na=False)
    if raw.empty:
        raise ValueError("No weather observations")
    raw.index = _timestamps(raw[schema.timestamp].str.strip(), schema)
    raw = raw.sort_index()
    span = pd.date_range(raw.index[0], raw.index[-1], freq="h", name="timestamp")
    values, flags = pd.DataFrame(index=span), pd.DataFrame(index=span)
    flags["missing_hour"] = ~span.isin(raw.index)
    for name, spec in schema.columns.items():
        source = raw[spec.source].str.strip().reindex(span)
        absent = source.isna() | source.isin(schema.missing) | source.eq("")
        numeric = pd.to_numeric(source.mask(absent), errors="coerce")
        scale, offset = UNITS[name][spec.unit]
        converted = numeric * scale + offset
        lower, upper = BOUNDS[name]
        invalid = ~absent & (~np.isfinite(converted) | ~converted.between(lower, upper))
        values[name] = converted.mask(invalid)
        flags[f"missing:{name}"] = absent
        flags[f"invalid:{name}"] = invalid
    # Do not fill precipitation or radiation fluxes, or carry across absent hours.
    states = ["air_temperature_c", "relative_humidity_pct", "wind_m_s"]
    filled = np.zeros((len(span), len(states)), dtype=bool)
    present = ~flags["missing_hour"].to_numpy()
    starts = np.flatnonzero(present & ~np.r_[False, present[:-1]])
    ends = np.flatnonzero(present & ~np.r_[present[1:], False]) + 1
    states_array = values[states].to_numpy(copy=True)
    for start, end in zip(starts, ends):
        states_array[start:end], filled[start:end] = forward_fill(
            states_array[start:end], schema.max_state_gap_hours
        )
    values[states] = states_array
    for i, name in enumerate(states):
        flags[f"filled:{name}"] = filled[:, i]
    return WeatherData(values, flags, {
        "source_label": source_label, "source_type": "synthetic_fixture" if synthetic else "station_export",
        "source_sha256": sha256(payload).hexdigest(),
        "source_rows": len(raw), "schema": asdict(schema), "timestamp_semantics": "UTC interval end",
    })


def check_hourly_configuration(reference_et: ReferenceET, assumptions: Assumptions):
    if assumptions.site.interval_hours != 1:
        raise ValueError("Feature preparation requires one-hour intervals")
    if reference_et.latitude_deg != assumptions.site.latitude_deg:
        raise ValueError("Reference ET latitude must match the site assumptions")
