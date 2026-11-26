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
class CalendarColumns:
    year: str
    day_of_year: str
    hour: str
    hour_encoding: str = "0-23"
    hhmm: bool = False

    def __post_init__(self):
        if self.hour_encoding not in ("0-23", "1-24"):
            raise ValueError("Specify 0-23 or 1-24 clock hours")


@dataclass(frozen=True)
class AZMETSchema:
    timestamp: str | None
    columns: dict[str, Column]
    convention: str
    timezone: str | None = None
    timestamp_format: str = "ISO8601"
    delimiter: str = ","
    missing: tuple[str, ...] = ("",)
    max_state_gap_hours: int = 0
    calendar: CalendarColumns | None = None
    headerless_columns: tuple[str, ...] | None = None

    def __post_init__(self):
        if (self.convention not in ("start", "end") or len(self.delimiter) != 1
                or not isinstance(self.max_state_gap_hours, int) or self.max_state_gap_hours < 0):
            raise ValueError("Invalid timestamp convention, delimiter, or gap limit")
        if not REQUIRED <= self.columns.keys() <= UNITS.keys():
            raise ValueError(f"Expected mappings for {sorted(REQUIRED)}")
        if (self.timestamp is None) == (self.calendar is None):
            raise ValueError("Supply exactly one timestamp column or calendar column mapping")
        if self.calendar is not None and self.timezone is None:
            raise ValueError("Calendar columns require an explicit source timezone")
        sources = list(self.timestamp_sources)
        for name, field_spec in self.columns.items():
            if field_spec.unit not in UNITS[name]:
                raise ValueError(f"Unsupported unit {field_spec.unit!r} for {name}")
            sources.append(field_spec.source)
        if len(sources) != len(set(sources)) or not all(sources):
            raise ValueError("Source headers must be nonempty and mapped at most once")

    @property
    def timestamp_sources(self):
        if self.calendar is None:
            return (self.timestamp,)
        return (self.calendar.year, self.calendar.day_of_year, self.calendar.hour)

    @classmethod
    def from_dict(cls, record):
        config = dict(record)
        config.setdefault("timestamp", None)
        config["columns"] = {name: Column(**column) for name, column in record["columns"].items()}
        if config.get("calendar") is not None:
            config["calendar"] = CalendarColumns(**config["calendar"])
        return cls(**config)


@dataclass
class WeatherData:
    values: pd.DataFrame
    flags: pd.DataFrame
    provenance: dict = field(default_factory=dict)


def _frame(payload, delimiter, headerless_columns=None):
    text = payload.decode("utf-8-sig")
    rows = csv.reader(iter(text.splitlines()), delimiter=delimiter)
    first = next(rows, [])
    headers = first if headerless_columns is None else list(headerless_columns)
    if (not first or not headers or len(first) != len(headers)
            or any(not isinstance(name, str) or not name.strip() for name in headers)
            or len(headers) != len(set(headers))):
        raise ValueError("CSV must have unique nonempty column headers matching the file width")
    for row in rows:
        if row and len(row) != len(headers):
            raise ValueError(f"Inconsistent CSV row width near line {rows.line_num}")
    options = {"header": None, "names": headers} if headerless_columns is not None else {}
    return pd.read_csv(BytesIO(payload), sep=delimiter, dtype=str, keep_default_na=False, **options)


def inspect_schema(path, delimiter=",", *, headerless=False):
    """Return actual file dimensions and per-column numeric coverage."""
    payload = Path(path).read_bytes()
    names = None
    if headerless:
        first = next(csv.reader(iter(payload.decode("utf-8-sig").splitlines()), delimiter=delimiter), [])
        names = [f"column_{i + 1}" for i in range(len(first))]
    frame = _frame(payload, delimiter, names)
    headers = frame.columns
    return {
        "sha256": sha256(payload).hexdigest(),
        "row_count": len(frame),
        "columns": {
            name: {"nonempty": int(frame[name].str.strip().ne("").sum()),
                   "numeric": int(pd.to_numeric(frame[name], errors="coerce").notna().sum())}
            for name in headers
        },
    }


def _timestamps(frame, schema):
    if schema.calendar is not None:
        parsed = _calendar_timestamps(frame, schema.calendar)
        index = parsed.tz_localize(schema.timezone, ambiguous="raise", nonexistent="raise").tz_convert("UTC")
    elif schema.timezone is None:
        raw = frame[schema.timestamp].str.strip()
        if not raw.str.contains(r"(?:Z|[+-]\d{2}:?\d{2})$", na=False).all():
            raise ValueError("Supply offset timestamps or an explicit timezone for naive timestamps")
        index = pd.DatetimeIndex(pd.to_datetime(raw, format=schema.timestamp_format, utc=True))
    else:
        raw = frame[schema.timestamp].str.strip()
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


def _calendar_timestamps(frame, calendar):
    columns = [frame[name].str.strip() for name in (calendar.year, calendar.day_of_year, calendar.hour)]
    if not all(column.str.fullmatch(r"[0-9]+").all() for column in columns):
        raise ValueError("Calendar timestamp components must be integers")
    year, day, hour = [column.astype(int).to_numpy() for column in columns]
    if calendar.hhmm:
        if (hour % 100 != 0).any():
            raise ValueError("Expected whole-hour HHMM records")
        hour = hour // 100
    midnight = pd.DatetimeIndex(pd.to_datetime(year.astype(str), format="%Y"))
    maximum_day = np.where(midnight.is_leap_year, 366, 365)
    minimum_hour = 0 if calendar.hour_encoding == "0-23" else 1
    if ((day < 1) | (day > maximum_day) | (hour < minimum_hour) |
            (hour > minimum_hour + 23)).any():
        raise ValueError("Calendar timestamp component outside its declared range")
    return midnight + pd.to_timedelta(day - 1, unit="D") + pd.to_timedelta(hour, unit="h")


def read_azmet(path, schema, *, source_label, synthetic=False):
    if not source_label.strip():
        raise ValueError("Supply a source label identifying the station/export or fixture")
    payload = Path(path).read_bytes()
    raw = _frame(payload, schema.delimiter, schema.headerless_columns)
    required = {*schema.timestamp_sources, *(spec.source for spec in schema.columns.values())}
    if missing := required - set(raw.columns):
        raise ValueError(f"Missing mapped CSV headers: {sorted(missing)}")
    if raw.empty:
        raise ValueError("No weather observations")
    raw.index = _timestamps(raw, schema)
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
