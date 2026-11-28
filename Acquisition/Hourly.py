"""Single-device causal aggregation with GPS fixes retained separately."""
from collections import Counter
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pandas as pd

from Acquisition.MappedPackets import FIELDS
from Data.AZMET import WeatherData


TEMPORAL_SENSORS = ("air_temperature_c", "relative_humidity_pct", "lwi", "soil_vwc")
AUXILIARY_SENSORS = ("soil_moisture_pct", "soil_temperature_c", "soil_ec_us_cm", "soil_ph", "leaf_mv")


def _aware(value):
    timestamp = pd.Timestamp(value)
    if pd.isna(timestamp) or timestamp.tzinfo is None:
        raise ValueError("Reading timestamp must have a UTC offset")
    return timestamp.tz_convert("UTC")


def session_hourly(path, leaf, soil, *, minimum_count=1, clock="received_at", device_id=None,
                   as_of=None, complete_only=True, duplicate_policy="mean_packets"):
    if clock not in ("received_at", "observed_at") or not isinstance(minimum_count, int) or minimum_count < 1:
        raise ValueError("Invalid clock or hourly coverage requirement")
    if duplicate_policy not in {"mean_packets", "median_timestamp"}:
        raise ValueError("Unknown duplicate timestamp policy")
    cutoff = _aware(as_of) if as_of is not None else None
    records, fixes, device_ids = [], [], set()
    counts = Counter()
    digest = sha256()
    with Path(path).open("rb") as file:
        for line in file:
            digest.update(line)
            message = json.loads(line)
            counts["records"] += 1
            for flag in message["flags"]:
                counts[f"packet_flag:{flag}"] += 1
            identity = message.get("device_id")
            if device_id is not None and identity != device_id:
                counts["other_device"] += 1
                continue
            if "duplicate_packet" in message["flags"]:
                continue
            if "missing:device_id" in message["flags"]:
                continue
            if clock not in message:
                counts["missing_clock"] += 1
                continue
            timestamp = _aware(message[clock])
            if cutoff is not None and (timestamp > cutoff or _aware(message["received_at"]) > cutoff):
                counts["after_cutoff"] += 1
                continue
            raw = message["measurements"]
            if not raw and identity is None:
                continue
            device_ids.add(identity)
            measured = {}
            for name, value in raw.items():
                if name in FIELDS:
                    lo, hi, _ = FIELDS[name]
                    if isinstance(value, (int, float)) and not isinstance(value, bool) and np.isfinite(value) and lo <= value <= hi:
                        measured[name] = value
                    else:
                        counts[f"invalid:{name}"] += 1
            if "leaf_mv" in measured:
                measured["lwi"] = leaf.apply(measured["leaf_mv"])
                if not min(leaf.wet_mv, leaf.dry_mv) <= measured["leaf_mv"] <= max(leaf.wet_mv, leaf.dry_mv):
                    counts["leaf_outside_endpoints"] += 1
            if "soil_moisture_pct" in measured:
                try:
                    measured["soil_vwc"] = soil.apply(measured["soil_moisture_pct"])
                except ValueError:
                    counts["soil_outside_calibrated_range"] += 1
            if {"latitude_deg", "longitude_deg"} <= measured.keys():
                fixes.append({"timestamp": timestamp, "received_at": _aware(message["received_at"]),
                              "device_id": identity,
                              "latitude_deg": measured["latitude_deg"],
                              "longitude_deg": measured["longitude_deg"]})
            records.append({"timestamp": timestamp, **{
                key: value for key, value in measured.items() if key in TEMPORAL_SENSORS + AUXILIARY_SENSORS
            }})
    if len(device_ids) > 1:
        raise ValueError("Session contains multiple devices; select one explicitly")
    if not records:
        raise ValueError("Session has no usable timestamped records")
    columns = list(TEMPORAL_SENSORS + AUXILIARY_SENSORS)
    frame = pd.DataFrame(records).set_index("timestamp").reindex(columns=columns).astype(float)
    # Distinct packets can share the same timestamp. Identity deduplication already ran.
    if duplicate_policy == "median_timestamp":
        frame = frame.groupby(level=0, sort=True).median()
    bins = frame.sort_index().resample("h", label="right", closed="right")
    coverage = bins.count()
    hourly = bins.mean().where(coverage >= minimum_count)
    flags = (coverage < minimum_count).add_prefix("insufficient:")
    flags["missing_hour"] = bins.size() == 0
    watermark = cutoff if cutoff is not None else frame.index.max()
    if complete_only:
        closed = hourly.index <= watermark
        counts["unfinished_hour_bins"] = int((~closed).sum())
        hourly, flags, coverage = hourly.loc[closed], flags.loc[closed], coverage.loc[closed]
        if not hourly.empty and cutoff is not None:
            span = pd.date_range(hourly.index[0], cutoff.floor("h"), freq="h", name="timestamp")
            counts["empty_hours_after_last_bin"] = len(span) - len(hourly)
            hourly = hourly.reindex(span)
            flags = flags.reindex(span, fill_value=True)
            coverage = coverage.reindex(span, fill_value=0)
        if hourly.empty:
            raise ValueError("Session has no closed hourly bins")
    gps = pd.DataFrame(fixes, columns=["timestamp", "received_at", "device_id", "latitude_deg", "longitude_deg"])
    provenance = {"source_type": "acquisition_session", "readings_sha256": digest.hexdigest(),
                  "timestamp_clock": clock, "aggregation": "right-closed UTC hourly means",
                  "duplicate_policy": duplicate_policy, "minimum_count": minimum_count, "device_id": next(iter(device_ids)),
                  "complete_hour_bins_only": complete_only, "as_of_utc": watermark.isoformat(),
                  "calibration": {"leaf": asdict(leaf), "soil": asdict(soil)},
                  "packet_counts": dict(counts)}
    return WeatherData(hourly, flags, provenance), gps, coverage
