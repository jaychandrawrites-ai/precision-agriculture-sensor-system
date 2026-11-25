"""Extract co-located measured values from GPS-bearing acquisition packets."""
from collections import Counter
from datetime import timedelta
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pandas as pd

from Acquisition.Hourly import _aware
from Acquisition.MappedPackets import FIELDS, GPS_FIELDS


MEASUREMENT_FIELDS = FIELDS.keys() - GPS_FIELDS


def spatial_observations(paths, feature, timestamp, *, clock="received_at", window_minutes=60,
                         as_of=None, device_ids=None):
    """Form a snapshot from (timestamp - window, timestamp] without joining across packets."""
    if feature not in MEASUREMENT_FIELDS:
        raise ValueError("Choose a spatial feature directly measured by a mapped sensor")
    if clock not in ("received_at", "observed_at") or not 0 < window_minutes <= 1440:
        raise ValueError("Invalid snapshot clock or window")
    end = _aware(timestamp)
    cutoff = _aware(as_of) if as_of is not None else end
    if cutoff < end:
        raise ValueError("Snapshot cutoff precedes the snapshot interval end")
    start = end - timedelta(minutes=window_minutes)
    low, high, _ = FIELDS[feature]
    counts, records, sources = Counter(), [], []
    paths = [Path(path) for path in paths]
    if not paths or len(set(path.resolve() for path in paths)) != len(paths):
        raise ValueError("Supply unique acquisition paths")
    selected_devices = set(device_ids) if device_ids is not None else None
    for path in paths:
        digest = sha256()
        with path.open("rb") as file:
            for line in file:
                digest.update(line)
                packet = json.loads(line)
                counts["records"] += 1
                if "duplicate_packet" in packet["flags"]:
                    counts["duplicate"] += 1
                    continue
                device = packet.get("device_id")
                if selected_devices is not None and device not in selected_devices:
                    counts["other_device"] += 1
                    continue
                if clock not in packet:
                    counts["missing_clock"] += 1
                    continue
                observed = _aware(packet[clock])
                if not start < observed <= end or _aware(packet["received_at"]) > cutoff:
                    counts["outside_snapshot"] += 1
                    continue
                values = packet["measurements"]
                if not (GPS_FIELDS | {feature}) <= values.keys():
                    counts["missing_co_located_measurement"] += 1
                    continue
                lat, lon, value = (values[key] for key in ("latitude_deg", "longitude_deg", feature))
                if (not all(isinstance(v, (int, float)) and not isinstance(v, bool)
                            for v in (lat, lon, value)) or not np.isfinite([lat, lon, value]).all()
                        or not -90 <= lat <= 90 or not -180 <= lon <= 180 or not low <= value <= high):
                    counts["invalid_measurement"] += 1
                    continue
                records.append((end, observed, device, lat, lon, value))
                counts["accepted"] += 1
        sources.append(digest.hexdigest())
    if len(set(sources)) != len(sources):
        raise ValueError("Identical acquisition contents supplied more than once")
    fields = ["timestamp", "observed_at", "device_id", "latitude_deg", "longitude_deg", feature]
    frame = pd.DataFrame(records, columns=fields)
    return frame, {"source_sha256": sources, "feature": feature,
                   "clock": clock, "start_exclusive_utc": start.isoformat(),
                   "end_inclusive_utc": end.isoformat(), "receipt_cutoff_utc": cutoff.isoformat(),
                   "counts": dict(counts)}
