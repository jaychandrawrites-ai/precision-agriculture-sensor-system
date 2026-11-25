"""Privacy-conscious per-node receipt, completeness, and GPS coverage."""
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
import math
from pathlib import Path

from Acquisition.Archives import _receive_time
from Acquisition.MappedPackets import FIELDS, GPS_FIELDS
from Utils.Serialization import decode_json


@dataclass
class ReceiptSummary:
    counts: Counter = field(default_factory=Counter)
    measurement_counts: Counter = field(default_factory=Counter)
    first: datetime | None = None
    last: datetime | None = None
    previous: datetime | None = None
    max_receipt_gap: float = 0.0
    gps_latest: datetime | None = None
    last_measurements: dict = field(default_factory=dict)

    def update(self, row, stamp, required):
        self.counts["received"] += 1
        self.first = stamp if self.first is None else min(stamp, self.first)
        self.last = stamp if self.last is None else max(stamp, self.last)
        if self.previous is not None:
            gap = (stamp - self.previous).total_seconds()
            if gap < 0:
                self.counts["receipt_out_of_order"] += 1
            self.max_receipt_gap = max(gap, self.max_receipt_gap)
        self.previous = stamp
        flags = row["flags"]
        if "duplicate_packet" in flags:
            self.counts["duplicate"] += 1
            return
        if "previous_boot_packet" in flags:
            self.counts["previous_boot"] += 1
            return
        values = row["measurements"]
        accepted_fields = set()
        for name in FIELDS.keys() & values.keys():
            minimum, maximum, _ = FIELDS[name]
            value = values[name]
            if (type(value) in (float, int) and minimum <= value <= maximum and math.isfinite(value)):
                accepted_fields.add(name)
        self.counts["usable_records"] += bool(accepted_fields)
        self.counts["required_complete"] += required <= accepted_fields
        if GPS_FIELDS <= accepted_fields:
            self.counts["gps_fixes"] += 1
            self.gps_latest = stamp if self.gps_latest is None else max(stamp, self.gps_latest)
        for name in sorted(accepted_fields):
            self.measurement_counts[name] += 1
            self.last_measurements[name] = max(self.last_measurements.get(name, stamp), stamp)

    def report(self, as_of, required):
        ages = {name: None if name not in self.last_measurements else
                (as_of - self.last_measurements[name]).total_seconds() for name in required}
        return {"counts": dict(self.counts), "measurement_counts": dict(self.measurement_counts),
                "last_measurement_age_seconds": ages, "max_receipt_gap_seconds": self.max_receipt_gap,
                "first_receipt_utc": self.first.isoformat(), "last_receipt_utc": self.last.isoformat(),
                "last_gps_receipt_utc": self.gps_latest.isoformat() if self.gps_latest else None}


def coverage_report(path, *, required=("air_temperature_c", "relative_humidity_pct"), as_of=None):
    required = set(required)
    if not required or not required <= FIELDS.keys():
        raise ValueError("Required measurements must name parsed sensor fields")
    cutoff = _receive_time(as_of).astimezone(timezone.utc) if as_of is not None else None
    newest = cutoff
    nodes = defaultdict(ReceiptSummary)
    ignored = 0
    with Path(path).open("rb") as file:
        for line in file:
            row = decode_json(line)
            stamp = _receive_time(row["received_at"]).astimezone(timezone.utc)
            if cutoff is not None and stamp > cutoff:
                ignored += 1
                continue
            newest = stamp if newest is None else max(newest, stamp)
            nodes[row.get("device_id") or "unidentified"].update(row, stamp, required)
    return {"receipt_cutoff_utc": newest.isoformat() if newest else None,
            "records_after_cutoff": ignored, "required": sorted(required),
            "devices": {name: record.report(newest, sorted(required)) for name, record in nodes.items()}}
