"""Live receiver accounting and safe, derived calibration diagnostics."""
from collections import Counter, deque
from datetime import datetime

from Acquisition.Calibration import LeafCalibration, SoilCalibration
from Acquisition.MappedPackets import FIELDS, GPS_FIELDS
from Acquisition.PacketParser import FIELDS as LEGACY_FIELDS


def units_for(schema):
    if schema is not None:
        return {field: {"source": value.unit, "normalized": next(iter(FIELDS[field][2]))}
                for field, value in schema.fields.items()}
    return {field: {"source": unit or "pH", "normalized": next(iter(FIELDS[field][2]))}
            for section in LEGACY_FIELDS.values() for field, unit, _, _ in section.values()}


def calibration_readout(configuration, measurements):
    report = {}
    for name, source, destination, factory in (
        ("leaf_calibration", "leaf_mv", "lwi", LeafCalibration),
        ("soil_calibration", "soil_moisture_pct", "soil_vwc", SoilCalibration),
    ):
        detail = {"state": "unavailable"}
        report[name] = detail
        if name not in configuration:
            continue
        try:
            calibration = factory(**configuration[name])
        except (TypeError, ValueError, AttributeError):
            detail["state"] = "invalid"
            continue
        detail["state"] = "available"
        if source not in measurements:
            continue
        detail["raw_field"] = source
        detail["raw_value"] = measurements[source]
        try:
            detail["calibrated_field"] = destination
            detail["calibrated_value"] = calibration.apply(measurements[source])
        except ValueError:
            detail["state"] = "outside_calibration_range"
    return report


class TraceDiagnostics:
    def __init__(self):
        self.successful = self.failed = self.duplicate = 0
        self.missing = Counter()
        self.device_counts = Counter()
        self.device_boots = {}
        self.boundaries = deque(maxlen=20)
        self.restart_count = 0
        self.gps_fixes = self.gps_invalid = 0
        self.last_error = None

    def error(self, *, message, timestamp: datetime, recoverable):
        self.last_error = {"message": message, "at": timestamp.isoformat(),
                           "recoverable": recoverable}

    def reading(self, parsed, received_at):
        flags = parsed["flags"]
        measurements = parsed["measurements"]
        device = parsed["device_id"]
        if device is not None:
            self.device_counts[device] += 1
        boot = parsed.get("boot_id")
        previous = self.device_boots.get(device)
        if boot is not None and device is not None and "previous_boot_packet" not in flags:
            self.device_boots[device] = boot
        if ("device_restart" in flags or
                (boot is not None and previous is not None and previous != boot
                 and "previous_boot_packet" not in flags)):
            self.restart_count += 1
            self.boundaries.append({"device_id": device, "boot_id": parsed.get("boot_id"),
                                    "at": received_at.isoformat()})
        if "duplicate_packet" in flags:
            self.duplicate += 1
            return
        if measurements:
            self.successful += 1
        else:
            self.failed += 1
        self.missing.update(flag.removeprefix("missing:") for flag in flags if flag.startswith("missing:"))
        if GPS_FIELDS <= measurements.keys():
            self.gps_fixes += 1
        elif any(flag.startswith(("invalid:gps", "incomplete:gps")) for flag in flags):
            self.gps_invalid += 1

    def snapshot(self):
        return {"successful_parses": self.successful, "parse_failures": self.failed,
                "duplicate_packets": self.duplicate, "missing_measurements": dict(self.missing),
                "device_packet_counts": dict(self.device_counts), "restart_count": self.restart_count,
                "recent_restart_boundaries": list(self.boundaries),
                "valid_gps_fixes": self.gps_fixes, "invalid_gps_reports": self.gps_invalid,
                "most_recent_error": self.last_error}
