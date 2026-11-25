"""Configurable JSON receiver envelope for a caller-verified packet protocol."""
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import math

from Acquisition.SensorData import SensorData


# Canonical ranges are acquisition checks, not calibration endpoints.
FIELDS = {
    "air_temperature_c": (-60, 65, {"C": (1, 0), "F": (5 / 9, -32 * 5 / 9)}),
    "relative_humidity_pct": (0, 100, {"%": (1, 0), "fraction": (100, 0)}),
    "soil_temperature_c": (-60, 85, {"C": (1, 0), "F": (5 / 9, -32 * 5 / 9)}),
    "soil_moisture_pct": (0, 100, {"%": (1, 0), "fraction": (100, 0)}),
    "soil_ec_us_cm": (0, 65535, {"uS/cm": (1, 0), "mS/cm": (1000, 0)}),
    "soil_ph": (0, 14, {"pH": (1, 0)}),
    "leaf_mv": (0, math.inf, {"mV": (1, 0), "V": (1000, 0)}),
    "latitude_deg": (-90, 90, {"deg": (1, 0)}),
    "longitude_deg": (-180, 180, {"deg": (1, 0)}),
}
GPS_FIELDS = {"latitude_deg", "longitude_deg"}


@dataclass(frozen=True)
class PacketField:
    key: str
    unit: str


@dataclass(frozen=True)
class PacketSchema:
    fields: dict[str, PacketField]
    packet_id_key: str | None = None
    device_id_key: str | None = None
    observed_at_key: str | None = None
    boot_id_key: str | None = None
    gps_fix_key: str | None = None
    gps_valid_values: tuple = (True,)
    duplicate_cache: int = 4096
    label: str = "caller_configured_json"
    constants: dict = field(default_factory=dict)

    def __post_init__(self):
        if not self.fields or not self.fields.keys() <= FIELDS.keys():
            raise ValueError("Packet mapping has unknown or missing field definitions")
        gps = self.fields.keys() & GPS_FIELDS
        if gps and gps != GPS_FIELDS:
            raise ValueError("Map latitude and longitude together")
        keys = [self.packet_id_key, self.device_id_key, self.observed_at_key,
                self.boot_id_key, self.gps_fix_key]
        for name, spec in self.fields.items():
            if spec.unit not in FIELDS[name][2]:
                raise ValueError(f"Unsupported packet unit for {name}")
            keys.append(spec.key)
        keys = [key for key in keys if key is not None]
        if (not all(isinstance(key, str) and key for key in keys)
                or len(set(keys)) != len(keys) or not isinstance(self.duplicate_cache, int)
                or self.duplicate_cache < 1 or not self.label.strip()):
            raise ValueError("Invalid packet keys, label, or deduplication cache size")
        if self.boot_id_key is not None and self.packet_id_key is None:
            raise ValueError("Boot identity requires a packet ID")
        if self.gps_fix_key is not None and gps != GPS_FIELDS:
            raise ValueError("A fix flag requires mapped GPS coordinates")

    @classmethod
    def from_dict(cls, record):
        return cls(**{**record, "fields": {
            name: PacketField(**spec) for name, spec in record["fields"].items()
        }})


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON key")
        result[key] = value
    return result


def _identity(message, key):
    if key is None:
        return None
    value = message.get(key)
    if isinstance(value, bool) or not isinstance(value, (str, int)) or not str(value).strip():
        return None
    return str(value)


class MappedPacketParser:
    def __init__(self, schema):
        self.schema = schema
        self.seen = OrderedDict()

    def finish(self, received_at):
        return None

    def feed(self, text, received_at):
        try:
            message = json.loads(text, object_pairs_hook=_unique_object)
        except (ValueError, RecursionError):
            return SensorData(received_at, {}, ("invalid_json",))
        if not isinstance(message, dict):
            return SensorData(received_at, {}, ("invalid_envelope",))
        for key, expected in self.schema.constants.items():
            if key not in message or type(message[key]) is not type(expected) or message[key] != expected:
                return SensorData(received_at, {}, ("unexpected_envelope",))
        flags, measurements = [], {}
        observed = None
        if self.schema.observed_at_key is not None:
            stamp = message.get(self.schema.observed_at_key)
            try:
                observed = datetime.fromisoformat(stamp)
                if observed.utcoffset() is None:
                    raise ValueError("Offset required")
                observed = observed.astimezone(timezone.utc)
            except (TypeError, ValueError, OverflowError):
                observed = None
                flags.append("invalid:observed_at")
        device = _identity(message, self.schema.device_id_key)
        identifier = _identity(message, self.schema.packet_id_key)
        boot = _identity(message, self.schema.boot_id_key)
        pairs = (("device_id", self.schema.device_id_key, device),
                 ("packet_id", self.schema.packet_id_key, identifier),
                 ("boot_id", self.schema.boot_id_key, boot))
        for name, key, value in pairs:
            if key is not None and value is None:
                flags.append(f"missing:{name}")
        key = (device, boot, identifier)
        if identifier is not None and not any(config and value is None for _, config, value in pairs):
            if key in self.seen:
                self.seen.move_to_end(key)
                return SensorData(received_at, {}, (*flags, "duplicate_packet"), observed, device, identifier)
            self.seen[key] = None
            if len(self.seen) > self.schema.duplicate_cache:
                self.seen.popitem(last=False)
        for name, spec in self.schema.fields.items():
            raw = message.get(spec.key)
            if raw is None:
                flags.append(f"missing:{name}")
                continue
            minimum, maximum, units = FIELDS[name]
            scale, offset = units[spec.unit]
            try:
                value = float(raw) * scale + offset
            except (TypeError, ValueError, OverflowError):
                value = math.nan
            if isinstance(raw, bool) or not math.isfinite(value) or not minimum <= value <= maximum:
                flags.append(f"invalid:{name}")
            else:
                measurements[name] = value
        if self.schema.fields.keys() & GPS_FIELDS:
            fix = message.get(self.schema.gps_fix_key)
            if self.schema.gps_fix_key is not None and not any(
                type(fix) is type(value) and fix == value for value in self.schema.gps_valid_values
            ):
                flags.append("invalid:gps_fix")
                for name in GPS_FIELDS:
                    measurements.pop(name, None)
            elif not GPS_FIELDS <= measurements.keys():
                flags.append("incomplete:gps")
                for name in GPS_FIELDS:
                    measurements.pop(name, None)
        return SensorData(received_at, measurements, tuple(flags), observed, device, identifier)
