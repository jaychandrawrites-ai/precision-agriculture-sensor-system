"""A normalized reading from a serial block or a mapped receiver packet."""
from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class SensorData:
    received_at: datetime
    measurements: dict[str, float]
    flags: tuple[str, ...]
    observed_at: datetime | None = None
    device_id: str | None = None
    packet_id: str | None = None
    boot_id: str | None = None
