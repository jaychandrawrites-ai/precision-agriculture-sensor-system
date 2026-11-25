"""A reading from one text block emitted by the checked-in firmware."""
from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class SensorData:
    received_at: datetime
    measurements: dict[str, float]
    flags: tuple[str, ...]
