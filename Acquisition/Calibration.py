"""Conversions for supplied calibration data."""
from dataclasses import dataclass
import math

from hyperparams import normalize_leaf_wetness


def leaf_wetness(raw_mv, calibration):
    if calibration.dry_mv is None or not all(map(
        math.isfinite, (raw_mv, calibration.dry_mv, calibration.wet_mv)
    )):
        raise ValueError("Supply finite raw and measured endpoint voltages")
    return normalize_leaf_wetness(raw_mv, calibration)




@dataclass(frozen=True)
class LeafCalibration:
    """Measured dry and wet PHYTOS31 endpoints with their provenance."""
    dry_mv: float
    wet_mv: float
    source: str

    def __post_init__(self):
        if (not all(math.isfinite(v) and v >= 0 for v in (self.dry_mv, self.wet_mv))
                or self.dry_mv == self.wet_mv or not self.source.strip()):
            raise ValueError("Supply distinct measured endpoint voltages and their source")

    def apply(self, raw_mv):
        return leaf_wetness(raw_mv, self)


@dataclass(frozen=True)
class SoilCalibration:
    """Convert a documented VWC percent or apply a supplied two-point calibration."""
    source: str
    points: tuple[tuple[float, float], tuple[float, float]] | None = None

    def __post_init__(self):
        if not self.source.strip():
            raise ValueError("Document the soil volumetric conversion source")
        if self.points is not None:
            (x1, y1), (x2, y2) = self.points
            if not 0 <= x1 < x2 <= 100 or not 0 <= y1 < y2 <= 1:
                raise ValueError("Expected increasing pairs of percent reading and volumetric fraction")

    def apply(self, raw_pct):
        if not 0 <= raw_pct <= 100:
            raise ValueError("Invalid soil moisture percent")
        if self.points is None:
            return raw_pct / 100
        (x1, y1), (x2, y2) = self.points
        if not x1 <= raw_pct <= x2:
            raise ValueError("Soil reading outside calibrated range")
        return y1 + (raw_pct - x1) * (y2 - y1) / (x2 - x1)
