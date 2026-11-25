"""Physical assumptions. Defaults are illustrative, not calibrated field measurements."""
from dataclasses import dataclass, asdict
import math

@dataclass(frozen=True)
class SoilAssumptions:
    field_capacity: float = 0.30
    wilting_point: float = 0.15
    root_depth_m: float = 0.6
    depletion_fraction: float = 0.55
    initial_depletion_fraction: float = 0.25
    crop_coefficient: float = 1.0

    def __post_init__(self):
        if not all(math.isfinite(value) for name, value in vars(self).items()
                   if name != "timezone" and value is not None):
            raise ValueError("Physical parameters must be finite")
        if not 0 <= self.wilting_point < self.field_capacity <= 1:
            raise ValueError("Invalid volumetric water limits")
        if self.root_depth_m <= 0 or not 0 < self.depletion_fraction < 1:
            raise ValueError("Invalid root zone")
        if not 0 <= self.initial_depletion_fraction <= 1 or self.crop_coefficient < 0:
            raise ValueError("Invalid initial depletion or crop coefficient")

@dataclass(frozen=True)
class SiteAssumptions:
    latitude_deg: float = 32.7
    elevation_m: float = 40.0
    interval_hours: float = 1.0
    timezone: str = "America/Phoenix"

    def __post_init__(self):
        if not all(math.isfinite(value) for name, value in vars(self).items()
                   if name != "timezone" and value is not None):
            raise ValueError("Physical parameters must be finite")
        if not -90 < self.latitude_deg < 90 or self.interval_hours <= 0:
            raise ValueError("Invalid site geometry or interval")

@dataclass(frozen=True)
class CropAssumptions:
    critical_temperature_c: float = 35.0
    canopy_stress_offset_c: float = 5.0
    heat_accumulation_per_hour: float = 1.0
    recovery_per_hour: float = 0.1

    def __post_init__(self):
        if not all(math.isfinite(value) for name, value in vars(self).items()
                   if name != "timezone" and value is not None):
            raise ValueError("Physical parameters must be finite")
        if min(self.canopy_stress_offset_c, self.heat_accumulation_per_hour,
               self.recovery_per_hour) < 0:
            raise ValueError("Negative crop model coefficient")

@dataclass(frozen=True)
class LeafAssumptions:
    dry_mv: float | None = None
    wet_mv: float | None = None
    storage_kg_m2: float = 0.05
    heat_capacity_j_m2_k: float = 1200.0
    absorptivity: float = 0.5
    emissivity: float = 0.96
    mass_transfer_kg_m2_s_kpa: float = 0.00001
    convection_w_m2_k: float = 12.0

    def __post_init__(self):
        if not all(math.isfinite(value) for name, value in vars(self).items()
                   if name != "timezone" and value is not None):
            raise ValueError("Physical parameters must be finite")
        if self.storage_kg_m2 <= 0 or self.heat_capacity_j_m2_k <= 0:
            raise ValueError("Invalid leaf storage")
        if not 0 <= self.absorptivity <= 1 or not 0 <= self.emissivity <= 1:
            raise ValueError("Invalid optical coefficient")
        if min(self.mass_transfer_kg_m2_s_kpa, self.convection_w_m2_k) < 0:
            raise ValueError("Negative transfer coefficient")
        if (self.dry_mv is None) != (self.wet_mv is None):
            raise ValueError("Supply both calibration endpoints")
        if self.dry_mv is not None and self.dry_mv == self.wet_mv:
            raise ValueError("Calibration endpoints must differ")

@dataclass(frozen=True)
class Assumptions:
    soil: SoilAssumptions = SoilAssumptions()
    site: SiteAssumptions = SiteAssumptions()
    crop: CropAssumptions = CropAssumptions()
    leaf: LeafAssumptions = LeafAssumptions()

    @classmethod
    def from_dict(cls, record):
        unknown = set(record) - {"soil", "site", "crop", "leaf"}
        if unknown:
            raise ValueError(f"Unknown assumption sections: {sorted(unknown)}")
        return cls(soil=SoilAssumptions(**record.get("soil", {})),
                   site=SiteAssumptions(**record.get("site", {})),
                   crop=CropAssumptions(**record.get("crop", {})),
                   leaf=LeafAssumptions(**record.get("leaf", {})))

    def metadata(self):
        return {"status": "illustrative_uncalibrated", **asdict(self)}
