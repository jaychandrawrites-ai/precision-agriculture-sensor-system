"""Describe calibration replicates and propagate their stated uncertainty."""
from dataclasses import asdict, dataclass
from math import hypot, isfinite, sqrt
from statistics import fmean, stdev

from Acquisition.Calibration import LeafCalibration


@dataclass(frozen=True)
class Endpoint:
    count: int
    mean_mv: float
    repeatability_sd_mv: float
    standard_error_mv: float


def _endpoint(replicates):
    readings = tuple(replicates)
    if len(readings) < 2 or any(type(value) not in (int, float) or not isfinite(value) or value < 0
                                for value in readings):
        raise ValueError("Supply at least two finite nonnegative measured voltages per endpoint")
    spread = stdev(readings)
    return Endpoint(len(readings), fmean(readings), spread, spread / sqrt(len(readings)))


@dataclass(frozen=True)
class LeafCalibrationReport:
    calibration: LeafCalibration
    dry: Endpoint
    wet: Endpoint

    def estimate(self, raw_mv, *, voltage_standard_uncertainty_mv):
        """Propagate independent reference and input uncertainty in the linear interior."""
        if (type(raw_mv) not in (int, float) or not isfinite(raw_mv) or raw_mv < 0 or
            type(voltage_standard_uncertainty_mv) not in (int, float) or
            not isfinite(voltage_standard_uncertainty_mv) or voltage_standard_uncertainty_mv < 0):
            raise ValueError("Supply finite nonnegative measured input and standard uncertainty")
        dry, wet = self.calibration.dry_mv, self.calibration.wet_mv
        span = wet - dry
        fraction = (raw_mv - dry) / span
        valid = 0 < fraction < 1
        uncertainty = (hypot(voltage_standard_uncertainty_mv,
                             (1 - fraction) * self.dry.standard_error_mv,
                             fraction * self.wet.standard_error_mv) / abs(span) if valid else None)
        return {"leaf_wetness": self.calibration.apply(raw_mv),
                "unclipped_fraction": fraction, "inside_calibration_span": valid,
                "linearized_standard_uncertainty": uncertainty,
                "uncertainty_model": "independent replicate means and supplied input standard uncertainty"}

    def as_record(self):
        return {"leaf_calibration": asdict(self.calibration),
                "endpoint_repeatability": {"dry": asdict(self.dry), "wet": asdict(self.wet)}}


def summarize_leaf_calibration(*, dry_mv, wet_mv, source):
    """Summarize caller-measured replicates. Reversed sensor polarity is supported."""
    dry = _endpoint(dry_mv)
    wet = _endpoint(wet_mv)
    calibration = LeafCalibration(dry.mean_mv, wet.mean_mv, source)
    return LeafCalibrationReport(calibration, dry, wet)
