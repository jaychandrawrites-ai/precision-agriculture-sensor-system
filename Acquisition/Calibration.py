"""Conversions for supplied calibration data."""
import math

from hyperparams import normalize_leaf_wetness


def leaf_wetness(raw_mv, calibration):
    if calibration.dry_mv is None or not all(map(
        math.isfinite, (raw_mv, calibration.dry_mv, calibration.wet_mv)
    )):
        raise ValueError("Supply finite raw and measured endpoint voltages")
    return normalize_leaf_wetness(raw_mv, calibration)
