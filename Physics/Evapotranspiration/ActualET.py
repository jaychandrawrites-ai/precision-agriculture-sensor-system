"""Crop coefficient and water balance estimates of actual ET."""
from Physics.Soil.WaterStress import depletion_from_moisture, stress_coefficient

def actual_et(eto_mm, depletion_mm, soil):
    if eto_mm < 0:
        raise ValueError("Negative reference ET")
    return eto_mm*soil.crop_coefficient*stress_coefficient(depletion_mm, soil)

def actual_et_from_moisture(eto_mm, theta, soil):
    return actual_et(eto_mm, depletion_from_moisture(theta, soil), soil)

def water_balance_et(theta_before, theta_after, root_depth_m,
                     precipitation_mm=0, irrigation_mm=0, runoff_mm=0, drainage_mm=0):
    """Signed interval estimate; negative values indicate unresolved balance."""
    if not 0 <= theta_before <= 1 or not 0 <= theta_after <= 1 or root_depth_m <= 0:
        raise ValueError("Invalid root-zone measurements")
    if min(precipitation_mm, irrigation_mm, runoff_mm, drainage_mm) < 0:
        raise ValueError("Water flux magnitudes must be nonnegative")
    storage_loss = 1000*root_depth_m*(theta_before-theta_after)
    return precipitation_mm+irrigation_mm-runoff_mm-drainage_mm+storage_loss
