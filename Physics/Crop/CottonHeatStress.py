"""Proposed heat stress index; not a validated AZMET cotton index."""
import math

def canopy_temperature(air_temperature_c, water_stress_coefficient, crop):
    if not 0 <= water_stress_coefficient <= 1:
        raise ValueError("Invalid stress coefficient")
    return air_temperature_c+crop.canopy_stress_offset_c*(1-water_stress_coefficient)

def advance_heat_stress(previous, canopy_c, interval_hours, crop):
    if previous < 0 or interval_hours <= 0:
        raise ValueError("Invalid heat stress state")
    forcing = crop.heat_accumulation_per_hour*max(0, canopy_c-crop.critical_temperature_c)
    recovery = crop.recovery_per_hour
    if recovery == 0:
        return previous+forcing*interval_hours
    decay = math.exp(-recovery*interval_hours)
    return previous*decay+forcing*(-math.expm1(-recovery*interval_hours))/recovery
