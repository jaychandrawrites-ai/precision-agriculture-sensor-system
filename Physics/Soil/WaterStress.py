"""FAO root-zone depletion and stress coefficient."""
from assumptions import SoilAssumptions

def total_available_water(soil: SoilAssumptions):
    return 1000*soil.root_depth_m*(soil.field_capacity-soil.wilting_point)

def readily_available_water(soil):
    return soil.depletion_fraction*total_available_water(soil)

def depletion_from_moisture(theta, soil):
    if not 0 <= theta <= 1:
        raise ValueError("Moisture must be volumetric fraction")
    raw = 1000*soil.root_depth_m*(soil.field_capacity-theta)
    return min(total_available_water(soil), max(0, raw))

def stress_coefficient(depletion_mm, soil):
    taw = total_available_water(soil)
    raw = readily_available_water(soil)
    depletion_mm = min(taw, max(0, depletion_mm))
    if depletion_mm <= raw:
        return 1.0
    return (taw-depletion_mm)/(taw-raw)
