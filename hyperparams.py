"""Shared physical feature transformations for training and inference."""
from Physics.Evapotranspiration.FAO56 import saturation_pressure, actual_pressure, reference_et_daily
from Physics.Evapotranspiration.Hargreaves import causal_interval_proxy
from Physics.Evapotranspiration.ActualET import actual_et_from_moisture, water_balance_et
from Physics.Soil.WaterStress import depletion_from_moisture, stress_coefficient
from Physics.Crop.CottonHeatStress import canopy_temperature, advance_heat_stress

def normalize_leaf_wetness(raw_mv, leaf):
    if leaf.dry_mv is None or leaf.wet_mv is None:
        raise ValueError("Leaf sensor calibration required")
    return min(1.0, max(0.0, (raw_mv-leaf.dry_mv)/(leaf.wet_mv-leaf.dry_mv)))

def sensor_features(temperature_c, humidity_pct, theta, raw_leaf_mv, eto_mm,
                    previous_heat_stress, assumptions):
    if not 0 <= humidity_pct <= 100:
        raise ValueError("Invalid humidity")
    depletion = depletion_from_moisture(theta, assumptions.soil)
    ks = stress_coefficient(depletion, assumptions.soil)
    eta = actual_et_from_moisture(eto_mm, theta, assumptions.soil)
    canopy = canopy_temperature(temperature_c, ks, assumptions.crop)
    heat = advance_heat_stress(previous_heat_stress, canopy,
                               assumptions.site.interval_hours, assumptions.crop)
    lwi = normalize_leaf_wetness(raw_leaf_mv, assumptions.leaf)
    return (temperature_c, humidity_pct, lwi, eto_mm, eta, heat)
