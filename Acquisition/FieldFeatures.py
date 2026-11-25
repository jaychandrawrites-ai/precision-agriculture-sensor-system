"""Transform aggregated Arduino readings into the trained six-feature contract."""
from dataclasses import asdict

import numpy as np
import pandas as pd

from config import FEATURES
from Data.AZMET import check_hourly_configuration
from Data.WeatherPhysics import CausalReferenceET
from Physics.Crop.CottonHeatStress import advance_heat_stress, canopy_temperature
from Physics.Evapotranspiration.ActualET import actual_et_from_moisture
from Physics.Soil.WaterStress import depletion_from_moisture, stress_coefficient


def field_features(hourly, assumptions, reference_et):
    check_hourly_configuration(reference_et, assumptions)
    reference = CausalReferenceET(reference_et)
    records, heat = [], None
    flags = hourly.flags.copy()
    flags["heat_restart"] = False
    flags["reference_et_unavailable"] = False
    for timestamp, row in hourly.values.iterrows():
        air, rh, lwi, theta = (row[name] for name in
                              ("air_temperature_c", "relative_humidity_pct", "lwi", "soil_vwc"))
        eto = reference.advance(timestamp, air)
        flags.loc[timestamp, "reference_et_unavailable"] = not np.isfinite(eto)
        eta = stress = np.nan
        if np.isfinite([air, theta, eto]).all():
            if heat is None:
                heat = 0.0
                flags.loc[timestamp, "heat_restart"] = True
            eta = actual_et_from_moisture(eto, theta, assumptions.soil)
            ks = stress_coefficient(depletion_from_moisture(theta, assumptions.soil), assumptions.soil)
            heat = advance_heat_stress(heat, canopy_temperature(air, ks, assumptions.crop),
                                       1.0, assumptions.crop)
            stress = heat
        else:
            heat = None
        records.append((air, rh, lwi, eto, eta, stress))
    features = pd.DataFrame(records, index=hourly.values.index, columns=FEATURES)
    provenance = {**hourly.provenance, "assumptions": assumptions.metadata(),
                  "reference_et": asdict(reference_et),
                  "feature_origin": dict(zip(FEATURES, (
                      "measured_hourly_mean", "measured_hourly_mean", "calibrated_hourly_mean",
                      "estimated_hargreaves", "estimated_from_measured_soil_moisture",
                      "simulated_cotton_heat_index",
                  )))}
    return features, flags, provenance
