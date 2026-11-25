"""Historical hourly feature generation with explicit model and scenario provenance."""
from dataclasses import asdict

import numpy as np
import pandas as pd

from config import FEATURES
from Data.AZMET import check_hourly_configuration
from Data.WeatherPhysics import CausalReferenceET
from Physics.Crop.CottonHeatStress import advance_heat_stress, canopy_temperature
from Physics.Leaf.EnergyBalance import LeafWeather, approximate_longwave
from Physics.Leaf.LeafWetness import LeafState, advance_leaf
from Physics.Soil.SoilWaterBalance import SoilBucket
from Physics.Soil.WaterStress import stress_coefficient


def historical_features(weather, assumptions, reference_et, *, rain_interception_fraction,
                        scenario_irrigation_mm=None, initial_leaf_wetness=0.0):
    check_hourly_configuration(reference_et, assumptions)
    if not 0 <= rain_interception_fraction <= 1 or not 0 <= initial_leaf_wetness <= 1:
        raise ValueError("Leaf interception and initial wetness must be fractions")
    mapped_irrigation = "irrigation_mm" in weather.values
    if mapped_irrigation == (scenario_irrigation_mm is not None):
        raise ValueError("Choose mapped irrigation or an explicit constant irrigation scenario")
    if not mapped_irrigation and (
        not np.isfinite(scenario_irrigation_mm) or scenario_irrigation_mm < 0
    ):
        raise ValueError("Invalid irrigation scenario")
    reference = CausalReferenceET(reference_et)
    leaf, bucket, heat = None, None, 0.0
    records = []
    flags = weather.flags.copy()
    flags["leaf_restart"] = False
    flags["soil_restart"] = False
    flags["reference_et_unavailable"] = False

    for timestamp, row in weather.values.iterrows():
        air, rh = row["air_temperature_c"], row["relative_humidity_pct"]
        eto = reference.advance(timestamp, air)
        if not np.isfinite(eto):
            flags.loc[timestamp, "reference_et_unavailable"] = True
        wind, solar, rain = row["wind_m_s"], row["shortwave_w_m2"], row["precipitation_mm"]
        irrigation = row["irrigation_mm"] if mapped_irrigation else scenario_irrigation_mm
        lwi = eta = stress = np.nan
        if np.isfinite([air, rh, wind, solar, rain]).all():
            if leaf is None:
                flags.loc[timestamp, "leaf_restart"] = True
                leaf = LeafState(air, initial_leaf_wetness)
            forcing = LeafWeather(air, rh, solar, approximate_longwave(air, rh), wind,
                                  rain * rain_interception_fraction / 3600)
            leaf, _ = advance_leaf(leaf, forcing, assumptions.leaf)
            lwi = leaf.wetness
        else:
            leaf = None
        if np.isfinite([eto, rain, irrigation, air]).all():
            if bucket is None:
                flags.loc[timestamp, "soil_restart"] = True
                bucket = SoilBucket(assumptions.soil)
            step = bucket.step(eto, precipitation_mm=rain, irrigation_mm=irrigation)
            eta = step.actual_et_mm
            ks = stress_coefficient(step.final_depletion_mm, assumptions.soil)
            heat = advance_heat_stress(heat, canopy_temperature(air, ks, assumptions.crop),
                                       1.0, assumptions.crop)
            stress = heat
        else:
            bucket, heat = None, 0.0
        records.append((air, rh, lwi, eto, eta, stress))

    features = pd.DataFrame(records, index=weather.values.index, columns=FEATURES)
    provenance = {**weather.provenance,
                  "assumptions": assumptions.metadata(),
                  "reference_et": asdict(reference_et),
                  "scenario": {"rain_interception_fraction": rain_interception_fraction,
                               "initial_leaf_wetness": initial_leaf_wetness,
                               "constant_irrigation_mm_per_hour": scenario_irrigation_mm},
                  "feature_origin": dict(zip(FEATURES, (
                      "source_or_flagged_causal_fill", "source_or_flagged_causal_fill",
                      "simulated_leaf_ode", "estimated_hargreaves",
                      "simulated_soil_bucket", "simulated_cotton_heat_index",
                  )))}
    return features, flags, provenance
