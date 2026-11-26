"""Matched comparison of two estimated hourly reference ET models."""
from dataclasses import asdict, dataclass
from datetime import timedelta
import math

import numpy as np
import pandas as pd

from Data.WeatherPhysics import CausalReferenceET
from Physics.Evapotranspiration.FAO56 import (actual_pressure, atmospheric_pressure,
                                             net_radiation_hourly, reference_et_hourly, wind_at_two_metres)
from Physics.Evapotranspiration.SolarRadiation import clear_sky_radiation, extraterrestrial_interval


@dataclass(frozen=True)
class RadiationAssumptions:
    elevation_m: float
    wind_height_m: float
    dark_cloud_ratio: float
    day_soil_flux_fraction: float = .1
    night_soil_flux_fraction: float = .5

    def __post_init__(self):
        if (not all(math.isfinite(x) for x in vars(self).values()) or self.wind_height_m <= .1
                or not .3 <= self.dark_cloud_ratio <= 1
                or not all(0 <= x <= 1 for x in (self.day_soil_flux_fraction,
                                                self.night_soil_flux_fraction))):
            raise ValueError("Explicit finite radiation, cloud and measurement assumptions are required")


def hourly_extraterrestrial(end, reference):
    solar_end = end + timedelta(minutes=reference.solar_offset_minutes)
    solar_start = solar_end - timedelta(hours=1)
    midnight = solar_start.normalize() + timedelta(days=1)
    boundaries = (solar_start, midnight, solar_end) if midnight < solar_end else (solar_start, solar_end)
    result = 0.0
    for left, right in zip(boundaries, boundaries[1:]):
        date = left.normalize()
        start, stop = ((t - date).total_seconds() / 3600 for t in (left, right))
        result += extraterrestrial_interval(reference.latitude_deg, left.dayofyear, start, stop)
    return result


def compare_et(weather, reference, radiation):
    """Compare estimates on matched observations without changing training features."""
    clock = CausalReferenceET(reference)
    pressure = atmospheric_pressure(radiation.elevation_m)
    estimates = []
    for timestamp, row in weather.values.iterrows():
        temperature = row["air_temperature_c"]
        humidity = row["relative_humidity_pct"]
        measured_wind = row["wind_m_s"]
        solar = row["shortwave_w_m2"] * 3600 / 1e6
        temperature_only = clock.advance(timestamp, temperature)
        pm = np.nan
        if np.isfinite([temperature, humidity, measured_wind, solar]).all():
            sky = clear_sky_radiation(hourly_extraterrestrial(timestamp, reference), radiation.elevation_m)
            wind = wind_at_two_metres(measured_wind, radiation.wind_height_m)
            vapor = actual_pressure(temperature, humidity)
            net = net_radiation_hourly(temperature, solar, sky, vapor,
                                       dark_cloud_ratio=radiation.dark_cloud_ratio)
            soil_fraction = (radiation.day_soil_flux_fraction if sky > 0
                             else radiation.night_soil_flux_fraction)
            pm = reference_et_hourly(temperature, net, soil_fraction * net, wind, vapor, pressure)
        estimates.append((temperature_only, pm))
    outputs = pd.DataFrame(estimates, index=weather.values.index,
                           columns=["hargreaves_mm_per_interval", "fao56_mm_per_interval"])
    matched = outputs.dropna()
    delta = matched.iloc[:, 1] - matched.iloc[:, 0]
    return outputs, {
        "interpretation": "estimate comparison, not validation against measured ET",
        "matched_hours": len(delta), "missing_hours": len(outputs) - len(delta),
        "fao56_minus_hargreaves_bias_mm": float(delta.mean()) if len(delta) else None,
        "difference_rmse_mm": float(np.sqrt(np.mean(delta**2))) if len(delta) else None,
        "radiation_assumptions": asdict(radiation), "reference_assumptions": asdict(reference),
        "source": weather.provenance,
    }
