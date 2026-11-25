"""Temperature-only ET proxy with causal history requirements."""
import math
from .SolarRadiation import extraterrestrial_daily, extraterrestrial_interval

def reference_et_daily(tmean_c, tmin_c, tmax_c, radiation_mj):
    if not tmin_c <= tmean_c <= tmax_c or radiation_mj < 0:
        raise ValueError("Invalid temperature statistics or radiation")
    equivalent_mm = 0.408*radiation_mj
    return max(0, 0.0023*(tmean_c+17.8)*math.sqrt(tmax_c-tmin_c)*equivalent_mm)

def causal_interval_proxy(temperatures, latitude_deg, day_of_year,
                          solar_hour_start, solar_hour_end, minimum_samples=24):
    """Trailing temperature history proxy, not measured hourly FAO-56 ET."""
    values = list(temperatures)
    if len(values) < minimum_samples or minimum_samples < 2:
        raise ValueError("Insufficient causal temperature history")
    values = values[-minimum_samples:]
    if not all(math.isfinite(x) for x in values):
        raise ValueError("Missing temperature history")
    daily_radiation = extraterrestrial_daily(latitude_deg, day_of_year)
    interval_radiation = extraterrestrial_interval(latitude_deg, day_of_year,
                                                  solar_hour_start, solar_hour_end)
    daily = reference_et_daily(sum(values)/len(values), min(values), max(values), daily_radiation)
    return daily*interval_radiation/daily_radiation if daily_radiation else 0.0
