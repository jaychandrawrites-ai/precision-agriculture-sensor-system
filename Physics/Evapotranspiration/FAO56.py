"""Daily FAO-56 Penman-Monteith reference evapotranspiration."""
import math

def saturation_pressure(temperature_c):
    if not -80 <= temperature_c <= 80:
        raise ValueError("Temperature outside supported range")
    return 0.6108*math.exp(17.27*temperature_c/(temperature_c+237.3))

def actual_pressure(temperature_c, relative_humidity_pct):
    if not 0 <= relative_humidity_pct <= 100:
        raise ValueError("Relative humidity must be a percentage")
    return saturation_pressure(temperature_c)*relative_humidity_pct/100

def saturation_slope(temperature_c):
    return 4098*saturation_pressure(temperature_c)/(temperature_c+237.3)**2

def atmospheric_pressure(elevation_m):
    base = (293-0.0065*elevation_m)/293
    if base <= 0:
        raise ValueError("Invalid elevation")
    return 101.3*base**5.26

def wind_at_two_metres(speed, measurement_height_m):
    if speed < 0 or measurement_height_m <= 0.1:
        raise ValueError("Invalid wind measurement")
    return speed*4.87/math.log(67.8*measurement_height_m-5.42)

def net_radiation_daily(solar_mj, clear_sky_mj, tmin_c, tmax_c, vapor_kpa, albedo=0.23):
    if solar_mj < 0 or clear_sky_mj <= 0 or vapor_kpa < 0 or not 0 <= albedo <= 1:
        raise ValueError("Invalid radiation input")
    if tmax_c < tmin_c:
        raise ValueError("Temperature bounds reversed")
    shortwave = (1-albedo)*solar_mj
    cloud_factor = 1.35*min(1, max(0.3, solar_mj/clear_sky_mj))-0.35
    longwave = 4.903e-9*((tmin_c+273.16)**4+(tmax_c+273.16)**4)/2
    longwave *= (0.34-0.14*math.sqrt(vapor_kpa))*cloud_factor
    return shortwave-longwave

def reference_et_daily(temperature_c, net_radiation_mj, soil_heat_mj,
                       wind_2m, saturation_kpa, actual_kpa, pressure_kpa):
    if wind_2m < 0 or pressure_kpa <= 0 or min(saturation_kpa, actual_kpa) < 0:
        raise ValueError("Invalid meteorological input")
    delta = saturation_slope(temperature_c)
    gamma = 0.000665*pressure_kpa
    deficit = max(0, saturation_kpa-actual_kpa)
    radiation = 0.408*delta*(net_radiation_mj-soil_heat_mj)
    aerodynamic = gamma*900/(temperature_c+273)*wind_2m*deficit
    denominator = delta+gamma*(1+0.34*wind_2m)
    return max(0, (radiation+aerodynamic)/denominator)


def net_radiation_hourly(temperature_c, solar_mj, clear_sky_mj, actual_kpa, *,
                        dark_cloud_ratio=None, albedo=0.23):
    """Net hourly radiation with explicit cloud information for dark intervals.

    Radiation inputs and output are in MJ m⁻² per hour. The dark cloud ratio
    is an explicitly supplied preceding or assumed shortwave/clear-sky ratio.
    """
    if (not all(math.isfinite(x) for x in (temperature_c, solar_mj, clear_sky_mj, actual_kpa, albedo))
            or not -80 <= temperature_c <= 80 or min(solar_mj, clear_sky_mj, actual_kpa) < 0
            or not 0 <= albedo <= 1):
        raise ValueError("Invalid hourly net radiation inputs")
    if clear_sky_mj == 0:
        if dark_cloud_ratio is None or not 0.3 <= dark_cloud_ratio <= 1:
            raise ValueError("A dark interval requires an explicit cloud ratio between 0.3 and 1")
        ratio = dark_cloud_ratio
    else:
        ratio = min(1., max(.3, solar_mj / clear_sky_mj))
    outgoing = 2.043e-10 * (temperature_c + 273.16)**4
    outgoing *= (0.34 - 0.14 * math.sqrt(actual_kpa)) * (1.35 * ratio - .35)
    return (1 - albedo) * solar_mj - outgoing


def reference_et_hourly(temperature_c, net_radiation_mj, soil_heat_mj, wind_2m, actual_kpa,
                        pressure_kpa, *, allow_negative=False):
    """FAO-56 Eq. 53 grass-reference ET in mm per hour.

    Rn/G are MJ m⁻² per hour, wind is m s⁻¹ at 2 m, and pressures are kPa.
    Signed output is available for comparing nighttime modeled fluxes.
    """
    if (not all(math.isfinite(x) for x in (temperature_c, net_radiation_mj, soil_heat_mj,
                                          wind_2m, actual_kpa, pressure_kpa))
            or min(wind_2m, actual_kpa) < 0 or pressure_kpa <= 0):
        raise ValueError("Invalid hourly reference ET inputs")
    saturation = saturation_pressure(temperature_c)
    delta = saturation_slope(temperature_c)
    gamma = .000665 * pressure_kpa
    deficit = max(0., saturation - actual_kpa)
    radiation = .408 * delta * (net_radiation_mj - soil_heat_mj)
    aerodynamic = gamma * 37 * wind_2m * deficit / (temperature_c + 273)
    et = (radiation + aerodynamic) / (delta + gamma * (1 + .34 * wind_2m))
    return et if allow_negative else max(0., et)
