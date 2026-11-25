"""FAO astronomical radiation, in MJ per square metre per day."""
import math
SOLAR_CONSTANT = 0.0820

def solar_geometry(latitude_deg, day_of_year):
    if not -90 < latitude_deg < 90 or not 1 <= day_of_year <= 366:
        raise ValueError("Invalid latitude or day")
    phi = math.radians(latitude_deg)
    angle = 2*math.pi*day_of_year/365
    distance = 1+0.033*math.cos(angle)
    declination = 0.409*math.sin(angle-1.39)
    sunset = math.acos(max(-1, min(1, -math.tan(phi)*math.tan(declination))))
    return phi, distance, declination, sunset

def extraterrestrial_daily(latitude_deg, day_of_year):
    phi, distance, declination, sunset = solar_geometry(latitude_deg, day_of_year)
    integral = sunset*math.sin(phi)*math.sin(declination)
    integral += math.cos(phi)*math.cos(declination)*math.sin(sunset)
    return max(0, 24*60/math.pi*SOLAR_CONSTANT*distance*integral)

def extraterrestrial_interval(latitude_deg, day_of_year, solar_hour_start, solar_hour_end):
    """Hours are local solar time, not civil clock time."""
    if not 0 <= solar_hour_start < solar_hour_end <= 24:
        raise ValueError("Invalid solar time interval")
    phi, distance, declination, sunset = solar_geometry(latitude_deg, day_of_year)
    start = max(-sunset, math.pi*(solar_hour_start-12)/12)
    end = min(sunset, math.pi*(solar_hour_end-12)/12)
    if end <= start:
        return 0.0
    integral = (end-start)*math.sin(phi)*math.sin(declination)
    integral += math.cos(phi)*math.cos(declination)*(math.sin(end)-math.sin(start))
    return max(0, 12*60/math.pi*SOLAR_CONSTANT*distance*integral)

def clear_sky_radiation(extraterrestrial_mj, elevation_m):
    if extraterrestrial_mj < 0:
        raise ValueError("Negative radiation")
    return max(0, (0.75+2e-5*elevation_m)*extraterrestrial_mj)
