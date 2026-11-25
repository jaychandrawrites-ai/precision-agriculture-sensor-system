"""Leaf energy and water exchange in SI units."""
from dataclasses import dataclass
import math
from Physics.Evapotranspiration.FAO56 import saturation_pressure,actual_pressure

SIGMA=5.670374419e-8
LATENT_HEAT=2.45e6

@dataclass(frozen=True)
class LeafWeather:
    air_c: float
    humidity_pct: float
    shortwave_w_m2: float
    longwave_w_m2: float
    wind_m_s: float=0.0
    intercepted_rain_kg_m2_s: float=0.0

    def __post_init__(self):
        if not all(math.isfinite(x) for x in vars(self).values()):
            raise ValueError("Nonfinite weather")
        if not -60<=self.air_c<=60 or not 0<=self.humidity_pct<=100:
            raise ValueError("Invalid air conditions")
        if min(self.shortwave_w_m2,self.longwave_w_m2,self.wind_m_s,
               self.intercepted_rain_kg_m2_s)<0:
            raise ValueError("Negative weather flux")

def approximate_longwave(air_c,humidity_pct):
    """Clear-sky Brutsaert approximation; clouds are not represented."""
    vapor=actual_pressure(air_c,humidity_pct)*10
    kelvin=air_c+273.15
    emissivity=min(1.0,1.24*(vapor/kelvin)**(1/7))
    return emissivity*SIGMA*kelvin**4

def exchange(leaf_c,wetness,weather,params):
    vapor_air=actual_pressure(weather.air_c,weather.humidity_pct)
    vapor_leaf=saturation_pressure(leaf_c)
    wind_factor=math.sqrt(1+weather.wind_m_s)
    evaporation=params.mass_transfer_kg_m2_s_kpa*wind_factor*(vapor_leaf-vapor_air)
    if evaporation>0:
        evaporation*=max(0,min(1,wetness))
    radiation=params.absorptivity*weather.shortwave_w_m2
    radiation+=params.emissivity*(weather.longwave_w_m2-SIGMA*(leaf_c+273.15)**4)
    convection=params.convection_w_m2_k*wind_factor*(weather.air_c-leaf_c)
    heat=radiation+convection-LATENT_HEAT*evaporation
    water=weather.intercepted_rain_kg_m2_s-evaporation
    return heat,water
