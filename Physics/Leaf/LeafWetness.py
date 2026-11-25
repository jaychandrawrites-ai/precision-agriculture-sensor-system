"""Coupled leaf temperature and bounded surface-water storage."""
from dataclasses import dataclass
import numpy as np
from Numerical.ODESolver import integrate
from .EnergyBalance import exchange

@dataclass(frozen=True)
class LeafState:
    temperature_c: float
    wetness: float

def advance_leaf(state,weather,params,interval_seconds=3600):
    if not 0<=state.wetness<=1 or interval_seconds<=0:
        raise ValueError("Invalid leaf state or duration")
    def rhs(t,y):
        temperature,wetness=y
        heat,water=exchange(float(temperature),float(wetness),weather,params)
        if (wetness<=0 and water<0) or (wetness>=1 and water>0):
            water=0.0
        return np.array([heat/params.heat_capacity_j_m2_k,
                         water/params.storage_kg_m2])
    def project(y):
        return np.array([y[0],np.clip(y[1],0,1)])
    result=integrate(rhs,[state.temperature_c,state.wetness],0,interval_seconds,
                     step=min(10,interval_seconds),project=project,max_step=60)
    temperature,wetness=result.states[-1]
    return LeafState(float(temperature),float(wetness)),result

def normalize_sensor(raw_mv,dry_mv,wet_mv):
    if not np.isfinite([raw_mv,dry_mv,wet_mv]).all() or dry_mv==wet_mv:
        raise ValueError("Invalid calibration")
    return float(np.clip((raw_mv-dry_mv)/(wet_mv-dry_mv),0,1))
