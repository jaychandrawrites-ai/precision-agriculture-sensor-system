"""Physical feasibility diagnostics for the six-feature schema."""
import numpy as np
from config import FEATURES

LOWER = (-60, 0, 0, 0, 0, 0)
UPPER = (65, 100, 1, np.inf, np.inf, np.inf)

def violations(values,crop_coefficient=None):
    x=np.asarray(values,dtype=float)
    if x.ndim<1 or x.shape[-1]!=len(FEATURES):
        raise ValueError("Expected six physical features")
    flags={
        "nonfinite":~np.isfinite(x).all(axis=-1),
        "temperature":(x[...,0]<-60)|(x[...,0]>65),
        "humidity":(x[...,1]<0)|(x[...,1]>100),
        "wetness":(x[...,2]<0)|(x[...,2]>1),
        "negative_et":(x[...,3]<0)|(x[...,4]<0),
        "negative_heat_stress":x[...,5]<0,
    }
    if crop_coefficient is not None:
        if crop_coefficient<0:
            raise ValueError("Negative crop coefficient")
        flags["eta_bound"]=x[...,4]>crop_coefficient*x[...,3]+1e-8
    return flags

def project(values,crop_coefficient=None):
    x=np.asarray(values,dtype=float).copy()
    if x.shape[-1]!=6 or not np.isfinite(x).all():
        raise ValueError("Cannot project invalid features")
    x = np.clip(x, LOWER, UPPER)
    if crop_coefficient is not None:
        if crop_coefficient<0:
            raise ValueError("Negative crop coefficient")
        x[...,4]=np.minimum(x[...,4],crop_coefficient*x[...,3])
    return x
