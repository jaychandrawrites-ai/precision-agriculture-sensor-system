"""Explicit Runge-Kutta integration with adaptive error control."""
from dataclasses import dataclass
import numpy as np

@dataclass
class IntegrationResult:
    times: np.ndarray
    states: np.ndarray
    accepted: int
    rejected: int

def rk4_step(rhs, t, y, h):
    k1 = np.asarray(rhs(t,y))
    k2 = np.asarray(rhs(t+h/2,y+h*k1/2))
    k3 = np.asarray(rhs(t+h/2,y+h*k2/2))
    k4 = np.asarray(rhs(t+h,y+h*k3))
    return y+h*(k1+2*k2+2*k3+k4)/6

def integrate(rhs, initial, start, end, step=1.0, rtol=1e-6,
              atol=1e-8, min_step=1e-8, max_steps=100000, project=None):
    """Step doubling estimates local error; projection occurs after acceptance."""
    if not end >= start or min(step,rtol,atol,min_step)<=0:
        raise ValueError("Invalid integration settings")
    y=np.asarray(initial,dtype=float).copy()
    if not np.isfinite(y).all():
        raise ValueError("Nonfinite initial state")
    t=float(start)
    times=[t]
    states=[y.copy()]
    accepted=rejected=0
    h=step
    while t<end:
        if accepted+rejected>=max_steps:
            raise RuntimeError("Integration budget exhausted")
        h=min(h,end-t)
        coarse=rk4_step(rhs,t,y,h)
        half=rk4_step(rhs,t,y,h/2)
        fine=rk4_step(rhs,t+h/2,half,h/2)
        error=(fine-coarse)/15
        scale=atol+rtol*np.maximum(np.abs(y),np.abs(fine))
        ratio=float(np.max(np.abs(error)/scale))
        if not np.isfinite(ratio):
            ratio=np.inf
        if ratio<=1:
            candidate=fine+error
            if project is not None:
                candidate=np.asarray(project(candidate),dtype=float)
            if candidate.shape!=y.shape or not np.isfinite(candidate).all():
                raise FloatingPointError("Invalid accepted state")
            t+=h
            y=candidate
            times.append(t)
            states.append(y.copy())
            accepted+=1
        else:
            rejected+=1
            if h<=min_step:
                raise RuntimeError("Cannot satisfy integration tolerance")
        factor=5.0 if ratio==0 else min(5.0,max(0.1,0.9*ratio**(-0.2)))
        h=max(min_step,h*factor)
    return IntegrationResult(np.asarray(times),np.stack(states),accepted,rejected)
