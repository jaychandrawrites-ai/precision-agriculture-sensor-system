"""Diagnostic equilibrium leaf temperature using bracketed root finding."""
from .EnergyBalance import exchange

def equilibrium_temperature(weather,params,wetness,tolerance=1e-5):
    low=max(-79.0,weather.air_c-40)
    high=min(79.0,weather.air_c+40)
    def residual(t):
        return exchange(t,wetness,weather,params)[0]
    a,b=residual(low),residual(high)
    if a*b>0:
        raise ValueError("Energy equilibrium not bracketed")
    for _ in range(100):
        middle=(low+high)/2
        c=residual(middle)
        if abs(c)<tolerance or high-low<tolerance:
            return middle
        if a*c<=0:
            high=middle
        else:
            low=middle
            a=c
    raise RuntimeError("Equilibrium solver failed")
