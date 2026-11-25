"""Finite differences for irregular temporal observations."""
import numpy as np

def derivative_weights(nodes, center, order):
    nodes=np.asarray(nodes,dtype=float)
    if nodes.ndim!=1 or not 0<=order<len(nodes):
        raise ValueError("Invalid derivative stencil")
    if len(np.unique(nodes))!=len(nodes):
        raise ValueError("Duplicate stencil coordinates")
    offsets=nodes-center
    scale=np.max(np.abs(offsets))
    if scale==0:
        return np.ones(1)
    z=offsets/scale
    matrix=np.vstack([z**k for k in range(len(z))])
    target=np.zeros(len(z))
    import math
    target[order]=math.factorial(order)/scale**order
    return np.linalg.solve(matrix,target)

def causal_derivatives(times, values, window=3, max_gap=None):
    times=np.asarray(times,dtype=float)
    values=np.asarray(values,dtype=float)
    if times.ndim!=1 or len(times)!=len(values) or window<3:
        raise ValueError("Invalid time series")
    if not np.isfinite(times).all() or np.any(np.diff(times)<=0):
        raise ValueError("Times must strictly increase")
    first=np.full_like(values,np.nan)
    second=np.full_like(values,np.nan)
    for i in range(window-1,len(times)):
        t=times[i-window+1:i+1]
        x=values[i-window+1:i+1]
        if max_gap is not None and np.max(np.diff(t))>max_gap:
            continue
        first[i]=np.tensordot(derivative_weights(t,t[-1],1),x,axes=1)
        second[i]=np.tensordot(derivative_weights(t,t[-1],2),x,axes=1)
    return first,second
