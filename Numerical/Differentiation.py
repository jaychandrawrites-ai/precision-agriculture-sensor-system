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


def jacobian(function, point, relative_step=None):
    """Centered finite differences with dimension-preserving output axes."""
    x = np.asarray(point, dtype=float)
    step = np.cbrt(np.finfo(float).eps) if relative_step is None else relative_step
    if x.ndim != 1 or not x.size or not np.isfinite(x).all() or not np.isfinite(step) or step <= 0:
        raise ValueError("Expected a finite vector and positive relative step")
    columns = []
    for i in range(len(x)):
        delta = step * max(1., abs(x[i]))
        plus, minus = x.copy(), x.copy()
        plus[i] += delta
        minus[i] -= delta
        if plus[i] == minus[i]:
            raise ValueError("Difference step is below floating point resolution")
        a, b = np.asarray(function(plus), dtype=float), np.asarray(function(minus), dtype=float)
        if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
            raise FloatingPointError("Invalid function values in finite differences")
        columns.append((a - b) / (plus[i] - minus[i]))
    return np.stack(columns, axis=-1)


def hessian(gradient, point, relative_step=None):
    """Differentiate a supplied scalar objective gradient and symmetrize."""
    matrix = jacobian(gradient, point, relative_step)
    if matrix.shape != (len(point), len(point)):
        raise ValueError("Gradient must have the same shape as the point")
    return (matrix + matrix.T) / 2


def hessian_vector(gradient, point, direction, relative_step=None):
    """Apply a Hessian to a vector from two centered gradient evaluations."""
    x, v = (np.asarray(value, dtype=float) for value in (point, direction))
    step = np.cbrt(np.finfo(float).eps) if relative_step is None else relative_step
    if (x.ndim != 1 or not x.size or x.shape != v.shape or not np.isfinite(x).all()
            or not np.isfinite(v).all() or not np.isfinite(step) or step <= 0):
        raise ValueError("Expected finite, matching vectors and a positive relative step")
    magnitude = np.max(np.abs(v))
    if magnitude == 0:
        return np.zeros_like(x)
    scale = step * max(1.0, np.max(np.abs(x)))
    delta = scale * (v / magnitude)
    if np.array_equal(x + delta, x - delta):
        raise ValueError("Difference step is below floating point resolution")
    a, b = (np.asarray(gradient(z), dtype=float) for z in (x + delta, x - delta))
    if a.shape != x.shape or b.shape != x.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise FloatingPointError("Invalid gradient in Hessian-vector product")
    return (a - b) * (magnitude / (2 * scale))
