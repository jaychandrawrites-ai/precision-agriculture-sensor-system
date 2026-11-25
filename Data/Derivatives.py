"""Causal 18-feature inputs and aligned next-sample targets."""
import numpy as np

def supervised(values, timestamps, interval_hours=1.0, tolerance_seconds=1.0):
    x = np.asarray(values, dtype=float)
    t = np.asarray(timestamps, dtype="datetime64[ns]")
    if x.ndim != 2 or x.shape[1] != 6 or len(x) != len(t):
        raise ValueError("Expected six features and aligned timestamps")
    if interval_hours <= 0 or len(x) < 4:
        raise ValueError("Need four samples and positive interval")
    gaps = np.diff(t)/np.timedelta64(1,"s")
    valid_gap = np.abs(gaps-3600*interval_hours) <= tolerance_seconds
    inputs, targets, indices = [], [], []
    for i in range(2,len(x)-1):
        if not valid_gap[i-2:i+1].all() or not np.isfinite(x[i-2:i+2]).all():
            continue
        v = (x[i]-x[i-1])/interval_hours
        a = (x[i]-2*x[i-1]+x[i-2])/interval_hours**2
        inputs.append(np.concatenate((x[i],v,a)))
        targets.append(x[i+1])
        indices.append(i+1)
    return (np.asarray(inputs).reshape(-1,18),
            np.asarray(targets).reshape(-1,6),np.asarray(indices,dtype=int))

def latest_input(history, interval_hours=1.0):
    x = np.asarray(history,dtype=float)
    if x.ndim != 2 or x.shape[1] != 6 or len(x)<3 or interval_hours<=0:
        raise ValueError("Need three aligned six-feature samples")
    a,b,c=x[-3:]
    if not np.isfinite([a,b,c]).all():
        raise ValueError("Nonfinite history")
    return np.concatenate((c,(c-b)/interval_hours,(c-2*b+a)/interval_hours**2))
