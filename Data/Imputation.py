"""Bounded causal forward filling with explicit quality flags."""
import numpy as np

def forward_fill(values,max_gap=2):
    x=np.asarray(values,dtype=float).copy()
    if x.ndim!=2 or max_gap<0:
        raise ValueError("Invalid imputation settings")
    filled=np.zeros(x.shape,dtype=bool)
    for j in range(x.shape[1]):
        last=np.nan
        age=max_gap+1
        for i in range(len(x)):
            if np.isfinite(x[i,j]):
                last=x[i,j]
                age=0
            else:
                age+=1
                x[i,j]=np.nan
                if age<=max_gap and np.isfinite(last):
                    x[i,j]=last
                    filled[i,j]=True
    return x,filled
