"""Trailing Hampel detector; never consults future measurements."""
import numpy as np

def hampel(values,window=12,threshold=3.5,minimum_history=5,noise_floor=1e-6):
    x=np.asarray(values,dtype=float)
    if x.ndim!=2 or not 2<=minimum_history<=window or threshold<=0 or noise_floor<=0:
        raise ValueError("Invalid Hampel settings")
    clean=x.copy()
    flagged=np.zeros(x.shape,dtype=bool)
    for i in range(len(x)):
        for j in range(x.shape[1]):
            history=clean[max(0,i-window):i,j]
            history=history[np.isfinite(history)]
            if len(history)<minimum_history or not np.isfinite(x[i,j]):
                continue
            center=np.median(history)
            scale=max(noise_floor,1.4826*np.median(np.abs(history-center)))
            if abs(x[i,j]-center)>threshold*scale:
                flagged[i,j]=True
                clean[i,j]=np.nan
    return clean,flagged
