"""Chronological splitting by target timestamp."""
import numpy as np

def split_indices(indices, train_fraction=0.7, validation_fraction=0.15):
    idx=np.asarray(indices,dtype=int)
    if idx.ndim!=1 or np.any(np.diff(idx)<=0):
        raise ValueError("Indices must strictly increase")
    if not 0<train_fraction<1 or not 0<validation_fraction<1-train_fraction:
        raise ValueError("Invalid split fractions")
    a=int(len(idx)*train_fraction)
    b=int(len(idx)*(train_fraction+validation_fraction))
    if not 0<a<b<len(idx):
        raise ValueError("Insufficient samples for three splits")
    return np.arange(a),np.arange(a,b),np.arange(b,len(idx))
