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


def calendar_cutoffs(timestamps, train_fraction=0.7, validation_fraction=0.15):
    """Declare split boundaries from the raw time span before filtering targets."""
    if not 0 < train_fraction < 1 or not 0 < validation_fraction < 1 - train_fraction:
        raise ValueError("Invalid split fractions")
    times = np.asarray(timestamps, dtype="datetime64[ns]")
    if times.ndim != 1 or not len(times) or np.isnat(times).any() or (np.diff(times) <= np.timedelta64(0, "ns")).any():
        raise ValueError("Expected increasing timestamps")
    # Anchor the half-open analysis span to its first observed hour.
    hours = int((times[-1] - times[0]) / np.timedelta64(1, "h")) + 1
    return tuple(times[0] + np.timedelta64(int(hours * fraction), "h")
                 for fraction in (train_fraction, train_fraction + validation_fraction))


def split_at(timestamps, target_indices, cutoffs):
    timestamps = np.asarray(timestamps, dtype="datetime64[ns]")
    cuts = np.asarray(cutoffs, dtype="datetime64[ns]")
    if cuts.shape != (2,) or np.isnat(cuts).any() or cuts[1] <= cuts[0]:
        raise ValueError("Expected two increasing calendar split boundaries")
    labels = np.searchsorted(cuts, timestamps[target_indices], side="right")
    splits = tuple(np.flatnonzero(labels == split) for split in range(3))
    if not all(len(rows) for rows in splits):
        raise ValueError("Calendar splits must contain usable examples in all three periods")
    return splits
