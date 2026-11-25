"""Deterministic mini-batch iteration without replacement."""
import numpy as np

class DataLoader:
    def __init__(self, inputs, targets, batch_size=128, shuffle=True, seed=42, weights=None):
        self.inputs=np.asarray(inputs)
        self.targets=np.asarray(targets)
        self.weights=np.ones(len(self.inputs)) if weights is None else np.asarray(weights)
        if len(self.inputs)!=len(self.targets) or self.weights.shape!=(len(self.inputs),):
            raise ValueError("Mismatched training arrays")
        if batch_size<1 or not np.isfinite(self.weights).all() or np.any(self.weights<0):
            raise ValueError("Invalid batch configuration")
        self.batch_size=batch_size
        self.shuffle=shuffle
        self.rng=np.random.default_rng(seed)

    def __len__(self):
        return (len(self.inputs)+self.batch_size-1)//self.batch_size

    def __iter__(self):
        order=self.rng.permutation(len(self.inputs)) if self.shuffle else np.arange(len(self.inputs))
        for start in range(0,len(order),self.batch_size):
            idx=order[start:start+self.batch_size]
            yield self.inputs[idx],self.targets[idx],self.weights[idx]
