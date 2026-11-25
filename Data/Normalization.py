"""Training-only standardization with portable statistics."""
import numpy as np

class Standardizer:
    def __init__(self, epsilon=1e-8):
        self.epsilon = epsilon
        self.mean = None
        self.scale = None

    def fit(self, values):
        x = np.asarray(values, dtype=float)
        if x.ndim != 2 or not len(x) or not np.isfinite(x).all():
            raise ValueError("Expected finite training matrix")
        self.mean = x.mean(axis=0)
        std = x.std(axis=0)
        self.scale = np.where(std < self.epsilon, 1.0, std)
        return self

    def transform(self, values):
        x = np.asarray(values, dtype=float)
        if self.mean is None or x.shape[-1] != len(self.mean):
            raise ValueError("Unfitted or incompatible standardizer")
        return (x-self.mean)/self.scale

    def inverse_transform(self, values):
        if self.mean is None:
            raise ValueError("Unfitted standardizer")
        return np.asarray(values)*self.scale+self.mean

    def state_dict(self):
        return {"mean": self.mean.tolist(), "scale": self.scale.tolist(),
                "epsilon": self.epsilon}

    @classmethod
    def from_state(cls, state):
        obj = cls(state["epsilon"])
        obj.mean = np.asarray(state["mean"], dtype=float)
        obj.scale = np.asarray(state["scale"], dtype=float)
        if obj.mean.shape != obj.scale.shape or not np.isfinite(obj.mean).all() or not np.isfinite(obj.scale).all() or np.any(obj.scale <= 0):
            raise ValueError("Invalid normalization state")
        return obj
