"""Learning rate policies with explicit step and epoch semantics."""
from __future__ import annotations
import math

class Scheduler:
    def __init__(self, optimizer):
        self.optimizer = optimizer
        self.base_rate = optimizer.learning_rate
        self.index = 0

    def rate(self, index):
        raise NotImplementedError

    def step(self):
        self.index += 1
        self.optimizer.learning_rate = self.rate(self.index)
        return self.optimizer.learning_rate

    def state_dict(self):
        return {"index": self.index, "base_rate": self.base_rate, "rate": self.optimizer.learning_rate}

    def load_state_dict(self, state):
        self.index = int(state["index"])
        self.base_rate = float(state["base_rate"])
        self.optimizer.learning_rate = float(state["rate"])

class CosineAnnealing(Scheduler):
    def __init__(self, optimizer, duration, minimum=1e-6, warmup=0):
        super().__init__(optimizer)
        if duration <= warmup or warmup < 0 or not 0 < minimum <= self.base_rate:
            raise ValueError("Invalid cosine schedule")
        self.duration = duration
        self.minimum = minimum
        self.warmup = warmup
        if warmup:
            optimizer.learning_rate = self.rate(0)

    def rate(self, index):
        if index < self.warmup:
            return self.base_rate*(index+1)/self.warmup
        progress = min(1.0, (index-self.warmup)/(self.duration-self.warmup))
        return self.minimum+0.5*(self.base_rate-self.minimum)*(1+math.cos(math.pi*progress))

class StepDecay(Scheduler):
    def __init__(self, optimizer, interval, gamma=0.5, minimum=1e-8):
        super().__init__(optimizer)
        if interval <= 0 or not 0 < gamma <= 1 or minimum <= 0:
            raise ValueError("Invalid step decay")
        self.interval = interval
        self.gamma = gamma
        self.minimum = minimum

    def rate(self, index):
        return max(self.minimum, self.base_rate*self.gamma**(index//self.interval))

class ReduceOnPlateau:
    def __init__(self, optimizer, patience=5, factor=0.5, minimum=1e-7, tolerance=1e-6):
        if patience < 1 or not 0 < factor < 1 or minimum <= 0 or tolerance < 0:
            raise ValueError("Invalid plateau schedule")
        self.optimizer = optimizer
        self.patience = patience
        self.factor = factor
        self.minimum = minimum
        self.tolerance = tolerance
        self.best = math.inf
        self.bad_epochs = 0

    def step(self, metric):
        if not math.isfinite(metric):
            raise ValueError("Validation metric must be finite")
        if metric < self.best-self.tolerance:
            self.best = metric
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1
            if self.bad_epochs >= self.patience:
                self.optimizer.learning_rate = max(self.minimum, self.optimizer.learning_rate*self.factor)
                self.bad_epochs = 0
        return self.optimizer.learning_rate

    def state_dict(self):
        return {"best": self.best, "bad_epochs": self.bad_epochs,
                "rate": self.optimizer.learning_rate}

    def load_state_dict(self, state):
        self.best = float(state["best"])
        self.bad_epochs = int(state["bad_epochs"])
        self.optimizer.learning_rate = float(state["rate"])
