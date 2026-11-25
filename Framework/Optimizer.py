"""Explicit optimizers with portable state and decoupled weight decay."""
from __future__ import annotations
import math

class Optimizer:
    def __init__(self, parameters, backend, learning_rate=0.001, weight_decay=0.0, max_norm=None):
        self.parameters = list(parameters)
        if len({id(p) for p in self.parameters}) != len(self.parameters):
            raise ValueError("Duplicate parameters")
        if not math.isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError("Invalid learning rate")
        if not math.isfinite(weight_decay) or weight_decay < 0:
            raise ValueError("Invalid weight decay")
        if max_norm is not None and max_norm <= 0:
            raise ValueError("Invalid gradient norm limit")
        self.backend = backend
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.max_norm = max_norm
        self.steps = 0
        self.state = [{} for _ in self.parameters]

    def zero_grad(self):
        for p in self.parameters:
            p.zero_grad()

    def step(self):
        with self.backend.scope():
            for p in self.parameters:
                self.backend.require_finite(p.grad, p.name)
            norm = self.backend.global_norm([p.grad for p in self.parameters])
            factor = 1.0 if self.max_norm is None else min(1.0, self.max_norm/max(norm, 1e-30))
            if not math.isfinite(norm):
                raise FloatingPointError("Gradient norm is nonfinite")
            self.steps += 1
            for p, state in zip(self.parameters, self.state):
                gradient = p.grad.astype("float64" if p.value.dtype.itemsize == 8 else "float32")*factor
                if p.decay and self.weight_decay:
                    p.value *= 1-self.learning_rate*self.weight_decay
                update = self.direction(gradient, state)
                p.value -= self.learning_rate*update
                self.backend.require_finite(p.value, p.name)
            return {"gradient_norm": norm, "clip_factor": factor, "step": self.steps}

    def direction(self, gradient, state):
        raise NotImplementedError

    def state_dict(self):
        return {
            "type": type(self).__name__,
            "steps": self.steps,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "max_norm": self.max_norm,
            "hyperparameters": self.hyperparameters(),
            "state": [{k: self.backend.to_host(v, copy=True) for k,v in s.items()} for s in self.state],
        }

    def hyperparameters(self):
        return {}

    def load_state_dict(self, record):
        if record["type"] != type(self).__name__ or len(record["state"]) != len(self.parameters):
            raise ValueError("Optimizer checkpoint incompatible")
        if record["hyperparameters"] != self.hyperparameters():
            raise ValueError("Optimizer hyperparameters differ")
        restored = []
        for p, state in zip(self.parameters, record["state"]):
            if any(v.shape != p.value.shape for v in state.values()):
                raise ValueError("Optimizer state shape mismatch")
            restored.append({k: self.backend.array(v, dtype=v.dtype, copy=True) for k,v in state.items()})
        self.state = restored
        self.steps = int(record["steps"])
        self.learning_rate = float(record["learning_rate"])
        self.weight_decay = float(record["weight_decay"])
        self.max_norm = record["max_norm"]

class SGD(Optimizer):
    def __init__(self, parameters, backend, learning_rate=0.01, momentum=0.0,
                 nesterov=False, **kwargs):
        super().__init__(parameters, backend, learning_rate, **kwargs)
        if not 0 <= momentum < 1 or (nesterov and momentum == 0):
            raise ValueError("Invalid momentum")
        self.momentum = momentum
        self.nesterov = nesterov

    def hyperparameters(self):
        return {"momentum": self.momentum, "nesterov": self.nesterov}

    def direction(self, gradient, state):
        if not self.momentum:
            return gradient
        if "velocity" not in state:
            state["velocity"] = self.backend.xp.zeros_like(gradient)
        velocity = state["velocity"]
        velocity *= self.momentum
        velocity += gradient
        return gradient+self.momentum*velocity if self.nesterov else velocity

class Adam(Optimizer):
    def __init__(self, parameters, backend, learning_rate=0.001, beta1=0.9,
                 beta2=0.999, epsilon=1e-8, amsgrad=False, **kwargs):
        super().__init__(parameters, backend, learning_rate, **kwargs)
        if not 0 <= beta1 < 1 or not 0 <= beta2 < 1 or epsilon <= 0:
            raise ValueError("Invalid Adam coefficients")
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.amsgrad = amsgrad

    def hyperparameters(self):
        return {"beta1": self.beta1, "beta2": self.beta2, "epsilon": self.epsilon,
                "amsgrad": self.amsgrad}

    def direction(self, gradient, state):
        xp = self.backend.xp
        if "first" not in state:
            state["first"] = xp.zeros_like(gradient)
            state["second"] = xp.zeros_like(gradient)
        first = state["first"]
        second = state["second"]
        first *= self.beta1
        first += (1-self.beta1)*gradient
        second *= self.beta2
        second += (1-self.beta2)*gradient*gradient
        if self.amsgrad:
            if "maximum" not in state:
                state["maximum"] = xp.zeros_like(second)
            xp.maximum(state["maximum"], second, out=state["maximum"])
            second = state["maximum"]
        corrected_first = first/(1-self.beta1**self.steps)
        corrected_second = second/(1-self.beta2**self.steps)
        return corrected_first/(xp.sqrt(corrected_second)+self.epsilon)

class RMSProp(Optimizer):
    def __init__(self, parameters, backend, learning_rate=0.001, alpha=0.99,
                 epsilon=1e-8, **kwargs):
        super().__init__(parameters, backend, learning_rate, **kwargs)
        if not 0 <= alpha < 1 or epsilon <= 0:
            raise ValueError("Invalid RMSProp coefficients")
        self.alpha = alpha
        self.epsilon = epsilon

    def hyperparameters(self):
        return {"alpha": self.alpha, "epsilon": self.epsilon}

    def direction(self, gradient, state):
        xp = self.backend.xp
        if "square" not in state:
            state["square"] = xp.zeros_like(gradient)
        square = state["square"]
        square *= self.alpha
        square += (1-self.alpha)*gradient**2
        return gradient/(xp.sqrt(square)+self.epsilon)
