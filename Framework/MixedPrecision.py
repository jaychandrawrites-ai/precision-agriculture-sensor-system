"""Dynamic loss scaling and explicit FP32 master parameter storage."""
from __future__ import annotations
import math

class LossScaler:
    def __init__(self, initial=1024.0, growth=2.0, backoff=0.5, interval=200,
                 minimum=1.0, maximum=65536.0):
        if not 0 < minimum <= initial <= maximum or growth <= 1 or not 0 < backoff < 1 or interval < 1:
            raise ValueError("Invalid loss scaler configuration")
        self.scale = float(initial)
        self.growth = growth
        self.backoff = backoff
        self.interval = interval
        self.minimum = minimum
        self.maximum = maximum
        self.good_steps = 0

    def scale_gradient(self, gradient):
        return gradient*self.scale

    def unscale(self, gradients, backend):
        gradients = list(gradients)
        with backend.scope():
            finite = all(bool(backend.scalar(backend.xp.isfinite(g).all())) for g in gradients)
            if finite:
                for g in gradients:
                    g /= self.scale
            return finite

    def update(self, success):
        if success:
            self.good_steps += 1
            if self.good_steps >= self.interval:
                self.scale = min(self.maximum, self.scale*self.growth)
                self.good_steps = 0
        else:
            self.scale = max(self.minimum, self.scale*self.backoff)
            self.good_steps = 0

    def state_dict(self):
        return {"scale": self.scale, "good_steps": self.good_steps}

    def load_state_dict(self, state):
        scale = float(state["scale"])
        if not math.isfinite(scale) or not self.minimum <= scale <= self.maximum:
            raise ValueError("Invalid restored scale")
        self.scale = scale
        self.good_steps = int(state["good_steps"])

class MasterParameters:
    def __init__(self, parameters, backend):
        from .Layers import Parameter
        self.model = list(parameters)
        self.backend = backend
        self.master = [Parameter(backend.array(p.value, dtype="float32", copy=True),
                                 backend.zeros(p.value.shape, dtype="float32"),
                                 p.name, p.decay) for p in self.model]

    def collect_gradients(self, scaler):
        with self.backend.scope():
            for model, master in zip(self.model, self.master):
                master.grad[...] = model.grad.astype("float32")
            return scaler.unscale([p.grad for p in self.master], self.backend)

    def synchronize_model(self):
        with self.backend.scope():
            for model, master in zip(self.model, self.master):
                model.value[...] = master.value.astype(model.value.dtype)

    def step(self, optimizer, scaler):
        if [id(p) for p in optimizer.parameters] != [id(p) for p in self.master]:
            raise ValueError("Optimizer must own master parameters")
        success = self.collect_gradients(scaler)
        if success:
            optimizer.step()
            self.synchronize_model()
        scaler.update(success)
        for p in self.model:
            p.zero_grad()
        return success
