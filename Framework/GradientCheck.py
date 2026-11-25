"""Finite-difference checks for deterministic manual backpropagation."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass
class CheckResult:
    parameter: str
    checked: int
    maximum_absolute: float
    maximum_relative: float
    failures: int

    @property
    def passed(self):
        return self.failures == 0

def check_model(model, x, upstream, backend, epsilon=1e-5, atol=1e-6,
                rtol=1e-3, samples=32, seed=42):
    if epsilon <= 0 or samples < 1:
        raise ValueError("Invalid gradient check settings")
    rng = np.random.default_rng(seed)
    model.zero_grad()
    y, cache = model.forward(x, training=False)
    if y.shape != upstream.shape:
        raise ValueError("Upstream shape mismatch")
    model.backward(upstream, cache)
    reports = []
    def objective():
        output, _ = model.forward(x, training=False)
        return float(backend.scalar((output*upstream).sum()))
    with backend.scope():
        for number, p in enumerate(model.parameters()):
            analytical = backend.to_host(p.grad, copy=True)
            indices = rng.choice(p.value.size, min(samples, p.value.size), replace=False)
            maximum_absolute = maximum_relative = 0.0
            failures = 0
            for flat in indices:
                index = np.unravel_index(flat, p.value.shape)
                original = float(backend.scalar(p.value[index]))
                step = epsilon*max(1.0, abs(original))
                try:
                    p.value[index] = original+step
                    plus = objective()
                    p.value[index] = original-step
                    minus = objective()
                finally:
                    p.value[index] = original
                numerical = (plus-minus)/(2*step)
                exact = float(analytical[index])
                absolute = abs(exact-numerical)
                relative = absolute/max(1e-12, abs(exact)+abs(numerical))
                maximum_absolute = max(maximum_absolute, absolute)
                maximum_relative = max(maximum_relative, relative)
                if not np.isfinite(absolute) or absolute > atol+rtol*max(abs(exact),abs(numerical)):
                    failures += 1
            reports.append(CheckResult(f"{number}:{p.name}", len(indices), maximum_absolute,
                                       maximum_relative, failures))
    return reports
