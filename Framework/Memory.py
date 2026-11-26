"""Activation-cache estimates and deterministic microbatch planning."""
from dataclasses import dataclass, fields, is_dataclass
import math


def _array_bytes(value, seen):
    if id(value) in seen:
        return 0
    seen.add(id(value))
    if hasattr(value, "nbytes"):
        return int(value.nbytes)
    if is_dataclass(value):
        return sum(_array_bytes(getattr(value, field.name), seen) for field in fields(value))
    if isinstance(value, dict):
        return sum(_array_bytes(x, seen) for x in value.values())
    if isinstance(value, (tuple, list)):
        return sum(_array_bytes(x, seen) for x in value)
    return 0


@dataclass(frozen=True)
class MemoryEstimate:
    parameter_bytes: int
    gradient_bytes: int
    optimizer_bytes: int
    snapshot_bytes: int
    activation_bytes_per_sample: int
    scratch_multiplier: float

    @property
    def fixed_bytes(self):
        return self.parameter_bytes + self.gradient_bytes + self.optimizer_bytes + self.snapshot_bytes

    @property
    def sample_bytes(self):
        return math.ceil(self.activation_bytes_per_sample * self.scratch_multiplier)

    def batch_bytes(self, count):
        if type(count) is not int or count < 1:
            raise ValueError("Batch count must be a positive integer")
        return self.fixed_bytes + self.sample_bytes * count


@dataclass(frozen=True)
class BatchPlan:
    requested_batch: int
    microbatch: int
    budget_bytes: int | None
    estimate_bytes: int

    @property
    def accumulated(self):
        return self.microbatch < self.requested_batch


def estimate_memory(model, backend, prototype, *, optimizer_slots=2, checkpoint_copy=True,
                    scratch_multiplier=3.0):
    """Estimate train buffers from a one-example forward pass.

    Count current parameters and gradients, anticipated optimizer states, a
    checkpoint copy, and the saved activation arrays. This excludes allocator
    overhead, static datasets and externally owned arrays. Scratch is a
    multiplier, not a measured hardware guarantee.
    """
    if (type(optimizer_slots) is not int or optimizer_slots < 0
            or not math.isfinite(scratch_multiplier) or scratch_multiplier < 1):
        raise ValueError("Invalid train memory estimate settings")
    with backend.scope():
        parameters = model.parameters()
        parameter_bytes = sum(p.value.nbytes for p in parameters)
        gradient_bytes = sum(p.grad.nbytes for p in parameters)
        state_bytes = sum(p.value.size * max(p.value.itemsize, 4) for p in parameters)
        example = backend.array(prototype)
        if example.ndim < 2 or len(example) != 1:
            raise ValueError("Memory prototype must contain exactly one sample")
        # Use training mode for dropout/batch caches without changing RNG or
        # running statistics of the model. Supported models use stateless
        # forwards except dropout; batch-normalization state is restored below.
        rng = backend.rng
        backend.rng = backend.xp.random.RandomState(0)
        state = [(layer, layer.running_mean.copy(), layer.running_var.copy())
                 for layer in getattr(model, "layers", ()) if hasattr(layer, "running_mean")]
        try:
            output, caches = model.forward(example, training=True)
        finally:
            backend.rng = rng
            for layer, mean, variance in state:
                layer.running_mean[...] = mean
                layer.running_var[...] = variance
    seen = {id(p.value) for p in parameters} | {id(p.grad) for p in parameters}
    sample_bytes = _array_bytes((example, output, caches), seen) + 2 * output.nbytes
    # Accumulation may need both the prior and prospective running sums.
    reserve = (parameter_bytes if checkpoint_copy else 0) + 2 * state_bytes
    return MemoryEstimate(parameter_bytes, gradient_bytes, optimizer_slots * state_bytes,
                          reserve, sample_bytes, scratch_multiplier)


def plan_batch(estimate, requested_batch, budget_bytes=None):
    """Keep the effective batch fixed and shrink only the microbatch."""
    if type(requested_batch) is not int or requested_batch < 1:
        raise ValueError("Requested batch must be a positive integer")
    if budget_bytes is None:
        return BatchPlan(requested_batch, requested_batch, None, estimate.batch_bytes(requested_batch))
    if type(budget_bytes) is not int or budget_bytes < 1:
        raise ValueError("Memory budget must be a positive integer byte count")
    if budget_bytes < estimate.batch_bytes(1):
        raise MemoryError(f"Estimated minimum training memory is {estimate.batch_bytes(1)} bytes")
    count = min(requested_batch, (budget_bytes - estimate.fixed_bytes) // estimate.sample_bytes)
    return BatchPlan(requested_batch, count, budget_bytes, estimate.batch_bytes(count))


def observed_process_peak():
    """Return the OS process resident-set high-water mark where available."""
    import sys
    try:
        import resource
    except ImportError:
        return None
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform.startswith("linux"):
        return {"bytes": int(value * 1024), "scope": "process_lifetime_resident_set"}
    if sys.platform == "darwin":
        return {"bytes": int(value), "scope": "process_lifetime_resident_set"}
    return None
