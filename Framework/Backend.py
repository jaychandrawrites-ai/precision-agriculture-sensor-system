"""Numerical execution backend for Jupiter.

NumPy is required. CuPy is optional. Arrays remain on the selected device
until an explicit host transfer is requested.
"""
from __future__ import annotations

import contextlib
import importlib
import logging
import time
from dataclasses import dataclass
from typing import Any, Iterator

import numpy as np

LOGGER = logging.getLogger(__name__)


class BackendError(RuntimeError):
    """Numerical backend configuration or execution failure."""


@dataclass(frozen=True)
class BackendConfig:
    device: str = "cpu"
    dtype: str = "float32"
    seed: int = 42
    allow_cpu_fallback: bool = False
    check_finite: bool = True

    def __post_init__(self):
        if self.dtype not in ("float16", "float32", "float64"):
            raise ValueError("Unsupported floating-point precision")
        if self.device != "cpu" and not (
            self.device == "cuda" or self.device.startswith("cuda:")
        ):
            raise ValueError("Expected cpu, cuda, or cuda:N")
        if self.seed < 0:
            raise ValueError("Seed must be nonnegative")


@dataclass(frozen=True)
class ArrayReport:
    shape: tuple
    dtype: str
    device: str
    bytes: int
    finite_count: int
    element_count: int
    minimum: float | None
    maximum: float | None
    mean: float | None
    std: float | None


class Backend:
    """Explicit array ownership, precision and reproducible random operations."""

    def __init__(self, config: BackendConfig | None = None):
        self.config = config or BackendConfig()
        self.xp = np
        self.device = "cpu"
        self.device_id = None
        self.fallback_reason = None
        if self.config.device.startswith("cuda"):
            self._initialize_cuda()
        self.dtype = self.xp.dtype(self.config.dtype)
        self.reseed(self.config.seed)

    def _initialize_cuda(self):
        try:
            cp = importlib.import_module("cupy")
            index = int(self.config.device.partition(":")[2] or "0")
            if not 0 <= index < cp.cuda.runtime.getDeviceCount():
                raise BackendError("CUDA device index is unavailable")
            with cp.cuda.Device(index):
                cp.zeros(1).sum().item()
            self.xp = cp
            self.device_id = index
            self.device = f"cuda:{index}"
        except Exception as exc:
            if not self.config.allow_cpu_fallback:
                raise BackendError("CUDA initialization failed") from exc
            self.fallback_reason = str(exc)
            LOGGER.warning("CUDA unavailable; using CPU: %s", exc)

    @property
    def is_gpu(self):
        return self.device_id is not None

    @contextlib.contextmanager
    def scope(self) -> Iterator[None]:
        if self.is_gpu:
            with self.xp.cuda.Device(self.device_id):
                yield
        else:
            yield

    def reseed(self, seed: int):
        if seed < 0:
            raise ValueError("Seed must be nonnegative")
        self.seed = seed
        with self.scope():
            self.rng = self.xp.random.RandomState(seed)

    def synchronize(self):
        if self.is_gpu:
            with self.scope():
                self.xp.cuda.get_current_stream().synchronize()

    def to_host(self, value, copy=False):
        if hasattr(value, "__cuda_array_interface__"):
            cp = importlib.import_module("cupy")
            result = cp.asnumpy(value)
        else:
            result = np.asarray(value)
        return result.copy() if copy else result

    def array(self, value, dtype=None, copy=False):
        target = self.dtype if dtype is None else dtype
        with self.scope():
            if not self.is_gpu and hasattr(value, "__cuda_array_interface__"):
                value = self.to_host(value)
            result = self.xp.asarray(value, dtype=target)
            if self.is_gpu and result.device.id != self.device_id:
                result = self.xp.array(self.to_host(result), dtype=target)
            return result.copy() if copy else result

    def scalar(self, value):
        return self.to_host(value).item()

    def zeros(self, shape, dtype=None):
        with self.scope():
            return self.xp.zeros(shape, dtype=dtype or self.dtype)

    def ones(self, shape, dtype=None):
        with self.scope():
            return self.xp.ones(shape, dtype=dtype or self.dtype)

    def full(self, shape, value, dtype=None):
        with self.scope():
            return self.xp.full(shape, value, dtype=dtype or self.dtype)

    def normal(self, shape, mean=0.0, std=1.0):
        if std < 0:
            raise ValueError("Standard deviation cannot be negative")
        with self.scope():
            return self.rng.normal(mean, std, size=shape).astype(self.dtype)

    def uniform(self, shape, low=0.0, high=1.0):
        if high <= low:
            raise ValueError("Invalid uniform interval")
        with self.scope():
            return self.rng.uniform(low, high, size=shape).astype(self.dtype)

    def permutation(self, size):
        with self.scope():
            return self.rng.permutation(size)

    def bernoulli(self, shape, probability):
        if not 0 <= probability <= 1:
            raise ValueError("Probability outside [0, 1]")
        return self.uniform(shape) < probability

    def require_finite(self, value, name="array"):
        with self.scope():
            if not bool(self.scalar(self.xp.all(self.xp.isfinite(value)))):
                raise FloatingPointError(f"{name} contains NaN or infinity")
        return value

    def validate(self, value, name="array", shape=None):
        if shape is not None and tuple(value.shape) != tuple(shape):
            raise ValueError(f"{name}: expected {shape}, got {value.shape}")
        if self.config.check_finite:
            self.require_finite(value, name)
        return value

    def matmul(self, left, right):
        with self.scope():
            result = self.xp.matmul(left, right)
            return self.validate(result, "matrix product")

    def stable_softmax(self, value, axis=-1):
        with self.scope():
            work = self.array(value, dtype="float64" if self.dtype.itemsize == 8 else "float32")
            self.require_finite(work, "softmax input")
            shifted = work - self.xp.max(work, axis=axis, keepdims=True)
            numerator = self.xp.exp(shifted)
            denominator = self.xp.sum(numerator, axis=axis, keepdims=True)
            return numerator / denominator

    def global_norm(self, arrays):
        with self.scope():
            total = self.xp.zeros((), dtype="float64")
            for array in arrays:
                work = self.array(array, dtype="float64")
                total += self.xp.sum(work * work)
            return float(self.scalar(self.xp.sqrt(total)))

    def clip_gradients(self, gradients, maximum_norm):
        if maximum_norm <= 0:
            raise ValueError("Maximum norm must be positive")
        gradients = list(gradients)
        norm = self.global_norm(gradients)
        if not np.isfinite(norm):
            raise FloatingPointError("Nonfinite gradient norm")
        factor = min(1.0, maximum_norm / max(norm, 1e-30))
        with self.scope():
            for gradient in gradients:
                gradient *= factor
        return norm, factor

    def report(self, value):
        with self.scope():
            array = self.xp.asarray(value)
            mask = self.xp.isfinite(array)
            count = int(self.scalar(mask.sum()))
            valid = array[mask].astype("float64")
            stats = (None, None, None, None)
            if count:
                stats = tuple(float(self.scalar(fn(valid))) for fn in (
                    self.xp.min, self.xp.max, self.xp.mean, self.xp.std
                ))
            return ArrayReport(tuple(array.shape), str(array.dtype),
                               self.device, int(array.nbytes), count,
                               int(array.size), *stats)

    def memory_report(self):
        if not self.is_gpu:
            return {"device": self.device, "pool_bytes": None,
                    "free_bytes": None, "total_bytes": None}
        with self.scope():
            free, total = self.xp.cuda.runtime.memGetInfo()
            return {"device": self.device,
                    "pool_bytes": int(self.xp.get_default_memory_pool().used_bytes()),
                    "free_bytes": int(free), "total_bytes": int(total)}

    def release_unused_memory(self):
        if self.is_gpu:
            with self.scope():
                self.synchronize()
                self.xp.get_default_memory_pool().free_all_blocks()
                self.xp.get_default_pinned_memory_pool().free_all_blocks()

    @contextlib.contextmanager
    def timed(self, name="operation"):
        record = {"name": name, "seconds": None}
        self.synchronize()
        start = time.perf_counter()
        try:
            yield record
        finally:
            self.synchronize()
            record["seconds"] = time.perf_counter() - start

    def metadata(self):
        return {"device": self.device, "dtype": str(self.dtype),
                "seed": self.seed, "library": self.xp.__name__,
                "library_version": self.xp.__version__,
                "fallback_reason": self.fallback_reason}


def create_backend(device="cpu", dtype="float32", seed=42,
                   allow_cpu_fallback=False):
    return Backend(BackendConfig(device=device, dtype=dtype, seed=seed,
                                 allow_cpu_fallback=allow_cpu_fallback))
