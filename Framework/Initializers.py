"""Reproducible initialization for dense and convolutional parameters."""
from __future__ import annotations

import math
from dataclasses import dataclass


def calculate_fans(shape, layout="dense"):
    shape = tuple(shape)
    if not shape or any(n <= 0 for n in shape):
        raise ValueError("Invalid parameter dimensions")
    if layout == "dense" and len(shape) == 2:
        return shape[0], shape[1]
    if layout == "conv" and len(shape) >= 3:
        receptive = math.prod(shape[2:])
        return shape[1]*receptive, shape[0]*receptive
    raise ValueError("Expected dense IO or convolution OI... layout")


def activation_gain(name, slope=0.01):
    gains = {"linear": 1.0, "sigmoid": 1.0, "tanh": 5/3,
             "relu": math.sqrt(2), "leaky_relu": math.sqrt(2/(1+slope*slope))}
    if name not in gains:
        raise ValueError(f"No defined initialization gain for {name}")
    return gains[name]


@dataclass(frozen=True)
class Initialization:
    method: str = "he_normal"
    gain: float = 1.0
    constant: float = 0.0

    def __post_init__(self):
        if not math.isfinite(self.gain) or self.gain < 0:
            raise ValueError("Gain must be finite and nonnegative")


def initialize(shape, backend, config=None, layout="dense"):
    config = config or Initialization()
    method = config.method
    gain = config.gain
    if method == "zeros":
        return backend.zeros(shape)
    if method == "ones":
        return backend.ones(shape)
    if method == "constant":
        return backend.full(shape, config.constant)
    fan_in, fan_out = calculate_fans(shape, layout)
    if method == "he_normal":
        return backend.normal(shape, std=gain*math.sqrt(2/fan_in))
    if method == "he_uniform":
        bound = gain*math.sqrt(6/fan_in)
        return backend.uniform(shape, -bound, bound) if bound else backend.zeros(shape)
    if method == "xavier_normal":
        return backend.normal(shape, std=gain*math.sqrt(2/(fan_in+fan_out)))
    if method == "xavier_uniform":
        bound = gain*math.sqrt(6/(fan_in+fan_out))
        return backend.uniform(shape, -bound, bound) if bound else backend.zeros(shape)
    if method == "lecun_normal":
        return backend.normal(shape, std=gain/math.sqrt(fan_in))
    if method == "orthogonal":
        return orthogonal(shape, backend, gain)
    raise ValueError(f"Unknown initializer: {method}")


def orthogonal(shape, backend, gain=1.0):
    if len(shape) < 2:
        raise ValueError("Orthogonal initialization requires a matrix")
    rows = shape[0]
    columns = math.prod(shape[1:])
    with backend.scope():
        xp = backend.xp
        a = backend.normal((max(rows, columns), min(rows, columns)))
        q, r = xp.linalg.qr(a.astype("float64"))
        sign = xp.sign(xp.diag(r))
        sign = xp.where(sign == 0, 1, sign)
        q = q*sign
        if rows < columns:
            q = q.T
        return (gain*q.reshape(shape)).astype(backend.dtype)


def initialize_dense(input_features, output_features, backend, config=None):
    weights = initialize((input_features, output_features), backend, config)
    bias = backend.zeros((output_features,))
    return weights, bias


def initialize_conv(input_channels, output_channels, kernel, backend, config=None):
    from .Math_Utilities import pair
    kh, kw = pair(kernel)
    weights = initialize((output_channels, input_channels, kh, kw),
                         backend, config, layout="conv")
    return weights, backend.zeros((output_channels,))
