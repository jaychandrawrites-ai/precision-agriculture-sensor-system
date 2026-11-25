"""Explicit tensor mathematics for manual differentiation.

Convolution uses NCHW layout and cross-correlation, matching common neural
network conventions. All operations accept an explicit Backend instance.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product
import math

from .Backend import Backend


def pair(value):
    if isinstance(value, int):
        value = (value, value)
    value = tuple(value)
    if len(value) != 2 or any(not isinstance(x, int) or x <= 0 for x in value):
        raise ValueError("Expected two positive integers")
    return value


def padding_tuple(value):
    if isinstance(value, int):
        value = (value,) * 4
    elif len(value) == 2:
        value = (value[0], value[0], value[1], value[1])
    value = tuple(value)
    if len(value) != 4 or any(not isinstance(x, int) or x < 0 for x in value):
        raise ValueError("Expected nonnegative top, bottom, left, right padding")
    return value


@dataclass(frozen=True)
class WindowGeometry:
    input_shape: tuple
    kernel: tuple
    stride: tuple
    dilation: tuple
    padding: tuple
    output_shape: tuple

    @classmethod
    def build(cls, input_shape, kernel, stride=1, padding=0, dilation=1):
        if len(input_shape) != 4 or any(x <= 0 for x in input_shape):
            raise ValueError("Expected positive NCHW input dimensions")
        kernel = pair(kernel)
        stride = pair(stride)
        dilation = pair(dilation)
        padding = padding_tuple(padding)
        n, c, h, w = input_shape
        kh, kw = kernel
        sh, sw = stride
        dh, dw = dilation
        top, bottom, left, right = padding
        effective_h = (kh - 1) * dh + 1
        effective_w = (kw - 1) * dw + 1
        oh = (h + top + bottom - effective_h) // sh + 1
        ow = (w + left + right - effective_w) // sw + 1
        if oh <= 0 or ow <= 0:
            raise ValueError("Kernel exceeds padded input")
        return cls(tuple(input_shape), kernel, stride, dilation, padding,
                   (n, c, oh, ow))

    @property
    def column_shape(self):
        n, c, oh, ow = self.output_shape
        kh, kw = self.kernel
        return n * oh * ow, c * kh * kw


def same_padding(height, width, kernel, stride=1, dilation=1):
    kh, kw = pair(kernel)
    sh, sw = pair(stride)
    dh, dw = pair(dilation)
    oh = math.ceil(height / sh)
    ow = math.ceil(width / sw)
    ph = max(0, (oh - 1) * sh + (kh - 1) * dh + 1 - height)
    pw = max(0, (ow - 1) * sw + (kw - 1) * dw + 1 - width)
    return ph // 2, ph - ph // 2, pw // 2, pw - pw // 2


def pad_nchw(x, padding, backend, value=0):
    top, bottom, left, right = padding_tuple(padding)
    with backend.scope():
        return backend.xp.pad(
            x, ((0, 0), (0, 0), (top, bottom), (left, right)),
            mode="constant", constant_values=value,
        )


def im2col(x, geometry, backend, pad_value=0):
    """Return one row per spatial window, ordered N, H, W."""
    if tuple(x.shape) != geometry.input_shape:
        raise ValueError("Input shape does not match geometry")
    n, c, oh, ow = geometry.output_shape
    kh, kw = geometry.kernel
    sh, sw = geometry.stride
    dh, dw = geometry.dilation
    with backend.scope():
        padded = pad_nchw(x, geometry.padding, backend, pad_value)
        windows = backend.xp.empty((n, oh, ow, c, kh, kw), dtype=x.dtype)
        for i in range(kh):
            for j in range(kw):
                patch = padded[:, :, i*dh:i*dh+oh*sh:sh,
                               j*dw:j*dw+ow*sw:sw]
                windows[:, :, :, :, i, j] = patch.transpose(0, 2, 3, 1)
        return windows.reshape(geometry.column_shape)


def col2im(columns, geometry, backend):
    """Adjoint of im2col: overlapping window contributions are summed."""
    if tuple(columns.shape) != geometry.column_shape:
        raise ValueError("Column shape does not match geometry")
    n, c, h, w = geometry.input_shape
    _, _, oh, ow = geometry.output_shape
    kh, kw = geometry.kernel
    sh, sw = geometry.stride
    dh, dw = geometry.dilation
    top, bottom, left, right = geometry.padding
    with backend.scope():
        padded = backend.xp.zeros(
            (n, c, h+top+bottom, w+left+right), dtype=columns.dtype)
        windows = columns.reshape(n, oh, ow, c, kh, kw)
        for i in range(kh):
            for j in range(kw):
                patch = windows[:, :, :, :, i, j].transpose(0, 3, 1, 2)
                padded[:, :, i*dh:i*dh+oh*sh:sh,
                       j*dw:j*dw+ow*sw:sw] += patch
        return padded[:, :, top:top+h, left:left+w]


@dataclass
class ConvCache:
    geometry: WindowGeometry
    columns: object
    weights: object
    has_bias: bool


def conv2d_forward(x, weights, bias, backend, stride=1, padding=0, dilation=1):
    if x.ndim != 4 or weights.ndim != 4:
        raise ValueError("Expected NCHW input and OIHW weights")
    filters, channels, kh, kw = weights.shape
    if channels != x.shape[1]:
        raise ValueError("Convolution channel mismatch")
    geometry = WindowGeometry.build(x.shape, (kh, kw), stride, padding, dilation)
    with backend.scope():
        columns = im2col(x, geometry, backend)
        flat_weights = weights.reshape(filters, -1)
        output = columns @ flat_weights.T
        if bias is not None:
            if bias.shape != (filters,):
                raise ValueError("Bias must contain one value per filter")
            output = output + bias
        n, _, oh, ow = geometry.output_shape
        output = output.reshape(n, oh, ow, filters).transpose(0, 3, 1, 2)
        backend.validate(output, "convolution output")
        return output, ConvCache(geometry, columns, weights, bias is not None)


def conv2d_backward(gradient, cache, backend):
    geometry = cache.geometry
    n, _, oh, ow = geometry.output_shape
    filters = cache.weights.shape[0]
    if gradient.shape != (n, filters, oh, ow):
        raise ValueError("Unexpected convolution gradient shape")
    with backend.scope():
        dy = gradient.transpose(0, 2, 3, 1).reshape(-1, filters)
        dw = (dy.T @ cache.columns).reshape(cache.weights.shape)
        db = dy.sum(axis=0) if cache.has_bias else None
        dx_columns = dy @ cache.weights.reshape(filters, -1)
        dx = col2im(dx_columns, geometry, backend)
        return dx, dw, db


@dataclass
class PoolCache:
    geometry: WindowGeometry
    indices: object


def maxpool_forward(x, backend, kernel=2, stride=None, padding=0):
    geometry = WindowGeometry.build(x.shape, kernel, stride or kernel, padding)
    n, c, oh, ow = geometry.output_shape
    kh, kw = geometry.kernel
    with backend.scope():
        columns = im2col(x, geometry, backend, pad_value=-float("inf"))
        windows = columns.reshape(n*oh*ow, c, kh*kw)
        indices = windows.argmax(axis=-1)
        values = backend.xp.take_along_axis(windows, indices[..., None], axis=-1)
        output = values.reshape(n, oh, ow, c).transpose(0, 3, 1, 2)
        backend.require_finite(output, "pool output")
        return output, PoolCache(geometry, indices)


def maxpool_backward(gradient, cache, backend):
    n, c, oh, ow = cache.geometry.output_shape
    kh, kw = cache.geometry.kernel
    if gradient.shape != (n, c, oh, ow):
        raise ValueError("Unexpected pooling gradient shape")
    with backend.scope():
        dy = gradient.transpose(0, 2, 3, 1).reshape(-1, c)
        windows = backend.xp.zeros((n*oh*ow, c, kh*kw), dtype=gradient.dtype)
        rows = backend.xp.arange(n*oh*ow)[:, None]
        channels = backend.xp.arange(c)[None, :]
        windows[rows, channels, cache.indices] = dy
        return col2im(windows.reshape(cache.geometry.column_shape),
                      cache.geometry, backend)


def unbroadcast(gradient, shape):
    """Sum a broadcast gradient back to its original operand shape."""
    shape = tuple(shape)
    if len(shape) > gradient.ndim:
        raise ValueError("Target has more dimensions than gradient")
    while gradient.ndim > len(shape):
        gradient = gradient.sum(axis=0)
    for axis, size in enumerate(shape):
        if size == 1 and gradient.shape[axis] != 1:
            gradient = gradient.sum(axis=axis, keepdims=True)
        elif size != gradient.shape[axis]:
            raise ValueError("Incompatible broadcast shape")
    return gradient.reshape(shape)


def logsumexp(x, backend, axis=-1, keepdims=False):
    with backend.scope():
        maximum = backend.xp.max(x, axis=axis, keepdims=True)
        shifted = x - maximum
        result = maximum + backend.xp.log(
            backend.xp.sum(backend.xp.exp(shifted), axis=axis, keepdims=True))
        return result if keepdims else backend.xp.squeeze(result, axis=axis)


def finite_difference_gradient(function, x, backend, epsilon=1e-5):
    """Central finite difference of a scalar function; intended for tests."""
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    with backend.scope():
        work = backend.array(x, dtype="float64", copy=True)
        gradient = backend.xp.zeros_like(work)
        for index in product(*(range(size) for size in work.shape)):
            original = float(backend.scalar(work[index]))
            step = epsilon * max(1.0, abs(original))
            try:
                work[index] = original + step
                positive = float(backend.scalar(function(work)))
                work[index] = original - step
                negative = float(backend.scalar(function(work)))
                gradient[index] = (positive-negative)/(2*step)
            finally:
                work[index] = original
        return gradient


def gradient_error(analytical, numerical, backend, epsilon=1e-12):
    with backend.scope():
        a = backend.array(analytical, dtype="float64")
        b = backend.array(numerical, dtype="float64")
        if a.shape != b.shape:
            raise ValueError("Gradient shapes differ")
        difference = a-b
        return {
            "relative_l2": backend.global_norm([difference]) /
                           max(epsilon, backend.global_norm([a])+backend.global_norm([b])),
            "maximum_absolute": float(backend.scalar(backend.xp.max(
                backend.xp.abs(difference)))),
        }


def backward_derivatives(values, dt, backend):
    """Return aligned values, first differences and second differences."""
    if dt <= 0 or values.shape[0] < 3:
        raise ValueError("Need positive interval and at least three samples")
    with backend.scope():
        current = values[2:]
        previous = values[1:-1]
        earlier = values[:-2]
        velocity = (current-previous)/dt
        acceleration = (current-2*previous+earlier)/(dt*dt)
        return current, velocity, acceleration


def laplacian_neumann(field, backend, dy=1.0, dx=1.0):
    """Symmetric five-point Laplacian on the final two axes."""
    if field.ndim < 2 or dx <= 0 or dy <= 0:
        raise ValueError("Invalid grid or spacing")
    with backend.scope():
        padding = [(0, 0)]*(field.ndim-2)+[(1, 1), (1, 1)]
        padded = backend.xp.pad(field, padding, mode="edge")
        center = padded[..., 1:-1, 1:-1]
        vertical = (padded[..., :-2, 1:-1]+padded[..., 2:, 1:-1]-2*center)/(dy*dy)
        horizontal = (padded[..., 1:-1, :-2]+padded[..., 1:-1, 2:]-2*center)/(dx*dx)
        return vertical+horizontal


def biharmonic_neumann(field, backend, dy=1.0, dx=1.0):
    return laplacian_neumann(laplacian_neumann(field, backend, dy, dx),
                             backend, dy, dx)
