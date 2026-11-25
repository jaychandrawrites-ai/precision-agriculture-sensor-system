"""Trainable layers with explicit caches and manual reverse differentiation."""
from __future__ import annotations
from dataclasses import dataclass
from .Initializers import initialize_dense, initialize_conv
from .Activations import make_activation
from .Math_Utilities import conv2d_forward, conv2d_backward, maxpool_forward, maxpool_backward
from .Regularization import Dropout

@dataclass
class Parameter:
    value: object
    grad: object
    name: str
    decay: bool = True

    def zero_grad(self):
        self.grad.fill(0)

class Layer:
    def parameters(self):
        return []

    def zero_grad(self):
        for parameter in self.parameters():
            parameter.zero_grad()

class Dense(Layer):
    def __init__(self, backend, inputs, outputs, initializer=None):
        self.backend = backend
        w, b = initialize_dense(inputs, outputs, backend, initializer)
        self.weight = Parameter(w, backend.zeros(w.shape), "weight")
        self.bias = Parameter(b, backend.zeros(b.shape), "bias", False)

    def parameters(self):
        return [self.weight, self.bias]

    def forward(self, x, training=True):
        if x.ndim != 2 or x.shape[1] != self.weight.value.shape[0]:
            raise ValueError("Dense input must be batch by input features")
        with self.backend.scope():
            y = x @ self.weight.value + self.bias.value
            return self.backend.validate(y, "dense output"), x

    def backward(self, dy, cache):
        with self.backend.scope():
            self.weight.grad += cache.T @ dy
            self.bias.grad += dy.sum(axis=0)
            return dy @ self.weight.value.T

class Conv2D(Layer):
    def __init__(self, backend, inputs, outputs, kernel, stride=1, padding=0,
                 dilation=1, initializer=None):
        self.backend = backend
        w, b = initialize_conv(inputs, outputs, kernel, backend, initializer)
        self.weight = Parameter(w, backend.zeros(w.shape), "weight")
        self.bias = Parameter(b, backend.zeros(b.shape), "bias", False)
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

    def parameters(self):
        return [self.weight, self.bias]

    def forward(self, x, training=True):
        return conv2d_forward(x, self.weight.value, self.bias.value,
                              self.backend, self.stride, self.padding, self.dilation)

    def backward(self, dy, cache):
        dx, dw, db = conv2d_backward(dy, cache, self.backend)
        with self.backend.scope():
            self.weight.grad += dw
            self.bias.grad += db
        return dx

class MaxPool(Layer):
    def __init__(self, backend, kernel=2, stride=None, padding=0):
        self.backend = backend
        self.kernel = kernel
        self.stride = stride
        self.padding = padding

    def forward(self, x, training=True):
        return maxpool_forward(x, self.backend, self.kernel, self.stride, self.padding)

    def backward(self, dy, cache):
        return maxpool_backward(dy, cache, self.backend)

class Flatten(Layer):
    def forward(self, x, training=True):
        return x.reshape(x.shape[0], -1), x.shape

    def backward(self, dy, cache):
        return dy.reshape(cache)

class ActivationLayer(Layer):
    def __init__(self, backend, name, **kwargs):
        self.activation = make_activation(name, backend, **kwargs)

    def forward(self, x, training=True):
        return self.activation.forward(x)

    def backward(self, dy, cache):
        return self.activation.backward(dy, cache)

class DropoutLayer(Layer):
    def __init__(self, backend, probability=0.1):
        self.dropout = Dropout(backend, probability)

    def forward(self, x, training=True):
        return self.dropout.forward(x, training)

    def backward(self, dy, cache):
        return self.dropout.backward(dy, cache)

@dataclass
class NormCache:
    normalized: object
    inverse_std: object
    axes: tuple
    broadcast_shape: tuple
    training: bool

class BatchNorm(Layer):
    """Normalize channels in NC or NCHW tensors."""
    def __init__(self, backend, channels, momentum=0.1, epsilon=1e-5):
        if channels <= 0 or not 0 < momentum <= 1 or epsilon <= 0:
            raise ValueError("Invalid normalization configuration")
        self.backend = backend
        self.channels = channels
        self.momentum = momentum
        self.epsilon = epsilon
        self.gamma = Parameter(backend.ones((channels,)), backend.zeros((channels,)), "gamma", False)
        self.beta = Parameter(backend.zeros((channels,)), backend.zeros((channels,)), "beta", False)
        self.running_mean = backend.zeros((channels,), dtype="float32")
        self.running_var = backend.ones((channels,), dtype="float32")

    def parameters(self):
        return [self.gamma, self.beta]

    def forward(self, x, training=True):
        if x.ndim not in (2, 4) or x.shape[1] != self.channels:
            raise ValueError("BatchNorm expects NC or NCHW")
        axes = (0,) + tuple(range(2, x.ndim))
        shape = (1, self.channels) + (1,)*(x.ndim-2)
        with self.backend.scope():
            xp = self.backend.xp
            work = x.astype("float64" if x.dtype.itemsize == 8 else "float32")
            if training:
                mean = work.mean(axis=axes, keepdims=True)
                variance = work.var(axis=axes, keepdims=True)
                self.running_mean *= 1-self.momentum
                self.running_mean += self.momentum*mean.reshape(-1)
                self.running_var *= 1-self.momentum
                self.running_var += self.momentum*variance.reshape(-1)
            else:
                mean = self.running_mean.reshape(shape)
                variance = self.running_var.reshape(shape)
            inverse = 1/xp.sqrt(variance+self.epsilon)
            normalized = (work-mean)*inverse
            y = normalized*self.gamma.value.reshape(shape)+self.beta.value.reshape(shape)
            return y.astype(x.dtype), NormCache(normalized, inverse, axes, shape, training)

    def backward(self, dy, cache):
        with self.backend.scope():
            axes = cache.axes
            normalized = cache.normalized
            self.gamma.grad += (dy*normalized).sum(axis=axes)
            self.beta.grad += dy.sum(axis=axes)
            g = dy*self.gamma.value.reshape(cache.broadcast_shape)
            if cache.training:
                g = g-g.mean(axis=axes, keepdims=True)-normalized*(g*normalized).mean(axis=axes, keepdims=True)
            return (g*cache.inverse_std).astype(dy.dtype)

class Sequential(Layer):
    def __init__(self, layers):
        self.layers = list(layers)

    def parameters(self):
        result = []
        seen = set()
        for layer in self.layers:
            for p in layer.parameters():
                if id(p) not in seen:
                    result.append(p)
                    seen.add(id(p))
        return result

    def forward(self, x, training=True):
        caches = []
        for layer in self.layers:
            x, cache = layer.forward(x, training)
            caches.append(cache)
        return x, caches

    def backward(self, dy, caches):
        if len(caches) != len(self.layers):
            raise ValueError("Cache count mismatch")
        for layer, cache in zip(reversed(self.layers), reversed(caches)):
            dy = layer.backward(dy, cache)
        return dy

    def state_dict(self, backend):
        state = {}
        for i, layer in enumerate(self.layers):
            for j, p in enumerate(layer.parameters()):
                state[f"{i}.{j}.{p.name}"] = backend.to_host(p.value, copy=True)
            if isinstance(layer, BatchNorm):
                state[f"{i}.running_mean"] = backend.to_host(layer.running_mean, copy=True)
                state[f"{i}.running_var"] = backend.to_host(layer.running_var, copy=True)
        return state

    def load_state_dict(self, state, backend):
        expected = self.state_dict(backend)
        if set(expected) != set(state):
            raise ValueError("Checkpoint keys mismatch")
        for key, value in state.items():
            if value.shape != expected[key].shape:
                raise ValueError(f"Checkpoint shape mismatch: {key}")
        with backend.scope():
            for i, layer in enumerate(self.layers):
                for j, p in enumerate(layer.parameters()):
                    p.value[...] = backend.array(state[f"{i}.{j}.{p.name}"], dtype=p.value.dtype)
                if isinstance(layer, BatchNorm):
                    layer.running_mean[...] = backend.array(state[f"{i}.running_mean"], dtype="float32")
                    layer.running_var[...] = backend.array(state[f"{i}.running_var"], dtype="float32")

class Residual(Layer):
    def __init__(self, branch, projection=None):
        self.branch = branch
        self.projection = projection

    def parameters(self):
        return self.branch.parameters() + ([] if self.projection is None else self.projection.parameters())

    def forward(self, x, training=True):
        y, cache = self.branch.forward(x, training)
        skip, skip_cache = (x, None) if self.projection is None else self.projection.forward(x, training)
        if y.shape != skip.shape:
            raise ValueError("Residual shape mismatch")
        return y+skip, (cache, skip_cache)

    def backward(self, dy, cache):
        main = self.branch.backward(dy, cache[0])
        skip = dy if self.projection is None else self.projection.backward(dy, cache[1])
        return main+skip
