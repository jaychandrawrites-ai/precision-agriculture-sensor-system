"""Six-output dense temporal model with optional projected residual blocks."""
from Framework.Layers import ActivationLayer, Dense, DropoutLayer, Residual, Sequential


class TemporalModel(Sequential):
    def __init__(self, backend, hidden, dropout=0.0, skips=()):
        self.backend = backend
        self.hidden = tuple(hidden)
        self.dropout = dropout
        self.skips = tuple(sorted(set(skips)))
        if (not self.hidden or min(self.hidden) < 1 or not 0 <= dropout < 1
                or any(not 0 <= index < len(self.hidden) for index in self.skips)):
            raise ValueError("Invalid temporal architecture")
        layers = []
        inputs = 18
        for index, width in enumerate(self.hidden):
            block = [Dense(backend, inputs, width), ActivationLayer(backend, "relu")]
            if dropout:
                block.append(DropoutLayer(backend, dropout))
            if index in self.skips:
                projection = Dense(backend, inputs, width) if inputs != width else None
                layers.append(Residual(Sequential(block), projection))
            else:
                layers.extend(block)
            inputs = width
        super().__init__([*layers, Dense(backend, inputs, 6)])

    def forward(self, x, training=True):
        with self.backend.scope():
            return super().forward(x, training)

    def backward(self, dy, cache):
        with self.backend.scope():
            return super().backward(dy, cache)

    @property
    def parameter_count(self):
        return sum(parameter.value.size for parameter in self.parameters())
