"""Optional direct forecast models using three observed states in time order."""
from Framework.Layers import ActivationLayer, Dense, Layer, Residual, Sequential
from Framework.SequenceLayers import Conv1D, GRU, LSTM, LayerNorm, MultiHeadAttention, SinusoidalPosition


class RestoreHistory(Layer):
    def forward(self, x, training=True):
        if x.ndim != 2 or x.shape[1] != 18:
            raise ValueError("Expected three flattened six-feature observations")
        return x.reshape(len(x), 3, 6), x.shape

    def backward(self, dy, cache):
        return dy.reshape(cache)


class LastObservation(Layer):
    def __init__(self, backend):
        self.backend = backend

    def forward(self, x, training=True):
        return x[:, -1], x.shape

    def backward(self, dy, cache):
        with self.backend.scope():
            sequence = self.backend.zeros(cache, dtype=dy.dtype)
            sequence[:, -1] = dy
            return sequence


class SequenceForecastModel(Sequential):
    """Causal sequence encoder with a direct six-feature, multi-lead head."""

    def __init__(self, backend, horizon, width, architecture):
        if min(horizon, width) < 1:
            raise ValueError("Invalid direct model size")
        self.horizon = horizon
        self.architecture = architecture
        layers = [RestoreHistory()]
        if architecture == "gru":
            layers.append(GRU(backend, 6, width))
        elif architecture in ("lstm", "lstm_attention"):
            layers.append(LSTM(backend, 6, width))
            if architecture == "lstm_attention":
                layers.extend((SinusoidalPosition(backend),
                               Residual(MultiHeadAttention(backend, width, 2)),
                               LayerNorm(backend, width)))
        elif architecture == "convolution":
            layers.extend((Conv1D(backend, 6, width, 2),
                           ActivationLayer(backend, "tanh"),
                           Residual(Conv1D(backend, width, width, 2))))
        elif architecture == "attention":
            layers.extend((Conv1D(backend, 6, width, 1),
                           SinusoidalPosition(backend),
                           Residual(MultiHeadAttention(backend, width, 2)),
                           LayerNorm(backend, width)))
        else:
            raise ValueError("Unknown sequence forecast architecture")
        layers.extend((LastObservation(backend), Dense(backend, width, 6 * horizon)))
        super().__init__(layers)
