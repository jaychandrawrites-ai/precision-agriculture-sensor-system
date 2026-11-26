"""Optional direct forecast models using three observed states in time order."""
from dataclasses import asdict, dataclass

from Framework.Layers import ActivationLayer, Dense, DropoutLayer, Layer, Residual, Sequential
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


FAMILIES = ("dense", "residual_dense", "gated", "gru", "lstm", "convolution",
            "attention", "gru_attention", "lstm_attention", "gru_convolution",
            "lstm_convolution")
DENSE_FAMILIES = FAMILIES[:3]


@dataclass(frozen=True)
class SequenceSpec:
    """Architecture choices for derivative and causal sequence direct models."""

    family: str
    width: int
    depth: int = 1
    activation: str = "tanh"
    residual: bool = True
    context: int = 3
    kernel: int = 2
    dilation: int = 1
    heads: int = 2
    dropout: float = 0

    def __post_init__(self):
        if self.family not in FAMILIES:
            raise ValueError("Unknown direct model family")
        size_values = (self.width, self.depth, self.context, self.kernel, self.dilation, self.heads)
        if (any(type(value) is not int for value in size_values) or not 1 <= self.width <= 256
                or not 1 <= self.depth <= 4 or not 3 <= self.context <= 168
                or not 1 <= self.kernel <= 31 or not 1 <= self.dilation <= 32
                or not 1 <= self.heads <= self.width
                or not 0 <= self.dropout < 1 or self.activation not in ("relu", "tanh", "silu", "gelu")):
            raise ValueError("Invalid direct model configuration")
        if "attention" in self.family and self.width % self.heads:
            raise ValueError("Attention width must divide into equal heads")
        if self.family in DENSE_FAMILIES and self.context != 3:
            raise ValueError("Derivative dense models require three observations")

    @property
    def receptive_field(self):
        return 1 + self.depth * self.dilation * (self.kernel - 1) if self.family == "convolution" else self.context

    def parameter_count(self, horizon):
        width = self.width
        output = (width + 1) * 6 * horizon
        if self.family in DENSE_FAMILIES:
            repeats = self.depth * (width + 1) * width
            return 19 * width + output + (2 * repeats + 12 if self.family == "gated" else repeats)
        total, inputs = 0, 6
        for _ in range(self.depth):
            if self.family.startswith("gru"):
                total += 3 * (inputs + width + 1) * width
            elif self.family.startswith("lstm"):
                total += 4 * (inputs + width + 1) * width
            else:
                kernel = self.kernel if self.family == "convolution" else 1
                total += (inputs * kernel + 1) * width
            inputs = width
        if "attention" in self.family:
            total += 4 * (width + 1) * width + 2 * width
        if self.family.endswith("_convolution"):
            total += (self.kernel * width + 1) * width
        return total + output

    def as_dict(self):
        return asdict(self)


class ConfiguredHistory(Layer):
    def __init__(self, context):
        self.context = context

    def forward(self, x, training=True):
        if x.ndim != 2 or x.shape[1] != self.context * 6:
            raise ValueError("Input length differs from model context")
        return x.reshape(len(x), self.context, 6), x.shape

    def backward(self, dy, cache):
        return dy.reshape(cache)


def configured_sequence(backend, horizon, spec):
    layers = [ConfiguredHistory(spec.context)]
    width, inputs = spec.width, 6
    for _ in range(spec.depth):
        if spec.family.startswith("gru"):
            block = GRU(backend, inputs, width)
        elif spec.family.startswith("lstm"):
            block = LSTM(backend, inputs, width)
        else:
            size = spec.kernel if spec.family == "convolution" else 1
            block = Conv1D(backend, inputs, width, size, dilation=spec.dilation)
        layers.append(Residual(block) if spec.residual and inputs == width else block)
        layers.append(ActivationLayer(backend, spec.activation))
        if spec.dropout:
            layers.append(DropoutLayer(backend, spec.dropout))
        inputs = width
    if "attention" in spec.family:
        block = MultiHeadAttention(backend, width, spec.heads)
        layers.extend((SinusoidalPosition(backend), Residual(block) if spec.residual else block,
                       LayerNorm(backend, width)))
    if spec.family.endswith("_convolution"):
        block = Conv1D(backend, width, width, spec.kernel, dilation=spec.dilation)
        layers.append(Residual(block) if spec.residual else block)
    layers.extend((LastObservation(backend), Dense(backend, width, 6 * horizon)))
    return Sequential(layers)
