"""Experimental direct multi-horizon forecasts with manual gradients."""
import numpy as np

from Data.Dataloader import DataLoader
from Data.Derivatives import lag_coefficients
from Data.Normalization import Standardizer
from Framework.Activations import make_activation
from Framework.Layers import Dense, Layer, Parameter, Residual, Sequential
from Framework.Optimizer import Adam
from Framework.Scheduler import CosineAnnealing
from Physics.Constraints import project
from Temporal.Losses import regression_loss
from Temporal.SequenceForecast import SequenceForecastModel


DIRECT_ARCHITECTURES = ("gated", "gru", "convolution", "attention", "lstm", "lstm_attention")


class DerivativeGain(Layer):
    def __init__(self, backend):
        self.gain = Parameter(backend.ones((12,)), backend.zeros((12,)), "gain", False)

    def parameters(self):
        return [self.gain]

    def forward(self, x, training=True):
        output = x.copy()
        output[:, 6:] *= self.gain.value
        return output, x

    def backward(self, dy, cache):
        self.gain.grad += (dy[:, 6:] * cache[:, 6:]).sum(axis=0)
        gradient = dy.copy()
        gradient[:, 6:] *= self.gain.value
        return gradient


class GatedDense(Layer):
    def __init__(self, backend, width):
        self.signal = Dense(backend, width, width)
        self.gate = Dense(backend, width, width)
        self.activation = make_activation("tanh", backend)
        self.sigmoid = make_activation("sigmoid", backend)

    def parameters(self):
        return self.signal.parameters() + self.gate.parameters()

    def forward(self, x, training=True):
        value, signal_cache = self.signal.forward(x, training)
        gate, gate_cache = self.gate.forward(x, training)
        signal, signal_activation = self.activation.forward(value)
        weight, gate_activation = self.sigmoid.forward(gate)
        return signal * weight, (signal, weight, signal_cache, gate_cache,
                                 signal_activation, gate_activation)

    def backward(self, dy, cache):
        signal, weight, signal_cache, gate_cache, signal_activation, gate_activation = cache
        ds = self.activation.backward(dy * weight, signal_activation)
        dg = self.sigmoid.backward(dy * signal, gate_activation)
        return self.signal.backward(ds, signal_cache) + self.gate.backward(dg, gate_cache)


class DirectModel(Sequential):
    architecture = "gated"

    def __init__(self, backend, horizon, width=32):
        if min(horizon, width) < 1:
            raise ValueError("Invalid direct model size")
        super().__init__([DerivativeGain(backend), Dense(backend, 18, width),
                          Residual(GatedDense(backend, width)), Dense(backend, width, horizon * 6)])
        self.horizon = horizon


def make_direct_model(backend, horizon, width, architecture="gated"):
    if architecture == "gated":
        return DirectModel(backend, horizon, width)
    return SequenceForecastModel(backend, horizon, width, architecture)


def direct_features(histories, architecture="gated"):
    histories = np.asarray(histories, dtype=float)
    if histories.ndim != 3 or histories.shape[1:] != (3, 6) or not np.isfinite(histories).all():
        raise ValueError("Expected finite [origin, 3, 6] observations")
    if architecture != "gated":
        return histories.reshape(len(histories), 18)
    return (lag_coefficients(.65) @ histories).reshape(len(histories), 18)


class DirectForecast:
    def __init__(self, model, backend, input_scaler, target_scaler):
        self.model, self.backend = model, backend
        self.inputs, self.targets = input_scaler, target_scaler

    def predict(self, histories, *, anchor=None):
        inputs = self.inputs.transform(direct_features(histories, self.model.architecture))
        with self.backend.scope():
            prediction, _ = self.model.forward(self.backend.array(inputs), training=False)
        output = self.backend.to_host(prediction).reshape(-1, self.model.horizon, 6)
        base = _forecast_anchor(histories, self.model.horizon, anchor)
        return project(base + output * self.targets.scale)

    def ood_score(self, histories):
        z = self.inputs.transform(direct_features(histories, self.model.architecture))
        return np.max(np.abs(z), axis=1)


def _forecast_anchor(histories, horizon, anchor):
    histories = np.asarray(histories)
    if anchor is None:
        return histories[:, -1, None, :]
    baseline = np.asarray(anchor, dtype=float)
    if baseline.shape != (len(histories), horizon, 6) or not np.isfinite(baseline).all():
        raise ValueError("Expected one finite seasonal forecast per origin, lead and feature")
    return baseline


def fit_direct(training, validation, target_scaler, backend, settings, *, width=32,
               horizon_weights=None, training_anchor=None, validation_anchor=None,
               architecture="gated"):
    if (not len(training.starts) or not len(validation.starts)
            or training.targets.shape[1:] != validation.targets.shape[1:]):
        raise ValueError("Need matching nonempty direct forecast windows")
    horizon = training.targets.shape[1]
    weight = np.ones(horizon) if horizon_weights is None else np.asarray(horizon_weights, dtype=float)
    if weight.shape != (horizon,) or not np.isfinite(weight).all() or (weight < 0).any() or not weight.any():
        raise ValueError("Use one nonnegative weight per horizon")
    model = make_direct_model(backend, horizon, width, architecture)
    features = direct_features(training.histories, architecture)
    inputs = Standardizer().fit(features)
    forecast = DirectForecast(model, backend, inputs, target_scaler)
    if (training_anchor is None) != (validation_anchor is None):
        raise ValueError("Provide the same anchor strategy for fit and validation")
    base = _forecast_anchor(training.histories, horizon, training_anchor)
    target = (training.targets - base) / target_scaler.scale
    batches = DataLoader(inputs.transform(features), target.reshape(len(target), -1),
                         settings.batch_size, seed=settings.seed)
    opt = Adam(model.parameters(), backend, learning_rate=settings.learning_rate,
               max_norm=settings.gradient_norm)
    schedule = CosineAnnealing(opt, settings.epochs, minimum=settings.learning_rate / 100)
    feature_weights = np.repeat(weight, 6)
    best, val_trace, train_trace, clipped, peak = np.inf, [], [], 0, 0
    with backend.scope():
        for _ in range(settings.epochs):
            loss_sum = 0.
            for x, y, _ in batches:
                opt.zero_grad()
                raw, cache = model.forward(backend.array(x))
                loss, gradient = regression_loss(raw, backend.array(y), backend,
                                                 feature_weights=feature_weights)
                loss_sum += loss * len(x)
                model.backward(gradient, cache)
                update = opt.step()
                peak = max(peak, update["gradient_norm"])
                clipped += update["clip_factor"] < 1
            train_trace.append(loss_sum / len(target))
            residual = (forecast.predict(validation.histories, anchor=validation_anchor) - validation.targets) / target_scaler.scale
            matrix = backend.array(residual.reshape(len(residual), -1))
            score, _ = regression_loss(matrix, backend.zeros(matrix.shape), backend,
                                       feature_weights=feature_weights)
            val_trace.append(score)
            if score < best:
                best = score
                weights = model.state_dict(backend)
            schedule.step()
    model.load_state_dict(weights, backend)
    return forecast, {"architecture": architecture, "training_loss": train_trace, "validation_loss": val_trace,
                      "parameters": sum(p.value.size for p in model.parameters()),
                      "clipped_updates": clipped, "peak_gradient_norm": peak}
