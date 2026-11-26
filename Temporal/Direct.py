"""Experimental direct multi-horizon forecasts with manual gradients."""
from dataclasses import asdict

import numpy as np

from Data.Dataloader import DataLoader
from Data.Derivatives import lag_coefficients
from Data.Normalization import Standardizer
from Framework.Activations import make_activation
from Framework.Layers import ActivationLayer, Dense, DropoutLayer, Layer, Parameter, Residual, Sequential
from Framework.Memory import estimate_memory, observed_process_peak, plan_batch
from Framework.Optimizer import Adagrad, Adam, GradientAccumulator, RMSProp, SGD
from Framework.Scheduler import CosineAnnealing, EarlyStopping, ReduceOnPlateau
from Physics.Constraints import project
from Temporal.Losses import regression_loss
from Temporal.SequenceForecast import DENSE_FAMILIES, SequenceForecastModel, SequenceSpec, configured_sequence


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
    def __init__(self, backend, width, activation="tanh"):
        self.signal = Dense(backend, width, width)
        self.gate = Dense(backend, width, width)
        self.activation = make_activation(activation, backend)
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
    if isinstance(architecture, SequenceSpec):
        spec = architecture
        if spec.family in DENSE_FAMILIES:
            layers = [DerivativeGain(backend)] if spec.family == "gated" else []
            layers.append(Dense(backend, 18, spec.width))
            for _ in range(spec.depth):
                if spec.family == "gated":
                    layer = GatedDense(backend, spec.width, spec.activation)
                else:
                    layer = Sequential([Dense(backend, spec.width, spec.width),
                                        ActivationLayer(backend, spec.activation)])
                layers.append(Residual(layer) if spec.residual else layer)
                if spec.dropout:
                    layers.append(DropoutLayer(backend, spec.dropout))
            layers.append(Dense(backend, spec.width, 6 * horizon))
            model = Sequential(layers)
        else:
            model = configured_sequence(backend, horizon, spec)
        model.architecture, model.horizon = spec, horizon
        if sum(p.value.size for p in model.parameters()) != spec.parameter_count(horizon):
            raise RuntimeError("Model parameter accounting does not match constructed layers")
        return model
    if architecture == "gated":
        return DirectModel(backend, horizon, width)
    return SequenceForecastModel(backend, horizon, width, architecture)


def direct_features(histories, architecture="gated"):
    histories = np.asarray(histories, dtype=float)
    context = architecture.context if isinstance(architecture, SequenceSpec) else 3
    family = architecture.family if isinstance(architecture, SequenceSpec) else architecture
    if histories.ndim != 3 or histories.shape[-1] != 6 or histories.shape[1] < context:
        raise ValueError(f"Expected at least {context} six-feature history observations")
    histories = histories[:, -context:]
    if not np.isfinite(histories).all():
        raise ValueError("Context contains nonfinite observations")
    if family not in DENSE_FAMILIES:
        return histories.reshape(len(histories), 6 * context)
    return (lag_coefficients(.65) @ histories).reshape(len(histories), 18)


class DirectForecast:
    def __init__(self, model, backend, input_scaler, target_scaler):
        self.model, self.backend = model, backend
        self.inputs, self.targets = input_scaler, target_scaler

    @property
    def context_hours(self):
        spec = self.model.architecture
        return spec.context if isinstance(spec, SequenceSpec) else 3

    def predict(self, histories, *, anchor=None, physical=True):
        inputs = self.inputs.transform(direct_features(histories, self.model.architecture))
        with self.backend.scope():
            prediction, _ = self.model.forward(self.backend.array(inputs), training=False)
        output = self.backend.to_host(prediction).reshape(-1, self.model.horizon, 6)
        base = _forecast_anchor(histories, self.model.horizon, anchor)
        raw = base + output * self.targets.scale
        return project(raw) if physical else raw

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
               architecture="gated", memory_budget_bytes=None, optimizer="adam", weight_decay=0.,
               scheduler="cosine", warmup_epochs=0, patience=None):
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
    normalized = inputs.transform(features)
    batches = DataLoader(normalized, target.reshape(len(target), -1), settings.batch_size,
                         seed=settings.seed)
    optimizers = {"adam": (Adam, 2), "adamw": (Adam, 2), "sgd": (SGD, 0),
                  "rmsprop": (RMSProp, 1), "adagrad": (Adagrad, 1)}
    if optimizer not in optimizers or scheduler not in ("cosine", "plateau"):
        raise ValueError("Unknown direct optimizer or scheduler")
    if optimizer == "adamw" and weight_decay == 0:
        weight_decay = .01
    factory, slots = optimizers[optimizer]
    opt = factory(model.parameters(), backend, learning_rate=settings.learning_rate,
                  max_norm=settings.gradient_norm, weight_decay=weight_decay)
    if scheduler == "plateau" and warmup_epochs:
        raise ValueError("Warmup is implemented for the cosine schedule")
    minimum = settings.learning_rate / 100
    schedule = (CosineAnnealing(opt, settings.epochs, minimum=minimum, warmup=warmup_epochs)
                if scheduler == "cosine" else ReduceOnPlateau(opt, minimum=minimum))
    stop_policy = None if patience is None else EarlyStopping(patience)
    estimate = plan = accumulator = None
    if memory_budget_bytes is not None:
        estimate = estimate_memory(model, backend, normalized[:1], optimizer_slots=slots)
        plan = plan_batch(estimate, settings.batch_size, memory_budget_bytes)
        if plan.accumulated:
            accumulator = GradientAccumulator(opt)
    feature_weights = np.repeat(weight, 6)
    best, val_trace, train_trace, clipped, peak = np.inf, [], [], 0, 0
    measured_pool = None
    with backend.scope():
        for _ in range(settings.epochs):
            loss_sum = 0.
            for x, y, _ in batches:
                for first in range(0, len(x), plan.microbatch if plan else len(x)):
                    last = min(first + plan.microbatch, len(x)) if plan else len(x)
                    opt.zero_grad()
                    raw, cache = model.forward(backend.array(x[first:last]))
                    loss, gradient = regression_loss(raw, backend.array(y[first:last]), backend,
                                                     feature_weights=feature_weights)
                    loss_sum += loss * (last - first)
                    model.backward(gradient, cache)
                    if accumulator is not None:
                        accumulator.add(last - first)
                    if plan is not None and backend.is_gpu:
                        used = backend.memory_report()["pool_bytes"]
                        measured_pool = used if measured_pool is None else max(used, measured_pool)
                if accumulator is not None:
                    accumulator.flush()
                update = opt.step()
                peak = max(peak, update["gradient_norm"])
                clipped += update["clip_factor"] < 1
            train_trace.append(loss_sum / len(target))
            val_sum = 0.
            count = len(validation.targets)
            batch = plan.microbatch if plan else count
            for first in range(0, count, batch):
                target_batch = validation.targets[first:first+batch]
                history_batch = validation.histories[first:first+batch]
                baseline = None if validation_anchor is None else validation_anchor[first:first+batch]
                residual = (forecast.predict(history_batch, anchor=baseline) - target_batch) / target_scaler.scale
                matrix = backend.array(residual.reshape(len(residual), -1))
                score, _ = regression_loss(matrix, backend.zeros(matrix.shape), backend,
                                           feature_weights=feature_weights)
                val_sum += score * len(target_batch)
            score = val_sum / count
            val_trace.append(score)
            improved, stop = stop_policy.observe(score) if stop_policy else (score < best, False)
            if improved:
                best = score
                weights = model.state_dict(backend)
            if scheduler == "cosine":
                schedule.step()
            else:
                schedule.step(score)
            if stop:
                break
    model.load_state_dict(weights, backend)
    report = {
        "architecture": architecture.as_dict() if isinstance(architecture, SequenceSpec) else architecture,
        "training_loss": train_trace, "validation_loss": val_trace,
        "parameters": sum(p.value.size for p in model.parameters()),
        "clipped_updates": clipped, "peak_gradient_norm": peak, "updates": opt.steps,
        "optimizer": optimizer, "scheduler": scheduler, "weight_decay": weight_decay,
        "early_stopping": stop_policy.state_dict() if stop_policy else None,
    }
    if plan:
        report["memory"] = {
            "estimate": asdict(estimate), "plan": asdict(plan),
            "process_peak": observed_process_peak(),
            "sampled_device_pool_peak_bytes": measured_pool,
        }
    return forecast, report
