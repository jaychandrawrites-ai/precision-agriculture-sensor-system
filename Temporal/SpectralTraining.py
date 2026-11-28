"""Manual spectral learning on complete chronological training windows."""
from time import perf_counter

import numpy as np

from Data.Normalization import Standardizer
from Framework.Backend import create_backend
from Framework.Optimizer import Adam
from Physics.Constraints import LOWER, UPPER
from Temporal.LayerDiagnostics import LayerDiagnostics
from Temporal.Model import TemporalModel
from Temporal.Spectral import HORIZON, SpectralForecaster, features, loss, sparse_events, residual_transform


def fit(training, validation, times, scaler, architecture, modes, settings,
        weights, augmentation=0., augmentation_bound=.25, *, diagnostics=False, residual_width=None, joint_epochs=0):
    backend = create_backend(settings.device, settings.dtype, settings.seed)
    rng = np.random.default_rng(settings.seed)
    events_rng = np.random.default_rng(settings.seed + 1)
    model = TemporalModel(backend, **architecture, dropout=settings.dropout,
                          input_size=22, output_size=6 * modes.width)
    opt = Adam(model.parameters(), backend, learning_rate=settings.learning_rate,
               max_norm=settings.gradient_norm)
    residual = None
    if residual_width is not None:
        residual = TemporalModel(backend, (residual_width,), activations=("tanh",),
                                 input_size=22, output_size=HORIZON * 6)
        residual_opt = Adam(residual.parameters(), backend, learning_rate=settings.learning_rate,
                            max_norm=settings.gradient_norm)
    train_inputs = features(training.histories, times[training.starts - 1])
    inputs = Standardizer().fit(train_inputs)
    basis = modes.anchored_basis(np.arange(1, HORIZON + 1))
    inverse = np.linalg.pinv(basis, rcond=1e-8)
    lower = backend.array(scaler.transform(LOWER))
    upper = backend.array(scaler.transform(UPPER))
    device_basis = backend.array(basis)

    def prepare(window, x):
        y = scaler.transform(window.targets)
        anchor = scaler.transform(window.histories[:, -1:])
        return inputs.transform(x), backend.array(y), backend.array(anchor), backend.array(inverse @ (y - anchor))

    train = prepare(training, train_inputs)
    valid = prepare(validation, features(validation.histories, times[validation.starts - 1]))
    best = float("inf")
    history = {"training_loss": [], "validation_loss": [], "gradient_norm": [],
               "validation_components": [], "event_counts": []}
    recorder = LayerDiagnostics(model, backend) if diagnostics else None
    history.update(layer_diagnostics=[], learning_rate=[], epoch_seconds=[], clipped_updates=0)
    index = 1
    start = perf_counter()
    def objective(batch, x, selection, phase, training):
        spectral_out, cache = model.forward(x, training=training and phase != "residual")
        coefficients = spectral_out.reshape(-1, modes.width, 6)
        arguments = (coefficients, batch[1][selection], batch[2][selection], device_basis,
                     batch[3][selection], lower, upper, weights, backend.xp)
        if phase == "spectral":
            score, gradient, terms = loss(*arguments)
        else:
            residual_out, residual_cache = residual.forward(x, training=training)
            correction, derivative = residual_transform(residual_out, backend.xp)
            score, gradient, terms, correction_gradient = loss(*arguments, correction=correction)
            if training:
                residual.backward((correction_gradient * derivative).reshape(residual_out.shape), residual_cache)
        if training and phase != "residual":
            model.backward(gradient.reshape(spectral_out.shape), cache)
        return score, terms

    phases = ["spectral"] * settings.epochs
    if residual is not None:
        phases += ["residual"] * settings.epochs + ["joint"] * joint_epochs
    history["phase"] = phases
    with backend.scope():
        for epoch, phase in enumerate(phases):
            if epoch == settings.epochs and residual is not None:
                model.load_state_dict(state, backend)
                history["spectral_best_epoch"] = best_epoch
                best = float("inf")
            tick = perf_counter()
            history["learning_rate"].append(opt.learning_rate)
            total = peak = count = 0
            indices = rng.permutation(len(train_inputs))
            for start_index in range(0, len(indices), settings.batch_size):
                selection = indices[start_index:start_index + settings.batch_size]
                x = train[0][selection].copy()
                if augmentation:
                    perturbation, hits = sparse_events(events_rng, index, x[:, :18].shape, augmentation,
                                                        augmentation_bound)
                    x[:, :18] += perturbation
                    index += perturbation.size
                    count += hits
                opt.zero_grad()
                if residual is not None:
                    residual_opt.zero_grad()
                score, _ = objective(train, backend.array(x), selection, phase, True)
                sample = recorder is not None and start_index == 0
                before = recorder.before() if sample else None
                updates = []
                if phase != "residual":
                    updates.append(opt.step())
                if phase != "spectral":
                    updates.append(residual_opt.step())
                peak = max(peak, *(update["gradient_norm"] for update in updates))
                history["clipped_updates"] += sum(update["clip_factor"] < 1 for update in updates)
                if sample:
                    history["layer_diagnostics"].append({
                        "epoch": epoch + 1, "layers": recorder.report(before),
                        "inference_activations": recorder.activations(backend.array(x[:8])),
                    })
                total += float(backend.scalar(score)) * len(selection)
            score, terms = objective(valid, backend.array(valid[0]), slice(None), phase, False)
            score = float(backend.scalar(score))
            history["training_loss"].append(total / len(train_inputs))
            history["validation_loss"].append(score)
            history["validation_components"].append({key: float(backend.scalar(value)) for key, value in terms.items()})
            history["event_counts"].append(count)
            history["gradient_norm"].append(peak)
            history["epoch_seconds"].append(perf_counter() - tick)
            if score < best:
                best, best_epoch = score, epoch + 1
                state = model.state_dict(backend)
                if phase != "spectral":
                    residual_state = residual.state_dict(backend)
        model.load_state_dict(state, backend)
        if residual is not None:
            residual.load_state_dict(residual_state, backend)
    history["training_seconds"] = perf_counter() - start
    history["augmentation_indices"] = index - 1
    return SpectralForecaster(model, backend, modes, inputs, scaler, residual), history, best_epoch
