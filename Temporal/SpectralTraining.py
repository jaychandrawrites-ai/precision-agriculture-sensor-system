"""Manual spectral learning on complete chronological training windows."""
from time import perf_counter

import numpy as np

from Data.Normalization import Standardizer
from Framework.Backend import create_backend
from Framework.Optimizer import Adam
from Physics.Constraints import LOWER, UPPER
from Temporal.LayerDiagnostics import LayerDiagnostics
from Temporal.Model import TemporalModel
from Temporal.Spectral import HORIZON, SpectralForecaster, features, loss, sparse_events


def fit(training, validation, times, scaler, architecture, modes, settings,
        weights, augmentation=0., augmentation_bound=.25, *, diagnostics=False):
    backend = create_backend(settings.device, settings.dtype, settings.seed)
    rng = np.random.default_rng(settings.seed)
    events_rng = np.random.default_rng(settings.seed + 1)
    model = TemporalModel(backend, **architecture, dropout=settings.dropout,
                          input_size=22, output_size=6 * modes.width)
    opt = Adam(model.parameters(), backend, learning_rate=settings.learning_rate,
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
    with backend.scope():
        for epoch in range(settings.epochs):
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
                output, cache = model.forward(backend.array(x))
                score, grad, _ = loss(output.reshape(-1, modes.width, 6),
                                      *(arr[selection] for arr in train[1:3]), device_basis,
                                      train[3][selection], lower, upper, weights, backend.xp)
                model.backward(grad.reshape(output.shape), cache)
                sample = recorder is not None and start_index == 0
                before = recorder.before() if sample else None
                update = opt.step()
                peak = max(peak, update["gradient_norm"])
                history["clipped_updates"] += update["clip_factor"] < 1
                if sample:
                    history["layer_diagnostics"].append({
                        "epoch": epoch + 1, "layers": recorder.report(before),
                        "inference_activations": recorder.activations(backend.array(x[:8])),
                    })
                total += float(backend.scalar(score)) * len(selection)
            output, _ = model.forward(backend.array(valid[0]), training=False)
            score, _, terms = loss(output.reshape(-1, modes.width, 6), *valid[1:3], device_basis,
                                   valid[3], lower, upper, weights, backend.xp)
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
        model.load_state_dict(state, backend)
    history["training_seconds"] = perf_counter() - start
    history["augmentation_indices"] = index - 1
    return SpectralForecaster(model, backend, modes, inputs, scaler), history, best_epoch
