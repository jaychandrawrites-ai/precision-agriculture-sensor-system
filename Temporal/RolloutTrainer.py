"""Matched-budget teacher-forced and autoregressive sequence training."""
from copy import deepcopy

from Data.Dataloader import DataLoader
from Data.Normalization import Standardizer
from Framework.Optimizer import Adam, SGD
from Framework.Scheduler import CosineAnnealing
from Temporal.Losses import regression_loss
from Temporal.Rollout import Rollout
from Temporal.Trainer import FitResult


def fit_rollout(model, dataset, training, validation, backend, settings, options,
                autoregressive=True, optimizer="adam", loss="huber", warmup_epochs=0):
    if not len(training.starts) or not len(validation.starts):
        raise ValueError("Need complete chronological training and validation windows")
    if not 0 <= warmup_epochs < settings.epochs:
        raise ValueError("Warmup must leave at least one training epoch")
    # The older backward slope equals the current slope minus acceleration (hourly).
    features = dataset.input_scaler.inverse_transform(dataset.inputs[dataset.train])
    features[:, 6:12] -= (1 - options.smoothing) * features[:, 12:18]
    input_scaler = Standardizer().fit(features)
    engine = Rollout(model, input_scaler, dataset.target_scaler, backend, options)
    model.rollout_options = options
    algorithms = {"adam": Adam, "sgd": SGD}
    opt = algorithms[optimizer](model.parameters(), backend, learning_rate=settings.learning_rate,
                                max_norm=settings.gradient_norm)
    schedule = CosineAnnealing(opt, settings.epochs, minimum=settings.learning_rate / 100)
    loader = DataLoader(training.histories, training.targets, settings.batch_size,
                        seed=settings.seed)
    val_targets = backend.array(dataset.target_scaler.transform(validation.targets).reshape(-1, 6))
    best = float("inf")
    peak_gradient_norm = 0.0
    clipped_updates = 0
    history = []
    for epoch in range(1, settings.epochs + 1):
        for inputs, targets, _ in loader:
            opt.zero_grad()
            engine.objective(inputs, targets, autoregressive=autoregressive and epoch > warmup_epochs,
                             loss=loss)
            update = opt.step()
            peak_gradient_norm = max(peak_gradient_norm, update["gradient_norm"])
            clipped_updates += update["clip_factor"] < 1
        predictions = engine.predict(validation.histories, validation.targets.shape[1])
        normalized = backend.array(dataset.target_scaler.transform(predictions).reshape(-1, 6))
        score, _ = regression_loss(normalized, val_targets, backend, kind=loss)
        history.append(score)
        schedule.step()
        if score < best:
            best = score
            best_epoch = epoch
            weights = model.state_dict(backend)
            opt_state = opt.state_dict()
            schedule_state = schedule.state_dict()
            shuffle_state = deepcopy(loader.rng.bit_generator.state)
            rng_state = deepcopy(backend.rng.get_state()) if not backend.is_gpu else None

    model.load_state_dict(weights, backend)
    result = FitResult(best, best_epoch, history, opt_state, schedule_state, shuffle_state, rng_state)
    diagnostics = {"updates": opt.steps, "peak_gradient_norm": peak_gradient_norm,
                   "clipped_updates": clipped_updates}
    return result, input_scaler, diagnostics
