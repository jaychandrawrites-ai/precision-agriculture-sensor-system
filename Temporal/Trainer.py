"""Manual mini-batch optimization with best validation model selection."""
from copy import deepcopy
from dataclasses import dataclass

from Data.Dataloader import DataLoader
from Framework.Optimizer import Adam, SGD
from Framework.Scheduler import CosineAnnealing
from Temporal.Losses import regression_loss


@dataclass
class FitResult:
    validation_loss: float
    best_epoch: int
    history: list
    optimizer_state: dict
    scheduler_state: dict
    shuffle_state: dict
    backend_rng_state: tuple | None


def fit(model, dataset, backend, settings, optimizer="adam", loss="huber"):
    algorithms = {"adam": Adam, "sgd": SGD}
    opt = algorithms[optimizer](model.parameters(), backend,
                                learning_rate=settings.learning_rate,
                                max_norm=settings.gradient_norm)
    schedule = CosineAnnealing(opt, settings.epochs, minimum=settings.learning_rate / 100)
    targets = dataset.target_scaler.transform(dataset.targets)
    train = dataset.train
    validation = dataset.validation
    loader = DataLoader(dataset.inputs[train], targets[train], settings.batch_size,
                        seed=settings.seed, weights=dataset.weights[train])
    val_inputs = backend.array(dataset.inputs[validation])
    val_targets = backend.array(targets[validation])
    best_loss = float("inf")
    history = []

    for epoch in range(1, settings.epochs + 1):
        for inputs, target, weights in loader:
            opt.zero_grad()
            prediction, cache = model.forward(backend.array(inputs), training=True)
            _, gradient = regression_loss(prediction, backend.array(target), backend,
                                          sample_weights=weights, kind=loss)
            model.backward(gradient, cache)
            opt.step()
        prediction, _ = model.forward(val_inputs, training=False)
        score, _ = regression_loss(prediction, val_targets, backend, kind=loss)
        history.append(score)
        schedule.step()
        if score < best_loss:
            best_loss = score
            best_epoch = epoch
            best_weights = model.state_dict(backend)
            optimizer_state = opt.state_dict()
            scheduler_state = schedule.state_dict()
            shuffle_state = deepcopy(loader.rng.bit_generator.state)
            backend_rng_state = deepcopy(backend.rng.get_state()) if not backend.is_gpu else None

    model.load_state_dict(best_weights, backend)
    return FitResult(best_loss, best_epoch, history, optimizer_state, scheduler_state,
                     shuffle_state, backend_rng_state)
