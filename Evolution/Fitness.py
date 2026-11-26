"""Validation-only candidate training under the same epoch and seed budgets."""
from dataclasses import dataclass, replace

from Evolution.Genome import WIDTHS
from Framework.Backend import create_backend
from Temporal.Model import TemporalModel
from Temporal.Rollout import RolloutOptions
from Temporal.RolloutTrainer import fit_rollout
from Temporal.Trainer import fit


@dataclass
class Candidate:
    genome: object
    score: float
    validation_loss: float
    parameters: int
    model_sample_steps: int = 0
    epochs: int = 0
    training_seconds: float | None = None
    prediction_batch_seconds: float | None = None
    prediction_batch_size: int | None = None
    raw_invalid_fraction: float | None = None


def validation_fitness(dataset, training, evolution, *, rollout_windows_pair=None,
                       parameter_count_limit=None):
    cap = evolution.candidate_parameter_step_limit
    if rollout_windows_pair is None:
        samples_per_epoch = len(dataset.train)
    else:
        train_windows, val_windows = rollout_windows_pair
        samples_per_epoch = len(train_windows.starts) * train_windows.targets.shape[1]
    if samples_per_epoch < 1:
        raise ValueError("Evolution requires complete training windows")
    widest = max(WIDTHS)
    # Upper bound for three hidden layers and a projection at each layer.
    maximum_parameters = 2 * (19 * widest + 2 * (widest + 1) * widest) + 6 * (widest + 1)
    if parameter_count_limit is not None:
        maximum_parameters = min(maximum_parameters, parameter_count_limit)
    if cap is not None and cap < maximum_parameters * samples_per_epoch:
        raise ValueError(f"Candidate work limit must be at least {maximum_parameters * samples_per_epoch}"
                         " parameter sample-steps to permit one epoch for every genome")

    def score(genome):
        architecture = genome.decode()
        backend = create_backend(training.device, training.dtype, training.seed)
        model = TemporalModel(backend, dropout=training.dropout, **architecture.model_arguments())
        count = model.parameter_count
        if parameter_count_limit is not None and count > parameter_count_limit:
            raise ValueError("Candidate architecture exceeds the parameter limit")
        epochs = evolution.candidate_epochs
        if cap is not None:
            epochs = min(epochs, cap // (count * samples_per_epoch))
        settings = replace(training, epochs=epochs)
        if rollout_windows_pair is None:
            result = fit(model, dataset, backend, settings)
        else:
            result, _, _ = fit_rollout(model, dataset, train_windows, val_windows, backend,
                                       settings, RolloutOptions(),
                                       warmup_epochs=settings.epochs // 3)
        adjusted = result.validation_loss + evolution.complexity_penalty * count / 1000
        return Candidate(genome, adjusted, result.validation_loss, count, epochs * samples_per_epoch, epochs)

    return score


def sequence_fitness(dataset, training, evolution, windows, *, parameter_count_limit=None):
    """Real direct-model validation under the shared context and target windows."""
    from statistics import median
    from time import perf_counter

    from Data.DataQuality import invalid_rows
    from Evolution.Genome import SEQUENCE_BIT_COUNT, Genome
    from Temporal.Direct import fit_direct

    train_windows, validation = windows
    horizon = train_windows.targets.shape[1]
    samples_per_epoch = len(train_windows.starts) * horizon
    if not samples_per_epoch or not len(validation.starts):
        raise ValueError("Evolution requires complete matched sequence windows")
    minimum = Genome((0,) * SEQUENCE_BIT_COUNT).decode().parameter_count(horizon)
    upper = min(50_000, parameter_count_limit) if parameter_count_limit is not None else 50_000
    cap = evolution.candidate_parameter_step_limit
    if upper < minimum:
        raise ValueError("Parameter cap excludes even the smallest dense candidate")
    if cap is not None:
        upper = min(upper, cap // samples_per_epoch)
        if upper < minimum:
            raise ValueError("Work cap cannot train the smallest dense candidate for one epoch")

    def score(genome):
        spec = genome.decode()
        size = spec.parameter_count(horizon)
        if size > upper:
            raise ValueError("Repair the candidate against its parameter and work caps before evaluation")
        epochs = evolution.candidate_epochs if cap is None else min(
            evolution.candidate_epochs, cap // (samples_per_epoch * size))
        backend = create_backend(training.device, training.dtype, training.seed)
        started = perf_counter()
        model, diagnostics = fit_direct(train_windows, validation, dataset.target_scaler, backend,
                                        replace(training, epochs=epochs), width=spec.width, architecture=spec)
        seconds = perf_counter() - started
        sample = validation.histories[:16]
        model.predict(sample)
        timings = []
        for _ in range(3):
            started = perf_counter()
            model.predict(sample)
            timings.append(perf_counter() - started)
        raw = model.predict(validation.histories, physical=False)
        rate = float(invalid_rows(raw.reshape(-1, 6)).mean())
        adjusted = min(diagnostics["validation_loss"]) + evolution.complexity_penalty * size / 1000
        return Candidate(genome, adjusted, min(diagnostics["validation_loss"]), size,
                         epochs * samples_per_epoch, epochs, seconds, median(timings), len(sample), rate)

    score.horizon = horizon
    score.max_parameters = upper
    return score
