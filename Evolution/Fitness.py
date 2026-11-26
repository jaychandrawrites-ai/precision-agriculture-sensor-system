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
