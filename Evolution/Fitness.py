"""Validation-only candidate training under the same epoch and seed budgets."""
from dataclasses import dataclass, replace

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


def validation_fitness(dataset, training, evolution, *, rollout_windows_pair=None):
    settings = replace(training, epochs=evolution.candidate_epochs)

    def score(genome):
        hidden, skips = genome.architecture()
        backend = create_backend(settings.device, settings.dtype, settings.seed)
        model = TemporalModel(backend, hidden, settings.dropout, skips)
        if rollout_windows_pair is None:
            result = fit(model, dataset, backend, settings)
            sample_steps = settings.epochs * len(dataset.train)
        else:
            train_windows, val_windows = rollout_windows_pair
            result, _, _ = fit_rollout(model, dataset, train_windows, val_windows, backend,
                                       settings, RolloutOptions(),
                                       warmup_epochs=settings.epochs // 3)
            sample_steps = settings.epochs * len(train_windows.starts) * train_windows.targets.shape[1]
        count = model.parameter_count
        adjusted = result.validation_loss + evolution.complexity_penalty * count / 1000
        return Candidate(genome, adjusted, result.validation_loss, count, sample_steps)

    return score
