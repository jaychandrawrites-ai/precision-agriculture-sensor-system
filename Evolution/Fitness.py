"""Validation-only candidate training under the same epoch and seed budgets."""
from dataclasses import dataclass, replace

from Framework.Backend import create_backend
from Temporal.Model import TemporalModel
from Temporal.Trainer import fit


@dataclass
class Candidate:
    genome: object
    score: float
    validation_loss: float
    parameters: int


def validation_fitness(dataset, training, evolution):
    settings = replace(training, epochs=evolution.candidate_epochs)

    def score(genome):
        hidden, skips = genome.architecture()
        backend = create_backend(settings.device, settings.dtype, settings.seed)
        model = TemporalModel(backend, hidden, settings.dropout, skips)
        result = fit(model, dataset, backend, settings)
        count = model.parameter_count
        adjusted = result.validation_loss + evolution.complexity_penalty * count / 1000
        return Candidate(genome, adjusted, result.validation_loss, count)

    return score
