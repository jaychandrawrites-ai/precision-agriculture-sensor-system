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
    estimated_training_bytes: int | None = None
    observed_process_peak_bytes: int | None = None
    prediction_warmup_seconds: float | None = None
    artifact_file: str | None = None
    artifact_sha256: str | None = None


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


def sequence_fitness(dataset, training, evolution, windows, *, parameter_count_limit=None,
                     artifact_directory=None, memory_budget_bytes=None):
    """Real direct-model validation under the shared context and target windows."""
    from hashlib import sha256
    from pathlib import Path
    from statistics import median
    from time import perf_counter

    from Data.DataQuality import invalid_rows
    from Evolution.Genome import SEQUENCE_BIT_COUNT, Genome
    from Framework.Memory import estimate_memory, observed_process_peak
    from Temporal.Direct import direct_features, fit_direct
    from Temporal.DirectCheckpoint import save_direct

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

    artifact_root = None if artifact_directory is None else Path(artifact_directory)
    if artifact_root is not None:
        artifact_root.mkdir(parents=True, exist_ok=True)

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
                                        replace(training, epochs=epochs), width=spec.width, architecture=spec,
                                        memory_budget_bytes=memory_budget_bytes)
        seconds = perf_counter() - started
        sample = validation.histories[:16]
        started = perf_counter()
        model.predict(sample)
        warmup_seconds = perf_counter() - started
        timings = []
        for _ in range(3):
            started = perf_counter()
            model.predict(sample)
            timings.append(perf_counter() - started)
        raw = model.predict(validation.histories, physical=False)
        rate = float(invalid_rows(raw.reshape(-1, 6)).mean())
        # CPU RSS is process-lifetime, not candidate-isolated. Keep it as diagnostic only.
        memory = estimate_memory(model.model, backend,
                                 model.inputs.transform(direct_features(sample[:1], spec)))
        planned_bytes = (diagnostics["memory"]["plan"]["estimate_bytes"] if "memory" in diagnostics
                         else memory.batch_bytes(training.batch_size))
        peak = observed_process_peak()
        filename = digest = None
        if artifact_root is not None:
            key = sha256(repr((spec.as_dict(), training.seed, epochs)).encode()).hexdigest()
            filename = f"{key}.json"
            path = artifact_root / filename
            save_direct(path, model, spec.width, provenance={"phase": "candidate_validation",
                        "seed": training.seed, "epochs": epochs, "architecture": spec.as_dict()})
            digest = sha256(path.read_bytes()).hexdigest()
        adjusted = min(diagnostics["validation_loss"]) + evolution.complexity_penalty * size / 1000
        return Candidate(genome, adjusted, min(diagnostics["validation_loss"]), size,
                         epochs * samples_per_epoch, epochs, seconds, median(timings), len(sample), rate,
                         planned_bytes, None if peak is None else peak["bytes"],
                         warmup_seconds, filename, digest)

    def verify_artifact(candidate):
        if artifact_root is None:
            return
        if candidate.artifact_file is None or Path(candidate.artifact_file).name != candidate.artifact_file:
            raise ValueError("Missing or invalid saved candidate reference")
        path = artifact_root / candidate.artifact_file
        if not path.is_file() or sha256(path.read_bytes()).hexdigest() != candidate.artifact_sha256:
            raise ValueError("Saved candidate checkpoint is missing or has changed")

    score.verify_artifact = verify_artifact
    score.horizon = horizon
    score.max_parameters = upper
    return score
