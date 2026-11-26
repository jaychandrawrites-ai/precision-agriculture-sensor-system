"""Verification of structured and adaptive architecture search."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from config import EvolutionConfig, TrainingConfig
from Data.Dataset import build_dataset
from Evolution.Fitness import Candidate, validation_fitness
from Evolution.Genome import Architecture, Genome
from Evolution.Resumable import SearchOptions, run_search
from experiments import compare
from Framework.Backend import create_backend
from main import synthetic_history
from QuantumInspired.Adaptive import adaptive_update, amplitude_diagnostics, binary_entropy
from QuantumInspired.QuantumGenome import QuantumGenome
from Temporal.Model import TemporalModel
from Temporal.Trainer import fit
from Utils.Checkpoints import load, save


def fixture():
    frame = synthetic_history(168)
    return frame.iloc[:, 1:].to_numpy(), frame.timestamp.to_numpy(dtype="datetime64[ns]")


def deterministic_score(genome):
    architecture = genome.decode()
    size = architecture.parameter_count
    loss = architecture.hidden[-1] / 128
    return Candidate(genome, loss + size * 1e-6, loss, size, 120, 3)


class StructuredSearchTests(unittest.TestCase):
    def test_round_trip_activation_checkpoint_and_analytic_parameter_count(self):
        values, times = fixture()
        dataset = build_dataset(values, times)
        settings = TrainingConfig(epochs=1, hidden=(16, 16, 32), batch_size=32, dropout=0)
        spec = Architecture(settings.hidden, (0, 1, 2), ("tanh", "silu", "gelu"))
        backend = create_backend(seed=settings.seed)
        model = TemporalModel(backend, **spec.model_arguments())
        self.assertEqual(model.parameter_count, spec.parameter_count)
        result = fit(model, dataset, backend, settings)
        before = backend.to_host(model.forward(backend.array(dataset.inputs[dataset.test]), False)[0])
        with TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.json"
            save(path, model, dataset, backend, result, settings)
            restored, runtime, _, _ = load(path)
            self.assertEqual(restored.activations, spec.activations)
            after = runtime.to_host(restored.forward(runtime.array(dataset.inputs[dataset.test]), False)[0])
            np.testing.assert_array_equal(after, before)

    def test_genome_projection_and_repair_preserve_valid_caps(self):
        original = Genome((1,) * 17)
        self.assertEqual(original.decode().activations, ("gelu",) * 3)
        self.assertEqual(original.decode().skips, (0, 1, 2))
        legacy = Genome((1,) * 10)
        self.assertEqual(legacy.decode().activations, ("relu",) * 3)
        rng = np.random.default_rng(10)
        for extended in (True, False):
            for _ in range(100):
                genome = Genome.random(rng, extended=extended)
                cap = int(rng.integers(406, 7000))
                revised = genome.repair(cap)
                self.assertLessEqual(revised.decode().parameter_count, cap)
                self.assertEqual(revised.repair(cap), revised)
                self.assertEqual(len(genome.bits), len(revised.bits))
        self.assertEqual(original.repair(406).decode().parameter_count, 406)
        with self.assertRaises(ValueError):
            original.repair(405)

    def test_normalization_entropy_relaxation_and_rotation_direction(self):
        np.testing.assert_array_equal(binary_entropy([0, .5, 1]), [0, 1, 0])
        weak = QuantumGenome(extended=True, angles=np.zeros(17))
        strong = QuantumGenome(extended=True)
        zero, one = Genome((0,) * 17), Genome((1,) * 17)
        candidates = [Candidate(zero, 2, 2, 100), Candidate(one, 1, 1, 100)]
        report = adaptive_update([weak, strong], list(enumerate(candidates)), candidates, one,
                                 .1, 0, .8)
        self.assertEqual(report["relaxed_individuals"], 1)
        self.assertTrue((weak.angles > .1).all())
        for item in (weak, strong):
            np.testing.assert_allclose((item.amplitudes ** 2).sum(axis=1), np.ones(17))
        neutral = amplitude_diagnostics([QuantumGenome(extended=True)])
        self.assertAlmostEqual(neutral["mean_population_bit_entropy"], 1)
        self.assertLess(report["mean_individual_bit_entropy"], 1)

    def test_resumed_hybrid_matches_full_run_and_keeps_candidate_budget(self):
        budget = EvolutionConfig(population=6, generations=5)
        options = SearchOptions(workers=2, extended=True, max_parameters=2500,
                                crossover="single_point", parent_selection="rank")
        with TemporaryDirectory() as directory:
            path = Path(directory) / "hybrid.json"
            complete = run_search("hybrid_amplitude", deterministic_score, budget, 42, None,
                                  "fixture", options)
            run_search("hybrid_amplitude", deterministic_score, budget, 42, path,
                       "fixture", options, stop_after=2)
            resumed = run_search("hybrid_amplitude", deterministic_score, budget, 42, path,
                                 "fixture", options)
        self.assertEqual(complete, resumed)
        _, generations, diagnostics = complete
        self.assertEqual([len(group) for group in generations], [6] * 5)
        self.assertEqual(sum(item["actual_evaluations"] for item in diagnostics), 30)
        self.assertEqual(diagnostics[0]["classical_offspring"], 0)
        self.assertTrue(all(item["classical_offspring"] == 1 for item in diagnostics[1:]))
        self.assertTrue(all(item.parameters <= 2500 for group in generations for item in group))

    def test_search_decisions_do_not_change_when_test_targets_change(self):
        values, times = fixture()
        settings = TrainingConfig(epochs=1, hidden=(16,), batch_size=64, dropout=0)
        budget = EvolutionConfig(population=2, generations=2, candidate_epochs=1)
        options = SearchOptions(extended=True, adaptive_amplitude=True, max_parameters=3000)
        report = compare(values, times, settings, budget, recursive=True, include_random=True,
                         include_hybrid=True, search_options=options)
        disturbed = values.copy()
        disturbed[-20:, 0] += 5
        second = compare(disturbed, times, settings, budget, recursive=True, include_random=True,
                         include_hybrid=True, search_options=options)
        for method in ("classical_ga", "quantum_inspired", "random", "hybrid_amplitude"):
            self.assertEqual(report[method]["winning_bits"], second[method]["winning_bits"])
            self.assertEqual(report[method]["selection_score"], second[method]["selection_score"])
            self.assertEqual(report[method]["actual_candidate_evaluations"], 4)
            self.assertEqual(report[method]["actual_candidate_model_sample_steps"],
                             report["random"]["actual_candidate_model_sample_steps"])

    def test_fitness_keeps_parameter_step_budget(self):
        dataset = build_dataset(*fixture())
        settings = TrainingConfig(epochs=1)
        genome = Genome((1,) * 17).repair(5000)
        cap = 5000 * len(dataset.train)
        budget = EvolutionConfig(candidate_epochs=8, candidate_parameter_step_limit=cap)
        result = validation_fitness(dataset, settings, budget, parameter_count_limit=5000)(genome)
        self.assertLessEqual(result.model_sample_steps * result.parameters, cap)
        self.assertLess(result.epochs, budget.candidate_epochs)
