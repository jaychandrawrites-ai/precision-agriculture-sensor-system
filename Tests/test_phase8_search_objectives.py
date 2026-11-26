"""Real recurrent recovery and deterministic constrained Pareto selection."""
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from config import EvolutionConfig, TrainingConfig
from Data.Dataset import build_dataset
from Data.RolloutWindows import rollout_windows
from Evolution.Fitness import Candidate, sequence_fitness
from Evolution.Genome import Genome
from Evolution.Pareto import ObjectivePolicy, crowding, fronts, rank_candidates, update_archive
from Evolution.Resumable import SearchOptions, run_search
from main import synthetic_history
from Utils.Serialization import read_json


class SearchObjectivesTests(unittest.TestCase):
    def test_constraints_dominate_and_crowding_prefers_objective_extremes(self):
        rows = [
            Candidate(Genome((0,) * 10), .3, .3, 100, 10, raw_invalid_fraction=.01,
                      estimated_training_bytes=700, prediction_batch_seconds=.02),
            Candidate(Genome((1,) * 10), .2, .2, 150, 10, raw_invalid_fraction=.01,
                      estimated_training_bytes=700, prediction_batch_seconds=.03),
            Candidate(Genome((0, 1) * 5), .1, .1, 250, 10, raw_invalid_fraction=.02,
                      estimated_training_bytes=900, prediction_batch_seconds=.01),
            Candidate(Genome((1, 0) * 5), .01, .01, 50, 10, raw_invalid_fraction=.5,
                      estimated_training_bytes=500, prediction_batch_seconds=.01),
        ]
        policy = ObjectivePolicy(max_raw_invalid_fraction=.25)
        self.assertEqual(fronts(rows, policy), [[0, 1, 2], [3]])
        distance = crowding(rows, [0, 1, 2], policy)
        self.assertEqual(distance[0], np.inf)
        self.assertEqual(distance[2], np.inf)
        self.assertTrue(np.isfinite(distance[1]))
        self.assertNotEqual(rank_candidates(rows, policy)[0], rows[3])
        constrained = ObjectivePolicy(max_latency_seconds=.025, max_raw_invalid_fraction=.25)
        self.assertEqual(update_archive([], rows, constrained), [rows[2], rows[0]])

    def test_recovered_real_models_retain_architectures_and_artifact_hashes(self):
        frame = synthetic_history(164)
        values, times = frame.iloc[:, 1:].to_numpy(), frame.timestamp.to_numpy(dtype="datetime64[ns]")
        dataset = build_dataset(values, times)
        windows = tuple(rollout_windows(dataset, rows, values, times, 2, context=24)
                        for rows in (dataset.train, dataset.validation))
        settings = TrainingConfig(batch_size=64, epochs=1, dropout=0)
        options = SearchOptions(sequence=True, cache=True)
        budget = EvolutionConfig(population=6, generations=2, candidate_epochs=1)
        policy = ObjectivePolicy()
        with TemporaryDirectory() as directory:
            root = Path(directory)
            artifacts = root / "models"
            scorer = sequence_fitness(dataset, settings, budget, windows, artifact_directory=artifacts)
            checkpoint = root / "search.json"
            complete = run_search("random", scorer, budget, 28, None, "fixture", options, policy=policy)
            run_search("random", scorer, budget, 28, checkpoint, "fixture", options, policy=policy, stop_after=1)
            resumed = run_search("random", scorer, budget, 28, checkpoint, "fixture", options, policy=policy)
            # Measured latency and RSS are diagnostics. Compare deterministic outcome and model bytes.
            self.assertEqual(complete[0].genome, resumed[0].genome)
            self.assertEqual([[row.genome for row in generation] for generation in complete[1]],
                             [[row.genome for row in generation] for generation in resumed[1]])
            self.assertEqual(complete[0].artifact_sha256, resumed[0].artifact_sha256)
            checkpoint_data = read_json(checkpoint)
            self.assertTrue(checkpoint_data["pareto_front"])
            self.assertTrue(any(row.genome.decode().family.startswith(("gru", "lstm"))
                                for generation in resumed[1] for row in generation))
            (artifacts / resumed[1][0][0].artifact_file).unlink()
            with self.assertRaisesRegex(ValueError, "checkpoint is missing"):
                run_search("random", scorer, budget, 28, checkpoint, "fixture", options, policy=policy)
