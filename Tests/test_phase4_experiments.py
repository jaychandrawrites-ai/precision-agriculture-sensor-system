from types import SimpleNamespace
import unittest

import numpy as np

from Evolution.Diagnostics import diversity, pareto_front
from Evolution.Genome import Genome
from Utils.Bootstrap import paired_rmse_difference


class ExperimentTests(unittest.TestCase):
    def test_pareto_and_diversity(self):
        candidates = [SimpleNamespace(genome=Genome((bit,) * 10), parameters=size,
                                     validation_loss=loss)
                      for bit, size, loss in ((0, 6, 4), (1, 10, 2), (1, 11, 3), (0, 14, 1))]
        self.assertEqual([x.parameters for x in pareto_front(candidates)], [6, 10, 14])
        self.assertAlmostEqual(diversity(candidates), 2 / 3)

    def test_paired_block_bootstrap(self):
        truth = np.ones((80, 3, 6))
        report = paired_rmse_difference(truth + 2, truth + 3, truth, block=24, draws=20)
        for key in ("lower", "upper", "difference"):
            np.testing.assert_array_equal(report[key], -np.ones((3, 6)))

    def test_resumed_search_matches_uninterrupted_all_modes(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory

        from config import EvolutionConfig
        from Evolution.Fitness import Candidate
        from Evolution.Resumable import SearchOptions, run_search

        def score(genome):
            dimensions = genome.architecture()[0]
            return Candidate(genome, float(sum(dimensions)), sum(dimensions), sum(dimensions), 10, 2)

        budget = EvolutionConfig(population=5, generations=4)
        options = SearchOptions(workers=2, cache=True)
        with TemporaryDirectory() as directory:
            for mode in ("classical_ga", "quantum_inspired", "random"):
                first = Path(directory) / (mode + "first.json")
                second = Path(directory) / (mode + "second.json")
                expected = run_search(mode, score, budget, 98, first, "fixture", options)
                partial = run_search(mode, score, budget, 98, second, "fixture", options, stop_after=2)
                self.assertEqual(len(partial[1]), 2)
                resumed = run_search(mode, score, budget, 98, second, "fixture", options)
                self.assertEqual(expected, resumed)
                with self.assertRaisesRegex(ValueError, "recipe"):
                    run_search(mode, score, budget, 98, second, "changed data", options)
