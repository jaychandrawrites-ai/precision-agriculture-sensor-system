"""Verify deployment and amplitude behavior on reproducible fixtures."""
from pathlib import Path
import tempfile
import unittest

import numpy as np

from config import EvolutionConfig, TrainingConfig
from Dashboard.Jobs import Jobs
from Data.Preprocessing import read_hourly
from Evolution.Genome import BIT_COUNT, Genome
from experiments import compare
from main import synthetic_history
from QuantumInspired.QuantumGenome import QuantumGenome
from Utils.ModelLoader import load_forecaster


class EvolvedRolloutTests(unittest.TestCase):
    def test_amplitude_sampling_and_repeated_rotation(self):
        probability = .2
        individual = QuantumGenome(angles=np.full(BIT_COUNT, np.arcsin(np.sqrt(probability))))
        rng = np.random.default_rng(892)
        observed = np.array([individual.measure(rng).bits for _ in range(3000)])
        np.testing.assert_allclose(observed.mean(axis=0), probability, atol=.035)
        zeros = Genome((0,) * BIT_COUNT)
        ones = Genome((1,) * BIT_COUNT)
        before = individual.amplitudes[:, 1] ** 2
        individual.rotate(zeros, ones, .05)
        self.assertTrue(((individual.amplitudes[:, 1] ** 2) > before).all())
        for _ in range(100):
            individual.rotate(zeros, ones, .05)
        np.testing.assert_allclose((individual.amplitudes ** 2).sum(axis=1), 1, atol=1e-15)
        self.assertTrue(((individual.amplitudes[:, 1] ** 2) < 1).all())

    def test_recursive_amplitude_winner_exports_and_imports(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "synthetic.csv"
            synthetic_history(540).to_csv(path, index=False)
            values, times = read_hourly(path)
            result = compare(values, times, TrainingConfig(epochs=1, rollout_steps=72, hidden=(8,)),
                             EvolutionConfig(population=2, generations=2, candidate_epochs=1),
                             recursive=True, export_directory=root / "models",
                             provenance={"synthetic": True})
            run = root / "models/quantum_inspired"
            self.assertEqual(len(result["quantum_inspired"]["test_rollout"]), 72)
            model = load_forecaster(run / "model.json")
            self.assertEqual(model.time_requirements()["validated_horizon_hours"], 72)
            output = model.forecast(values[-3:], 72)
            self.assertEqual(output.shape, (72, 6))
            self.assertTrue(np.isfinite(output).all())
            for horizon in (0, 73, 1.5, True, "12"):
                with self.subTest(horizon=horizon), self.assertRaisesRegex(ValueError, "integer horizons"):
                    model.forecast(values[-3:], horizon)
            jobs = Jobs(root / "dashboard")
            try:
                key = jobs.import_training_run(run)
                self.assertTrue(jobs.artifact(key, "diagnostics/evolution.png").is_file())
                self.assertTrue(jobs.artifact(key, "diagnostics/internal.png").is_file())
            finally:
                jobs.close()
