import tempfile
import unittest
from pathlib import Path

import numpy as np

from Examples.SyntheticJoint import compare, fixture
from Framework.Backend import Backend, BackendConfig
from Temporal.JointTraining import JointTrainer
from Temporal.Model import TemporalModel


class JointTrainingTests(unittest.TestCase):
    def test_combined_gradient_and_exact_recovery(self):
        backend = Backend(BackendConfig(dtype="float64"))
        scales, train, _ = fixture()
        model = TemporalModel(backend, (5,), activations=("tanh",))
        trainer = JointTrainer(model, scales)
        trainer.objective(*train, gradients=True)
        for parameter in model.parameters():
            index = (0,) * parameter.value.ndim
            original = parameter.value[index]
            h = 1e-5
            parameter.value[index] = original + h
            upper = trainer.objective(*train)[2]["loss"]
            parameter.value[index] = original - h
            lower = trainer.objective(*train)[2]["loss"]
            parameter.value[index] = original
            np.testing.assert_allclose(parameter.grad[index], (upper-lower)/(2*h), atol=1e-7)
        trainer.step(*train)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "joint.json"
            trainer.save(path)
            restored = JointTrainer.load(path, Backend(BackendConfig(dtype="float64")))
            self.assertEqual(trainer.step(*train), restored.step(*train))
            for left, right in zip(model.parameters(), restored.model.parameters()):
                np.testing.assert_array_equal(left.value, right.value)

    def test_independent_fixture_and_reproducible_joint_experiment(self):
        _, train, test = fixture()
        self.assertFalse(np.shares_memory(train[2], test[2]))
        self.assertEqual(train[2].shape[:2], (72, 3))
        self.assertEqual(test[2].shape[:2], (24, 1))
        first = compare(40)
        second = compare(40)
        for name, result in first["models"].items():
            self.assertLess(result["final_loss"], result["initial_loss"])
            np.testing.assert_array_equal(result["held_out_site_rmse"],
                                          second["models"][name]["held_out_site_rmse"])
