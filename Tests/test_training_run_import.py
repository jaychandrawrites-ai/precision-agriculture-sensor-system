import tempfile
from pathlib import Path
import unittest

from config import TrainingConfig
from Dashboard.Jobs import Jobs
from main import synthetic_history
from train import train


class TrainingRunImportTests(unittest.TestCase):
    def test_cli_import_restores_model_and_internal_diagnostics(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "hourly.csv"
            synthetic_history(240).to_csv(source, index=False)
            run = root / "run"
            run.mkdir()
            opts = TrainingConfig(epochs=2)
            recovery = run / "resume.json"
            args = dict(stable_rollout=True, diagnostics_dir=run / "diagnostics",
                        diagnostic_every=1, recovery_path=recovery)
            original = train(source, run / "model.json", opts, **args)
            recovered = train(source, run / "model.json", opts, **args)
            self.assertEqual(original["fit_diagnostics"]["layer_diagnostics"],
                             recovered["fit_diagnostics"]["layer_diagnostics"])
            self.assertEqual(len(recovered["fit_diagnostics"]["learning_rate"]), 2)
            layers = recovered["fit_diagnostics"]["layer_diagnostics"][0]
            self.assertTrue(layers["layers"])
            self.assertTrue(layers["inference_activations"])
            self.assertTrue(all(row["gradient"]["nonfinite"] == 0 for row in layers["layers"].values()))
            jobs = Jobs(root / "dashboard")
            try:
                key = jobs.import_training_run(run)
                self.assertEqual(jobs.import_training_run(run), key)
                self.assertTrue(jobs.artifact(key, "diagnostics/internal.png").is_file())
            finally:
                jobs.close()
            jobs = Jobs(root / "dashboard")
            try:
                self.assertTrue(jobs.artifact(key, "model.json").is_file())
            finally:
                jobs.close()
