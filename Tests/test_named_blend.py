import tempfile
import unittest
from pathlib import Path

import numpy as np

from blend_forecast import fit
from direct_forecast import train
from config import TrainingConfig
from main import synthetic_history
from Utils.ModelLoader import load_forecaster


class NamedBlendTests(unittest.TestCase):
    def test_portable_blend_uses_disjoint_validation_and_survives_member_removal(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            frame = synthetic_history(240)
            source = root / "weather.csv"
            frame.to_csv(source, index=False)
            members = [root / "gated.json", root / "recurrent.json"]
            for seed, member in enumerate(members):
                train(source, member, TrainingConfig(epochs=1, seed=seed), 3, 8)
            out = root / "blend.json"
            fit(source, members, out, start=str(frame.timestamp.iloc[207].tz_localize(None)),
                end=str(frame.timestamp.iloc[229].tz_localize(None)), steps=3)
            for member in members:
                member.unlink()
            model = load_forecaster(out)
            history = frame.iloc[-3:, 1:].to_numpy()
            actual = model.forecast(history, 2)
            self.assertEqual(actual.shape, (2, 6))
            self.assertTrue(np.isfinite(actual).all())
            with self.assertRaisesRegex(ValueError, "Blend supports"):
                model.forecast(history, 4)
