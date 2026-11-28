from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import pandas as pd

from assumptions import Assumptions
from config import FEATURES
from Data.WeatherPhysics import ReferenceET
from Examples.SyntheticPhase3 import session_fixture
from Examples.SyntheticTrace import run
from Pipeline import forecast_session


class ShortSessionTests(unittest.TestCase):
    def test_hourly_checkpoint_uses_explicit_adjacent_context(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            run(root / "model")
            session, _, config = session_fixture(root, hours=2)
            path = session / "readings.jsonl"
            dates = pd.date_range(end="2020-06-30T23:00:00Z", periods=25, freq="h")
            config["historical_context"] = {
                "source": "Synthetic fixture",
                "assumptions": Assumptions().metadata(),
                "reference_et": asdict(ReferenceET(Assumptions().site.latitude_deg, **config["reference_et"])),
                "rows": [{"timestamp": time.isoformat(), **dict(zip(FEATURES, [25, 65, .2, .1, .08, 0]))}
                         for time in dates],
            }
            forecast, report, _, _ = forecast_session(path, config, root / "model/synthetic-model.json", 24)
            self.assertEqual(forecast.shape, (24, 6))
            self.assertEqual(report["field_provenance"]["historical_context"]["hours"], 25)
            context = config["historical_context"]
            last = context["rows"][-1]
            context["rows"].append({**last, "timestamp": "2020-07-02T00:00:00Z"})
            context["match_visit"] = True
            matched, _, _, _ = forecast_session(path, config, root / "model/synthetic-model.json", 24)
            pd.testing.assert_frame_equal(forecast, matched)
            context["rows"].pop()
            context["rows"].pop()
            with self.assertRaisesRegex(ValueError, "one hour before"):
                forecast_session(path, config, root / "model/synthetic-model.json", 24)
