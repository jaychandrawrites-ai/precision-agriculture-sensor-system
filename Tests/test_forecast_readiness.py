from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from Data.ForecastHistory import inspect_history, require_history
from Examples.SyntheticPhase3 import session_fixture
from Pipeline import check_session


class ForecastReadinessTests(unittest.TestCase):
    def test_hour_gaps_and_invalid_features_block_forecast(self):
        values = [[25, 50, .5, 0, 0, 0]] * 3
        hours = np.array(["2020-01-01T10", "2020-01-01T11", "2020-01-01T12"], dtype="datetime64[h]")
        self.assertTrue(inspect_history(values, hours)["ready"])
        invalid = np.array(values, dtype=float)
        invalid[-1, 2] = 1.5
        self.assertEqual(inspect_history(invalid, hours)["invalid_features"], ["lwi"])
        hours[-1] += np.timedelta64(1, "h")
        with self.assertRaisesRegex(ValueError, "consecutive"):
            require_history(values, hours)

    def test_synthetic_session_reports_warmup_without_gps(self):
        with TemporaryDirectory() as root:
            for hours, expected in ((25, False), (26, True)):
                directory = Path(root) / str(hours)
                directory.mkdir()
                session, _, config = session_fixture(directory, hours)
                report = check_session(session / "readings.jsonl", config)
                self.assertIs(report["ready"], expected)
                self.assertEqual(report["minimum_temperature_history_hours"], 26)
                self.assertEqual(report["checkpoint_physics"], "unchecked")
                self.assertNotIn("gps", str(report))
