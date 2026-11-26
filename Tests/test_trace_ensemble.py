"""End-to-end modeled spread without reporting synthetic coordinates."""
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from Examples.SyntheticTrace import run
from Utils.Serialization import read_json


class TraceEnsembleTests(unittest.TestCase):
    def test_shared_draw_axis_and_time_index_survive_survey_mapping(self):
        with TemporaryDirectory() as directory:
            output = Path(directory) / "synthetic"
            report = run(output, ensemble=True)
            self.assertEqual(report["ready_locations"], 5)
            manifest = read_json(output / "ensemble-maps.json")
            self.assertEqual(len(manifest["valid_at"]), 6)
            self.assertEqual(len(manifest["features"]), 6)
            self.assertEqual(len(set(manifest["checkpoint_sha256"])), 2)
            self.assertNotIn("latitude", str(manifest))
            with np.load(output / "ensemble-maps.npz", allow_pickle=False) as arrays:
                self.assertEqual(arrays["site_member_forecasts"].shape, (2, 5, 6, 6))
                self.assertEqual(arrays["mean"].shape, (6, 25, 25, 6))
                self.assertTrue(np.any(arrays["spread"] > 0))
                self.assertTrue(np.all(arrays["lower"] <= arrays["upper"]))
