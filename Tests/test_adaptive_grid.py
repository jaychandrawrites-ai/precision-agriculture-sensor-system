from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from adaptive_map import adaptive_map
from Spatial.AdaptiveGrid import Refinement, refine


class AdaptiveGridTests(unittest.TestCase):
    def test_refinement_conserves_area_and_reports_budget(self):
        result = refine([0, 0, 16, 16], lambda p: (p[:, 0] / 16, np.zeros(len(p))),
                        Refinement(max_leaves=10, max_variation=.05))
        self.assertEqual(result["leaf_count"], 10)
        self.assertGreater(result["budget_limited_cells"], 0)
        self.assertEqual(sum((a[2]-a[0])*(a[3]-a[1]) for cell in result["cells"]
                             for a in [cell["bounds_m"]]), 256)

    def test_kriging_entry_point_and_polygon_mask(self):
        with TemporaryDirectory() as path:
            source = Path(path) / "measured.csv"
            source.write_text("east_m,north_m,moisture\n0,0,11\n0,16,13\n16,0,14\n16,16,15\n")
            result = adaptive_map(source, {"feature": "moisture", "bounds_m": [0, 0, 16, 16],
                "covariance": {"range_m": 10, "nugget": .01},
                "field_polygon_m": [[0, 0], [16, 0], [0, 16]],
                "refinement": {"max_depth": 3, "max_variation": 1.2}})
        self.assertGreater(result["leaf_count"], 4)
        self.assertTrue(any(not cell["inside"] for cell in result["cells"]))
        self.assertTrue(all(cell["center_variance"] >= 0 for cell in result["cells"]))
