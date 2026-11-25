import hashlib
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import pandas as pd

from Data.Registry import DatasetReference, DatasetRegistry
from Data.Studies import distribution_profile, seasonal_drift
from Examples.SyntheticPhase3 import historical_fixture


class RegistryTests(unittest.TestCase):
    def test_idempotent_versioned_import_and_export(self):
        with TemporaryDirectory() as temp:
            folder = Path(temp)
            source, config = historical_fixture(folder, 48)
            config.update(station_id="simulated", source_sha256=hashlib.sha256(source.read_bytes()).hexdigest())
            registry = DatasetRegistry(folder / "registry")
            self.assertEqual(registry.references(), [])
            reference = registry.import_archive(source, config)
            self.assertEqual(reference, DatasetReference.parse(str(reference)))
            self.assertEqual(registry.import_archive(source, config), reference)
            self.assertEqual(registry.references(), [reference])
            features, manifest = registry.load(reference)
            self.assertEqual(len(features), 48)
            self.assertTrue(manifest["synthetic"])
            registry.verify(reference, source)
            output = folder / "export.csv"
            registry.export(reference, output, start="2020-06-01T17:00:00-07:00",
                            end="2020-06-01T18:00:00-07:00")
            self.assertEqual(len(pd.read_csv(output)), 1)
            with self.assertRaisesRegex(ValueError, "timezone"):
                registry.export(reference, output, start="2020-06-01")
            other = registry.import_archive(source, {**config, "source_label": "Alternate fixture label"})
            self.assertNotEqual(other, reference)
            self.assertEqual(len(registry.references()), 2)

    def test_hashes_and_duplicate_observations_rejected(self):
        with TemporaryDirectory() as temp:
            folder = Path(temp)
            source, config = historical_fixture(folder, 48)
            config["station_id"] = "simulated"
            registry = DatasetRegistry(folder / "registry")
            with self.assertRaisesRegex(ValueError, "Archive hash"):
                registry.import_archive(source, {**config, "source_sha256": "0" * 64})
            records = source.read_text().splitlines()
            reference = registry.import_archive(source, config)
            source.write_text("\n".join([*records, records[-1]]) + "\n")
            with self.assertRaisesRegex(ValueError, "Supplied source"):
                registry.verify(reference, source)
            with self.assertRaisesRegex(ValueError, "Duplicate source timestamps"):
                registry.import_archive(source, config)
            self.assertEqual(registry.references(), [reference])
            path = folder / "registry" / str(reference) / "hourly.csv"
            path.write_text("edited\n")
            with self.assertRaisesRegex(ValueError, "artifact changed"):
                registry.verify(reference)
        with self.assertRaises(ValueError):
            DatasetReference.parse("../anything")

    def test_study_calendar_and_missingness(self):
        index = pd.date_range("2020-03-01T06:00:00Z", periods=5, freq="h")
        first = pd.DataFrame({"air_temperature_c": [35, 36, 37, np.nan, 34]}, index=index)
        temp = "America/Phoenix"
        stats = distribution_profile(first, timezone=temp)
        self.assertEqual(stats["longest_consecutive_hot_hours"], 2)
        self.assertEqual(stats["longest_incomplete_feature_run"], 1)
        self.assertEqual(stats["months"]["2"]["hours"], 2)
        self.assertEqual(stats["months"]["3"]["hours"], 3)
        report = seasonal_drift(first, first + 1, reference_timezone=temp,
                                target_timezone=temp, min_hours=2)
        # March has 2 aligned finite observations after one NaN is removed.
        fields = report["seasons"]["MAM"]["features"]["air_temperature_c"]
        self.assertEqual(fields["reference_finite_hours"], 2)
        self.assertAlmostEqual(fields["target_minus_reference_mean"], 1)
        self.assertGreater(fields["empirical_cdf_distance"], 0)
