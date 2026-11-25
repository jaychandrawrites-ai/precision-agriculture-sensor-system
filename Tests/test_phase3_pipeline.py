from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from Acquisition.AcquisitionController import capture
from Acquisition.Calibration import LeafCalibration, SoilCalibration
from Acquisition.DataLogger import replay
from Acquisition.Hourly import session_hourly
from Acquisition.MappedPackets import MappedPacketParser, PacketSchema
from CombinedLoss import combined_loss
from config import EvolutionConfig, FEATURES, TrainingConfig
from Examples.SyntheticPhase3 import historical_fixture, session_fixture, spatial_fixture
from experiments import compare
from Framework.Backend import create_backend
from main import synthetic_history
from Pipeline import forecast_session, map_measurements, train_historical
from Utils.ModelLoader import load_forecaster
from Utils.Serialization import read_arrays, read_json, write_arrays, write_json


class SensorTests(unittest.TestCase):
    def test_duplicate_replay_preserves_bytes_and_session_metadata(self):
        with TemporaryDirectory() as tmp:
            directory = Path(tmp)
            session, schema, _ = session_fixture(directory, hours=9)
            duplicate = directory / "replay"
            parser = MappedPacketParser(PacketSchema.from_dict(schema))
            capture(replay(session / "raw.jsonl"), duplicate, parser=parser,
                    metadata={"kind": "synthetic_fixture"})
            for name in ("raw.jsonl", "readings.jsonl"):
                self.assertEqual((session / name).read_bytes(), (duplicate / name).read_bytes())
            rows = (duplicate / "readings.jsonl").read_text().splitlines()
            self.assertEqual(len(rows), 10)
            self.assertIn("duplicate_packet", json.loads(rows[6])["flags"])
            record = json.loads(rows[5])["measurements"]
            self.assertEqual(record["soil_ec_us_cm"], 500)
            self.assertEqual(record["leaf_mv"], 410)
            self.assertEqual(read_json(duplicate / "session.json")["kind"], "synthetic_fixture")

    def test_mapped_packet_calibration_and_gps_validation(self):
        with TemporaryDirectory() as tmp:
            _, schema, _ = session_fixture(Path(tmp), hours=1)
            parser = MappedPacketParser(PacketSchema.from_dict(schema))
            stamp = datetime(2020, 1, 1, tzinfo=timezone.utc)
            record = {"kind": "fixture", "id": 1, "node": "a", "boot": "a",
                      "when": stamp.isoformat(), "fix": 1, "lat": 0, "lon": 0,
                      "t": True, "leaf": "Infinity"}
            reading = parser.feed(json.dumps(record), stamp)
            self.assertIn("invalid:gps_fix", reading.flags)
            self.assertIn("invalid:leaf_mv", reading.flags)
            self.assertIn("invalid:air_temperature_c", reading.flags)
            self.assertNotIn("latitude_deg", reading.measurements)
            record["boot"] = "b"
            record["fix"] = True
            reading = parser.feed(json.dumps(record), stamp)
            self.assertNotIn("duplicate_packet", reading.flags)
            self.assertEqual(reading.measurements["latitude_deg"], 0)
            self.assertEqual(parser.feed('{"id":1,"id":2}', stamp).flags, ("invalid_json",))
        calibration = LeafCalibration(600, 300, "synthetic calibration")
        self.assertAlmostEqual(calibration.apply(450), 0.5)
        self.assertEqual(calibration.apply(100), 1)
        with self.assertRaises(ValueError):
            LeafCalibration(300, 300, "invalid synthetic calibration")
        soil = SoilCalibration("synthetic lab reference", ((20, 0.1), (60, 0.3)))
        self.assertAlmostEqual(soil.apply(40), 0.2)
        with self.assertRaises(ValueError):
            soil.apply(70)

    def test_hourly_cutoff_excludes_future_and_late_received_packets(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "readings.jsonl"
            start = datetime(2020, 1, 1, tzinfo=timezone.utc)

            def row(minute, arrival, temperature):
                return {"observed_at": (start + timedelta(minutes=minute)).isoformat(),
                        "received_at": (start + timedelta(minutes=arrival)).isoformat(),
                        "flags": [], "measurements": {"air_temperature_c": temperature}}

            data = [row(0, 0, 10), row(55, 120, 30), row(60, 60, 20), row(65, 65, 40)]
            path.write_text("".join(json.dumps(record) + "\n" for record in data))
            options = {"leaf": LeafCalibration(300, 600, "synthetic"),
                       "soil": SoilCalibration("synthetic"), "clock": "observed_at"}
            cutoff = start + timedelta(minutes=60)
            frame, gps, counts = session_hourly(path, as_of=cutoff.isoformat(), **options)
            self.assertTrue(gps.empty)
            np.testing.assert_array_equal(frame.values.air_temperature_c, [10, 20])
            self.assertEqual(frame.provenance["packet_counts"]["after_cutoff"], 2)
            np.testing.assert_array_equal(counts.air_temperature_c, [1, 1])
            snapshot, _, _ = session_hourly(path, **options)
            self.assertEqual(snapshot.provenance["packet_counts"]["unfinished_hour_bins"], 1)
            self.assertEqual(snapshot.values.index[-1], cutoff)


class IntegrationTests(unittest.TestCase):
    def test_train_once_then_repeat_field_forecast_and_independent_map(self):
        with TemporaryDirectory() as tmp:
            directory = Path(tmp)
            source, history_config = historical_fixture(directory)
            session, _, field = session_fixture(directory)
            settings = TrainingConfig(hidden=(12,), epochs=3, batch_size=16, dropout=0)
            checkpoint = directory / "temporal.json"
            result = train_historical(source, history_config, checkpoint, settings)
            self.assertEqual(result["provenance"]["source_type"], "synthetic_fixture")
            self.assertEqual(load_forecaster(checkpoint).provenance["source_sha256"],
                             result["provenance"]["source_sha256"])
            source.unlink()
            first, report, gps, _ = forecast_session(session / "readings.jsonl", field, checkpoint, 6)
            second, _, _, _ = forecast_session(session / "readings.jsonl", field, checkpoint, 6)
            np.testing.assert_array_equal(first, second)
            self.assertEqual(list(first.columns), list(FEATURES))
            self.assertEqual(len(gps), 32)
            self.assertTrue(report["checkpoint_physics_checked"])
            changed = {**field, "reference_et": {**field["reference_et"], "history_hours": 25}}
            with self.assertRaisesRegex(ValueError, "history length"):
                forecast_session(session / "readings.jsonl", changed, checkpoint, 6)
            source, mapping = spatial_fixture(directory)
            result = map_measurements(source, mapping, directory / "map")
            self.assertEqual(result["measurement_count"], 4)
            keys = {"east_m", "north_m", "variational", "idw", "kriging", "mean",
                    "std", "lower", "upper", "split_rhat"}
            maps = read_arrays(directory / "map" / "field.npz", keys)
            np.testing.assert_array_equal(maps["variational"][0, [0, -1]], [25, 29])
            self.assertEqual(maps["std"][0, 0], 0)
            self.assertGreater(maps["std"][1, 1], 0)

    def test_objectives_and_serialization(self):
        backend = create_backend(dtype="float64")
        args = {"temporal": ([[4.0, 8.0]], [[2.0, 4.0]], [2.0, 4.0]),
                "spatial": ([[2.0]], [[1.0]], [2.0]), "temporal_weight": 2}
        objective = combined_loss(backend, kind="mse", **args)
        self.assertAlmostEqual(objective.loss, 2.25)
        np.testing.assert_allclose(objective.temporal_gradient, [[1, 0.5]])
        np.testing.assert_allclose(objective.spatial_gradient, [[0.5]])
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "sample.json"
            write_json(path, {"a": np.array([1, 2])})
            with self.assertRaises(ValueError):
                write_json(path, {"bad": np.nan})
            self.assertEqual(read_json(path), {"a": [1, 2]})
            with self.assertRaises(ValueError):
                write_arrays(Path(tmp) / "bad.npz", array=np.array([object()]))

    def test_evolutionary_selection_ignores_held_out_targets(self):
        frame = synthetic_history()
        values = frame.iloc[:, 1:].to_numpy()
        timestamps = frame.timestamp.to_numpy(dtype="datetime64[ns]")
        altered = values.copy()
        altered[-36:, 0] += 8
        settings = TrainingConfig(hidden=(12,), epochs=2, batch_size=32, dropout=0)
        budget = EvolutionConfig(population=2, generations=1, candidate_epochs=2)
        first = compare(values, timestamps, settings, budget, recursive=True)
        second = compare(altered, timestamps, settings, budget, recursive=True)
        for method in ("classical_ga", "quantum_inspired"):
            self.assertEqual(first[method]["selection_score"], second[method]["selection_score"])
            self.assertEqual(first[method]["hidden"], second[method]["hidden"])
            self.assertEqual(first[method]["candidate_model_sample_steps"], 3840)
            self.assertNotEqual(first[method]["test"], second[method]["test"])
