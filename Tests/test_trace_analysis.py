"""AI Trace analytics with explicitly synthetic observations."""
from copy import deepcopy
import json
from pathlib import Path
from unittest.mock import patch
from tempfile import TemporaryDirectory
import unittest

from Acquisition.AITrace import TraceStore, _time
from Acquisition.TraceAnalysis import observational_map, survey_forecasts
from Examples.SyntheticPhase3 import historical_fixture, session_fixture
from Pipeline import train_historical
from config import FEATURES, TrainingConfig


class MemoryStore:
    def __init__(self):
        self.locations = []
        self.rows = {}

    def add(self, name, histories, **location_fields):
        segments = []
        for i, rows in enumerate(histories):
            code = f"{name}/{i}"
            self.rows[code] = rows
            segments.append({"id": code, "started_at": rows[0]["received_at"],
                             "ended_at": rows[-1]["received_at"]})
        self.locations.append({"id": name, "label": name, "segments": segments, **location_fields})

    def status(self, survey_id):
        return {"locations": self.locations}

    def segment_readings(self, survey_id, segment_id):
        return self.rows[segment_id]


class TraceAnalysisTests(unittest.TestCase):
    def test_forecasts_stay_within_segments_and_devices(self):
        with TemporaryDirectory() as root:
            directory = Path(root)
            session, _, config = session_fixture(directory, hours=32)
            readings = [json.loads(line) for line in (session / "readings.jsonl").read_text().splitlines()]
            store = MemoryStore()
            store.add("synthetic complete", [readings])
            store.add("synthetic revisit", [readings[:16], readings[16:]])
            adjacent_device = deepcopy(readings)
            adjacent_device[-1]["device_id"] = "other-synthetic-device"
            store.add("synthetic mixed devices", [adjacent_device])
            restart = deepcopy(readings)
            restart[16]["flags"].append("device_restart")
            store.add("synthetic restarted", [restart])
            reconnect = deepcopy(readings)
            reconnect[16]["flags"].append("receiver_reconnected")
            store.add("synthetic reconnected", [reconnect])
            source, historical = historical_fixture(directory)
            checkpoint = directory / "model.json"
            train_historical(source, historical, checkpoint,
                             TrainingConfig(hidden=(4,), epochs=1, batch_size=32, dropout=0))
            result = survey_forecasts(store, "synthetic", config, checkpoint, 2)
        self.assertEqual(result["ready_locations"], 1)
        complete, revisit, mixed, restart, reconnect = result["locations"]
        point = complete["segments"][0]
        self.assertEqual(point["checkpoint_physics"], "matched")
        self.assertEqual(len(point["forecast"]), 2)
        self.assertEqual(point["forecast"][0]["timestamp"], "2020-07-02T08:00:00+00:00")
        self.assertEqual(set(point["forecast"][0]), {"timestamp", *FEATURES})
        self.assertEqual(len(revisit["segments"]), 2)
        self.assertTrue(all(not row["ready"] and "forecast" not in row for row in revisit["segments"]))
        self.assertIn("separate device histories", mixed["segments"][0]["reason"])
        self.assertIn("restart or receiver boundary", restart["segments"][0]["reason"])
        self.assertIn("restart or receiver boundary", reconnect["segments"][0]["reason"])

    def test_real_store_readiness_does_not_join_a_revisit(self):
        with TemporaryDirectory() as root:
            directory = Path(root)
            session, _, config = session_fixture(directory, hours=32)
            readings = [json.loads(line) for line in (session / "readings.jsonl").read_text().splitlines()]
            store = TraceStore(directory / "trace")
            clock = [readings[0]["received_at"]]
            with patch("Acquisition.AITrace._time", side_effect=lambda value=None: _time(value or clock[0])):
                survey = store.create("synthetic trace")["id"]
                original = store.start(survey, "synthetic location")["active_location_id"]
                for record in readings[:16]:
                    store.ingest(survey, None, record["received_at"], record)
                clock[0] = readings[15]["received_at"]
                store.stop(survey)
                clock[0] = readings[16]["received_at"]
                store.start(survey, location_id=original)
                for record in readings[16:]:
                    store.ingest(survey, None, record["received_at"], record)
                clock[0] = readings[-1]["received_at"]
                store.stop(survey)
                store.finish(survey)
            result = survey_forecasts(store, survey, config)
            self.assertEqual(result["ready_locations"], 0)
            self.assertEqual(len(result["locations"][0]["segments"]), 2)
            self.assertTrue(all(not item["ready"] and item["reference_et_unavailable_hours"] > 0
                                for item in result["locations"][0]["segments"]))

    def test_map_preserves_times_and_omits_missing_coordinates(self):
        def row(minute, temp, values, receipt_minute=None):
            receipt_minute = receipt_minute if receipt_minute is not None else minute
            return {"observed_at": f"2020-07-01T02:{minute:02}:00+00:00",
                    "received_at": f"2020-07-01T02:{receipt_minute:02}:00+00:00",
                    "measurements": {"air_temperature_c": temp, **values}, "flags": []}

        store = MemoryStore()
        store.add("synthetic gps", [[row(40, 26, {"latitude_deg": 20.5, "longitude_deg": 19.0}),
                                    row(41, 31, {"latitude_deg": 20.5, "longitude_deg": 19.0}, 59)]])
        store.add("synthetic manual", [[row(10, 24, {})], [row(45, 27, {})]],
                  latitude_deg=20.5, longitude_deg=19.0001, coordinate_source="manual")
        store.add("synthetic unavailable", [[row(46, 25, {})]])
        store.add("synthetic too old", [[row(7, 28, {"latitude_deg": 20.52, "longitude_deg": 19.0})]])
        store.add("synthetic other reading", [[row(47, 29, {"latitude_deg": 20.501, "longitude_deg": 19.0})]])
        result = observational_map(store, "synthetic", "air_temperature_c", "2020-07-01T02:50:00Z",
                                   15, clock="observed_at", grid_size=5)
        self.assertEqual(result["provenance"], "measured_window_composite")
        self.assertEqual(result["excluded_locations"], 2)
        self.assertEqual([entry["value"] for entry in result["samples"]], [26, 27, 29])
        self.assertEqual(result["samples"][0]["sample_at"], "2020-07-01T02:40:00+00:00")
        self.assertEqual(result["samples"][1]["coordinate_source"], "manual_location")
        self.assertIn("segment_id", result["samples"][1])
        self.assertNotIn("latitude_deg", str(result))
        self.assertEqual(len(result["estimate"]), 5)
        self.assertEqual(len(result["estimate"][0]), 5)
