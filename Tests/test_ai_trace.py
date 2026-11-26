from datetime import datetime, timezone
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from Acquisition.AITrace import TraceStore
from Acquisition.TraceImport import import_csv


class SurveyArchiveTests(unittest.TestCase):
    def test_five_locations_revisit_recovery_and_verified_archive(self):
        timestamp = datetime(2025, 1, 1, tzinfo=timezone.utc)
        with TemporaryDirectory() as temp:
            directory = Path(temp)
            first = TraceStore(directory / "source")
            survey = first.create("Synthetic survey")["id"]
            locations = []
            for index in range(5):
                started = first.start(survey, label=f"Site {index}", coordinates=[0, index / 1000] if index else None)
                locations.append(started["active_location_id"])
                self.assertTrue(first.ingest(survey, bytes([index, 255]), timestamp,
                                            {"measurements": {"air_temperature_c": 20 + index}, "flags": []}))
                stopped = first.stop(survey)
                self.assertIsNone(stopped["active_segment_id"])
                self.assertFalse(first.ingest(survey, b"between", timestamp))
            segment = first.start(survey, location_id=locations[0])["active_segment_id"]
            first.ingest(survey, b"revisited", timestamp)
            restarted = TraceStore(directory / "source")
            state = restarted.status(survey)
            self.assertIsNone(state["active_segment_id"])
            self.assertEqual(state["locations"][0]["segments"][-1]["status"], "interrupted")
            self.assertEqual(restarted.segment_readings(survey, segment), [])
            self.assertEqual(len(state["locations"]), 5)
            self.assertEqual(state["locations"][0]["coordinate_source"], "unavailable")
            self.assertEqual(state["locations"][1]["coordinate_source"], "manual")
            restarted.finish(survey)
            archive = restarted.export(survey)
            restored = TraceStore(directory / "restored")
            restored.restore(archive)
            self.assertEqual(restored.export(survey), archive)
            self.assertEqual(len(restored.export(survey, "raw").splitlines()), 6)
            self.assertEqual(len(restored.export(survey, "csv", locations[0]).splitlines()), 2)
            damaged = json.loads(archive)
            damaged["survey"]["label"] = "tampered"
            with self.assertRaisesRegex(ValueError, "checksum"):
                restored.restore(damaged)

    def test_manual_import_units_and_explicit_segments(self):
        with TemporaryDirectory() as directory:
            store = TraceStore(directory)
            lines = ("location,visit,when,temp\nLake,a,2025-01-01T00:00Z,68\n"
                     "Lake,b,2025-01-01T01:00Z,77\n")
            survey = import_csv(store, lines, "Synthetic import", {
                "location": "location", "segment": "visit", "observed_at": "when",
                "fields": {"air_temperature_c": {"column": "temp", "unit": "F"}},
            })
            first, second = survey["locations"][0]["segments"]
            self.assertNotEqual(first["id"], second["id"])
            reading = store.segment_readings(survey["id"], first["id"])[0]
            self.assertAlmostEqual(reading["measurements"]["air_temperature_c"], 20)
            self.assertIn("receipt_is_import_time", reading["flags"])
            self.assertEqual(first["raw_count"], 0)

    def test_replay_keeps_original_segment_time(self):
        import base64
        from Acquisition.TraceImport import replay_segment

        with TemporaryDirectory() as tmp:
            source = Path(tmp) / "saved.jsonl"
            stamp = "2025-01-01T02:00:00+00:00"
            source.write_text(json.dumps({"received_at": stamp,
                                          "line_b64": base64.b64encode(b'{"t":22}').decode()}) + "\n")
            store = TraceStore(Path(tmp) / "trace")
            survey = store.create("Replay")["id"]
            result = replay_segment(store, survey, source, label="First location", schema={
                "fields": {"air_temperature_c": {"key": "t", "unit": "C"}},
            })
            segment = result["locations"][0]["segments"][0]
            self.assertEqual(segment["started_at"], "2025-01-01T02:00:00.000000+00:00")
            self.assertEqual(segment["ended_at"], segment["started_at"])
            self.assertEqual(segment["raw_count"], 1)
            self.assertEqual(store.segment_readings(survey, segment["id"])[0]["measurements"]["air_temperature_c"], 22)
