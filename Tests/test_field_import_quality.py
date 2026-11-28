"""Synthetic-only checks for offline field quality and irregular time math."""
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from Acquisition.AITrace import TraceStore
from Acquisition.ElapsedTime import derivatives
from Acquisition.TraceImport import import_csv


class FieldQualityTests(unittest.TestCase):
    def test_timezone_and_duplicate_rows_remain_visible(self):
        mapping = {"location_label": "Synthetic site", "observed_at": "time", "timezone": "America/Phoenix",
                   "gps": {"minimum_samples": 2}, "fields": {
                       "air_temperature_c": {"column": "temp", "unit": "C"},
                       "latitude_deg": {"column": "lat", "unit": "deg"},
                       "longitude_deg": {"column": "lon", "unit": "deg"}}}
        payload = ("time,temp,lat,lon\n2026-11-01T10:00:00,20,0,0\n2026-11-01T10:00:00,21,0,.00001\n"
                   "2026-11-01T10:04:00,22,0,.00002\n")
        with TemporaryDirectory() as root:
            store = TraceStore(root)
            with self.assertRaisesRegex(ValueError, "timezone"):
                import_csv(store, payload, "Synthetic", {**mapping, "timezone": None})
            survey = import_csv(store, payload, "Synthetic", mapping)
            segment = survey["locations"][0]["segments"][0]
            report = segment["calibration"]["csv_quality"]
            self.assertEqual(report["records"], 3)
            self.assertEqual(report["duplicate_timestamp_rows"], 2)
            self.assertEqual(report["max_gap_seconds"], 240.)
            self.assertEqual(report["data_use"], "inference_only")
            self.assertIsNotNone(report["gps"]["location"])
            rows = store.segment_readings(survey["id"], segment["id"])
            self.assertEqual(len(rows), 3)
            self.assertTrue(rows[0]["observed_at"].startswith("2026-11-01T17:00"))
            self.assertIn("duplicate_csv_timestamp", rows[0]["flags"])

    def test_unequal_timestep_quadratic_and_duplicate_gap(self):
        hours = np.array([0, 1, 3, 3, 20])
        times = np.datetime64("2020-01-01") + hours.astype("timedelta64[h]")
        velocity, acceleration = derivatives((hours**2)[:, None], times, maximum_gap_seconds=10800)
        self.assertEqual(velocity[2, 0], 4)
        self.assertEqual(acceleration[2, 0], 2)
        self.assertTrue(np.isnan(velocity[3:]).all())
        self.assertTrue(np.isnan(acceleration[3:]).all())

    def test_native_readiness_requires_fresh_complete_cluster(self):
        from datetime import datetime, timedelta, timezone

        from Acquisition.AquaMap import NATIVE
        from Acquisition.TraceReceiver import TraceReceiver

        with TemporaryDirectory() as root:
            store = TraceStore(root)
            survey = store.create("Synthetic receiver")["id"]
            receiver = TraceReceiver(store, native_gps_samples=2, native_fresh_seconds=5)
            receiver._survey, receiver._schema, receiver._state = survey, NATIVE, "connected"
            now = datetime.now(timezone.utc)
            with self.assertRaisesRegex(ValueError, "fresh complete"):
                receiver.start(survey, label="Site")
            receiver.feed(survey, b"# diagnostics only", now)
            self.assertFalse(receiver.diagnostics(survey)["native_readiness"]["ready"])
            for sample in (1, 2):
                line = f"RX,-90,3,AQ,1,{sample},0,0,30,24,450,6.8,22,55,350"
                receiver.feed(survey, line.encode(), now - timedelta(seconds=10))
            self.assertFalse(receiver.diagnostics(survey)["native_readiness"]["ready"])
            for sample in (3, 4):
                line = f"RX,-90,3,AQ,1,{sample},0,0,30,24,450,6.8,22,55,350"
                receiver.feed(survey, line.encode(), now)
            self.assertTrue(receiver.diagnostics(survey)["native_readiness"]["ready"])
            self.assertIsNotNone(receiver.start(survey, label="Site")["active_segment_id"])

    def test_batch_preview_is_read_only_and_composite_retains_separate_visits(self):
        from Acquisition.TraceAnalysis import session_composite
        from Acquisition.TraceImport import import_csv_batch

        header = ("point,timestamp,latitude,longitude,gps_sats,soil_moisture,soil_temp,ec,ph,"
                  "air_temp,air_humidity,leaf_wetness,radio_rssi,radio_snr\n")

        def fixture(lon):
            return (header + "".join(
                f"{i},2026-11-01T10:00:{i:02},0,{lon},,30,24,400,7,22,50,350,-90,4\n"
                for i in range(6))).encode()

        files = [("one.csv", fixture(0), None), ("two.csv", fixture(.0004), "Second visit")]
        with TemporaryDirectory() as root:
            store = TraceStore(root)
            preview = import_csv_batch(store, files, "Synthetic", "America/Phoenix", preview=True)
            self.assertEqual(store.list_surveys(), [])
            self.assertEqual([item["overlapping_visits"] for item in preview], [1, 1])
            self.assertEqual(preview[0]["missing_fields"]["gps_satellites"], 6)
            self.assertNotIn("location", preview[0]["gps"])
            with self.assertRaisesRegex(ValueError, "distinct"):
                import_csv_batch(store, files + [files[0]], "Synthetic", "America/Phoenix")
            self.assertEqual(store.list_surveys(), [])
            survey = import_csv_batch(store, files, "Synthetic", "America/Phoenix")
            mapped = session_composite(store, survey["id"], "air_temperature_c", grid_size=9,
                                       domain={"center_lat_lon": [0, 0], "radius_m": 75}, diffusion={})
            self.assertEqual(mapped["provenance"], "non_simultaneous_observed_session_composite")
            self.assertEqual(len(mapped["samples"]), 2)
            self.assertIsNone(mapped["estimate"][0][0])
