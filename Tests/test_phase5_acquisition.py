"""Synthetic archive, acquisition coverage and receiver-quality fixtures."""
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event
import unittest

from Acquisition.AcquisitionController import capture
from Acquisition.Archives import RawArchiveError, inspect_raw, inspect_session, raw_records
from Acquisition.CalibrationReport import summarize_leaf_calibration
from Acquisition.Coverage import coverage_report
from Acquisition.MappedPackets import MappedPacketParser, PacketField, PacketSchema
from Acquisition.Recovery import recover_session


NOW = datetime(2020, 1, 1, tzinfo=timezone.utc)
TEXT = {"parser": "checked_in_arduino_text", "fixture": "synthetic"}


class AcquisitionArchiveTests(unittest.TestCase):
    def test_manifest_verifies_and_detects_derived_tampering(self):
        with TemporaryDirectory() as folder:
            source = Path(folder, "source")
            capture(iter([(NOW, b"--- AIR ---\r\n"), (NOW, b"Temperature: 25 C\r\n"),
                          (NOW, b"==========\r\n"), (NOW, bytes([0xff, 0]))]),
                    source, metadata=TEXT)
            first = inspect_session(source)
            self.assertTrue(first["verified"])
            self.assertEqual(first["raw"]["records"], 4)
            self.assertEqual(list(raw_records(source / "raw.jsonl"))[-1][1], bytes([0xff, 0]))
            readings = source / "readings.jsonl"
            readings.write_bytes(readings.read_bytes() + readings.read_bytes())
            self.assertIn("readings_manifest_mismatch", inspect_session(source)["findings"])
            output = Path(folder, "recovered")
            recover_session(source, output)
            self.assertTrue(inspect_session(output)["verified"])
            self.assertEqual(first["readings"]["sha256"], inspect_session(output)["readings"]["sha256"])

    def test_interrupted_capture_and_torn_tail_recover_only_complete_records(self):
        def interrupted():
            yield NOW, b"--- AIR ---\n"
            yield NOW + timedelta(minutes=1), b"Temperature: 21 C\n"
            raise RuntimeError("synthetic transport cut")

        with TemporaryDirectory() as folder:
            source = Path(folder, "source")
            with self.assertRaisesRegex(RuntimeError, "transport cut"):
                capture(interrupted(), source, metadata=TEXT)
            report = inspect_session(source)
            self.assertIn("capture_status:failed", report["findings"])
            raw = source / "raw.jsonl"
            torn = b'{"received_at":"2020-01'
            raw.write_bytes(raw.read_bytes() + torn)
            original = raw.read_bytes()
            with self.assertRaisesRegex(RawArchiveError, "line 3"):
                inspect_raw(raw)
            output = Path(folder, "new")
            result = recover_session(source, output, accept_torn_tail=True)
            self.assertEqual(result["source_raw"]["incomplete_final_bytes"], len(torn))
            self.assertEqual(result["source_raw"]["records"], 2)
            self.assertEqual(raw.read_bytes(), original)
            self.assertTrue(inspect_session(output)["verified"])

    def test_complete_capture_rejects_changed_raw_and_corrupt_middle(self):
        with TemporaryDirectory() as folder:
            source = Path(folder, "source")
            capture([(NOW, b"first byte payload"), (NOW, b"second")], source, metadata=TEXT)
            raw = source / "raw.jsonl"
            rows = raw.read_bytes().splitlines(keepends=True)
            altered_record = json.loads(rows[0])
            altered_record["line_b64"] = "YQ=="
            raw.write_bytes((json.dumps(altered_record) + "\n").encode() + rows[1])
            self.assertIn("raw_manifest_mismatch", inspect_session(source)["findings"])
            output = Path(folder, "recovered")
            with self.assertRaisesRegex(RawArchiveError, "disagrees"):
                recover_session(source, output)
            self.assertFalse(output.exists())
            raw.write_bytes(b"corrupt\n" + rows[1])
            with self.assertRaisesRegex(RawArchiveError, "line 1"):
                inspect_raw(raw, accept_torn_tail=True)

    def test_stop_manifest_finalizes_without_advancing_ignored_timestamp(self):
        flag = Event()

        def lines():
            yield NOW, b"--- AIR ---\n"
            flag.set()
            yield NOW + timedelta(hours=5), b"Temperature: 88 C\n"

        with TemporaryDirectory() as folder:
            source = Path(folder, "source")
            capture(lines(), source, flag, metadata=TEXT)
            verified = inspect_session(source)
            self.assertEqual(verified["status"], "stopped")
            self.assertEqual(verified["raw"]["records"], 1)
            row = json.loads((source / "readings.jsonl").read_text())
            self.assertEqual(row["received_at"], NOW.isoformat())


class ReceiverQualityTests(unittest.TestCase):
    def schema(self):
        return PacketSchema(
            fields={
                "air_temperature_c": PacketField("temp", "C"),
                "relative_humidity_pct": PacketField("rh", "%"),
                "latitude_deg": PacketField("lat", "deg"),
                "longitude_deg": PacketField("lon", "deg"),
                "gps_satellites": PacketField("sat", "count"),
                "gps_hdop": PacketField("hdop", "ratio"),
            },
            device_id_key="node", boot_id_key="boot", gps_fix_key="valid_fix",
            packet_id_key="seq", sequence_key="seq", sequence_modulus=8,
            observed_at_key="observed", maximum_clock_lead_seconds=10,
            gps_min_satellites=4, gps_max_hdop=2.0,
        )

    def record(self, stamp, counter, **arguments):
        return {"node": "synthetic-node", "boot": "synthetic-boot", "seq": counter,
                "valid_fix": True, "observed": stamp.isoformat(), "temp": 21.3, "rh": 40,
                "lat": 1.25, "lon": 2.5, "sat": 7, "hdop": .5, **arguments}

    def test_quality_drops_bad_fix_and_bad_clock_but_retains_other_sensor_values(self):
        parser = MappedPacketParser(self.schema())
        record = self.record(NOW + timedelta(seconds=11), 0, sat=3.5, hdop=4.5)
        parsed = parser.feed(json.dumps(record), NOW)
        self.assertEqual(parsed.measurements["air_temperature_c"], 21.3)
        self.assertNotIn("latitude_deg", parsed.measurements)
        self.assertIn("invalid:gps_satellites", parsed.flags)
        self.assertIn("invalid:gps_satellite_quality", parsed.flags)
        self.assertIn("invalid:gps_hdop_quality", parsed.flags)
        self.assertIn("observed_at_ahead_of_receipt", parsed.flags)
        self.assertIsNone(parsed.observed_at)

    def test_sequence_wrap_is_not_mistaken_for_a_duplicate_packet(self):
        parser = MappedPacketParser(self.schema())
        for counter in (*range(8), 0):
            parsed = parser.feed(json.dumps(self.record(NOW, counter)), NOW)
            self.assertNotIn("duplicate_packet", parsed.flags)
        parsed = parser.feed(json.dumps(self.record(NOW, 0)), NOW)
        self.assertIn("duplicate_packet", parsed.flags)

    def test_coverage_reports_ages_instead_of_exact_gps_coordinates(self):
        stamps = [NOW + timedelta(minutes=i) for i in [0, 1, 2, 3]]
        rows = [self.record(stamps[0], 0), self.record(stamps[1], 0),
                self.record(stamps[2], 2, valid_fix=False, rh=None),
                self.record(stamps[3], 3)]
        with TemporaryDirectory() as folder:
            output = Path(folder, "fixture")
            capture([(stamp, json.dumps(row).encode()) for stamp, row in zip(stamps, rows)],
                    output, parser=MappedPacketParser(self.schema()))
            coverage = coverage_report(output / "readings.jsonl", as_of=stamps[2].isoformat())
        self.assertEqual(coverage["records_after_cutoff"], 1)
        node = coverage["devices"]["synthetic-node"]
        self.assertEqual(node["counts"]["duplicate"], 1)
        self.assertEqual(node["counts"]["required_complete"], 1)
        self.assertEqual(node["counts"]["gps_fixes"], 1)
        self.assertEqual(node["last_measurement_age_seconds"]["relative_humidity_pct"], 120)
        self.assertNotIn("1.25", json.dumps(coverage))


class CalibrationReplicateTests(unittest.TestCase):
    def test_reversed_polarity_uses_measured_replicates(self):
        report = summarize_leaf_calibration(dry_mv=[598, 602], wet_mv=[299, 301],
                                            source="synthetic fixture")
        self.assertEqual(report.as_record()["leaf_calibration"]["dry_mv"], 600)
        result = report.estimate(450, voltage_standard_uncertainty_mv=3)
        self.assertEqual(result["leaf_wetness"], .5)
        self.assertAlmostEqual(result["linearized_standard_uncertainty"], (10.25 ** .5) / 300)
        outside = report.estimate(250, voltage_standard_uncertainty_mv=3)
        self.assertEqual(outside["leaf_wetness"], 1)
        self.assertIsNone(outside["linearized_standard_uncertainty"])

    def test_needs_measured_non_degenerate_replicate_groups(self):
        with self.assertRaisesRegex(ValueError, "at least two"):
            summarize_leaf_calibration(dry_mv=[500], wet_mv=[600, 601], source="synthetic")
        with self.assertRaisesRegex(ValueError, "distinct"):
            summarize_leaf_calibration(dry_mv=[500, 505], wet_mv=[501, 504], source="synthetic")


if __name__ == "__main__":
    unittest.main()
