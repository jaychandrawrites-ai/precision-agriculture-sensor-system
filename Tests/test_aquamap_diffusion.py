"""Firmware-derived replay and numerical diffusion checks."""
from datetime import datetime, timedelta, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
import base64
import json
import unittest

import numpy as np

from Acquisition.AITrace import TraceStore
from Acquisition.AquaMap import AquaMapParser
from Acquisition.TraceAnalysis import observational_map
from Acquisition.TraceImport import replay_segment


class AquaMapDiffusionTests(unittest.TestCase):
    def test_firmware_packet_contract(self):
        parser = AquaMapParser()
        now = datetime.now(timezone.utc)
        line = "RX,-92,5.2,AQ,1,2,0.00000,0.00100,30.0,24.0,450,6.8,22.50,52.2,360"
        self.assertIsNone(parser.feed("# listening", now))
        reading = parser.feed(line, now)
        self.assertIsNone(reading.observed_at)
        self.assertEqual(reading.packet_id, "2")
        self.assertEqual(reading.measurements["soil_ec_us_cm"], 450.)
        self.assertEqual(reading.measurements["radio_rssi_dbm"], -92)
        self.assertIn("duplicate_packet", parser.feed(line, now).flags)
        missing = parser.feed("RX,-92,5.2,AQ,1,3,,,,,,,,,", now)
        self.assertIn("missing:leaf_mv", missing.flags)
        self.assertNotIn("latitude_deg", missing.measurements)
        self.assertIn("device_restart", parser.feed(line, now).flags)
        self.assertIn("invalid_aquamap_frame", parser.feed("bad", now).flags)

    def test_separate_replayed_locations_map_inside_radius(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            store = TraceStore(root / "trace")
            survey = store.create("Firmware-derived replay")["id"]
            now = datetime.now(timezone.utc)
            for index, coordinate in enumerate((-.0003, .0003)):
                packet = (f"RX,-92,5.2,AQ,1,{index+1},0.00000,{coordinate:.5f},"
                          f"30,24,450,6.8,{20+index*5},52.2,360\r\n")
                raw = root / f"visit-{index}.jsonl"
                raw.write_text(json.dumps({"received_at": now.isoformat(),
                                           "line_b64": base64.b64encode(packet.encode()).decode()}) + "\n")
                replay_segment(store, survey, raw, label=f"Site {index}", schema="aquamap_rx_csv")
            mapped = observational_map(
                store, survey, "air_temperature_c", now + timedelta(seconds=1), grid_size=9,
                domain={"center_lat_lon": [0, 0], "radius_m": 70}, diffusion={"iterations": 30})
            self.assertEqual(mapped["interpolation"], "diffusion")
            self.assertEqual(mapped["diffusion"]["seed_cells"], 2)
            self.assertLessEqual(mapped["diffusion"]["final_energy"], mapped["diffusion"]["initial_energy"])
            self.assertIsNone(mapped["estimate"][0][0])
            self.assertTrue(20 <= np.asarray(mapped["estimate"])[4, 4] <= 25)
