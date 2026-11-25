"""Generate labeled synthetic inputs for exercising the Phase 3 entry points."""
import argparse
from datetime import datetime, timedelta, timezone
import json
import math
from pathlib import Path

import pandas as pd

from Acquisition.AcquisitionController import capture
from Acquisition.MappedPackets import MappedPacketParser, PacketSchema
from Pipeline import forecast_session, map_measurements, train_historical
from config import TrainingConfig
from Utils.Serialization import write_json


def historical_fixture(directory, hours=100):
    source = directory / "synthetic-weather.csv"
    times = pd.date_range("2020-06-01T00:00:00Z", periods=hours, freq="h")
    rows = []
    for i, timestamp in enumerate(times):
        sun = max(0, math.sin(2 * math.pi * (i - 13) / 24))
        rows.append({
            "time": timestamp.isoformat(), "temp_f": 77 + 16 * sun,
            "humidity_fraction": 0.65 - 0.30 * sun, "solar_w": 700 * sun,
            "wind_m_s": 1.7, "rain_in": 0.04 if i in (34, 66) else 0,
        })
    pd.DataFrame(rows).to_csv(source, index=False)
    fields = {
        "air_temperature_c": ("temp_f", "F"),
        "relative_humidity_pct": ("humidity_fraction", "fraction"),
        "shortwave_w_m2": ("solar_w", "W/m2"),
        "wind_m_s": ("wind_m_s", "m/s"),
        "precipitation_mm": ("rain_in", "in/hour"),
    }
    configuration = {
        "source_label": "Synthetic weather fixture for software verification",
        "synthetic": True,
        "schema": {"timestamp": "time", "convention": "end", "max_state_gap_hours": 1,
                   "columns": {name: {"source": source, "unit": unit}
                               for name, (source, unit) in fields.items()}},
        "reference_et": {"solar_offset_minutes": -420},
        "scenario": {"rain_interception_fraction": 0.05, "scenario_irrigation_mm": 0.0},
    }
    return source, configuration


def session_fixture(directory, hours=32):
    fields = {
        "air_temperature_c": {"key": "t", "unit": "C"},
        "relative_humidity_pct": {"key": "rh", "unit": "%"},
        "soil_moisture_pct": {"key": "sw", "unit": "%"},
        "soil_temperature_c": {"key": "st", "unit": "C"},
        "soil_ec_us_cm": {"key": "ec", "unit": "mS/cm"},
        "soil_ph": {"key": "ph", "unit": "pH"},
        "leaf_mv": {"key": "leaf", "unit": "V"},
        "latitude_deg": {"key": "lat", "unit": "deg"},
        "longitude_deg": {"key": "lon", "unit": "deg"},
    }
    schema = {"fields": fields, "label": "synthetic fixture JSON",
              "packet_id_key": "id", "device_id_key": "node", "observed_at_key": "when",
              "boot_id_key": "boot", "gps_fix_key": "fix", "constants": {"kind": "fixture"}}
    start = datetime(2020, 7, 1, tzinfo=timezone.utc)
    lines = []
    for i in range(hours):
        stamp = start + timedelta(hours=i)
        sun = max(0, math.sin(2 * math.pi * (i - 13) / 24))
        data = {"id": i, "node": "synthetic-node", "boot": "example-boot",
                "when": stamp.isoformat(), "kind": "fixture",
                "t": 25 + 9 * sun, "rh": 65 - 30 * sun, "sw": 26,
                "st": 23, "ec": 0.5, "ph": 7.2, "leaf": 0.35 + 0.06 * (1 - sun),
                "lat": 0.0, "lon": 0.0, "fix": True}
        lines.append((stamp, (json.dumps(data) + "\n").encode("utf-8")))
    if hours > 6:
        lines.insert(6, lines[5])
    capture(lines, directory / "session", parser=MappedPacketParser(PacketSchema.from_dict(schema)),
            metadata={"fixture": "synthetic"})
    config = {
        "reference_et": {"solar_offset_minutes": -420},
        "leaf_calibration": {"dry_mv": 300, "wet_mv": 600,
                             "source": "SYNTHETIC fixture values for software verification"},
        "soil_calibration": {"source": "SYNTHETIC fixture VWC percent"},
        "hourly": {"clock": "observed_at"},
    }
    return directory / "session", schema, config


def spatial_fixture(directory):
    timestamp = "2020-07-01T12:00:00Z"
    source = directory / "synthetic-field.csv"
    pd.DataFrame([
        (timestamp, 0.0, 0.0, 25.0),
        (timestamp, 0.0, 0.00005, 29.0),
        (timestamp, 0.00005, 0.0, 27.0),
        (timestamp, 0.00005, 0.00005, 30.0),
    ], columns=["timestamp", "latitude_deg", "longitude_deg", "air_temperature_c"]).to_csv(
        source, index=False
    )
    config = {"feature": "air_temperature_c", "timestamp": timestamp, "origin_lat_lon": [0, 0],
              "east_m": [0, 3, 6], "north_m": [0, 3, 6], "kriging_range_m": 4,
              "hard_seeds": True, "sampling": {"draws": 80, "burn_in": 40, "seed": 19}}
    return source, config


def run(directory):
    directory.mkdir()
    weather, historical = historical_fixture(directory)
    session, packet_schema, field = session_fixture(directory)
    measurements, spatial = spatial_fixture(directory)
    for name, record in (("historical-mapping", historical), ("packet-schema", packet_schema),
                         ("field-settings", field), ("spatial-settings", spatial)):
        write_json(directory / f"{name}.json", record)
    checkpoint = directory / "temporal.json"
    training = train_historical(weather, historical, checkpoint,
                                TrainingConfig(hidden=(16,), epochs=4, batch_size=16, dropout=0))
    forecast, field_report, gps, _ = forecast_session(session / "readings.jsonl", field, checkpoint, 6)
    forecast.to_csv(directory / "forecast.csv")
    gps.to_csv(directory / "gps-fixes.csv", index=False)
    spatial_report = map_measurements(measurements, spatial, directory / "map")
    report = {"fixture": "synthetic software verification", "training": training,
              "field": field_report, "spatial": spatial_report}
    write_json(directory / "report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    run(args.output)


if __name__ == "__main__":
    main()
