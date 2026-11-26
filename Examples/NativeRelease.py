"""Use a saved model with firmware-shaped synthetic AquaMap replay."""
import argparse
import base64
from datetime import datetime, timedelta, timezone
import json
import math
from pathlib import Path

from Acquisition.AITrace import TraceStore
from Acquisition.TraceAnalysis import forecast_map, survey_forecasts
from Acquisition.TraceImport import replay_segment
from Utils.Serialization import write_json


def run(checkpoint, output):
    output.mkdir()
    store = TraceStore(output / "trace")
    survey = store.create("Synthetic firmware-derived acceptance survey")["id"]
    start = datetime(2020, 7, 1, tzinfo=timezone.utc)
    for site, longitude in enumerate((-.0004, 0., .0004)):
        capture = output / f"synthetic-visit-{site}.jsonl"
        with capture.open("w") as stream:
            for hour in range(33):
                sun = max(0, math.sin(2 * math.pi * (hour - 13) / 24))
                payload = (f"RX,-92,5.2,AQ,{site+1},{hour+1},0.00000,{longitude:.5f},"
                           f"26,23,500,7.2,{25+9*sun+site:.2f},{65-30*sun:.1f},380")
                stamp = start + timedelta(hours=hour)
                stream.write(json.dumps({
                    "received_at": stamp.isoformat(),
                    "line_b64": base64.b64encode((payload + "\r\n").encode()).decode(),
                }) + "\n")
        replay_segment(store, survey, capture, label=f"Synthetic site {site}", schema="aquamap_rx_csv")
    config = {
        "reference_et": {"solar_offset_minutes": -420},
        "leaf_calibration": {"dry_mv": 300, "wet_mv": 600, "source": "SYNTHETIC software fixture"},
        "soil_calibration": {"source": "SYNTHETIC software fixture"},
        "hourly": {"clock": "received_at"},
    }
    forecasts = survey_forecasts(store, survey, config, checkpoint, steps=6)
    if forecasts["ready_locations"] != 3:
        raise ValueError(f"Expected three available synthetic visits: {forecasts['locations']}")
    issue = forecasts["locations"][0]["segments"][0]["forecast"][0]["timestamp"]
    from Acquisition.Hourly import _aware
    origin = _aware(issue) - timedelta(hours=1)
    maps = {str(lead): forecast_map(
        store, survey, forecasts, "air_temperature_c", origin + timedelta(hours=lead), origin,
        grid_size=21, domain={"center_lat_lon": [0, 0], "radius_m": 70}, diffusion={"iterations": 800}
    ) for lead in range(1, 7)}
    (output / "survey.json").write_bytes(store.export(survey))
    write_json(output / "forecasts.json", forecasts)
    write_json(output / "diffusion-maps.json", {"synthetic": True, "maps": maps})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    run(args.checkpoint, args.output)
