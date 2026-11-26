"""Exercise survey capture, observed mapping and saved-model prediction offline."""
import argparse
from pathlib import Path

from Acquisition.AITrace import TraceStore
from Acquisition.TraceAnalysis import ensemble_forecast_maps, forecast_map, observational_map, survey_forecasts
from Acquisition.TraceImport import replay_segment
from Examples.SyntheticPhase3 import historical_fixture, session_fixture
from Pipeline import train_historical
from Utils.Serialization import write_arrays, write_json
from config import TrainingConfig


def run(output, *, ensemble=False):
    output.mkdir()
    session, schema, configuration = session_fixture(output, 32)
    trace = TraceStore(output / "trace")
    survey_id = trace.create("SYNTHETIC five-location survey")["id"]
    # Reuse an explicitly synthetic stream, changing only the manually assigned
    # location. Omit the fixture GPS fields so the manual locations are exercised.
    for name in ("latitude_deg", "longitude_deg"):
        schema["fields"].pop(name)
    schema.pop("gps_fix_key")
    for index in range(5):
        # Seed each location with manual coordinates; replay creates a distinct visit.
        result = trace.start(survey_id, label=f"Synthetic site {index}",
                             coordinates=[0, index * .0002])
        location = result["active_location_id"]
        trace.stop(survey_id, interrupted=True)
        replay_segment(trace, survey_id, session / "raw.jsonl", location_id=location, schema=schema)
    trace.finish(survey_id)
    source, mapping = historical_fixture(output)
    checkpoint = output / "synthetic-model.json"
    train_historical(source, mapping, checkpoint, TrainingConfig(epochs=3, hidden=(16,), dropout=0))
    members = []
    if ensemble:
        other = output / "synthetic-model-2.json"
        train_historical(source, mapping, other, TrainingConfig(epochs=3, hidden=(16,), dropout=0, seed=17))
        members.append(other)
    forecasts = survey_forecasts(trace, survey_id, configuration, checkpoint, steps=6, ensemble=members)
    if ensemble:
        arrays, metadata = ensemble_forecast_maps(trace, survey_id, forecasts,
                                                   "2020-07-02T07:00:00Z", steps=6)
        write_arrays(output / "ensemble-maps.npz", **arrays)
        write_json(output / "ensemble-maps.json", metadata)
    mapped = observational_map(trace, survey_id, "air_temperature_c", "2020-07-02T07:00:00Z")
    predicted_map = forecast_map(trace, survey_id, forecasts, "air_temperature_c",
                                 "2020-07-02T10:00:00Z", "2020-07-02T07:00:00Z")
    write_json(output / "forecast-map.json", predicted_map)
    (output / "survey.json").write_bytes(trace.export(survey_id))
    write_json(output / "forecasts.json", forecasts)
    write_json(output / "observed-map.json", mapped)
    write_json(output / "field-settings.json", configuration)
    return forecasts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--ensemble", action="store_true")
    args = parser.parse_args()
    report = run(args.output, ensemble=args.ensemble)
    print(f"Synthetic forecast-eligible locations: {report['ready_locations']}")


if __name__ == "__main__":
    main()
