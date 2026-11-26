"""Offline research workflow with explicitly synthetic weather and sensor data."""
import argparse
from pathlib import Path

from Acquisition.AITrace import TraceStore
from Acquisition.TraceAnalysis import ensemble_forecast_maps, survey_forecasts
from Data.AZMET import inspect_schema
from Data.Preprocessing import read_hourly
from Evolution.Resumable import SearchOptions
from Evolution.SequenceSearch import compare_sequences
from Examples.SyntheticPhase3 import historical_fixture
from Examples.SyntheticTrace import run as synthetic_trace
from Utils.Serialization import read_json, write_arrays, write_json
from azmet import prepare
from config import EvolutionConfig, TrainingConfig
from direct_forecast import train
from forecast import forecast_csv
from research_figures import render
from spatial_figures import render as render_map


def run(output):
    output.mkdir()
    source, mapping = historical_fixture(output, hours=360)
    write_json(output / "schema.json", inspect_schema(source))
    features, _, report = prepare(source, mapping)
    prepared = output / "hourly.csv"
    features.to_csv(prepared)
    write_json(output / "preparation.json", report)
    settings = TrainingConfig(epochs=1, rollout_steps=6)
    checkpoint = output / "direct.json"
    fitted = train(prepared, checkpoint, settings, 6, width=8)
    write_json(output / "training.json", fitted)
    values, timestamps = read_hourly(prepared)
    winners = output / "winners"
    search = compare_sequences(values, timestamps, settings,
                               EvolutionConfig(population=2, generations=1, candidate_epochs=1),
                               SearchOptions(max_parameters=3000), export_directory=winners,
                               provenance={"synthetic": True})
    write_json(output / "search.json", search)
    winner = winners / "classical_ga.json"
    forecast_csv(prepared, winner, 6).to_csv(output / "forecast.csv", index=False)

    survey = output / "survey"
    synthetic_trace(survey)
    store = TraceStore(survey / "trace", recover=False)
    identifier = read_json(survey / "forecasts.json")["survey_id"]
    issue = "2020-07-02T07:00:00Z"
    forecasts = survey_forecasts(store, identifier, read_json(survey / "field-settings.json"),
                                 winner, steps=6, ensemble=[checkpoint])
    if forecasts["ready_locations"] != 5:
        raise RuntimeError("Synthetic survey could not load both model checkpoints")
    arrays, metadata = ensemble_forecast_maps(
        store, identifier, forecasts, issue, steps=6, grid_size=15,
        domain={"center_lat_lon": [0, .0004], "radius_m": 70},
        predictive={"method": "idw", "draws": 20, "feature_scales": [1.] * 6})
    metadata["synthetic"] = True
    write_json(output / "radius.json", metadata)
    write_arrays(output / "radius.npz", **arrays)
    for kind, artifact in (("training", "training.json"), ("forecast", "forecast.csv"),
                           ("pareto", "search.json")):
        render(output / artifact, output / f"{kind}-figure", kind)
    render_map(output / "radius.json", output / "spatial-figure", lead=6, feature="air_temperature_c")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    run(parser.parse_args().output)
