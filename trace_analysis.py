"""Check AI Trace segments, forecast from saved models, or map survey observations."""
import argparse
from pathlib import Path

from Acquisition.AITrace import TraceStore
from Acquisition.TraceAnalysis import ensemble_forecast_maps, forecast_map, observational_map, survey_forecasts
from Utils.Serialization import read_json, write_arrays, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("readiness", "forecast", "map", "forecast-map", "ensemble-map"):
        command = commands.add_parser(name)
        command.add_argument("survey_id")
        command.add_argument("output", type=Path)
        if name == "map":
            command.add_argument("--feature", required=True)
            command.add_argument("--timestamp", required=True)
            command.add_argument("--window-minutes", type=float, default=60)
            command.add_argument("--clock", choices=("received_at", "observed_at"), default="received_at")
        elif name == "ensemble-map":
            command.add_argument("--forecast-report", type=Path, required=True)
            command.add_argument("--adaptive", type=Path)
            command.add_argument("--origin", required=True)
            command.add_argument("--steps", type=int, default=6)
        elif name == "forecast-map":
            command.add_argument("--forecast-report", type=Path, required=True)
            command.add_argument("--feature", required=True)
            command.add_argument("--valid-at", required=True)
            command.add_argument("--origin", required=True)
        else:
            command.add_argument("--configuration", type=Path, required=True)
            command.add_argument("--checkpoint", type=Path, required=name == "forecast")
        if name == "forecast":
            command.add_argument("--steps", type=int, default=6)
            command.add_argument("--member", type=Path, action="append", default=[])
    args = parser.parse_args()
    store = TraceStore(args.root, recover=False)
    if args.command == "ensemble-map":
        arrays, result = ensemble_forecast_maps(store, args.survey_id, read_json(args.forecast_report),
                                                args.origin, steps=args.steps,
                                                adaptive=read_json(args.adaptive) if args.adaptive else None)
        write_arrays(args.output.with_suffix(".npz"), **arrays)
    elif args.command == "map":
        result = observational_map(store, args.survey_id, args.feature, args.timestamp,
                                   args.window_minutes, args.clock)
    elif args.command == "forecast-map":
        result = forecast_map(store, args.survey_id, read_json(args.forecast_report),
                              args.feature, args.valid_at, args.origin)
    else:
        result = survey_forecasts(store, args.survey_id, read_json(args.configuration),
                                  args.checkpoint, getattr(args, "steps", None), ensemble=getattr(args, "member", ()))
    write_json(args.output, result)


if __name__ == "__main__":
    main()
