"""Calibrate, assess or mechanically verify offline Jupiter forecasts."""
import argparse
from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd

from config import FEATURES
from Data.ForecastHistory import inspect_history
from Data.Preprocessing import read_hourly
from Temporal.HorizonVerification import InferenceArtifact, VerificationEnsemble, verify_leads
from Temporal.ReliabilityGate import ReliabilityProfile, eligible_origins, ensemble_disagreement
from Utils.Serialization import read_json, write_json


def artifacts(checkpoints, device):
    results = tuple(InferenceArtifact(path, device) for path in checkpoints)
    if not results or len({artifact.checksum for artifact in results}) != len(results):
        raise ValueError("Use distinct checkpoints")
    return results


def fit_profile(source, models, *, steps, stride=25, tail=.05, station=(), kind="historical",
                calibration_id=None):
    values, times = read_hourly(source)
    maximum = min(model.limit for model in models)
    if type(steps) is not int or not 1 <= steps <= maximum:
        raise ValueError(f"The provided models share only leads 1–{maximum}")
    context = max(model.context for model in models)
    options = {"history_hours": context, "stride": stride, "tail": tail, "station_ids": station,
               "source_kind": kind, "calibration_id": calibration_id, "supported_steps": steps}
    profile = ReliabilityProfile.fit(values, times, **options)
    if len(models) > 1:
        spread = []
        for origin in eligible_origins(values, times, history_hours=context, stride=stride):
            forecasts = np.stack([model.forecast(values[:origin+1], times[:origin+1], steps)
                                  for model in models])
            spread.append(ensemble_disagreement(forecasts, profile.scale))
        profile = ReliabilityProfile.fit(values, times, **options, disagreements=spread)
    return {"profile": profile.to_record(), "checkpoint_hashes": [model.checksum for model in models],
            "calibration_sha256": sha256(Path(source).read_bytes()).hexdigest(),
            "calibration_stride_hours": stride}


def assess_data(values, times, record, models, *, steps, station=None, kind="historical",
                calibration_id=None, diagnostic_only=False):
    if record["checkpoint_hashes"] != [model.checksum for model in models]:
        raise ValueError("Reliability profile is bound to another checkpoint or checkpoint order")
    profile = ReliabilityProfile.from_record(record["profile"])
    latest = pd.Timestamp(times[-1], tz="UTC") if len(times) else None
    if latest is not None and latest <= pd.Timestamp(profile.calibration_last_utc):
        raise ValueError("An assessed origin must follow the designated calibration period")
    metadata = {"steps": steps, "model_max_steps": min(model.limit for model in models),
                "station_id": station, "source_kind": kind, "calibration_id": calibration_id,
                "enforce": not diagnostic_only}
    result = profile.assess(values, times, **metadata)
    if result["decision"] == "forecast":
        predictions = np.stack([model.forecast(values, times, steps) for model in models])
        spread = ensemble_disagreement(predictions, profile.scale) if len(models) > 1 else None
        result = profile.assess(values, times, **metadata, disagreement=spread)
        if result["decision"] == "forecast":
            result["point_forecast"] = predictions.mean(axis=0)
    result["artifact_hashes"] = record["checkpoint_hashes"]
    return result


def evaluate_policy(values, times, record, models, *, steps, stride=72, **metadata):
    """Compare policy decisions on fixed post-calibration origins with available labels."""
    if type(steps) is not int or not 1 <= steps <= min(model.limit for model in models):
        raise ValueError("Verification horizon is unsupported by the selected models")
    if type(stride) is not int or stride < steps:
        raise ValueError("Verification stride must be at least the forecast horizon")
    profile = ReliabilityProfile.from_record(record["profile"])
    calibration_end = pd.Timestamp(profile.calibration_last_utc).tz_localize(None).to_datetime64()
    base = eligible_origins(values, times, history_hours=profile.history_hours, stride=1)
    candidates = [int(i) for i in base if times[i] > calibration_end and i + steps < len(times)]
    scores, counts, accepted = [], {}, []
    structural_exclusions = 0
    previous = None
    for origin in candidates:
        if previous is not None and times[origin] - previous < np.timedelta64(stride, "h"):
            continue
        segment = values[origin:origin+steps+1]
        future = slice(origin-1, origin+steps+1)
        if not inspect_history(values[future], times[future], hours=steps+2)["ready"]:
            continue
        issue, available = times[:origin+1], values[:origin+1]
        allowed = assess_data(available, issue, record, models, steps=steps, **metadata)
        baseline = assess_data(available, issue, record, models, steps=steps, diagnostic_only=True, **metadata)
        if baseline["decision"] == "abstain":
            structural_exclusions += 1
            # Apply the same spacing to each attempted origin, including exclusions.
            previous = times[origin]
            for item in baseline["issues"]:
                counts[item["code"]] = counts.get(item["code"], 0) + 1
            continue
        previous = times[origin]
        for item in allowed["issues"]:
            counts[item["code"]] = counts.get(item["code"], 0) + 1
        error = baseline["point_forecast"] - segment[1:]
        scores.append(error)
        if allowed["decision"] == "forecast":
            accepted.append(error)
    if not scores:
        raise ValueError("No complete post-calibration verification trajectories")
    def metrics(errors):
        if not errors:
            return None
        stack = np.asarray(errors)
        return {"rmse_by_lead_feature": np.sqrt((stack**2).mean(axis=0)),
                "mae_by_lead_feature": np.abs(stack).mean(axis=0)}
    return {
        "verification_origins": len(scores), "accepted_origins": len(accepted),
        "acceptance_rate": len(accepted) / len(scores), "issue_counts": counts,
        "structurally_excluded_origins": structural_exclusions,
        "all_mechanical_forecasts": metrics(scores), "accepted_forecasts": metrics(accepted),
        "features": list(FEATURES), "forecast_horizon": steps, "stride_hours": stride,
        "interpretation": "Descriptive post-calibration verification, not a guarantee under transfer.",
        "checkpoint_hashes": record["checkpoint_hashes"],
    }


def main():
    command = argparse.ArgumentParser(description=__doc__)
    subcommands = command.add_subparsers(dest="mode", required=True)
    for mode in ("fit", "assess", "evaluate", "verify-leads"):
        parser = subcommands.add_parser(mode)
        parser.add_argument("source", type=Path)
        parser.add_argument("output", type=Path)
        parser.add_argument("--checkpoint", type=Path, action="append", required=True)
        parser.add_argument("--device", default="cpu")
        if mode != "verify-leads":
            parser.add_argument("--steps", type=int, required=True)
            parser.add_argument("--source-kind", default="historical")
            parser.add_argument("--calibration-id")
    subcommands.choices["fit"].add_argument("--station", action="append", default=[])
    subcommands.choices["fit"].add_argument("--stride", type=int, default=25)
    subcommands.choices["fit"].add_argument("--tail", type=float, default=.05)
    for mode in ("assess", "evaluate"):
        subcommands.choices[mode].add_argument("--station")
        subcommands.choices[mode].add_argument("--profile", type=Path, required=True)
    subcommands.choices["evaluate"].add_argument("--stride", type=int, default=72)
    subcommands.choices["assess"].add_argument("--diagnostic-only", action="store_true")
    args = command.parse_args()
    models = artifacts(args.checkpoint, args.device)
    if args.mode == "fit":
        report = fit_profile(args.source, models, steps=args.steps, stride=args.stride, tail=args.tail,
                             station=args.station, kind=args.source_kind, calibration_id=args.calibration_id)
    else:
        values, times = read_hourly(args.source)
        if args.mode == "assess":
            report = assess_data(values, times, read_json(args.profile), models, steps=args.steps,
                                 station=args.station, kind=args.source_kind, calibration_id=args.calibration_id,
                                 diagnostic_only=args.diagnostic_only)
        elif args.mode == "evaluate":
            report = evaluate_policy(values, times, read_json(args.profile), models, steps=args.steps,
                                     stride=args.stride, station=args.station, kind=args.source_kind,
                                     calibration_id=args.calibration_id)
            report["verification_sha256"] = sha256(args.source.read_bytes()).hexdigest()
        else:
            report = {"checks": [verify_leads(model, values, times) for model in models]}
            if len(models) > 1:
                report["checks"].append(verify_leads(VerificationEnsemble(models), values, times))
    write_json(args.output, report)


if __name__ == "__main__":
    main()
