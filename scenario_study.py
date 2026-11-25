"""Compare explicit historical soil/leaf parameter scenarios on matched hours."""
import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from assumptions import Assumptions
from config import FEATURES
from Data.AZMET import AZMETSchema, read_azmet
from Data.HistoricalFeatures import historical_features
from Data.WeatherPhysics import ReferenceET
from Utils.Serialization import read_json, write_json


@dataclass
class ScenarioOutput:
    features: object
    flags: object
    provenance: dict


def _comparison(reference, alternative):
    a, b = reference.to_numpy(dtype=float), alternative.to_numpy(dtype=float)
    report = {}
    for i, name in enumerate(FEATURES):
        valid = np.isfinite(a[:, i]) & np.isfinite(b[:, i])
        delta = b[valid, i] - a[valid, i]
        if len(delta):
            report[name] = {"hours": len(delta), "mean_difference": float(delta.mean()),
                            "rms_difference": float(np.sqrt(np.mean(delta**2))),
                            "largest_absolute_difference": float(np.max(np.abs(delta)))}
        else:
            report[name] = {"hours": 0}
    return report


def compare_scenarios(source, metadata, alternatives):
    if not isinstance(alternatives, dict) or "reference" in alternatives:
        raise ValueError("Use named scenarios other than the reserved 'reference'")
    schema = AZMETSchema.from_dict(metadata["schema"])
    weather = read_azmet(source, schema, source_label=metadata["source_label"],
                         synthetic=metadata.get("synthetic", False))
    base_assumptions = metadata.get("assumptions", {})
    base_scenario = metadata["scenario"]
    runs = {}
    for name, changes in {"reference": {}, **alternatives}.items():
        if not isinstance(changes, dict) or set(changes) - {"assumptions", "scenario"}:
            raise ValueError("A scenario can change physical assumptions or forcing parameters")
        changed = changes.get("assumptions", {})
        physical = Assumptions.from_dict({
            compartment: {**base_assumptions.get(compartment, {}), **changed.get(compartment, {})}
            for compartment in base_assumptions.keys() | changed.keys()
        })
        if physical.site != Assumptions.from_dict(base_assumptions).site:
            raise ValueError("Scenarios for one archive must preserve its site and clock")
        reference = ReferenceET(physical.site.latitude_deg, **metadata["reference_et"])
        result = historical_features(weather, physical, reference,
                                     **{**base_scenario, **changes.get("scenario", {})})
        runs[name] = ScenarioOutput(*result)
    return {name: {"comparison_with_reference": _comparison(runs["reference"].features, item.features),
                   "resets": {kind: int(item.flags[kind].sum())
                              for kind in ("leaf_restart", "soil_restart", "reference_et_unavailable")},
                   "provenance": item.provenance} for name, item in runs.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("mapping", type=Path)
    parser.add_argument("scenarios", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    write_json(args.output, compare_scenarios(args.source, read_json(args.mapping),
                                             read_json(args.scenarios)))


if __name__ == "__main__":
    main()
