"""Score predictive draws at explicitly held-out locations."""
import argparse
from hashlib import sha256
from pathlib import Path

from Spatial.CovarianceStudy import CovarianceScenario, covariance_scenario_fields, posterior_predictive_checks
from Spatial.Kriging import CovarianceModel
from Utils.Serialization import read_json, write_json


def evaluate(specification):
    import numpy as np

    train, held = specification["conditioning"], specification["held_out"]
    fit_ids, held_ids = train["ids"], held["ids"]
    points, queries = np.asarray(train["points"], dtype=float), np.asarray(held["points"], dtype=float)
    if (len(fit_ids) != len(points) or len(held_ids) != len(queries)
            or len(set(fit_ids + held_ids)) != len(fit_ids) + len(held_ids)
            or set(map(tuple, points)) & set(map(tuple, queries))):
        raise ValueError("Conditioning and holdout IDs and coordinates must be disjoint")
    scenarios = [CovarianceScenario(item["id"], CovarianceModel(**item["covariance"]))
                 for item in specification["scenarios"]]
    seed = specification.get("seed", 42)
    result = covariance_scenario_fields(
        points, train["values"], queries, scenarios, measurement_variance=train.get("variance"),
        trend=specification.get("trend", "constant"), draws=specification.get("draws", 200),
        seed=seed)
    verification = posterior_predictive_checks(
        result.draws, held["values"], observation_variance=held.get("variance"), seed=seed + 1)
    for identifier, diagnostics in zip(held_ids, verification["site_diagnostics"]):
        diagnostics["id"] = identifier
    return {"verification": verification, "scenarios": result.diagnostics,
            "conditional_variance": result.conditional_variance,
            "between_scenario_variance": result.scenario_variance}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("specification", type=Path)
    parser.add_argument("report", type=Path)
    args = parser.parse_args()
    report = evaluate(read_json(args.specification))
    report["specification_sha256"] = sha256(args.specification.read_bytes()).hexdigest()
    write_json(args.report, report)


if __name__ == "__main__":
    main()
