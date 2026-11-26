"""Propagate caller-declared physical parameter scenarios through archived weather."""
import argparse
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import NormalDist
from time import perf_counter

import numpy as np
import pandas as pd

from assumptions import Assumptions
from config import FEATURES
from Data.AZMET import AZMETSchema, read_azmet
from Data.HistoricalFeatures import historical_features
from Data.WeatherPhysics import ReferenceET
from Utils.Serialization import atomic_output, read_json, write_json


SCENARIO_PARAMETERS = {"scenario_irrigation_mm", "rain_interception_fraction", "initial_leaf_wetness"}
PHYSICAL_SECTIONS = {"soil", "leaf", "crop"}


@dataclass(frozen=True)
class ParameterRange:
    path: str
    lower: float
    upper: float

    def __post_init__(self):
        parts = self.path.split(".")
        if (len(parts) != 2 or parts[0] not in PHYSICAL_SECTIONS | {"scenario"}
                or not np.isfinite([self.lower, self.upper]).all() or self.lower >= self.upper):
            raise ValueError("Expected a varied soil, leaf, crop or scenario parameter range")


def stratified_sample(parameters, draws, seed, *, latent_correlation=None):
    if (type(draws) is not int or not 4 <= draws <= 256 or not parameters or
            len({p.path for p in parameters}) != len(parameters)):
        raise ValueError("Use 4–256 draws and unique parameter paths")
    rng = np.random.default_rng(seed)
    u = np.column_stack([(rng.permutation(draws) + rng.random(draws)) / draws for _ in parameters])
    if latent_correlation is not None:
        matrix = np.asarray(latent_correlation, dtype=float)
        if (matrix.shape != (len(parameters), len(parameters)) or not np.isfinite(matrix).all()
                or not np.allclose(matrix, matrix.T, atol=1e-12, rtol=0)
                or not np.allclose(np.diag(matrix), 1., atol=1e-12, rtol=0)):
            raise ValueError("Latent correlation must be a symmetric unit-diagonal matrix")
        try:
            factor = np.linalg.cholesky(matrix)
        except np.linalg.LinAlgError as error:
            raise ValueError("Latent correlation must be positive definite") from error
        distribution = NormalDist()
        latent = np.vectorize(distribution.inv_cdf, otypes=[float])(u) @ factor.T
        ranks = np.argsort(latent, axis=0, kind="stable")
        # Copula ranks rearrange the independent strata while preserving every marginal stratum.
        np.put_along_axis(u, ranks, np.sort(u, axis=0), axis=0)
    bounds = np.array([(entry.lower, entry.upper) for entry in parameters])
    return bounds[:, 0] + u * np.diff(bounds, axis=1)[:, 0]


def _configured(base, scenario, parameters, draw):
    physical = asdict(base)
    scenario = dict(scenario)
    for spec, value in zip(parameters, draw):
        section, name = spec.path.split(".")
        if section == "scenario":
            if name not in SCENARIO_PARAMETERS:
                raise ValueError(f"Unknown scenario parameter {name}")
            scenario[name] = float(value)
        else:
            if name not in physical[section] or name in {"dry_mv", "wet_mv"}:
                raise ValueError(f"Unsupported parameter {spec.path}")
            physical[section][name] = float(value)
    return Assumptions.from_dict(physical), scenario


def _ranks(inputs, summaries):
    x = pd.DataFrame(inputs).rank(method="average").to_numpy()
    y = pd.DataFrame(summaries).rank(method="average").to_numpy()
    x -= x.mean(axis=0)
    y -= y.mean(axis=0)
    denom = np.sqrt((x**2).sum(axis=0)[:, None] * (y**2).sum(axis=0))
    score = np.divide(x.T @ y, denom, out=np.zeros_like(denom), where=denom > 0)
    return score, denom > 0


def physical_ensemble(weather, base, reference, scenario, parameters, *, draws=16, seed=42, mass=.9, latent_correlation=None):
    """Report scenario spread on common hours for train-independent forcing."""
    if not 0 < mass < 1:
        raise ValueError("Central mass must be between zero and one")
    parameters = tuple(parameters)
    samples = stratified_sample(parameters, draws, seed, latent_correlation=latent_correlation)
    try:
        configured = [_configured(base, scenario, parameters, draw) for draw in samples]
    except (ValueError, TypeError) as exc:
        raise ValueError("A declared range generated an invalid physical configuration") from exc
    started = perf_counter()
    forecasts = []
    for assumptions, forcing in configured:
        output, _, _ = historical_features(weather, assumptions, reference, **forcing)
        forecasts.append(output.to_numpy(dtype=float))
    cube = np.asarray(forecasts)
    common = np.isfinite(cube).all(axis=0)
    # Compare each feature only on hours where every scenario supplies that feature.
    stats = np.full((3, *common.shape), np.nan)
    sensitivity_input = np.full((draws, len(FEATURES)), np.nan)
    tail = (1-mass)/2
    for index, eligible in enumerate(common.T):
        if eligible.any():
            values = cube[:, eligible, index]
            stats[:, eligible, index] = np.quantile(values, [tail, .5, 1-tail], axis=0)
            sensitivity_input[:, index] = values.mean(axis=1)
    scores, defined = _ranks(samples, sensitivity_input)
    bands = pd.DataFrame(index=weather.values.index)
    for label, values in zip(("lower", "median", "upper"), stats):
        for index, feature in enumerate(FEATURES):
            bands[f"{feature}_{label}"] = values[:, index]
    result = {"draws": draws, "seed": seed, "central_mass": mass,
              "design": ("independently stratified uniform assumption ranges" if latent_correlation is None
                         else "Gaussian copula rank dependence with uniform stratified marginals"),
              "latent_correlation_assumption": latent_correlation,
              "interpretation": "sensitivity to supplied scenarios, without empirical calibration",
              "source": weather.provenance, "reference": asdict(reference),
              "base_physics": base.metadata(), "base_scenario": scenario,
              "runtime_seconds": perf_counter() - started,
              "parameters": [asdict(spec) for spec in parameters],
              "sampled_parameters": samples,
              "features": {
                  feature: {"common_hours": int(common[:, i].sum()),
                            "spearman_correlation_of_mean": {
                                p.path: float(scores[j, i]) if defined[j, i] else None
                                for j, p in enumerate(parameters)}}
                  for i, feature in enumerate(FEATURES)
              }}
    return bands, result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("mapping", type=Path)
    parser.add_argument("ranges", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--draws", type=int, default=16)
    parser.add_argument("--latent-correlation", type=Path, help="JSON matrix in range-file parameter order")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    mapping = read_json(args.mapping)
    base = Assumptions.from_dict(mapping.get("assumptions", {}))
    reference = ReferenceET(base.site.latitude_deg, **mapping["reference_et"])
    weather = read_azmet(args.source, AZMETSchema.from_dict(mapping["schema"]),
                         source_label=mapping["source_label"], synthetic=mapping.get("synthetic", False))
    parameters = [ParameterRange(**item) for item in read_json(args.ranges)]
    bands, summary = physical_ensemble(weather, base, reference, mapping["scenario"], parameters,
                                       draws=args.draws, seed=args.seed, latent_correlation=
                                       read_json(args.latent_correlation) if args.latent_correlation else None)
    with atomic_output(args.output.with_suffix(".csv")) as handle:
        bands.to_csv(handle)
    write_json(args.output, summary)


if __name__ == "__main__":
    main()
