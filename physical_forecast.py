"""Apply shared physical scenarios to saved-checkpoint inference and optional ensemble maps."""
import argparse
from pathlib import Path
import re

import pandas as pd

from config import FEATURES
from physical_ensemble import ParameterRange
from Physics.ForecastScenarios import load_site, scenario_forecasts, utc_hour
from spatial_forecast import map_draw_file
from Utils.ModelLoader import load_forecaster
from Utils.Serialization import atomic_output, read_json, write_arrays, write_json


def _from_spec(path, spec):
    result = Path(spec)
    return result if result.is_absolute() else path.parent / result


def _rows(predictions, sites, report, steps):
    issue_at = pd.Timestamp(report["issue_at"])
    for draw, identifier in enumerate(report["draw_ids"]):
        for site_index, (name, site) in enumerate(sites.items()):
            for lead in range(steps):
                valid_at = issue_at + pd.to_timedelta(lead + 1, unit="h")
                yield {"draw_id": identifier, "site_id": name, "lead_hours": lead+1,
                       "forecast_origin": issue_at.isoformat(), "valid_at": valid_at.isoformat(),
                       "value_origin": "model_forecast", **(site.position or {}),
                       **dict(zip(FEATURES, predictions[draw, site_index, lead]))}


def run_experiment(spec_path, output, *, figures=False):
    spec_path = Path(spec_path)
    spec = read_json(spec_path)
    issue_at = utc_hour(spec["issue_at"])
    sites = {}
    for site_id, item in spec["sites"].items():
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", site_id):
            raise ValueError("Site identifiers must be alphanumeric with optional '.', '_' and '-'")
        sites[site_id] = load_site(
            _from_spec(spec_path, item["source"]),
            read_json(_from_spec(spec_path, item["configuration"])),
            item["kind"], issue_at, gps_freshness_hours=spec.get("gps_freshness_hours", 1)
        )
    steps = spec["steps"]
    map_spec = spec.get("map")
    if map_spec is not None:
        leads = map_spec.get("leads", [])
        if (not leads or len(set(leads)) != len(leads)
                or any(type(lead) is not int or not 1 <= lead <= steps for lead in leads)):
            raise ValueError("Choose unique map leads within the requested forecast horizon")
        if len(sites) < 2 or any(site.position is None for site in sites.values()):
            raise ValueError("Forecast maps require at least two field sites with recent GPS at issue time")
    params = [ParameterRange(**item) for item in spec["ranges"]]
    forecaster = load_forecaster(_from_spec(spec_path, spec["checkpoint"]), spec.get("device", "cpu"))
    predictions, histories, report = scenario_forecasts(
        sites, forecaster, params, issue_at, steps=steps, draws=spec.get("draws", 16),
        seed=spec.get("seed", 42), mass=spec.get("central_mass", .9),
        latent_correlation=spec.get("latent_correlation"), perturb=spec.get("local_derivatives", True)
    )
    output = Path(output)
    output.mkdir()
    trajectories = pd.DataFrame(_rows(predictions, sites, report, steps))
    with atomic_output(output / "trajectories.csv") as stream:
        trajectories.to_csv(stream, index=False)
    write_arrays(output / "trajectories.npz", forecasts=predictions, histories=histories,
                 parameter_draws=report["parameter_draws"])
    report["site_axis"] = list(sites)
    report["forecast_leads_hours"] = list(range(1, steps+1))
    report["maps"] = {}
    if map_spec is not None:
        settings = read_json(_from_spec(spec_path, map_spec["configuration"]))
        for lead in map_spec["leads"]:
            source = output / f"map-input-{lead:03d}.csv"
            with atomic_output(source) as stream:
                trajectories.loc[trajectories["lead_hours"] == lead].to_csv(stream, index=False)
            report["maps"][str(lead)] = map_draw_file(source, settings, output / f"map-{lead:03d}")
    write_json(output / "report.json", report)
    if figures:
        from Physics.ScenarioFigures import render_scenario_figures

        render_scenario_figures(report, output / "figures")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("specification", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--figures", action="store_true", help="Write PNG/SVG figures with matplotlib")
    args = parser.parse_args()
    run_experiment(args.specification, args.output, figures=args.figures)


if __name__ == "__main__":
    main()
