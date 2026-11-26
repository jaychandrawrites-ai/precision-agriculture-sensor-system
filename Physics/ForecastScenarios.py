"""Propagate explicitly configured physical uncertainty into checkpoint forecasts."""
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

from Acquisition.Calibration import LeafCalibration, SoilCalibration
from Acquisition.FieldFeatures import field_features
from Acquisition.Hourly import session_hourly
from assumptions import Assumptions
from config import FEATURES
from Data.AZMET import AZMETSchema, read_azmet
from Data.ForecastHistory import require_history
from Data.HistoricalFeatures import historical_features
from Data.WeatherPhysics import ReferenceET
from physical_ensemble import _configured, _ranks, stratified_sample


PARAMETER_UNITS = {
    "soil.root_depth_m": "m",
    "scenario.scenario_irrigation_mm": "mm per hour",
    "measurement.soil_vwc_bias": "m³/m³",
    "leaf.storage_kg_m2": "kg/m²",
    "leaf.heat_capacity_j_m2_k": "J/(m² K)",
    "leaf.mass_transfer_kg_m2_s_kpa": "kg/(m² s kPa)",
    "leaf.convection_w_m2_k": "W/(m² K)",
    "crop.critical_temperature_c": "°C",
    "crop.canopy_stress_offset_c": "°C difference",
    "crop.heat_accumulation_per_hour": "index/(°C hour)",
    "crop.recovery_per_hour": "1/hour",
}
FIELD_FIXED = {"soil.initial_depletion_fraction"}


def utc_hour(value):
    time = pd.Timestamp(value)
    if pd.isna(time) or time.tzinfo is None:
        raise ValueError("Forecast issuance needs an explicit time zone")
    time = time.tz_convert("UTC")
    if time != time.floor("h"):
        raise ValueError("Forecast issuance must be an exact hourly UTC boundary")
    return time


def _recent_fix(gps, issue_at, freshness_hours):
    if gps.empty:
        return None
    eligible = gps[(gps["timestamp"] <= issue_at)
                   & (gps["received_at"] <= issue_at)
                   & (gps["timestamp"] >= issue_at - pd.to_timedelta(float(freshness_hours), unit="h"))]
    if eligible.empty:
        return None
    point = eligible.sort_values(["timestamp", "received_at"], kind="stable").iloc[-1]
    return {name: float(point[name]) for name in ("latitude_deg", "longitude_deg")}


@dataclass
class ForecastSite:
    kind: str
    weather: object
    base: Assumptions
    reference: ReferenceET
    forcing: dict
    position: dict | None = None

    def uses(self, path):
        if self.kind == "historical":
            return not path.startswith("measurement.")
        return not path.startswith(("leaf.", "scenario.")) and path not in FIELD_FIXED

    def prepare(self, params=(), sample=()):
        active = [(param, value) for param, value in zip(params, sample) if self.uses(param.path)]
        model_parameters = [(param, value) for param, value in active
                            if not param.path.startswith("measurement.")]
        assumptions, forcing = _configured(self.base, self.forcing,
                                           [p for p, _ in model_parameters],
                                           [x for _, x in model_parameters])
        if self.kind == "historical":
            return historical_features(self.weather, assumptions, self.reference, **forcing)
        bias = next((value for param, value in active
                     if param.path == "measurement.soil_vwc_bias"), 0.0)
        weather = self.weather
        if bias:
            adjusted = weather.values.copy()
            moisture = adjusted["soil_vwc"] + bias
            if ((moisture < 0) | (moisture > 1)).any():
                raise ValueError("A soil-moisture scenario leaves the physical [0, 1] range")
            adjusted["soil_vwc"] = moisture
            weather = replace(weather, values=adjusted)
        return field_features(weather, assumptions, self.reference)


def load_site(path, configuration, kind, issue_at, *, gps_freshness_hours=1):
    base = Assumptions.from_dict(configuration.get("assumptions", {}))
    reference = ReferenceET(base.site.latitude_deg, **configuration["reference_et"])
    if kind == "historical":
        weather = read_azmet(Path(path), AZMETSchema.from_dict(configuration["schema"]),
                             source_label=configuration["source_label"],
                             synthetic=configuration.get("synthetic", False))
        rows = weather.values.index <= issue_at
        if not rows.any():
            raise ValueError("No historical observations as of forecast issuance")
        weather = replace(weather, values=weather.values.loc[rows], flags=weather.flags.loc[rows])
        return ForecastSite(kind, weather, base, reference, configuration["scenario"])
    if kind != "field":
        raise ValueError("Each site kind must be historical or field")
    hourly = configuration.get("hourly", {})
    if hourly.get("complete_only", True) is not True:
        raise ValueError("Field scenario inference requires closed hourly bins")
    if "as_of" in hourly and utc_hour(hourly["as_of"]) != issue_at:
        raise ValueError("Field configuration as_of must equal the common issue time")
    weather, gps, _ = session_hourly(
        path, LeafCalibration(**configuration["leaf_calibration"]),
        SoilCalibration(**configuration["soil_calibration"]), **{**hourly, "as_of": issue_at.isoformat()}
    )
    return ForecastSite(kind, weather, base, reference, {}, _recent_fix(gps, issue_at, gps_freshness_hours))


def _history(site, issue_at, params=(), sample=()):
    features, _, provenance = site.prepare(params, sample)
    if features.empty or features.index[-1] != issue_at:
        raise ValueError("A site lacks observations ending at the common issue time")
    return require_history(features.to_numpy(), features.index.to_numpy(dtype="datetime64[ns]")), provenance


def _rank_results(parameters, values, inputs):
    flat = values.reshape(len(values), -1)
    score, defined = _ranks(inputs, flat)
    shape = (len(parameters), *values.shape[1:])
    # Undefined rank correlations are JSON nulls, rather than false zero effects.
    entries = np.where(defined, score.astype(object), None).reshape(shape)
    return {param.path: row.tolist() for param, row in zip(parameters, entries)}


def _local_derivatives(site, issue_at, forecaster, steps, params, nominal_parameters):
    results = {}
    for index, param in enumerate(params):
        if not site.uses(param.path):
            results[param.path] = {"status": "not_applicable_to_feature_generator"}
            continue
        # Use a finite central perturbation relative to the declared range.
        center = nominal_parameters.copy()
        step = (param.upper - param.lower) / 100
        try:
            left, right = center.copy(), center.copy()
            left[index] -= step
            right[index] += step
            a, _ = _history(site, issue_at, params, left)
            b, _ = _history(site, issue_at, params, right)
        except ValueError as exc:
            results[param.path] = {"status": "invalid_central_perturbation", "reason": str(exc)}
            continue
        gradient = (forecaster.forecast(b, steps) - forecaster.forecast(a, steps)) / (2 * step)
        results[param.path] = {"status": "estimated", "parameter_delta": step,
                               "derivative_by_history_hour": (b - a) / (2 * step),
                               "derivative_by_lead": gradient}
    return results


def scenario_forecasts(sites, forecaster, parameters, issue_at, *, steps, draws=16, seed=42,
                       mass=.9, latent_correlation=None, perturb=True):
    """Return coherent [scenario, site, lead, feature] draws and provenance."""
    issue_at = utc_hour(issue_at)
    params = tuple(parameters)
    if (not sites or type(steps) is not int or steps < 1 or not 0 < mass < 1):
        raise ValueError("Supply sites, a positive lead count, and a central mass in (0, 1)")
    for param in params:
        if not any(site.uses(param.path) for site in sites.values()):
            raise ValueError(f"No site uses parameter {param.path}")
    samples = stratified_sample(params, draws, seed, latent_correlation=latent_correlation)
    middle = np.array([(p.lower + p.upper) / 2 for p in params])
    forecasts, histories, summary = [], [], {}
    for name, site in sites.items():
        baseline, provenance = _history(site, issue_at)
        compatibility = forecaster.require_compatible_field(provenance)
        nominal = forecaster.forecast(baseline, steps)
        histories_site = np.stack([_history(site, issue_at, params, sample)[0] for sample in samples])
        predictions = np.stack([forecaster.forecast(history, steps) for history in histories_site])
        histories.append(histories_site)
        forecasts.append(predictions)
        tail = (1 - mass) / 2
        summary[name] = {
            "kind": site.kind, "input_provenance": provenance, "baseline_checkpoint_physics_checked": compatibility,
            "gps_at_issuance": site.position, "nominal_forecast": nominal,
            "applicable_parameters": [p.path for p in params if site.uses(p.path)],
            "history": {"minimum_by_hour_feature": histories_site.min(axis=0),
                        "maximum_by_hour_feature": histories_site.max(axis=0),
                        "rank_correlation_by_hour_feature": _rank_results(params, histories_site, samples)},
            "forecast": {"quantiles_by_lead_feature": np.quantile(predictions, [tail, .5, 1 - tail], axis=0),
                         "rank_correlation_by_lead_feature": _rank_results(params, predictions, samples)},
        }
        if perturb:
            summary[name]["local_derivatives_at_range_midpoints"] = _local_derivatives(
                site, issue_at, forecaster, steps, params, middle
            )
    cube = np.stack(forecasts, axis=1)
    report = {
        "issue_at": issue_at.isoformat(), "features": list(FEATURES), "sites": summary,
        "checkpoint_sha256": forecaster.checkpoint_sha256, "draw_count": draws, "seed": seed,
        "central_mass": mass, "latent_correlation_assumption": latent_correlation,
        "parameter_scenarios": [{**asdict(param), "unit": PARAMETER_UNITS.get(param.path, "fraction")}
                                for param in params],
        "parameter_draws": samples, "draw_ids": [f"physical-{i:03d}" for i in range(draws)],
        "interpretation": ("Scenario-derived variation conditional on a fixed checkpoint. "
                           "Forecast sensitivity may reflect learned cross-feature coupling. "
                           "These envelopes are not empirically calibrated prediction intervals."),
        "axis_order": ["physical_draw", "site", "lead", "feature"],
    }
    return cube, np.stack(histories, axis=1), report
