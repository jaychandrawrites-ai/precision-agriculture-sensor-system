"""Integrated leaf flux accounting and composable physical balance checks."""
from dataclasses import asdict, dataclass
import math

import numpy as np

from Numerical.ODESolver import integrate
from Physics.Leaf.EnergyBalance import exchange
from Physics.Leaf.LeafWetness import LeafState


@dataclass(frozen=True)
class LeafBudget:
    elapsed_seconds: float
    storage_delta_kg_m2: float
    intercepted_rain_kg_m2: float
    condensation_kg_m2: float
    evaporation_kg_m2: float
    rejected_water_kg_m2: float
    heat_input_j_m2: float
    heat_storage_delta_j_m2: float
    water_residual_kg_m2: float
    energy_residual_j_m2: float


def audited_leaf_step(initial, weather, params, seconds=3600, *, max_step=60):
    """Integrate fluxes alongside the original leaf state in controlled experiments.

    Positive rejected water represents runoff at saturation plus water removed
    by bound projection. A negative value would indicate a lower-bound
    correction, which the caller can inspect as a numerical warning.
    """
    if not (isinstance(initial, LeafState) and 0 <= initial.wetness <= 1
            and math.isfinite(seconds) and seconds > 0 and math.isfinite(max_step) and max_step > 0):
        raise ValueError("Use a physical leaf state and finite positive durations")
    # State components after temperature and wetness are cumulative physical fluxes.
    def rhs(_, y):
        temperature, wetness = y[:2]
        heat, net_water = exchange(float(temperature), float(wetness), weather, params)
        rain = weather.intercepted_rain_kg_m2_s
        atmospheric = net_water - rain
        limited = (wetness <= 0 and net_water < 0) or (wetness >= 1 and net_water > 0)
        reject = net_water if limited else 0.0
        return np.array([heat / params.heat_capacity_j_m2_k,
                         (net_water - reject) / params.storage_kg_m2,
                         rain, max(atmospheric, 0.0), max(-atmospheric, 0.0), reject, heat])

    def project(y):
        bounded = np.clip(y[1], 0, 1)
        y[5] += (y[1] - bounded) * params.storage_kg_m2
        y[1] = bounded
        return y

    start = [initial.temperature_c, initial.wetness, 0, 0, 0, 0, 0]
    result = integrate(rhs, start, 0, seconds, step=min(seconds, max_step, 10.0),
                       project=project, max_step=max_step)
    temperature, wetness, rain, condensation, evaporation, rejected, heat = result.states[-1]
    storage_delta = (wetness - initial.wetness) * params.storage_kg_m2
    energy_delta = (temperature - initial.temperature_c) * params.heat_capacity_j_m2_k
    final = LeafState(float(temperature), float(wetness))
    budget = LeafBudget(seconds, storage_delta, rain, condensation, evaporation, rejected, heat,
                        energy_delta, storage_delta - (rain + condensation - evaporation - rejected),
                        energy_delta - heat)
    return final, budget


def audit_leaf_regime(initial, weather_sequence, params, seconds=3600):
    """Report cumulative balances and a coarser/finer numerical endpoint comparison."""
    state, refined = initial, initial
    budgets = []
    for weather in weather_sequence:
        state, budget = audited_leaf_step(state, weather, params, seconds)
        refined, _ = audited_leaf_step(refined, weather, params, seconds, max_step=30)
        budgets.append(budget)
    if not budgets:
        raise ValueError("A regime requires at least one forcing interval")
    return {
        "final_state": asdict(state),
        "fine_endpoint_minus_main": {"temperature_c": refined.temperature_c - state.temperature_c,
                                      "wetness": refined.wetness - state.wetness},
        "balance_totals": {key: sum(getattr(item, key) for item in budgets)
                           for key in asdict(budgets[0])},
        "absolute_step_residual_max": {
            "water_kg_m2": max(abs(item.water_residual_kg_m2) for item in budgets),
            "energy_j_m2": max(abs(item.energy_residual_j_m2) for item in budgets),
        },
    }


def audit_soil_regime(bucket, samples):
    """Advance one declared forcing sequence and verify root-zone water accounting."""
    steps = [bucket.step(**inputs) for inputs in samples]
    if not steps:
        raise ValueError("A regime requires at least one forcing interval")
    sums = {name: sum(getattr(step, name) for step in steps)
            for name in ("actual_et_mm", "effective_input_mm", "drainage_mm", "residual_mm")}
    return {
        "hours": len(steps), "initial_depletion_mm": steps[0].initial_depletion_mm,
        "final_depletion_mm": steps[-1].final_depletion_mm, "capacity_mm": bucket.capacity,
        "totals": sums, "absolute_step_residual_max_mm": max(abs(step.residual_mm) for step in steps),
        "depletion_within_bounds": all(0 <= step.final_depletion_mm <= bucket.capacity for step in steps),
    }


def audit_crop_regime(initial, canopy_temperatures, crop, *, hours=1):
    """Accumulate heat forcing and exponential recovery in index units."""
    from Physics.Crop.CottonHeatStress import advance_heat_stress

    canopy_temperatures = list(canopy_temperatures)
    if (not math.isfinite(initial) or initial < 0 or not math.isfinite(hours) or hours <= 0
            or not canopy_temperatures or not all(math.isfinite(value) for value in canopy_temperatures)):
        raise ValueError("Use a nonnegative starting index and finite temperatures and durations")
    index, forcing_total, recovery_total, max_residual = initial, 0.0, 0.0, 0.0
    for temperature in canopy_temperatures:
        forcing = crop.heat_accumulation_per_hour * max(0, temperature - crop.critical_temperature_c)
        change = advance_heat_stress(index, temperature, hours, crop) - index
        # Integrate recovery independently using the analytical state trajectory.
        if crop.recovery_per_hour:
            loss = 0.0
            elapsed = crop.recovery_per_hour * hours
            for transition in (.06943184420297371, .33000947820757187,
                               .6699905217924281, .9305681557970262):
                step = transition * hours
                weight = .1739274225687269 if transition < .1 or transition > .9 else .3260725774312731
                loss += weight * advance_heat_stress(index, temperature, step, crop)
            loss *= elapsed
        else:
            loss = 0.0
        forcing_total += forcing * hours
        recovery_total += loss
        max_residual = max(max_residual, abs(change - (forcing * hours - loss)))
        index += change
    return {"initial_index": initial, "final_index": index, "integrated_forcing_index": forcing_total,
            "integrated_recovery_index": recovery_total, "absolute_step_residual_max_index": max_residual,
            "residual_index": index - initial - (forcing_total - recovery_total)}
