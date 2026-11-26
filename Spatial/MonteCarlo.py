"""Random-scan Metropolis for the classical density proportional to exp(-E/T)."""
from dataclasses import dataclass
import math

import numpy as np

from Spatial.EnergyFunctional import EnergyFunctional
from Spatial.Hamiltonian import hessian_columns
from Spatial.NoiseGenerator import chains


@dataclass(frozen=True)
class SamplingConfig:
    draws: int = 100
    burn_in: int = 100
    thin: int = 2
    chain_count: int = 4
    temperature: float = 1.0
    proposal_scale: float = 1.0
    seed: int = 42
    proposal: str = "normal"
    adapt: bool = False
    target_acceptance: float = 0.44

    def __post_init__(self):
        counts = (self.draws, self.burn_in, self.thin, self.chain_count)
        if (not all(isinstance(v, int) for v in counts) or self.draws < 2 or self.burn_in < 0
                or min(self.thin, self.chain_count) < 1
                or not all(math.isfinite(v) and v > 0
                           for v in (self.temperature, self.proposal_scale))):
            raise ValueError("Invalid Metropolis settings")
        if self.proposal not in ("normal", "uniform") or not 0 < self.target_acceptance < 1:
            raise ValueError("Invalid proposal or target acceptance")
        if self.adapt and self.burn_in < 1:
            raise ValueError("Adaptation requires burn-in sweeps")


@dataclass
class Samples:
    fields: np.ndarray
    energy_trace: np.ndarray
    acceptance: np.ndarray
    burn_in_acceptance: np.ndarray
    proposal_scales: np.ndarray | None = None


def metropolis(energy: EnergyFunctional, initial, settings=SamplingConfig()):
    """Return [chain, draw, north, east] samples with fixed cells exactly enforced."""
    backend = energy.backend
    fixed = backend.to_host(energy.fixed).ravel()
    free = np.flatnonzero(~fixed)
    fidelity = backend.to_host(energy.fidelity).ravel()
    energy.require_anchored()
    shape = energy.grid.shape
    columns = hessian_columns(energy)
    scales = np.zeros(len(fixed))
    for index in free:
        diagonal = columns[index][2]
        scales[index] = settings.proposal_scale * math.sqrt(settings.temperature / diagonal)
    projected = energy.project(backend.array(initial))
    initial_host = backend.to_host(projected).astype(float)
    if initial_host.shape != shape or not np.isfinite(initial_host).all():
        raise ValueError("Expected a finite initial field")
    fields = np.empty((settings.chain_count, settings.draws, *shape))
    energies = np.empty((settings.chain_count, settings.draws))
    rates = np.zeros((settings.chain_count, 2))
    sweeps = settings.burn_in + settings.draws * settings.thin
    # Compute the initial gradient in float64 consistently for all backends.
    anchor = backend.to_host(energy.targets).ravel()
    adapted = np.full(settings.chain_count, settings.proposal_scale)
    for chain, rng in enumerate(chains(settings.seed, settings.chain_count)):
        field = initial_host.ravel().copy()
        gradient = -fidelity * anchor
        for i, (nodes, weights, _) in enumerate(columns):
            gradient[nodes] += field[i] * weights
        objective = 0.5 * (field @ gradient - (fidelity * anchor) @ field +
                           (fidelity * anchor) @ anchor)
        for sweep in range(sweeps):
            phase = int(sweep >= settings.burn_in)
            indices = rng.choice(free, size=len(free), replace=True) if len(free) else free
            noise = (rng.normal(size=len(free)) if settings.proposal == "normal"
                     else rng.uniform(-math.sqrt(3), math.sqrt(3), len(free)))
            increments = noise * scales[indices] * (adapted[chain] / settings.proposal_scale)
            before = rates[chain, phase]
            cutoffs = -settings.temperature * rng.exponential(size=len(free))
            for index, increment, cutoff in zip(indices, increments, cutoffs):
                nodes, weights, diagonal = columns[index]
                change = increment * (gradient[index] + 0.5 * increment * diagonal)
                if -change >= cutoff:
                    field[index] += increment
                    gradient[nodes] += increment * weights
                    objective += change
                    rates[chain, phase] += 1
            if not phase and settings.adapt and len(free):
                observed = (rates[chain, phase] - before) / len(free)
                log_scale = math.log(adapted[chain]) + (observed - settings.target_acceptance) / math.sqrt(sweep + 1)
                adapted[chain] = math.exp(float(np.clip(log_scale, -6, 6)))
            if phase and (sweep - settings.burn_in + 1) % settings.thin == 0:
                draw = (sweep - settings.burn_in) // settings.thin
                fields[chain, draw] = field.reshape(shape)
                energies[chain, draw] = objective
    if not np.isfinite(fields).all() or not np.isfinite(energies).all():
        raise FloatingPointError("Metropolis sampling produced nonfinite values")
    proposals = len(free) * np.array([settings.burn_in, settings.draws * settings.thin])
    np.divide(rates, proposals, out=rates, where=proposals > 0)
    return Samples(fields, energies, rates[:, 1], rates[:, 0], adapted)
