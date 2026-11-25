"""Tiled Metropolis with Hessian-preconditioned symmetric block proposals."""
from dataclasses import dataclass

import numpy as np

from Spatial.GaussianField import field_system
from Spatial.MonteCarlo import Samples
from Spatial.NoiseGenerator import chains


@dataclass(frozen=True)
class BlockSampling:
    tile: int = 3
    draws: int = 500
    burn_in: int = 150
    thin: int = 2
    chain_count: int = 4
    scale: float = 1.5
    temperature: float = 1.0
    target_acceptance: float = 0.35
    adapt: bool = True
    seed: int = 42

    def __post_init__(self):
        limits = (self.tile, self.draws, self.burn_in, self.thin, self.chain_count)
        if (any(type(number) is not int for number in limits)
                or not 1 <= self.tile <= 8 or self.draws < 2 or self.burn_in < 0
                or min(self.thin, self.chain_count) < 1
                or not np.isfinite([self.scale, self.temperature, self.target_acceptance]).all()
                or min(self.scale, self.temperature) <= 0 or not 0 < self.target_acceptance < 1):
            raise ValueError("Invalid block sampling settings")


@dataclass(frozen=True)
class _Tile:
    cells: np.ndarray
    neighborhood: np.ndarray
    stencil: np.ndarray
    precision: np.ndarray
    move: np.ndarray


def _tiles(shape, tile, fixed, columns, temperature):
    layout = np.arange(len(fixed)).reshape(shape)
    for north in range(0, shape[0], tile):
        for east in range(0, shape[1], tile):
            block = layout[north:north+tile, east:east+tile].ravel()
            cells = block[~fixed[block]]
            if not len(cells):
                continue
            neighborhood = np.unique(np.concatenate([columns[index][0] for index in cells]))
            stencil = np.zeros((len(neighborhood), len(cells)))
            for col, node in enumerate(cells):
                indices, weights, _ = columns[node]
                stencil[np.searchsorted(neighborhood, indices), col] = weights
            precision = stencil[np.searchsorted(neighborhood, cells)]
            chol = np.linalg.cholesky(precision)
            move = np.linalg.solve(chol.T, np.eye(len(cells)))
            move *= np.sqrt(temperature / len(cells))
            yield _Tile(cells, neighborhood, stencil, precision, move)


def block_metropolis(energy, initial, settings=BlockSampling()):
    """Run independent chains, adapting within burn-in then freezing scales."""
    columns, fixed, anchor, info = field_system(energy)
    free = ~fixed
    fidelity = energy.backend.to_host(energy.fidelity).ravel()
    smoothing = energy.weights.gradient > 0 or energy.weights.curvature > 0
    if free.any() and not ((smoothing and (fixed.any() or (fidelity > 0).any()))
                           or (fidelity[free] > 0).all()):
        raise ValueError("Field has an improper conditional distribution")
    shape = energy.grid.shape
    initial = np.asarray(energy.backend.to_host(initial), dtype=float)
    if initial.shape != shape or not np.isfinite(initial).all():
        raise ValueError("Expected a finite initial field")
    try:
        tiles = list(_tiles(shape, settings.tile, fixed, columns, settings.temperature))
    except np.linalg.LinAlgError as error:
        raise ValueError("Block covariance is not positive definite") from error

    fields = np.empty((settings.chain_count, settings.draws, *shape))
    energy_trace = np.empty((settings.chain_count, settings.draws))
    rates = np.zeros((settings.chain_count, 2))
    scales = np.empty(settings.chain_count)
    total_sweeps = settings.burn_in + settings.draws * settings.thin
    target = energy.backend.to_host(energy.targets).ravel()
    offset = 0.5 * (info @ target)
    for chain, rng in enumerate(chains(settings.seed, settings.chain_count)):
        field = initial.ravel().copy()
        field[fixed] = anchor[fixed]
        grad = -info.copy()
        for i, (indexes, weights, _) in enumerate(columns):
            grad[indexes] += weights * field[i]
        objective = 0.5 * (field @ grad - field @ info) + offset
        log_scale = np.log(settings.scale)
        for sweep in range(total_sweeps):
            accepted = 0
            for i in rng.permutation(len(tiles)):
                tile = tiles[i]
                step = (tile.move @ rng.normal(size=len(tile.cells))) * np.exp(log_scale)
                delta = step @ (grad[tile.cells] + 0.5 * (tile.precision @ step))
                if delta <= settings.temperature * rng.exponential():
                    field[tile.cells] += step
                    grad[tile.neighborhood] += tile.stencil @ step
                    objective += delta
                    accepted += 1
            phase = int(sweep >= settings.burn_in)
            rates[chain, phase] += accepted
            if phase == 0 and settings.adapt and tiles:
                error = accepted / len(tiles) - settings.target_acceptance
                log_scale = np.clip(log_scale + error / np.sqrt(1+sweep), -6, 6)
            step_from_burn = sweep - settings.burn_in + 1
            if phase and step_from_burn % settings.thin == 0:
                draw = step_from_burn // settings.thin - 1
                fields[chain, draw] = field.reshape(shape)
                energy_trace[chain, draw] = objective
        scales[chain] = np.exp(log_scale)
    proposals = len(tiles) * np.array([settings.burn_in, settings.draws * settings.thin])
    np.divide(rates, proposals, out=rates, where=proposals > 0)
    if not np.isfinite(fields).all() or not np.isfinite(energy_trace).all():
        raise FloatingPointError("Nonfinite samples from block Metropolis")
    return Samples(fields, energy_trace, rates[:, 1], rates[:, 0], scales)
