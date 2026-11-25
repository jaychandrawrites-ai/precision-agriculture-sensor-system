"""Shared fixed-length encoding for equal-budget architecture searches."""
from dataclasses import dataclass

import numpy as np

BIT_COUNT = 10
WIDTHS = (16, 32, 64, 128)


@dataclass(frozen=True)
class Genome:
    bits: tuple[int, ...]

    def __post_init__(self):
        if len(self.bits) != BIT_COUNT or any(bit not in (0, 1) for bit in self.bits):
            raise ValueError("Invalid genome")

    @classmethod
    def random(cls, rng):
        return cls(tuple(map(int, rng.integers(0, 2, BIT_COUNT))))

    def architecture(self):
        depth = 1 + min(2, 2 * self.bits[0] + self.bits[1])
        widths = tuple(WIDTHS[2 * self.bits[i] + self.bits[i + 1]]
                       for i in range(2, 2 + 2 * depth, 2))
        skips = tuple(i for i in range(depth) if self.bits[8 + i % 2])
        return widths, skips

    def offspring(self, other, rng, mutation_probability):
        point = rng.integers(1, BIT_COUNT)
        bits = np.array(self.bits[:point] + other.bits[point:])
        bits ^= rng.random(BIT_COUNT) < mutation_probability
        return Genome(tuple(map(int, bits)))
