"""Classical real amplitude pairs and fitness-guided rotations."""
import math

import numpy as np

from Evolution.Genome import BIT_COUNT, Genome


class QuantumGenome:
    def __init__(self):
        self.angles = np.full(BIT_COUNT, math.pi / 4)

    @property
    def amplitudes(self):
        return np.column_stack((np.cos(self.angles), np.sin(self.angles)))

    def measure(self, rng):
        bits = rng.random(BIT_COUNT) < self.amplitudes[:, 1] ** 2
        return Genome(tuple(map(int, bits)))

    def rotate(self, observed, guide, radians):
        if not 0 <= radians <= math.pi / 2:
            raise ValueError("Invalid rotation")
        direction = np.array(guide.bits) - np.array(observed.bits)
        self.angles = np.clip(self.angles + radians * direction, 0.01, math.pi / 2 - 0.01)
