"""Classical real amplitude pairs and fitness-guided rotations."""
import math

import numpy as np

from Evolution.Genome import BIT_COUNT, EXTENDED_BIT_COUNT, SEQUENCE_BIT_COUNT, Genome


class QuantumGenome:
    def __init__(self, *, extended=False, sequence=False, angles=None):
        count = SEQUENCE_BIT_COUNT if sequence else EXTENDED_BIT_COUNT if extended else BIT_COUNT
        self.angles = (np.full(count, math.pi / 4) if angles is None
                       else np.asarray(angles, dtype=float).copy())
        if (self.angles.shape != (count,) or not np.isfinite(self.angles).all()
                or np.any(self.angles < 0) or np.any(self.angles > math.pi / 2)):
            raise ValueError("Amplitudes require one finite angle in [0, π/2] per bit")

    @property
    def amplitudes(self):
        return np.column_stack((np.cos(self.angles), np.sin(self.angles)))

    def measure(self, rng):
        bits = rng.random(len(self.angles)) < self.amplitudes[:, 1] ** 2
        return Genome(tuple(map(int, bits)))

    def rotate(self, observed, guide, radians):
        if (not 0 <= radians <= math.pi / 2
                or len(observed.bits) != len(self.angles) or len(guide.bits) != len(self.angles)):
            raise ValueError("Invalid rotation or incompatible genome")
        direction = np.array(guide.bits) - np.array(observed.bits)
        self.angles = np.clip(self.angles + radians * direction, 0.01, math.pi / 2 - 0.01)

    def relax(self, fraction):
        if not 0 <= fraction <= 1:
            raise ValueError("Invalid relaxation fraction")
        self.angles += fraction * (math.pi / 4 - self.angles)
