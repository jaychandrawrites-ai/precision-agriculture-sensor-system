"""Shared fixed-length encoding for equal-budget architecture searches."""
from dataclasses import dataclass

import numpy as np

BIT_COUNT = 10
EXTENDED_BIT_COUNT = 17
WIDTHS = (16, 32, 64, 128)
ACTIVATIONS = ("relu", "tanh", "silu", "gelu")


@dataclass(frozen=True)
class Architecture:
    hidden: tuple[int, ...]
    skips: tuple[int, ...]
    activations: tuple[str, ...]

    @property
    def parameter_count(self):
        count, previous = 0, 18
        for index, width in enumerate(self.hidden):
            count += (previous + 1) * width
            if index in self.skips and previous != width:
                count += (previous + 1) * width
            previous = width
        return count + (previous + 1) * 6

    def model_arguments(self):
        return {"hidden": self.hidden, "skips": self.skips, "activations": self.activations}


@dataclass(frozen=True)
class Genome:
    bits: tuple[int, ...]

    def __post_init__(self):
        if len(self.bits) not in (BIT_COUNT, EXTENDED_BIT_COUNT) or any(bit not in (0, 1) for bit in self.bits):
            raise ValueError("Invalid genome")

    @classmethod
    def random(cls, rng, *, extended=False):
        count = EXTENDED_BIT_COUNT if extended else BIT_COUNT
        return cls(tuple(map(int, rng.integers(0, 2, count))))

    def architecture(self):
        """Return widths and residual blocks for callers using the original API."""
        decoded = self.decode()
        return decoded.hidden, decoded.skips

    def decode(self):
        depth = 1 + min(2, 2 * self.bits[0] + self.bits[1])
        widths = tuple(WIDTHS[2 * self.bits[i] + self.bits[i + 1]]
                       for i in range(2, 2 + 2 * depth, 2))
        if len(self.bits) == BIT_COUNT:
            skips = tuple(i for i in range(depth) if self.bits[8 + i % 2])
            activations = ("relu",) * depth
        else:
            skips = tuple(i for i in range(depth) if self.bits[8 + i])
            activations = tuple(ACTIVATIONS[2 * self.bits[i] + self.bits[i + 1]]
                                for i in range(11, 11 + 2 * depth, 2))
        return Architecture(widths, skips, activations)

    def repair(self, max_parameters):
        """Shrink a genome without random draws when it exceeds a parameter cap."""
        if max_parameters is None:
            return self
        minimum = Architecture((min(WIDTHS),), (), ("relu",)).parameter_count
        if type(max_parameters) is not int or max_parameters < minimum:
            raise ValueError(f"Parameter cap must be at least {minimum}")
        genome = self
        while (size := genome.decode().parameter_count) > max_parameters:
            depth = len(genome.decode().hidden)
            choices = []
            for i in range(depth):
                offset = 2 + 2 * i
                code = 2 * genome.bits[offset] + genome.bits[offset + 1]
                for smaller in range(code):
                    bits = list(genome.bits)
                    bits[offset:offset + 2] = divmod(smaller, 2)
                    choices.append(Genome(tuple(bits)))
            for i in range(8, 10 if len(genome.bits) == BIT_COUNT else 11):
                if genome.bits[i]:
                    bits = list(genome.bits)
                    bits[i] = 0
                    choices.append(Genome(tuple(bits)))
            if depth > 1:
                choices.append(Genome((*divmod(depth - 2, 2), *genome.bits[2:])))
            cheaper = [(item.decode().parameter_count, item) for item in choices
                       if item.decode().parameter_count < size]
            feasible = [item for item in cheaper if item[0] <= max_parameters]
            genome = (max(feasible, key=lambda item: item[0]) if feasible
                      else min(cheaper, key=lambda item: item[0]))[1]
        return genome

    def offspring(self, other, rng, mutation_probability):
        if len(self.bits) != len(other.bits) or not 0 <= mutation_probability <= 1:
            raise ValueError("Incompatible genome lengths or mutation probability")
        point = rng.integers(1, len(self.bits))
        bits = np.array(self.bits[:point] + other.bits[point:])
        bits ^= rng.random(len(bits)) < mutation_probability
        return Genome(tuple(map(int, bits)))
