"""Shared fixed-length encoding for equal-budget architecture searches."""
from dataclasses import dataclass

import numpy as np

from Temporal.SequenceForecast import DENSE_FAMILIES, FAMILIES, SequenceSpec

BIT_COUNT = 10
EXTENDED_BIT_COUNT = 17
SEQUENCE_BIT_COUNT = 20
MAX_SEQUENCE_CONTEXT = 24
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
        if len(self.bits) not in (BIT_COUNT, EXTENDED_BIT_COUNT, SEQUENCE_BIT_COUNT) or any(bit not in (0, 1) for bit in self.bits):
            raise ValueError("Invalid genome")

    @classmethod
    def random(cls, rng, *, extended=False, sequence=False):
        count = SEQUENCE_BIT_COUNT if sequence else EXTENDED_BIT_COUNT if extended else BIT_COUNT
        return cls(tuple(map(int, rng.integers(0, 2, count))))

    def architecture(self):
        """Return widths and residual blocks for callers using the original API."""
        decoded = self.decode()
        if isinstance(decoded, SequenceSpec):
            raise ValueError("Use decode() for a sequence architecture")
        return decoded.hidden, decoded.skips

    def decode(self):
        if len(self.bits) == SEQUENCE_BIT_COUNT:
            def code(start, stop):
                return sum(bit << exponent for exponent, bit in enumerate(reversed(self.bits[start:stop])))

            family = FAMILIES[code(0, 4) % len(FAMILIES)]
            return SequenceSpec(
                family=family, width=(8, 16, 24, 32)[code(4, 6)], depth=code(6, 8) + 1,
                activation=ACTIVATIONS[code(8, 10)], residual=bool(self.bits[10]) or family == "residual_dense",
                context=3 if family in DENSE_FAMILIES else (3, 6, 12, MAX_SEQUENCE_CONTEXT)[code(11, 13)],
                kernel=(2, 3, 4, 5)[code(13, 15)], dilation=(1, 2, 3, 4)[code(15, 17)],
                heads=(1, 2, 4, 8)[code(17, 19)], dropout=self.bits[19] * .1,
            )
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

    def repair(self, max_parameters, horizon=1):
        """Shrink a genome without random draws when it exceeds a parameter cap."""
        if max_parameters is None:
            return self
        if len(self.bits) == SEQUENCE_BIT_COUNT:
            minimal = Genome((0,) * SEQUENCE_BIT_COUNT)
            if type(max_parameters) is not int or max_parameters < minimal.decode().parameter_count(horizon):
                raise ValueError("Parameter cap cannot accommodate a minimum direct dense model")
            genome = self
            while (size := genome.decode().parameter_count(horizon)) > max_parameters:
                choices = []
                for start, end in ((4, 6), (6, 8), (13, 15)):
                    for lower in range(4):
                        change = divmod(lower, 2)
                        candidate = Genome((*genome.bits[:start], *change, *genome.bits[end:]))
                        count = candidate.decode().parameter_count(horizon)
                        if count < size:
                            choices.append((count, candidate))
                if not choices:
                    return minimal
                feasible = [item for item in choices if item[0] <= max_parameters]
                genome = (max(feasible, key=lambda item: item[0]) if feasible
                          else min(choices, key=lambda item: item[0]))[1]
            return genome
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
