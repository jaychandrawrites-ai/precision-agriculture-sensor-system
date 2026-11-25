"""Independent reproducible pseudorandom streams for spatial sampling."""
import numpy as np


def chains(seed, count):
    if count < 1:
        raise ValueError("At least one chain is required")
    return [np.random.default_rng(child) for child in np.random.SeedSequence(seed).spawn(count)]
