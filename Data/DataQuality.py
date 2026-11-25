"""Row quality flags for the physical feature contract."""
import numpy as np

from Physics.Constraints import violations


def invalid_rows(values):
    return np.logical_or.reduce(tuple(violations(values).values()))
