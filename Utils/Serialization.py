"""Atomically publish portable JSON and numeric-only NumPy artifacts."""
from contextlib import contextmanager
import json
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np


@contextmanager
def atomic_output(destination, *, binary=False):
    destination = Path(destination)
    mode = "wb" if binary else "w"
    encoding = None if binary else "utf-8"
    with NamedTemporaryFile(mode=mode, encoding=encoding, newline=None if binary else "",
                            dir=destination.parent,
                            prefix=f".{destination.name}.", delete=False) as file:
        temporary = Path(file.name)
        try:
            yield file
            file.close()
            temporary.replace(destination)
        finally:
            file.close()
            temporary.unlink(missing_ok=True)


def _json_default(value):
    if isinstance(value, (np.ndarray, np.generic)):
        return value.tolist()
    raise TypeError(f"Unsupported JSON value of type {type(value).__name__}")


def write_json(path, record, *, indent=2):
    with atomic_output(path) as file:
        json.dump(record, file, default=_json_default, allow_nan=False, indent=indent)
        file.write("\n")


def _unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_constant(token):
    raise ValueError(f"Unsupported JSON number {token}")


def decode_json(payload):
    return json.loads(payload, object_pairs_hook=_unique_pairs, parse_constant=_reject_constant)


def read_json(path):
    return decode_json(Path(path).read_bytes())


def write_arrays(path, **arrays):
    numeric = {name: np.asarray(array) for name, array in arrays.items()}
    if not all(array.dtype.kind in "biuf" for array in numeric.values()):
        raise ValueError("Expected real numeric or boolean arrays")
    with atomic_output(path, binary=True) as file:
        np.savez_compressed(file, **numeric)


def read_arrays(path, expected):
    with np.load(path, allow_pickle=False) as file:
        if set(file.files) != set(expected):
            raise ValueError("Unexpected array artifact keys")
        result = {name: file[name] for name in expected}
        if any(array.dtype.kind not in "biuf" for array in result.values()):
            raise ValueError("Expected real numeric or boolean arrays")
        return result
