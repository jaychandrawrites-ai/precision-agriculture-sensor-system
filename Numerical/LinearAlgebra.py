"""Real matrix tooling for NumPy and CuPy, with explicit precision handling."""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Protocol

import numpy as np


class LinearOperator(Protocol):
    """Square matrix-free maps and sparse operators are interchangeable in Krylov solves."""

    def __call__(self, vector):
        ...


def _matrix(value, backend, *, square=False):
    array = backend.array(value, dtype="float64" if backend.dtype.itemsize == 8 else "float32")
    if array.ndim < 2 or min(array.shape[-2:]) < 1:
        raise ValueError("Expected a nonempty array of matrices")
    if square and array.shape[-2] != array.shape[-1]:
        raise ValueError("Expected square matrices")
    return backend.require_finite(array, "matrix")


def cholesky(value, backend, *, diagonal_shift=0.0):
    """Return lower factors of symmetric positive definite matrices.

    Optional diagonal_shift is absolute in the input matrix's units.
    """
    if not math.isfinite(diagonal_shift) or diagonal_shift < 0:
        raise ValueError("Diagonal shift must be finite and nonnegative")
    with backend.scope():
        xp = backend.xp
        matrix = _matrix(value, backend, square=True)
        magnitude = xp.max(xp.abs(matrix), axis=(-2, -1), keepdims=True)
        tolerance = 16 * xp.finfo(matrix.dtype).eps * magnitude
        if backend.scalar((xp.abs(matrix - matrix.swapaxes(-2, -1)) > tolerance).any()):
            raise ValueError("Cholesky requires symmetric matrices")
        adjusted = matrix + diagonal_shift * xp.eye(matrix.shape[-1], dtype=matrix.dtype)
        factor = xp.linalg.cholesky(adjusted)
        backend.require_finite(factor, "Cholesky factor")
        return factor


def solve_cholesky(factor, rhs, backend):
    """Solve one or more right-hand sides from lower Cholesky factors."""
    with backend.scope():
        xp = backend.xp
        lower = _matrix(factor, backend, square=True)
        b = backend.array(rhs, dtype=lower.dtype)
        backend.require_finite(b, "right hand side")
        vector = b.ndim == lower.ndim - 1
        rhs_matrix = b[..., None] if vector else b
        if (rhs_matrix.ndim != lower.ndim or rhs_matrix.shape[:-2] != lower.shape[:-2]
                or rhs_matrix.shape[-2] != lower.shape[-1] or not rhs_matrix.shape[-1]):
            raise ValueError("Right-hand side must match factor batch and row dimensions")
        solution = xp.linalg.solve(lower.swapaxes(-2, -1), xp.linalg.solve(lower, rhs_matrix))
        backend.require_finite(solution, "solution")
        return solution[..., 0] if vector else solution


def qr(value, backend):
    """Reduced QR on the final two dimensions, preserving the selected device."""
    with backend.scope():
        matrix = _matrix(value, backend)
        return backend.xp.linalg.qr(matrix, mode="reduced")


def svd(value, backend, *, vectors=True):
    """Reduced SVD, optionally returning singular values only."""
    with backend.scope():
        matrix = _matrix(value, backend)
        return backend.xp.linalg.svd(matrix, full_matrices=False, compute_uv=vectors)


@dataclass(frozen=True)
class Conditioning:
    singular_values: object
    rank: object
    condition: object
    rank_threshold: object


def conditioning(value, backend, *, relative_tolerance=None):
    """Compute effective rank and 2-norm condition number per matrix."""
    if relative_tolerance is not None and (
        not math.isfinite(relative_tolerance) or relative_tolerance < 0
    ):
        raise ValueError("Relative tolerance must be finite and nonnegative")
    with backend.scope():
        xp = backend.xp
        matrix = _matrix(value, backend)
        singular = xp.linalg.svd(matrix, full_matrices=False, compute_uv=False)
        largest, smallest = singular[..., 0], singular[..., -1]
        if relative_tolerance is None:
            relative_tolerance = max(matrix.shape[-2:]) * xp.finfo(matrix.dtype).eps
        tolerance = largest * relative_tolerance
        rank = (singular > tolerance[..., None]).sum(axis=-1)
        usable = smallest > tolerance
        ratio = largest / xp.where(usable, smallest, 1)
        return Conditioning(singular, rank, xp.where(usable, ratio, xp.inf), tolerance)


class SparseOperator:
    """Coordinate sparse matrix with coalesced entries and transpose action.

    Constructing the sparsity structure is a host operation. Multiplication
    uses only backend arrays and supports one or many right-hand sides.
    """

    def __init__(self, rows, columns, values, shape, backend):
        if len(shape) != 2 or any(not isinstance(n, int) or n < 1 for n in shape):
            raise ValueError("Sparse matrix shape must contain two positive dimensions")
        row = backend.to_host(rows)
        col = backend.to_host(columns)
        data = backend.to_host(values)
        if (row.ndim != 1 or row.shape != col.shape or row.shape != data.shape
                or (row.size and (row.dtype.kind not in "iu" or col.dtype.kind not in "iu"))
                or data.dtype.kind not in "biuf" or not np.isfinite(data).all()):
            raise ValueError("Invalid sparse coordinates or values")
        if ((row < 0).any() or (row >= shape[0]).any() or (col < 0).any() or (col >= shape[1]).any()):
            raise ValueError("Sparse coordinate out of range")
        if row.size:
            order = np.lexsort((col, row))
            row, col, data = row[order], col[order], data[order]
            split = np.r_[0, np.flatnonzero((row[1:] != row[:-1]) | (col[1:] != col[:-1])) + 1]
            row, col = row[split], col[split]
            data = np.add.reduceat(data.astype("float64"), split)
            nonzero = data != 0
            row, col, data = row[nonzero], col[nonzero], data[nonzero]
        self.backend = backend
        self.shape = tuple(shape)
        self.row = backend.array(row, dtype="int64")
        self.column = backend.array(col, dtype="int64")
        self.data = backend.array(data, dtype="float64" if backend.dtype.itemsize == 8 else "float32")
        backend.require_finite(self.data, "sparse coefficients")

    @property
    def nnz(self):
        return int(self.data.size)

    def __call__(self, value):
        return self.apply(value)

    def apply(self, value, *, transpose=False):
        with self.backend.scope():
            xp = self.backend.xp
            array = self.backend.array(value, dtype=self.data.dtype)
            rows, columns = self.shape[::-1] if transpose else self.shape
            target, source = (self.column, self.row) if transpose else (self.row, self.column)
            if array.ndim == 0 or array.shape[0] != columns:
                raise ValueError("Sparse product expects the matrix dimension on the first axis")
            result = xp.zeros((rows, *array.shape[1:]), dtype=array.dtype)
            products = self.data.reshape((-1,) + (1,) * (array.ndim - 1)) * array[source]
            xp.add.at(result, target, products)
            return self.backend.validate(result, "sparse product")

    def diagonal(self):
        with self.backend.scope():
            result = self.backend.zeros((min(self.shape),), dtype=self.data.dtype)
            same = self.row == self.column
            result[self.row[same]] = self.data[same]
            return result


@dataclass(frozen=True)
class CGResult:
    solution: object
    iterations: int
    converged: bool
    reason: str
    residual_norms: tuple[float, ...]
    target_norm: float


def conjugate_gradient(operator: LinearOperator, rhs, backend, *, initial=None,
                       preconditioner=None, rtol=1e-6, atol=0.0, iterations=None, refresh=25):
    """Preconditioned conjugate gradient for symmetric positive definite maps.

    The optional preconditioner applies the inverse of a symmetric positive
    definite approximation. Reported convergence confirms the true residual.
    """
    if (not math.isfinite(rtol) or not math.isfinite(atol) or min(rtol, atol) < 0
            or max(rtol, atol) == 0 or not isinstance(refresh, int) or refresh < 1):
        raise ValueError("Invalid residual tolerances or refresh interval")
    with backend.scope():
        xp = backend.xp
        dtype = "float64" if backend.dtype.itemsize == 8 else "float32"
        b = backend.array(rhs, dtype=dtype)
        if b.ndim != 1 or not b.size:
            raise ValueError("Conjugate gradient requires a nonempty vector")
        if iterations is None:
            iterations = 10 * len(b)
        if not isinstance(iterations, int) or iterations < 1:
            raise ValueError("Iteration limit must be positive")
        x = xp.zeros_like(b) if initial is None else backend.array(initial, dtype=dtype, copy=True)
        if x.shape != b.shape:
            raise ValueError("Initial state and right-hand side must match")
        backend.require_finite(x, "initial state")
        backend.require_finite(b, "right hand side")

        def action(fn, value):
            result = backend.array(fn(value), dtype=dtype)
            if result.shape != b.shape:
                raise ValueError("Linear operator returned a mismatched vector")
            return backend.require_finite(result, "linear operator output")

        def magnitude(value):
            return backend.global_norm((value,))

        def dot(a, b):
            return float(backend.scalar(xp.dot(a.astype("float64"), b.astype("float64"))))

        threshold = max(atol, rtol * magnitude(b))
        residual = b - action(operator, x)
        history = [magnitude(residual)]
        if history[-1] <= threshold:
            return CGResult(x, 0, True, "residual_tolerance", tuple(history), threshold)
        z = residual if preconditioner is None else action(preconditioner, residual)
        rho = dot(residual, z)
        direction = z.copy()
        for step in range(1, iterations + 1):
            if not math.isfinite(rho) or rho <= 0:
                return CGResult(x, step - 1, False, "nonpositive_preconditioner", tuple(history), threshold)
            product = action(operator, direction)
            curvature = dot(direction, product)
            if not math.isfinite(curvature) or curvature <= 0:
                return CGResult(x, step - 1, False, "nonpositive_curvature", tuple(history), threshold)
            factor = rho / curvature
            x += factor * direction
            residual = residual - factor * product
            norm = magnitude(residual)
            restarted = norm <= threshold or step % refresh == 0 or step == iterations
            if restarted:
                residual = b - action(operator, x)
                norm = magnitude(residual)
            history.append(norm)
            if norm <= threshold:
                return CGResult(x, step, True, "residual_tolerance", tuple(history), threshold)
            z = residual if preconditioner is None else action(preconditioner, residual)
            next_rho = dot(residual, z)
            direction = z.copy() if restarted else z + (next_rho / rho) * direction
            rho = next_rho
        return CGResult(x, iterations, False, "iteration_limit", tuple(history), threshold)
