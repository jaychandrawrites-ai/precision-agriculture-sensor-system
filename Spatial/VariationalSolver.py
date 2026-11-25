"""Projected descent with a spectral stability bound and residual diagnostics."""
from dataclasses import dataclass
import math


@dataclass
class VariationalResult:
    field: object
    energies: list[float]
    iterations: int
    converged: bool
    gradient_rms: float
    step: float


def minimize(energy, initial, *, iterations=500, tolerance=1e-6, step=None):
    if iterations < 1 or not math.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("Invalid solver budget or tolerance")
    backend = energy.backend
    with backend.scope():
        field = energy.project(backend.array(initial, copy=True))
        backend.require_finite(field, "initial spatial field")
        free = ~energy.fixed
        count = int(backend.scalar(free.sum()))
        if not count:
            components, _ = energy.value_gradient(field)
            return VariationalResult(field, [sum(components)], 0, True, 0.0, 0.0)
        if energy.lipschitz <= 0:
            raise ValueError("Objective has no active terms")
        step = 0.9 / energy.lipschitz if step is None else step
        if not 0 < step < 2 / energy.lipschitz:
            raise ValueError("Explicit step exceeds stability bound")
        energies = []
        for index in range(iterations + 1):
            components, gradient = energy.value_gradient(field)
            energies.append(sum(components))
            gradient *= free
            rms = backend.global_norm([gradient]) / math.sqrt(count)
            if rms <= tolerance or index == iterations:
                return VariationalResult(field, energies, index, rms <= tolerance, rms, step)
            field = energy.project(field - step * gradient)


def minimize_conjugate(energy, initial, *, iterations=500, tolerance=1e-6,
                       precondition=False):
    """Find a constrained minimum with matrix-free conjugate gradients.

    Directions vanish on fixed cells. A nonpositive curvature along a
    nonzero residual indicates degeneracy or loss of numerical precision.
    """
    if iterations < 1 or not math.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("Invalid solver budget or tolerance")
    backend = energy.backend
    with backend.scope():
        field = energy.project(backend.array(initial, copy=True))
        backend.require_finite(field, "initial spatial field")
        free = ~energy.fixed
        count = int(backend.scalar(free.sum()))
        components, gradient = energy.value_gradient(field)
        energies = [sum(components)]
        if not count:
            return VariationalResult(field, energies, 0, True, 0.0, 0.0)
        inverse = None
        if precondition:
            from Spatial.Hamiltonian import hessian_columns

            diagonal = backend.array([column[2] for column in hessian_columns(energy)]).reshape(field.shape)
            if bool(backend.scalar((diagonal[free] <= 0).any())):
                raise FloatingPointError("Free-cell Jacobi diagonal must be positive")
            inverse = 1 / backend.xp.where(free, diagonal, 1)

        def prepare(residual):
            transformed = residual if inverse is None else residual * inverse
            return transformed, backend.scalar((residual * transformed).sum())

        residual = -gradient * free
        transformed, squared = prepare(residual)
        direction = transformed.copy()
        rms = backend.global_norm([residual]) / math.sqrt(count)
        if rms <= tolerance:
            return VariationalResult(field, energies, 0, True, rms, 0.0)
        if energy.lipschitz <= 0:
            raise ValueError("Objective has no active terms")
        step = 0.0
        for index in range(1, iterations + 1):
            hessian_direction = energy.hessian_vector(direction) * free
            curvature = backend.scalar((direction * hessian_direction).sum())
            if not math.isfinite(curvature) or curvature <= 0:
                raise FloatingPointError("Nonpositive curvature in free conjugate direction")
            step = squared / curvature
            field = energy.project(field + step * direction)
            components, exact = energy.value_gradient(field)
            energies.append(sum(components))
            residual -= step * hessian_direction
            exact *= -free.astype(field.dtype)
            rms = backend.global_norm([exact]) / math.sqrt(count)
            if rms <= tolerance:
                return VariationalResult(field, energies, index, True, rms, step)
            # Periodic replacement prevents iterative residual drift.
            if index % 25 == 0:
                residual = exact
                transformed, squared = prepare(residual)
                direction = transformed.copy()
            else:
                transformed, new_squared = prepare(residual)
                if new_squared == 0:
                    residual = exact
                    transformed, squared = prepare(residual)
                    direction = transformed.copy()
                else:
                    direction = transformed + (new_squared / squared) * direction
                    squared = new_squared
        return VariationalResult(field, energies, iterations, False, rms, step)
