"""Limited-memory BFGS and Armijo backtracking for differentiable objectives."""
from collections import deque
from dataclasses import dataclass

import numpy as np


@dataclass
class OptimizationResult:
    point: np.ndarray
    loss_trace: list[float]
    gradient_norm: float
    status: str
    evaluations: int


def lbfgs(objective, initial, *, iterations=200, memory=8, tolerance=1e-6,
          max_backtracks=30):
    """Minimize a scalar objective returning its value and analytic gradient."""
    x = np.asarray(initial, dtype=float).copy()
    if (x.ndim != 1 or not x.size or not np.isfinite(x).all()
            or min(iterations, memory, max_backtracks) < 1
            or not np.isfinite(tolerance) or tolerance <= 0):
        raise ValueError("Invalid objective initialization or optimizer settings")

    def evaluate(point):
        value, gradient = objective(point)
        gradient = np.asarray(gradient, dtype=float)
        if gradient.shape != x.shape:
            raise ValueError("Objective gradient must match the point")
        return float(value), gradient

    value, gradient = evaluate(x)
    if not np.isfinite(value) or not np.isfinite(gradient).all():
        raise ValueError("Initial objective and gradient must be finite")
    history, pairs = [value], deque(maxlen=memory)
    evaluations = 1
    status = "iteration_limit"
    for _ in range(iterations):
        if np.linalg.norm(gradient, ord=np.inf) <= tolerance:
            status = "converged"
            break
        q = gradient.copy()
        coefficients = []
        for displacement, change, rho in reversed(pairs):
            alpha = rho * (displacement @ q)
            q -= alpha * change
            coefficients.append(alpha)
        if pairs:
            s, y, _ = pairs[-1]
            q *= (s @ y) / (y @ y)
        for (s, y, rho), alpha in zip(pairs, reversed(coefficients)):
            q += s * (alpha - rho * (y @ q))
        direction = -q
        slope = gradient @ direction
        if not np.isfinite(slope) or slope >= 0:
            pairs.clear()
            direction = -gradient
            slope = -(gradient @ gradient)
        step = 1.0
        for _ in range(max_backtracks):
            next_x = x + step * direction
            next_value, next_gradient = evaluate(next_x)
            evaluations += 1
            if (np.isfinite(next_value) and np.isfinite(next_gradient).all()
                    and next_value <= value + 1e-4 * step * slope
                    and not np.array_equal(next_x, x)):
                break
            step *= 0.5
        else:
            status = "line_search_failed"
            break
        s, y = next_x - x, next_gradient - gradient
        curvature = s @ y
        if curvature > max(1e-14 * np.linalg.norm(s) * np.linalg.norm(y), 0):
            pairs.append((s, y, 1 / curvature))
        x, value, gradient = next_x, next_value, next_gradient
        history.append(value)
    norm = float(np.linalg.norm(gradient, ord=np.inf))
    if norm <= tolerance:
        status = "converged"
    return OptimizationResult(x, history, norm, status, evaluations)
