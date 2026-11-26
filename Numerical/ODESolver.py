"""Explicit Runge-Kutta integration with adaptive error control."""
from dataclasses import dataclass
import numpy as np

@dataclass
class IntegrationResult:
    times: np.ndarray
    states: np.ndarray
    accepted: int
    rejected: int

def rk4_step(rhs, t, y, h):
    k1 = np.asarray(rhs(t,y))
    k2 = np.asarray(rhs(t+h/2,y+h*k1/2))
    k3 = np.asarray(rhs(t+h/2,y+h*k2/2))
    k4 = np.asarray(rhs(t+h,y+h*k3))
    return y+h*(k1+2*k2+2*k3+k4)/6

def integrate(rhs, initial, start, end, step=1.0, rtol=1e-6,
              atol=1e-8, min_step=1e-8, max_steps=100000, project=None, max_step=np.inf):
    """Step doubling estimates local error; projection occurs after acceptance."""
    if not end >= start or min(step,rtol,atol,min_step)<=0 or not max_step >= min_step:
        raise ValueError("Invalid integration settings")
    y=np.asarray(initial,dtype=float).copy()
    if not np.isfinite(y).all():
        raise ValueError("Nonfinite initial state")
    t=float(start)
    times=[t]
    states=[y.copy()]
    accepted=rejected=0
    h=step
    while t<end:
        if accepted+rejected>=max_steps:
            raise RuntimeError("Integration budget exhausted")
        h=min(h,end-t,max_step)
        coarse=rk4_step(rhs,t,y,h)
        half=rk4_step(rhs,t,y,h/2)
        fine=rk4_step(rhs,t+h/2,half,h/2)
        error=(fine-coarse)/15
        scale=atol+rtol*np.maximum(np.abs(y),np.abs(fine))
        ratio=float(np.max(np.abs(error)/scale))
        if not np.isfinite(ratio):
            ratio=np.inf
        if ratio<=1:
            candidate=fine+error
            if project is not None:
                candidate=np.asarray(project(candidate),dtype=float)
            if candidate.shape!=y.shape or not np.isfinite(candidate).all():
                raise FloatingPointError("Invalid accepted state")
            t+=h
            y=candidate
            times.append(t)
            states.append(y.copy())
            accepted+=1
        else:
            rejected+=1
            if h<=min_step:
                raise RuntimeError("Cannot satisfy integration tolerance")
        factor=5.0 if ratio==0 else min(5.0,max(0.1,0.9*ratio**(-0.2)))
        h=max(min_step,h*factor)
    return IntegrationResult(np.asarray(times),np.stack(states),accepted,rejected)


def integrate_implicit(rhs, initial, start, end, *, step=1.0, jac=None,
                       rtol=1e-4, atol=1e-7, min_step=1e-9, max_steps=100000,
                       newton_iterations=12):
    """Adaptive backward Euler with step doubling and damped Newton solves."""
    from Numerical.Differentiation import jacobian

    parameters = [start, end, step, rtol, atol, min_step]
    if (not np.isfinite(parameters).all() or end < start
            or min(step, rtol, atol, min_step) <= 0 or step < min_step
            or min(max_steps, newton_iterations) < 1):
        raise ValueError("Invalid implicit integration settings")
    state = np.asarray(initial, dtype=float)
    if state.ndim != 1 or not state.size or not np.isfinite(state).all():
        raise ValueError("Expected a finite initial state vector")
    identity = np.eye(len(state))

    def evaluate(time, value):
        result = np.asarray(rhs(time, value), dtype=float)
        if result.shape != state.shape:
            raise ValueError("ODE right hand side must match the state")
        return result

    def backward_euler(time, old, dt):
        candidate = old.copy()
        def residual(value):
            return value - old - dt * evaluate(time, value)
        tolerance = (atol + rtol * np.maximum(np.abs(old), 1e-8)) * 0.025
        for _ in range(newton_iterations):
            error = residual(candidate)
            size = np.max(np.abs(error) / tolerance)
            if not np.isfinite(size):
                return None
            if size <= 1:
                return candidate
            if jac is None:
                matrix = jacobian(residual, candidate)
            else:
                derivative = np.asarray(jac(time, candidate), dtype=float)
                if derivative.shape != identity.shape or not np.isfinite(derivative).all():
                    raise ValueError("ODE Jacobian must be finite and square")
                matrix = identity - dt * derivative
            try:
                direction = np.linalg.solve(matrix, -error)
            except np.linalg.LinAlgError:
                return None
            for factor in (1, .5, .25, .125, .0625):
                updated = candidate + factor * direction
                if np.max(np.abs(residual(updated)) / tolerance) < size:
                    candidate = updated
                    break
            else:
                return None
        return None

    t, dt = float(start), step
    times, states = [t], [state.copy()]
    accepted = rejected = 0
    while t < end:
        if accepted + rejected >= max_steps:
            raise RuntimeError("Implicit integration budget exhausted")
        dt = min(dt, end - t)
        if t + dt <= t:
            raise RuntimeError("Step is below time precision")
        coarse = backward_euler(t + dt, state, dt)
        half = backward_euler(t + dt / 2, state, dt / 2)
        fine = None if half is None else backward_euler(t + dt, half, dt / 2)
        ratio = np.inf
        if coarse is not None and fine is not None:
            weight = atol + rtol * np.maximum(np.abs(fine), np.abs(state))
            ratio = np.max(np.abs(fine - coarse) / weight)
        if ratio <= 1:
            t, state = t + dt, fine
            times.append(t)
            states.append(state)
            accepted += 1
        else:
            rejected += 1
            if dt <= min_step:
                raise RuntimeError("Cannot satisfy implicit integration tolerance")
        dt = max(min_step, dt * (4 if ratio == 0 else np.clip(0.9 / np.sqrt(ratio), .2, 4)))
    return IntegrationResult(np.asarray(times), np.stack(states), accepted, rejected)


# Dormand-Prince 5(4) tableau. The final stage is evaluated at the fifth-order solution.
_DOPRI_STAGES = (
    (1 / 5, (1 / 5,)),
    (3 / 10, (3 / 40, 9 / 40)),
    (4 / 5, (44 / 45, -56 / 15, 32 / 9)),
    (8 / 9, (19372 / 6561, -25360 / 2187, 64448 / 6561, -212 / 729)),
    (1, (9017 / 3168, -355 / 33, 46732 / 5247, 49 / 176, -5103 / 18656)),
)
_DOPRI_FIFTH = (35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84)
_DOPRI_ERROR = tuple(
    high - low for high, low in zip(
        (*_DOPRI_FIFTH, 0),
        (5179 / 57600, 0, 7571 / 16695, 393 / 640, -92097 / 339200, 187 / 2100, 1 / 40),
    )
)


def integrate_rk45(rhs, initial, start, end, *, step=1.0, rtol=1e-6, atol=1e-8,
                   min_step=1e-10, max_step=np.inf, max_steps=100000, project=None):
    """Adaptive Dormand-Prince integration for smooth, nonstiff real ODEs.

    An optional projection is applied only after the embedded error estimate
    accepts a step. It invalidates endpoint derivatives when it changes the
    state, and the error estimate does not control projection error.
    """
    settings = [start, end, step, rtol, atol, min_step]
    if (not np.isfinite(settings).all() or end < start or min(step, rtol, atol, min_step) <= 0
            or not max_step >= min_step or not isinstance(max_steps, int) or max_steps < 1):
        raise ValueError("Invalid RK45 settings")
    state = np.asarray(initial, dtype=float).copy()
    if not state.size or not np.isfinite(state).all():
        raise ValueError("Expected nonempty, finite initial state")

    def evaluate(time, point):
        result = np.asarray(rhs(time, point), dtype=float)
        if result.shape != state.shape:
            raise ValueError("ODE right hand side must match the state")
        return result

    time = float(start)
    progress = min(max(step, min_step), max_step)
    times, states = [time], [state.copy()]
    accepted = rejected = 0
    first = None
    while time < end:
        if accepted + rejected >= max_steps:
            raise RuntimeError("RK45 integration budget exhausted")
        next_time = min(end, time + progress)
        dt = next_time - time
        if dt <= 0:
            raise RuntimeError("Step is below time precision")
        if first is None:
            first = evaluate(time, state)
        stages = [first]
        for fraction, weights in _DOPRI_STAGES:
            trial = state + dt * sum(weight * value for weight, value in zip(weights, stages))
            stages.append(evaluate(time + fraction * dt, trial))
        proposal = state + dt * sum(weight * value for weight, value in zip(_DOPRI_FIFTH, stages))
        last = evaluate(next_time, proposal)
        stages.append(last)
        error = dt * sum(weight * value for weight, value in zip(_DOPRI_ERROR, stages))
        scale = atol + rtol * np.maximum(np.abs(state), np.abs(proposal))
        ratio = float(np.max(np.abs(error) / scale))
        if not np.isfinite(ratio) or not np.isfinite(last).all():
            ratio = np.inf
        if ratio <= 1:
            candidate = proposal if project is None else np.asarray(project(proposal.copy()), dtype=float)
            if candidate.shape != state.shape or not np.isfinite(candidate).all():
                raise FloatingPointError("Invalid accepted state")
            first = last if np.array_equal(candidate, proposal) else None
            state, time = candidate, next_time
            states.append(state.copy())
            times.append(time)
            accepted += 1
        else:
            rejected += 1
            if dt <= min_step:
                raise RuntimeError("Cannot satisfy RK45 integration tolerance")
        scale_factor = 5.0 if ratio == 0 else min(5.0, max(0.2, 0.9 * ratio**(-0.2)))
        progress = min(max_step, max(min_step, dt * scale_factor))
    return IntegrationResult(np.asarray(times), np.stack(states), accepted, rejected)
