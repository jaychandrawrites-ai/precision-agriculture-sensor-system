"""Shared causal forecast and manually differentiated unrolled objective."""
from dataclasses import dataclass
import math

from Data.Derivatives import lag_coefficients
from Physics.Constraints import LOWER, UPPER
from Temporal.Losses import regression_loss


@dataclass(frozen=True)
class RolloutOptions:
    output: str = "delta"
    smoothing: float = 0.65
    acceleration_limit: float | None = 2.0
    delta_limit: float | None = None

    def __post_init__(self):
        if self.output not in ("state", "delta") or not 0 <= self.smoothing <= 1:
            raise ValueError("Invalid rollout options")
        if self.acceleration_limit is not None and (
            not math.isfinite(self.acceleration_limit) or self.acceleration_limit <= 0
        ):
            raise ValueError("Acceleration limit must be finite and positive")

        if self.delta_limit is not None and (self.output != "delta"
                or not math.isfinite(self.delta_limit) or self.delta_limit <= 0):
            raise ValueError("Delta limit requires delta output and a finite positive bound")

    @classmethod
    def for_horizon(cls, hours):
        # Keep short-horizon training unchanged and bound long unrolled increments.
        return cls(delta_limit=1.0) if hours > 24 else cls()


class Rollout:
    def __init__(self, model, input_scaler, target_scaler, backend, options,
                 interval_hours=1.0, crop_coefficient=None):
        self.model = model
        self.backend = backend
        self.options = options
        self.coefficients = backend.array(lag_coefficients(options.smoothing, interval_hours))
        self.input_mean = backend.array(input_scaler.mean)
        self.input_scale = backend.array(input_scaler.scale)
        self.mean = backend.array(target_scaler.mean)
        self.scale = backend.array(target_scaler.scale)
        self.lower = backend.array(LOWER)
        self.upper = backend.array(UPPER)
        if crop_coefficient is not None and (
            crop_coefficient < 0 or not math.isfinite(crop_coefficient)
        ):
            raise ValueError("Invalid crop coefficient")
        self.crop = crop_coefficient

    def normalized_inputs(self, history):
        xp = self.backend.xp
        features = (self.coefficients @ history).reshape(len(history), 18)
        inputs = (features - self.input_mean) / self.input_scale
        self.backend.require_finite(inputs, "normalized rollout input")
        derivative = None
        limit = self.options.acceleration_limit
        if limit is not None:
            robust = xp.tanh(inputs[:, 12:] / limit)
            inputs[:, 12:] = limit * robust
            derivative = 1 - robust**2
        return inputs, derivative

    def _step(self, history, training):
        xp = self.backend.xp
        inputs, derivative = self.normalized_inputs(history)
        output, cache = self.model.forward(inputs, training=training)
        output_derivative = 1.0
        if self.options.delta_limit is not None:
            self.backend.require_finite(output, "unbounded rollout output")
            limit = self.options.delta_limit
            bounded = xp.tanh(output / limit)
            output_derivative = 1 - bounded**2
            output = limit * bounded
        if self.options.output == "delta":
            raw = history[:, -1] + output * self.scale
        else:
            raw = self.mean + output * self.scale
        self.backend.require_finite(raw, "raw rollout state")
        projected = xp.clip(raw, self.lower, self.upper)
        eta_limited = None
        if self.crop is not None:
            cap = self.crop * projected[:, 3]
            eta_limited = projected[:, 4] > cap
            projected[:, 4] = xp.minimum(projected[:, 4], cap)
        return projected, raw, (cache, derivative, eta_limited, output_derivative), inputs

    def predict(self, histories, steps, return_diagnostics=False):
        with self.backend.scope():
            xp = self.backend.xp
            rolling = self.backend.array(histories)
            if rolling.ndim != 3 or rolling.shape[1:] != (3, 6) or steps < 1 or not len(rolling):
                raise ValueError("Expected batch of three-state histories and positive horizon")
            self.backend.require_finite(rolling, "initial rollout state")
            states, raw_states, diagnostics = [], [], []
            for _ in range(steps):
                state, raw, _, inputs = self._step(rolling, training=False)
                states.append(state)
                if return_diagnostics:
                    raw_states.append(raw)
                    diagnostics.append({
                        "projected_rows": int(self.backend.scalar(xp.any(state != raw, axis=1).sum())),
                        "projected_by_feature": self.backend.to_host((state != raw).sum(axis=0)).tolist(),
                        "max_abs_normalized_input": float(self.backend.scalar(xp.abs(inputs).max())),
                        "max_abs_normalized_raw_step": float(self.backend.scalar(
                            xp.abs((raw - rolling[:, -1]) / self.scale).max()
                        )),
                    })
                rolling = xp.concatenate((rolling[:, 1:], state[:, None]), axis=1)
            forecasts = self.backend.to_host(xp.stack(states, axis=1))
            if return_diagnostics:
                return forecasts, self.backend.to_host(xp.stack(raw_states, axis=1)), diagnostics
            return forecasts

    def objective(self, histories, targets, *, autoregressive=True, training=True,
                  backward=True, loss="huber"):
        """Score raw proposals and differentiate through projected feedback.

        Raw loss retains a corrective signal when an output saturates. Future
        losses propagate through the actual projected states used as inputs.
        Parameter gradients accumulate into existing buffers.
        """
        with self.backend.scope():
            xp = self.backend.xp
            rolling = self.backend.array(histories)
            targets = self.backend.array(targets)
            if (rolling.ndim != 3 or rolling.shape[1:] != (3, 6) or not len(rolling)
                    or targets.ndim != 3 or targets.shape[0] != len(rolling)
                    or not targets.shape[1] or targets.shape[2] != 6):
                raise ValueError("Expected aligned history and target batches")
            steps = targets.shape[1]
            tape = []
            total = 0.0
            for step in range(steps):
                next_state, raw, context, _ = self._step(rolling, training)
                target = targets[:, step]
                score, gradient = regression_loss((raw - target) / self.scale,
                                                  xp.zeros_like(raw), self.backend, kind=loss)
                total += score / steps
                if backward:
                    tape.append((raw, context, gradient / (steps * self.scale)))
                feedback = next_state if autoregressive else target
                rolling = xp.concatenate((rolling[:, 1:], feedback[:, None]), axis=1)

            if backward:
                carry = xp.zeros_like(rolling)
                for raw, (cache, derivative, eta_limited, output_derivative), direct in reversed(tape):
                    # Reverse the three-state shift and the optional ETa coupling.
                    feedback_grad = carry[:, 2].copy() if autoregressive else xp.zeros_like(raw)
                    if eta_limited is not None:
                        feedback_grad[:, 3] += self.crop * feedback_grad[:, 4] * eta_limited
                        feedback_grad[:, 4] *= ~eta_limited
                    interior = (raw > self.lower) & (raw < self.upper)
                    raw_grad = direct + feedback_grad * interior
                    input_grad = self.model.backward(raw_grad * self.scale * output_derivative, cache)
                    if derivative is not None:
                        input_grad[:, 12:] *= derivative
                    input_grad = (input_grad / self.input_scale).reshape(-1, 3, 6)
                    previous = self.coefficients.T @ input_grad
                    if self.options.output == "delta":
                        previous[:, 2] += raw_grad
                    if autoregressive:
                        previous[:, 1:] += carry[:, :2]
                    carry = previous
            return total
