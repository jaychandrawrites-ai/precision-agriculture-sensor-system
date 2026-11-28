"""Manual regression losses with sample and feature weighting."""
from __future__ import annotations

def regression_loss(prediction, target, backend, sample_weights=None,
                    feature_weights=None, kind="huber", delta=1.0):
    if prediction.shape != target.shape or prediction.ndim != 2:
        raise ValueError("Expected matching batch by feature arrays")
    if kind not in ("mse", "huber") or delta <= 0:
        raise ValueError("Invalid regression loss")
    with backend.scope():
        xp = backend.xp
        n, d = prediction.shape
        if not n or not d:
            raise ValueError("Empty batch")
        s = backend.ones((n,)) if sample_weights is None else backend.array(sample_weights)
        f = backend.ones((d,)) if feature_weights is None else backend.array(feature_weights)
        if s.shape != (n,) or f.shape != (d,):
            raise ValueError("Weight shape mismatch")
        backend.require_finite(s)
        backend.require_finite(f)
        if bool(backend.scalar(xp.any(s < 0))) or bool(backend.scalar(xp.any(f < 0))):
            raise ValueError("Negative loss weight")
        weights = s[:, None]*f[None, :]
        denominator = float(backend.scalar(weights.sum()))
        if denominator <= 0:
            raise ValueError("Batch has no positive weight")
        residual = prediction-target
        if kind == "mse":
            element = residual**2
            derivative = 2*residual
        else:
            magnitude = xp.abs(residual)
            quadratic = xp.minimum(magnitude, delta)
            element = 0.5 * quadratic**2 + delta * (magnitude - quadratic)
            derivative = xp.clip(residual, -delta, delta)
        loss = float(backend.scalar((element*weights).sum()/denominator))
        gradient = derivative*weights/denominator
        backend.require_finite(gradient)
        return loss, gradient

def mse(prediction, target, backend, **kwargs):
    return regression_loss(prediction, target, backend, kind="mse", **kwargs)

def huber(prediction, target, backend, **kwargs):
    return regression_loss(prediction, target, backend, kind="huber", **kwargs)

def cross_entropy_logits(logits, labels, backend, class_weights=None):
    """Optional classification loss; not used for continuous ET forecasts."""
    from Framework.Math_Utilities import logsumexp
    with backend.scope():
        xp = backend.xp
        if logits.ndim != 2 or not logits.shape[0] or logits.shape[1] < 2:
            raise ValueError("Invalid logits")
        labels = backend.array(labels, dtype="int64")
        n, c = logits.shape
        if labels.shape != (n,) or bool(backend.scalar(xp.any((labels < 0)|(labels >= c)))):
            raise ValueError("Invalid class labels")
        weights = backend.ones((c,)) if class_weights is None else backend.array(class_weights)
        if weights.shape != (c,) or bool(backend.scalar(xp.any(weights < 0))):
            raise ValueError("Invalid class weights")
        backend.require_finite(weights)
        rows = xp.arange(n)
        sample = weights[labels]
        total = float(backend.scalar(sample.sum()))
        if total <= 0:
            raise ValueError("Zero class weight")
        logp = logits-logsumexp(logits, backend, axis=1, keepdims=True)
        loss = -float(backend.scalar((sample*logp[rows, labels]).sum()/total))
        gradient = xp.exp(logp)
        gradient[rows, labels] -= 1
        gradient *= sample[:, None]/total
        return loss, gradient
