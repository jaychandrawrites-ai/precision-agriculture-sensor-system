"""Bounded samples of actual layer values and optimizer updates."""
import numpy as np


class LayerDiagnostics:
    def __init__(self, model, backend):
        self.model, self.backend = model, backend
        self.layers = [(f"{i}:{type(layer).__name__}", layer) for i, layer in enumerate(model.layers)]

    def arrays(self, layer, attribute):
        return [self.backend.to_host(getattr(param, attribute), copy=True).astype(float)
                for param in layer.parameters()]

    @staticmethod
    def stats(values):
        flat = np.concatenate([value.ravel() for value in values])
        finite = flat[np.isfinite(flat)]
        return {"count": int(flat.size), "nonfinite": int(flat.size-finite.size),
                "mean": float(finite.mean()) if finite.size else None,
                "std": float(finite.std()) if finite.size else None,
                "zero_fraction": float((finite == 0).mean()) if finite.size else None,
                "l2": float(np.linalg.norm(finite)), "max_abs": float(np.max(np.abs(finite))) if finite.size else None,
                "sample": finite[::max(1, (len(finite)+127)//128)].tolist()}

    def before(self):
        return {name: (self.arrays(layer, "value"), self.arrays(layer, "grad"))
                for name, layer in self.layers if layer.parameters()}

    def report(self, before):
        result = {}
        for name, layer in self.layers:
            if name not in before:
                continue
            old, gradients = before[name]
            current = self.arrays(layer, "value")
            updates = self.stats([new-old_value for new, old_value in zip(current, old)])
            ratio_base = self.stats(old)["l2"]
            result[name] = {"gradient": self.stats(gradients), "parameters": self.stats(current),
                            "update": updates, "update_to_weight_norm": updates["l2"]/ratio_base if ratio_base else None}
            biases = [value for value, param in zip(current, layer.parameters()) if value.ndim == 1]
            if biases:
                result[name]["bias"] = self.stats(biases)
        return result

    def activations(self, inputs):
        values = {}
        with self.backend.scope():
            for name, layer in self.layers:
                inputs, _ = layer.forward(inputs, training=False)
                values[name] = self.stats([self.backend.to_host(inputs)])
        return values
