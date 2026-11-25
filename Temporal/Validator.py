"""Prediction and physical-unit evaluation on a selected chronological split."""
from Physics.Constraints import project
from Temporal.Evaluator import evaluate


def predict_rows(model, dataset, rows, backend):
    predictions, _ = model.forward(backend.array(dataset.inputs[rows]), training=False)
    return dataset.target_scaler.inverse_transform(backend.to_host(predictions))


def evaluate_rows(model, dataset, rows, backend):
    predictions = project(predict_rows(model, dataset, rows, backend))
    return evaluate(predictions, dataset.targets[rows], dataset.persistence[rows])


def evaluate_rollouts(model, dataset, rows, values, timestamps, backend, steps):
    """Score fixed-origin rollouts whose complete horizons fall inside a split."""
    from itertools import groupby

    import numpy as np

    from Temporal.Predictor import forecast

    if steps < 1:
        raise ValueError("Rollout horizon must be positive")
    indices = dataset.target_indices[rows]
    predictions = []
    targets = []
    origins = []
    for _, group in groupby(enumerate(indices), key=lambda item: item[1] - item[0]):
        block = [index for _, index in group]
        for start in block[:max(0, len(block) - steps + 1)]:
            history = values[start-3:start]
            if (not np.isfinite(history).all()
                    or not (np.diff(timestamps[start-3:start+steps])
                            == np.timedelta64(1, "h")).all()):
                continue
            predictions.append(forecast(model, history, dataset.input_scaler,
                                        dataset.target_scaler, backend, steps))
            targets.append(values[start:start+steps])
            origins.append(history[-1])
    return [
        evaluate(np.asarray(predictions)[:, horizon], np.asarray(targets)[:, horizon], origins)
        for horizon in range(steps)
    ] if predictions else []
