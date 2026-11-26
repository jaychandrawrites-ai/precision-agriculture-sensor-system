"""Joint training with held-out times and independently noisy survey labels."""
import argparse

import numpy as np

from Framework.Backend import Backend, BackendConfig
from spatial_forecast import interpolation_weights
from Temporal.JointTraining import JointTrainer, JointWeights
from Temporal.Model import TemporalModel
from Utils.Serialization import write_json


def fixture(seed=82):
    rng = np.random.default_rng(seed)
    sources = np.array([[0., 0.], [1., 0.], [0., 1.], [1., 1.]])
    queries = np.array([[.2, .3], [.7, .4], [.5, .8], [.9, .6]])
    scales = np.array([5., 15., .2, .08, .05, .35])

    def truth(hours, sites):
        east, north = sites.T
        daily = np.sin(hours[:, None] * (2 * np.pi / 24))
        signal = daily + .27 * east - .41 * north + .16 * east * north
        baseline = [24., 65., .4, .23, .12, .7]
        return np.asarray(baseline) + signal[..., None] * scales

    issues = np.arange(96)
    now, previous, earlier = [truth(issues - offset, sources) for offset in range(3)]
    inputs = np.concatenate((now / scales, (now - previous) / scales,
                             (now - 2 * previous + earlier) / scales), axis=2)
    temporal = truth(issues + 1, sources)
    survey = truth(issues + 1, queries)
    temporal += rng.normal(size=temporal.shape) * scales * .015
    survey += rng.normal(size=survey.shape) * scales * .025
    weights = interpolation_weights(sources, queries, 2.)
    graph = np.array([[1., -1., 0.], [0., 1., -1.]])
    train = (inputs[:72], temporal[:72], survey[:72, :3], weights[:3], graph)
    test = (inputs[72:], temporal[72:], survey[72:, 3:], weights[3:], np.zeros((0, 1)))
    return scales, train, test


def compare(epochs=160, seed=82):
    if epochs < 1:
        raise ValueError("Provide a positive epoch budget")
    scales, train, test = fixture(seed)
    report = {"synthetic": True, "seed": seed, "updates_per_model": epochs, "models": {}}
    for method, weights in (("temporal_only", JointWeights(1, 0, 0, 0)), ("joint", JointWeights())):
        backend = Backend(BackendConfig(dtype="float64", seed=seed))
        model = JointTrainer(TemporalModel(backend, (24,), activations=("tanh",)),
                             scales, weights=weights, learning_rate=.025)
        initial = model.objective(*train)[2]["loss"]
        for _ in range(epochs):
            last = model.step(*train)
        source, spatial, _ = model.objective(*test)
        report["models"][method] = dict(
            initial_loss=initial, final_loss=last["loss"],
            temporal_rmse=np.sqrt(np.mean((source - test[1])**2, axis=(0, 1))),
            held_out_site_rmse=np.sqrt(np.mean((spatial - test[2])**2, axis=(0, 1))))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output")
    parser.add_argument("--epochs", type=int, default=160)
    parser.add_argument("--seed", type=int, default=82)
    args = parser.parse_args()
    write_json(args.output, compare(args.epochs, args.seed))
