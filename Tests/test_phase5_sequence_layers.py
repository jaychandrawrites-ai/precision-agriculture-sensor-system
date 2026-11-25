import unittest

import numpy as np

from Framework.Backend import BackendError, create_backend
from Framework.GradientCheck import check_model
from Framework.Math_Utilities import finite_difference_gradient
from Framework.SequenceLayers import Conv1D, GRU, LayerNorm, MultiHeadAttention, SinusoidalPosition


class SequenceLayerTests(unittest.TestCase):
    def setUp(self):
        self.backend = create_backend(dtype="float64", seed=19)

    def verify_adjoint(self, layer, x, kw=None):
        kw = kw or {}
        output, cache = layer.forward(x, training=False, **kw)
        upstream = self.backend.normal(output.shape)
        layer.zero_grad()
        adjoint = layer.backward(upstream, cache)
        numeric = finite_difference_gradient(
            lambda z: (layer.forward(z, training=False, **kw)[0] * upstream).sum(),
            x, self.backend,
        )
        np.testing.assert_allclose(adjoint, numeric, rtol=3e-5, atol=2e-7)
        if not kw:
            for result in check_model(layer, x, upstream, self.backend,
                                      atol=2e-7, rtol=3e-5):
                self.assertTrue(result.passed, result)

    def test_layer_norm_statistics_and_gradients(self):
        layer = LayerNorm(self.backend, 4)
        x = self.backend.normal((2, 3, 4))
        result, _ = layer.forward(x)
        np.testing.assert_allclose(result.mean(-1), 0, atol=1e-14)
        variance = x.var(-1)
        np.testing.assert_allclose(result.var(-1), variance / (variance + layer.epsilon))
        self.verify_adjoint(layer, x)

    def test_temporal_convolution_respects_dilation_and_prefix_causality(self):
        layer = Conv1D(self.backend, 2, 3, 3, dilation=2)
        x = self.backend.normal((2, 6, 2))
        original, _ = layer.forward(x)
        perturbed = x.copy()
        perturbed[:, 3:] += 100
        changed, _ = layer.forward(perturbed)
        np.testing.assert_array_equal(original[:, :3], changed[:, :3])
        self.assertEqual(original.shape, (2, 6, 3))
        self.verify_adjoint(layer, x)

    def test_attention_masks_control_inputs_outputs_and_gradients(self):
        layer = MultiHeadAttention(self.backend, 4, 2)
        x = self.backend.normal((2, 4, 4))
        original, _ = layer.forward(x)
        perturbed = x.copy()
        perturbed[:, 2:] += 2
        changed, _ = layer.forward(perturbed)
        np.testing.assert_array_equal(original[:, :2], changed[:, :2])
        valid = [[True, True, False, True], [False, False, False, False]]
        masked, cache = layer.forward(x, valid=valid)
        np.testing.assert_array_equal(masked[~np.asarray(valid)], 0)
        dx = layer.backward(np.ones_like(masked), cache)
        np.testing.assert_array_equal(dx[~np.asarray(valid)], 0)
        self.verify_adjoint(layer, x, {"valid": valid})
        self.verify_adjoint(layer, x)

    def test_gru_adjoint_and_masked_state_retention(self):
        layer = GRU(self.backend, 2, 3)
        x = self.backend.normal((2, 4, 2))
        self.verify_adjoint(layer, x)
        initial = self.backend.normal((2, 3))
        valid = [[False, True, False, True], [True, True, True, False]]
        y, cache = layer.forward(x, initial=initial, valid=valid)
        np.testing.assert_array_equal(y[0, 0], initial[0])
        np.testing.assert_array_equal(y[0, 2], y[0, 1])
        dy = self.backend.normal(y.shape)
        final = self.backend.normal(initial.shape)
        dx, dh = layer.backward_with_state(dy, cache, final_gradient=final)
        np.testing.assert_array_equal(dx[~np.asarray(valid)], 0)

        def objective(state):
            result, _ = layer.forward(x, initial=state, valid=valid)
            return (result*dy).sum() + (result[:, -1]*final).sum()

        expected = finite_difference_gradient(objective, initial, self.backend)
        np.testing.assert_allclose(dh, expected, rtol=1e-6, atol=1e-7)

    def test_positions_match_contiguous_windows(self):
        x = np.zeros((2, 8, 5))
        whole, _ = SinusoidalPosition(self.backend, 10).forward(x)
        layer = SinusoidalPosition(self.backend, 14)
        tail, cache = layer.forward(x[:, 4:])
        np.testing.assert_array_equal(whole[:, 4:], tail)
        self.assertTrue(np.isfinite(tail).all())
        self.assertIs(layer.backward(tail, cache), tail)

    def test_cpu_cuda_attention_agree_with_shared_weights(self):
        try:
            cuda = create_backend(device="cuda", dtype="float64")
        except BackendError:
            self.skipTest("CUDA backend unavailable")
        cpu_layer = MultiHeadAttention(self.backend, 4, 2)
        gpu_layer = MultiHeadAttention(cuda, 4, 2)
        for cpu, gpu in zip(cpu_layer.parameters(), gpu_layer.parameters()):
            with cuda.scope():
                gpu.value[...] = cuda.array(cpu.value)
        x = self.backend.normal((2, 3, 4))
        cpu_y, cpu_cache = cpu_layer.forward(x)
        gpu_y, gpu_cache = gpu_layer.forward(cuda.array(x))
        np.testing.assert_allclose(cpu_y, cuda.to_host(gpu_y), atol=1e-11)
        cpu_grad = cpu_layer.backward(np.ones_like(cpu_y), cpu_cache)
        gpu_grad = gpu_layer.backward(cuda.ones(gpu_y.shape), gpu_cache)
        np.testing.assert_allclose(cpu_grad, cuda.to_host(gpu_grad), atol=1e-10)
