import unittest

import numpy as np

from Framework.Backend import BackendError, create_backend
from Framework.SequenceLayers import Conv1D, GRU, LayerNorm, MultiHeadAttention


class BackendEquivalenceTests(unittest.TestCase):
    def test_sequence_forward_and_adjoints_match_across_devices(self):
        try:
            gpu = create_backend(device="cuda", dtype="float64")
        except BackendError:
            self.skipTest("CUDA or its optional CuPy backend is unavailable")
        cpu = create_backend(dtype="float64")
        rng = np.random.default_rng(712)
        x = rng.normal(size=(2, 5, 4))
        constructors = (
            lambda backend: LayerNorm(backend, 4),
            lambda backend: Conv1D(backend, 4, 4, 2),
            lambda backend: GRU(backend, 4, 4),
            lambda backend: MultiHeadAttention(backend, 4, 2),
        )
        for constructor in constructors:
            host, device = constructor(cpu), constructor(gpu)
            with self.subTest(layer=type(host).__name__):
                with gpu.scope():
                    for a, b in zip(host.parameters(), device.parameters(), strict=True):
                        b.value[:] = gpu.array(a.value)
                actual_host, cache_host = host.forward(cpu.array(x))
                actual_device, cache_device = device.forward(gpu.array(x))
                np.testing.assert_allclose(actual_host, gpu.to_host(actual_device), rtol=1e-8, atol=1e-9)
                dy = rng.normal(size=actual_host.shape)
                dx_host = host.backward(cpu.array(dy), cache_host)
                dx_device = device.backward(gpu.array(dy), cache_device)
                np.testing.assert_allclose(dx_host, gpu.to_host(dx_device), rtol=1e-8, atol=1e-9)
                for a, b in zip(host.parameters(), device.parameters(), strict=True):
                    np.testing.assert_allclose(a.grad, gpu.to_host(b.grad), rtol=1e-8, atol=1e-9)
