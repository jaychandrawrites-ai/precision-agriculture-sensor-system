import unittest

import numpy as np

from Framework.Backend import create_backend
from Framework.Math_Utilities import (
    batched_matmul_backward,
    batched_matmul_forward,
    contract_backward,
    contract_forward,
    finite_difference_gradient,
)
from Numerical.LinearAlgebra import (
    SparseOperator, cholesky, conditioning, conjugate_gradient, qr, solve_cholesky, svd,
)


class LinearAlgebraTests(unittest.TestCase):
    def setUp(self):
        self.backend = create_backend(dtype="float64", seed=7)

    def verify_pair(self, a, b, forward, backward, reference):
        value, cache = forward(a, b)
        np.testing.assert_allclose(value, reference, atol=1e-12)
        upstream = self.backend.normal(value.shape)
        ga, gb = backward(upstream, cache, self.backend)
        for analytic, operand, fn in ((ga, a, lambda z: forward(z, b)),
                                       (gb, b, lambda z: forward(a, z))):
            numeric = finite_difference_gradient(lambda z: (fn(z)[0]*upstream).sum(), operand, self.backend)
            np.testing.assert_allclose(analytic, numeric, rtol=1e-6, atol=1e-7)

    def test_batch_matmul_adjoint_with_broadcasting_and_vectors(self):
        for a_shape, b_shape in (((2, 1, 2, 3), (1, 4, 3, 2)),
                                 ((3,), (2, 3, 2)), ((2, 3, 2), (2,)), ((3,), (3,))):
            with self.subTest(a_shape=a_shape, b_shape=b_shape):
                a, b = self.backend.normal(a_shape), self.backend.normal(b_shape)
                self.verify_pair(a, b, lambda x, y: batched_matmul_forward(x, y, self.backend),
                                 batched_matmul_backward, a @ b)

    def test_tensor_contraction_adjoint_axis_order_and_scalar(self):
        a = self.backend.normal((2, 3, 4))
        b = self.backend.normal((4, 5, 2))
        axes = ((-1, 0), (0, -1))
        self.verify_pair(a, b, lambda x, y: contract_forward(x, y, axes, self.backend),
                         contract_backward, np.tensordot(a, b, axes=axes))
        scalar = self.backend.array(2)
        self.verify_pair(scalar, b, lambda x, y: contract_forward(x, y, 0, self.backend),
                         contract_backward, 2*b)
        with self.assertRaises(ValueError):
            contract_forward(a, b, ((-1, 2), (0, 1)), self.backend)

    def test_decompositions_and_conditioning(self):
        zero = np.zeros((3, 3))
        matrix = self.backend.normal((2, 3, 3))
        a = matrix @ matrix.swapaxes(-1, -2) + np.eye(3)
        lower = cholesky(a, self.backend)
        np.testing.assert_allclose(lower @ lower.swapaxes(-1, -2), a, atol=1e-12)
        rhs = self.backend.normal((2, 3))
        solution = solve_cholesky(lower, rhs, self.backend)
        np.testing.assert_allclose((a @ solution[..., None])[..., 0], rhs, atol=1e-12)
        q, r = qr(matrix, self.backend)
        np.testing.assert_allclose(q @ r, matrix, atol=1e-12)
        np.testing.assert_allclose(q.swapaxes(-1, -2) @ q,
                                   np.broadcast_to(np.eye(3), (2, 3, 3)), atol=1e-12)
        u, singular, vt = svd(matrix, self.backend)
        np.testing.assert_allclose((u * singular[:, None, :]) @ vt, matrix, atol=1e-12)
        report = conditioning(np.stack([np.eye(3), zero, np.diag([1, 1e-12, 0])]), self.backend)
        np.testing.assert_array_equal(report.rank, [3, 0, 2])
        np.testing.assert_array_equal(report.condition, [1, np.inf, np.inf])
        with self.assertRaisesRegex(ValueError, "symmetric"):
            cholesky([[1, 1], [0, 1]], self.backend)

    def test_sparse_forward_transpose_and_rectangular_diagonal(self):
        operator = SparseOperator([0, 0, 0, 1, 1], [0, 0, 2, 1, 1],
                                  [1, 2, 3, -1, 1], (2, 3), self.backend)
        matrix = np.array([[3, 0, 3], [0, 0, 0]])
        self.assertEqual(operator.nnz, 2)
        np.testing.assert_array_equal(operator(np.arange(6).reshape(3, 2)), matrix @ np.arange(6).reshape(3, 2))
        np.testing.assert_array_equal(operator.apply([1, 2], transpose=True), matrix.T @ [1, 2])
        np.testing.assert_array_equal(operator.diagonal(), [3, 0])
        self.assertEqual(SparseOperator([], [], [], (2, 2), self.backend).nnz, 0)

    def test_pcg_matches_reference_and_checks_true_residual(self):
        scales = np.geomspace(1, 1e4, 20)
        operator = SparseOperator(list(range(20)), list(range(20)), scales,
                                  (20, 20), self.backend)
        rhs = np.linspace(1, 2, 20)
        result = conjugate_gradient(operator, rhs, self.backend, rtol=1e-11,
                                     preconditioner=lambda r: r / operator.diagonal())
        self.assertTrue(result.converged)
        self.assertEqual(result.iterations, 1)
        np.testing.assert_allclose(result.solution, rhs / scales, atol=1e-11)
        self.assertLessEqual(np.linalg.norm(operator(result.solution) - rhs), result.target_norm)
        self.assertEqual(conjugate_gradient(lambda x: -x, rhs, self.backend).reason,
                         "nonpositive_curvature")
        self.assertEqual(conjugate_gradient(operator, rhs, self.backend,
                                             preconditioner=lambda r: -r).reason,
                         "nonpositive_preconditioner")
