# Jupiter mathematics

## Manual differentiation

For a linear layer, `Y = XW + b`, reverse propagation maps a gradient
`G` to `GWᵀ`, `XᵀG` and the row sum of `G`. Batched tensor products
sum their gradients over broadcast axes back to each operand's shape.
Phase 5 adds causal convolution, GRU, normalization and attention. Causal
attention masks keys later than each query before the softmax. Its reverse
pass applies the masked softmax Jacobian and the projection adjoints.

The direct residual objective predicts an increment to a per-origin anchor.
The training loss compares the physical target minus the anchor to the
network's normalized increment. Default direct inference uses the latest
state. The seasonal experiment supplies an explicitly computed daily anchor.

## Numerical tools

`Numerical` includes step-doubling RK4, Dormand–Prince RK45, implicit
backward Euler, finite-difference Jacobians and directional Hessian products,
matrix-free preconditioned conjugate gradients, and L-BFGS with line search.
Sparse operators expose their action on vectors. Numerical tests use known
solutions, adjoint identities and finite-difference gradients.

## Spatial models

The quadratic field objective combines a gradient term, a squared
Laplacian term and an observation squared residual weighted by precision.
Fixed cells are excluded from free updates. Robust fitting replaces the
quadratic observation loss with pseudo-Huber fidelity, and its graph
Laplacian uses directional edge conductance.

Gaussian geostatistics conditions on measurements using a declared
covariance, observation variance and constant or affine trend. Stable solves
use matrix factorizations and report added jitter. Block Metropolis targets
a classical Gibbs distribution under the specified energy. Small grids have
a dense analytical Gaussian reference. Adaptive proposals tune during
burn-in only. Neither the sampler nor amplitude-based neural search
implements physical quantum computation.
