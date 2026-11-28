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

Phase 6's experimental LSTM uses cell state
`c_t = f_t c_(t−1) + i_t g_t` and hidden state
`h_t = o_t tanh(c_t)`. The manual backward pass accumulates both state
adjoints from the end of the sequence. The recurrent-attention variant
adds a causal attention layer to the LSTM sequence before its direct head.

`adaptive_map.py` refines a bounded quadtree using model variation,
conditional standard deviation, vicinity to observations, or a polygon
boundary. Polygon membership refers to cell centers. Conditional kriging
variance measures uncertainty under the specified covariance model.

## Evolution of derivative-informed recursive architectures

The input is a current six-variable state, smoothed backward slope, and
backward second difference. Each 18-vector uses training-only scaling.
The acceleration inputs are smoothly bounded after normalization. The
network predicts an increment in units of the target scale. For new
training runs above 24 hours the normalized increment is
`L tanh(network_output / L)`, with L = 1. This is a numerical trust
bound on proposals, with no clipping of observed targets. The derivative
through it is `1 - tanh(network_output / L)^2`. The projected six-variable
state enters the next three-state history. Backpropagation reverses the
history shift and applies the projection Jacobian to feedback gradients,
while the direct loss still compares the pre-projection state with the
observed target. Checkpoints store the option. Existing checkpoints load
with their original unbounded increment rule.

The recursive architecture genome encodes depth, widths and residual
connections. Extended genomes encode independent activations. Candidate
weights are initialized from the controlled training seed, then optimized
by manual backpropagation. Crossover and mutation act on architecture
bits. Neither reproduced genotypes nor final refits inherit weight tensors.

```text
initialize seeded genomes or angle vectors
for each generation:
    decode candidate genomes, repairing against the shared size cap
    train from fresh initialization on chronological training windows
    rank by validation loss plus a parameter penalty
    retain the best validation record
    GA: keep elite genotypes, select parents, cross over, flip bits
    QIGA: rotate sampled angle vectors toward the best genotype
    hybrid: inject GA offspring and rotate the sampled angle vectors
refit the winning architecture on the training split
evaluate on the final time partition and save a portable checkpoint
```

The quantum-inspired operations are the real amplitude representation
`(cos θ, sin θ)`, Bernoulli measurement with `P(1) = sin² θ`, and
angle rotation. A mismatch with the guide moves θ toward the guide bit.
Angle bounds preserve nonzero alternative probabilities. The adaptive
variant adjusts rotation by rank and progress and relaxes concentrated
vectors toward π/4. The algorithm uses classical random numbers, training,
ranking, selection and mutation. The preserved QIGA elite is a best-so-far
record, not a guaranteed member of the next sampled population.
