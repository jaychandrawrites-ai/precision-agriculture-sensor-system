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
network predicts a per-step tendency. For training above 24 hours the
raw proposal is `x + r(μ - x) + s L tanh(network_output / L)` with
`r = 1 - 2^(-1/12)`, `L = 2`, training state mean μ, and per-feature s
equal to the RMS of training-hour changes after subtracting relaxation.
A fixed neural tendency cannot cause unbounded drift because the
coefficient on the previous state is `1 - r`. This does not establish
trajectory skill or guarantee that a trajectory avoids a physical bound.
The derivative of the bounded output is `1 - tanh(network_output / L)^2`.
The relaxation and fitted scales are stored in checkpoints. Short
historical recipes retain their target-scale increment and older
long-horizon checkpoints retain their saved configuration. The projected six-variable
state enters the next three-state history. Backpropagation reverses the
history shift and applies the projection Jacobian to feedback gradients,
while the direct loss still compares the pre-projection state with the
observed target. Observed targets are unchanged.

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

## Spectral evolutionary forecaster

A 22-input dense network receives the six states, smoothed backward first
derivatives, backward second derivatives, and daily and annual sine/cosine
pairs. Calendar hours count from the Unix epoch in UTC, with periods 24
and 8,765.82 hours. Feature and target normalization use training only.

For each environmental variable the network emits a constant and real
and imaginary coefficient pairs. The real basis uses
`exp(-γt) cos(2πt/P)` and `-exp(-γt) sin(2πt/P)`. The full Fourier
configuration emits 90 neurons for seven periods and six variables.
Damped configurations can add an 18-hour transient with a 12-hour
e-folding time. The daily component stays undamped.

Let B(t) be the row of basis values and z₀ be the most recent normalized
state. The reconstructed trajectory is
`z(t) = z₀ + [B(t) - exp(-t/3) B(0)] C`.
The exponential term imposes the observed origin exactly and fades away,
while preserving the constant and daily mode at long leads. Leads are
1 through 72 hours. The forecast is reconstructed independently of the
preceding predicted state.

Shape fitness combines normalized value MSE, identifiable coefficient
error, first-difference MSE, squared pre-projection violations, and a
frequency-sensitive regularizer. Coefficient labels use the SVD
pseudoinverse of the actual anchored basis, including for nonorthogonal
damped modes. The coefficient gradient includes the transposed basis
times the state gradient. The difference gradient accumulates with
opposite signs at successive leads. The final physical projection applies
during inference.

The spectral search uses 17 architecture bits and three frequency-group
bits. It retains a 24-hour mode, can add slower and faster mode groups,
and can enable damping in the damped search arm. The hybrid QIGA starts
with neutral amplitudes, samples genomes, trains fresh networks, ranks
validation fitness plus a complexity penalty, rotates toward the best
genome, and injects mutated and crossed offspring in later generations.
Search state is saved at generation boundaries.

Optional augmentation counts individual normalized state/derivative inputs
in training traversal order, starting at x = 1 and continuing across
batches and epochs. The event probability is 1/x. Its sign is symmetric
and its unbounded magnitude is nx. The resulting mean is zero, expected
absolute magnitude is n and variance is n²x. The implementation caps the
magnitude at b, so the latter two quantities become min(nx,b)/x and
min(nx,b)²/x. The cap leaves event probability unchanged. Validation,
testing, calendar inputs, and inference use zero augmentation.

### Joint level and curve objective

For each origin and feature, split the normalized trajectory error e into
its lead mean m and the centered error e − m. Their mean squares sum
to the existing value MSE. The curve preset adds distinct weights for
these orthogonal terms, giving effective value weights of 1.0 on level
and 1.5 on centered shape. It also increases slope weight and adds
squared second-difference error to score changes in slope around turns.
The adjoint for this last term scatters the second-difference gradient
with weights [1, −2, 1] before applying the spectral basis transpose.
The older objective remains available with the new weights at zero.

### Direct nonlinear residual experiment

The hybrid adds a separate one-hidden-layer tanh network to the spectral
branch. It receives the same 22 issuance features and emits 432 values
for 72 hourly leads and six features. In normalized units the raw
forecast is the anchored spectral reconstruction plus 0.5 tanh(u).
A zero-lead forecast is the observed anchor. Neither branch consumes
future observations or earlier predicted states. The physical projection
runs once after addition. The magnitude penalty is 0.2 mean(correction²).
The residual gradient is the trajectory gradient plus the penalty
gradient, multiplied by 0.5(1 − tanh²(u)). The spectral branch uses the
existing basis transpose and coefficient loss.

The spectral branch trains first, then stays frozen while the residual
learns against its training-origin forecasts. Optional joint epochs
update both networks. The best validation state across residual-only and
joint epochs is exported. Two existing genome bits also encode the
residual width (8, 16, 24 or 32), coupling width selection with the encoded
spectral architecture. The sparse control augments training inputs only.

## QSEN v3 local release

QSEN v3 (Quantum Spectral Evolutionary Network) is Jupiter's
quantum-inspired spectral and neural residual forecaster for classical
hardware. The [QSEN local guide](QSEN_LOCAL_GUIDE.md) is the current
entry point for installation, training, dashboard import and release
checks. Earlier commands and results describe reference experiments.
