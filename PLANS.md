# Jupiter implementation roadmap

## Objective
Train a temporal forecasting model once from historical AZMET data and subsequently forecast from Arduino sessions without live AZMET input or automatic retraining. Use GPS separately for spatial mapping. Target roughly 40,000 lines only as a planning estimate, not a completion criterion.

## Data contract
Six ordered physical features: air temperature C, relative humidity percent, leaf wetness index 0–1, reference ET mm/interval, actual ET mm/interval, proposed cotton heat stress. Construct 18 inputs from current values, backward first differences and backward second differences using three consecutive hourly samples. Predict next six values, recursively update derivatives. No future leakage. Chronological train/validation/test; normalization fit on training only. Compare persistence baseline.

Historical data: inspect actual AZMET CSV before mapping columns. Derive LWI from coupled leaf temperature and surface water ODE; use soil bucket scenarios for ETa, not observed historical ETa. Use shared heat stress definition. Hargreaves proxy requires adequate causal temperature history; align training and inference estimation methods and time conventions. FAO56 available for reference. Document all assumptions, uncertainty, units and domain shift. Do not invent missing source data.

Arduino: SEN0604 soil moisture/temp/EC/pH, SHT45 air temperature/RH, PHYTOS31 leaf wetness, PA1010D GPS. Inspect actual AquaMap firmware when supplied. Expected packet format requires verification. Preserve raw measurements, timestamps, quality flags. Calibration endpoints must be measured, not fabricated. GPS excluded from primary temporal input.

## Phases
1. Audit current framework and physics modules. Finish Data/Dataset.py, Preprocessing.py, DataQuality.py, Filters.py; verify causal resampling, missing-data logic, and feature ordering. Add requirements and package imports.
2. Acquisition: SerialReader, PacketParser, SensorData, Calibration, DataLogger, SessionManager, AcquisitionController. Graceful stop, raw logs, duplicate and invalid packet handling. Offline replay.
3. Temporal: Model, Trainer, Evaluator, Validator, Predictor, Losses. Dense network with manual gradients, Huber and MSE, quality weighting, Adam/SGD, clipping, scheduling, checkpointing, recursive rollout. Test gradient and shape contracts.
4. Evolution: Genome, Population, GeneticAlgorithm, Fitness, Mutation, Crossover. Evolve layer widths, depth and valid skip connections. Default population 8, generations 10, 20 candidate epochs. Validation fitness with complexity penalty. Untouched test split.
5. QuantumInspired: Qubit, RotationGate, Measurement, QuantumGenome, QuantumPopulation, QuantumEvolution. Classical amplitude pairs normalized to one, sample bits with squared amplitudes, fitness-guided rotation. Compare classical GA under equal budgets.
6. Spatial: SpatialGrid, SeedManager, NoiseGenerator, MonteCarlo, Hamiltonian, EnergyFunctional, BoundaryConditions, PDESolver, VariationalSolver, SpatialLoss, Interpolation, SpatialModel, Uncertainty. GPS to local metric coordinates. Field energy = gradient penalty + curvature penalty + seed fidelity. Stable PDE descent, do not claim literal quantum physics. Hard/soft constraints, IDW and kriging baselines, held-out GPS validation. Metropolis sampling with burn-in and diagnostics; distinguish uncertainty from random restarts. Explicit spatial grid shape convention.
7. Integration: Pipeline, CombinedLoss, SpatiotemporalModel. Separate historical temporal training and field spatial training. Do not fabricate spatial labels from one weather station. Optional joint fine-tuning only with real field data.
8. Utils: Checkpoints, Serialization, Diagnostics, Metrics, Logger, ModelLoader. Portable saved state including assumptions, normalization, architecture, RNG and optimizer. Trusted pickle only or safer serialization. TensorBoard optional. Server jobs and visualization after core pipeline.
9. Entry points train.py, acquire.py, forecast.py, main.py. Independent commands, no implicit retraining. README with runnable setup, usage and limits.
10. Experiments and Tests: physics unit checks, gradients, optimizer convergence, evolution invariants, spatial seed fidelity, integration smoke tests. Compare fixed network, classical GA and quantum-inspired GA with interpolation and variational reconstruction. RMSE, MAE, R2, persistence skill, parameter count and compute budget. Use synthetic fixtures only explicitly labeled as such.

## Completion criteria
A clean checkout installs dependencies, passes relevant tests and runs an explicitly synthetic offline end-to-end demonstration. Real dataset and sensor runs are separate and must be documented as pending if inputs are unavailable. Report actual test outputs and unresolved blockers. Preserve all original hardware source. Do not overwrite main automatically.

## Phase 4 follow-up

The development branch includes the verified Tucson 2019 hourly mapping,
separate experimental direct and hybrid studies, CPU epoch recovery, resumable
validation-only architecture search, robust anisotropic reconstruction,
burn-in-only Monte Carlo adaptation, multi-site forecasts and the loopback
research dashboard. The Arduino source remains available on both development
and archival branches.

Prioritize independent, multi-station temporal evaluation and verified
calibration before using forecasts for field decisions. The Tucson benchmark
favored seasonal persistence at 12, 24 and 48 hours. Additional work includes
evaluation on independent seasons, integrating model uncertainty into spatial
maps, adaptive grids, sampling measurement and parameter uncertainty, validating
the AquaMap protocol with hardware, and optional joint fitting with genuine
paired field labels. GPU consistency and real serial tests require the target
devices. Track these outcomes and maintain checked-in fixtures as explicitly
synthetic data.

## Phase 5 completion checklist

Implemented the verified multi-station archive catalog, versioned registry and
rolling-origin reports. The default derivative model remains unchanged.
Experimental sequence models, a saved seasonal residual ensemble and a frozen
Maricopa transfer study are available. Spatial tooling includes robust and
heteroscedastic reconstruction, anisotropic geostatistics, block posterior
sampling and coherent forecast-draw propagation. Acquisition archives can be
verified and recovered. The local dashboard can persist and resume results.

Priorities for the next run:

1. Predeclare new independent station-years and benchmark short-horizon
   persistence, daily persistence and expanding harmonic fits before investing
   in further neural refinements. Report dependence-sensitive effect bounds.
2. Verify AquaMap receiver payloads and calibration with real instruments, then
   independently measure temporal and spatial field outcomes. Preserve original
   Arduino firmware and differentiate observed, estimated and simulated inputs.
3. Validate full spatial predictive uncertainty with paired field labels.
   Further spatial work includes adaptive grid refinement and coherent
   parameter scenarios with independently supported prior weights.
4. Verify browser rendering, actual live serial workflows and CUDA parity
   on appropriate hardware. Extend comparisons under equal measured training
   work and add publication figure outputs where results justify them.
