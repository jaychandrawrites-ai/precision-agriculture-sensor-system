# Jupiter build log

## 2026-11-25

Started from `jupiter-development` at `4f468ee`. Read AGENTS.md and
PLANS.md and audited the existing Python modules and Arduino sketch.

- **Phase 1:** Implemented a six-feature hourly CSV contract, physical quality
  flags, causal filters segmented at missing hours, consecutive lag windows,
  chronological splits and training-only scaling. Reused existing physics,
  resampling, imputation and derivative implementations. Real AZMET column
  inspection and historical feature derivation are pending.
- **Phase 2:** Implemented live serial capture and offline replay for the
  checked-in sketch, including byte-preserving raw logs, receive timestamps,
  invalid and duplicate field flags, and final partial-block handling. Device
  timestamps and packet IDs are absent in this sketch. AquaMap, GPS, PHYTOS31
  endpoint measurements and live hardware validation remain pending.
- **Phase 3:** Implemented dense regression with manual gradients, optional
  residual blocks, Huber/MSE, quality weighting, Adam/SGD, gradient clipping,
  scheduling, best-validation checkpoint selection and held-out one-step and
  recursive reporting against persistence.
- **Phases 4 and 5:** Implemented validation-only classical GA and a classical
  quantum-inspired amplitude-pair search with equal candidate budgets and a
  parameter-count penalty. Verified on small synthetic budgets. Full default
  budget comparisons and held-out winner evaluation are pending.
- **Phase 6:** Added local metric GPS projection, east/north grid, IDW baseline,
  and hard/soft variational reconstruction through the existing stable PDE
  solver. Kriging, held-out GPS experiments and Metropolis uncertainty remain
  pending.
- **Phases 7–9:** Connected the offline temporal training, saved model and
  independent recursive forecast commands. Added acquisition and demo entry
  points. The versioned JSON artifact saves model, normalization, illustrative
  assumptions, optimizer and scheduler snapshots, shuffle RNG and CPU backend
  RNG. Training resume and GPU RNG restoration are pending. Joint or field
  integration remains pending.
- **Phase 10:** Added tests for chronology, causal features, resampling, gaps,
  checkpoint round trip, projected residual gradients, raw replay, equal
  evolutionary budgets, normalized amplitudes and spatial seed fidelity.

### Verification

- `OPENBLAS_NUM_THREADS=1 python -m unittest discover -s Tests -v` passed 8
  tests under NumPy 2.3.5 and pandas 2.2.3.
- `OPENBLAS_NUM_THREADS=1 python main.py demo` trained on 165 examples,
  validated on 36, tested on 36, saved and reloaded the model, and emitted six
  recursive forecasts. The fixture gave 0.433 °C one-step temperature RMSE
  and 70.481 °C six-step temperature RMSE, with forecasts reaching physical
  bounds. Recursive stability and real-data performance remain open.
- `python -m compileall -q .` passed. Live serial hardware, CUDA and real
  dataset runs were unavailable for this verification.

### Follow-up verification

- Added ordinary kriging with a supplied covariance range and a held-out
  site evaluator for kriging, IDW and variational reconstruction. The spatial
  verification used synthetic sites. Field experiments and Metropolis
  uncertainty remain pending.
- Added `experiments.py` and completed a default search-budget run on the
  synthetic fixture. Both searches used 80 candidate fits and 1,600
  candidate epochs, followed by 30 final-fit epochs per method.

| Method | Parameters | One-step temperature RMSE (°C) | Six-step temperature RMSE (°C) |
| --- | ---: | ---: | ---: |
| Fixed | 3,494 | 2.149 | 59.897 |
| Classical GA | 19,782 | 0.899 | 43.728 |
| Classical amplitude search | 26,822 | 0.907 | 56.766 |

The comparison used the configured default dropout. The standalone synthetic
demo used dropout disabled. All methods showed large recursive errors on the
fixture, so the results do not establish multi-hour forecast reliability.

A fresh Python virtual environment installed `requirements.txt` with NumPy
2.5.5, pandas 2.3.3 and pyserial 3.5. All 9 tests passed with deprecation
warnings treated as errors. The synthetic demo and bytecode compilation also
passed in that environment.

## 2026-11-25 — Recursive forecasting stability

Reproduced the original seed-42 demo with `python main.py demo --legacy`.
Temperature RMSE was **0.433 °C** over all 36 one-step cases and **70.481 °C**
at six hours over 31 common complete forecast windows. On those same 31
windows the legacy one-hour RMSE was 0.450 °C.

The revised model predicts six deltas in units of the training target standard
deviations and adds them to the current state. The current and previous causal
backward slopes contribute 65% and 35%, respectively. The standardized second
difference passes through `2*tanh(acceleration/2)`. This bounds an input feature,
while observed targets remain in their original physical units for scoring.
Training fits input and output scaling from training rows. The first 10 of
30 epochs use teacher forcing. The remaining 20 backpropagate through six
recursive steps, including the history shift, delta addition, normalization,
smoothing, acceleration saturation and projection. Loss is evaluated on the
raw proposals so saturation still receives a corrective signal. Recursive
feedback and validation use the existing physical bounds. An optional supplied
crop coefficient adds an ETa/ET₀ cap with a coupled gradient. Inference starts
from three observations and advances using predictions alone.

### Matched six-hour study

Ran `OPENBLAS_NUM_THREADS=1 python rollout_study.py --seeds 42 43 44 --steps 6`
on the unchanged 240-hour periodic synthetic fixture. The chronological target
splits remained 165/36/36. All matched variants used 160 training windows, 31
validation windows, and 31 test windows, with six steps per window. Test forecast
origins ran from January 9 11:00 through January 10 17:00 UTC in the dated
synthetic fixture. Every potential complete window in each split was used.
All matched neural variants used the same 3,494 parameters, 30 epochs, batch
size 32, no dropout, 150 updates, and 28,800 training sample-steps per seed.
Teacher variants processed six observed histories per window, with one loss per
history. Unrolled variants processed six recursive proposals per window. All
matched variants selected epochs by projected six-step validation Huber loss.
The historical reference used its original 4,950 training sample-steps and
one-step validation objective so it could reproduce the original behavior.
Seasonal persistence needed zero fallbacks.

Mean per-seed test temperature RMSE in °C, with forecast horizons in hours.

| Method | 1 | 2 | 3 | 4 | 5 | 6 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Persistence | 1.500 | 2.928 | 4.262 | 5.484 | 6.588 | 7.571 |
| Seasonal persistence | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| Historical reference | 0.704 | 3.871 | 24.027 | 50.464 | 63.565 | 65.888 |
| State, raw, teacher | 1.330 | 7.777 | 31.570 | 57.984 | 61.520 | 61.605 |
| State, raw, unrolled | 6.562 | 26.476 | 43.290 | 50.859 | 52.486 | 47.861 |
| State, robust, teacher | 0.730 | 3.115 | 5.413 | 9.046 | 12.521 | 13.405 |
| State, robust, unrolled | 4.339 | 4.833 | 5.279 | 5.250 | 4.983 | 4.084 |
| State, robust, warmup | 2.032 | 3.955 | 4.309 | 4.434 | 4.873 | 4.408 |
| Delta, raw, teacher | 0.531 | 2.730 | 9.497 | 34.465 | 45.134 | 54.010 |
| Delta, raw, unrolled | 5.140 | 19.324 | 33.688 | 51.016 | 55.600 | 54.218 |
| Delta, robust, teacher | 0.568 | 2.196 | 4.887 | 8.388 | 11.283 | 16.690 |
| Delta, robust, unrolled | 2.832 | 3.758 | 3.273 | 4.080 | 4.858 | 4.511 |
| Delta, robust, warmup | 1.253 | 2.806 | 4.009 | 4.136 | 4.582 | 4.874 |
| Delta, smoothing only, warmup | 1.333 | 3.987 | 9.500 | 15.251 | 26.789 | 33.498 |
| Delta, acceleration limit only, warmup | 1.524 | 2.854 | 4.158 | 4.159 | 5.683 | 6.924 |

At six hours the adopted robust delta warmup model scored 4.335, 4.666, and
5.620 °C for seeds 42, 43, and 44. The state robust unrolled variant scored
better at six hours on average and worse in the first five hours. The CLI
uses the warmup delta configuration because it beat ordinary persistence at
all six horizons on average. The test results influenced this choice. New
independent data is needed to confirm that result. The exactly repeating
fixture favors seasonal persistence, which beats every neural variant here.

For seed 42 the historical reference's six-hour temperature RMSE before
projection was 326.077 °C. Its maximum absolute standardized input was 450.70
at hour six. For the adopted model, projected and raw temperature RMSE both
measured 4.335 °C and the maximum standardized input was 6.18. Projection
still changed at least one output in 24 of 31 hour-six forecasts. Raw
unrolled variants remained unstable. Seed-42 raw state training reached a
pre-clipping gradient norm of 9,922 and clipped all 150 updates. The adopted
model peaked at 50.50 and clipped 96 updates.

A separate run with `--steps 24` used 13 shared complete test windows and the
same training and validation procedure. The adopted model averaged 4.908,
15.781, and 46.407 °C temperature RMSE at 6, 12, and 24 hours. Ordinary
persistence averaged 8.152, 10.930, and effectively 0 °C. Long-horizon
forecasts still drift on this fixture. AZMET and calibrated field validation
remain pending.

### Verification

- The initial float64 finite-difference test used absolute-only tolerance and
  flagged two recursive gradients. Their differences were about 3e-7 for
  derivatives of magnitude 20 and 83. The final test uses relative tolerance
  1e-6 with absolute tolerance 1e-7. It covers state and delta modes, teacher
  forcing and recursive feedback, bound saturation and the coupled ET cap.
- The 13-test suite passed, including version 1 and 2 checkpoint forecasts,
  matched unchanged targets, and a seasonal-baseline test that replaces all
  future observations with missing values.
- The 13 tests also passed in the clean virtual environment with deprecation
  warnings treated as errors. The default train and standalone forecast CLIs
  completed a checkpoint round trip, and bytecode compilation passed.

## 2026-11-25 — Phase 3 implementation expansion

Base `b3c0792`. The baseline suite passed 13 tests. Development used
incremental commits on `jupiter-development`. Arduino and PlatformIO
source files were preserved.

### Files and change size

- Historical preparation and chronology. Created `Data/AZMET.py`,
  `Data/WeatherPhysics.py`, `Data/HistoricalFeatures.py`,
  `Data/QualityReport.py`, and `azmet.py`. Updated `Data/Dataset.py`,
  `Data/TimeSeries.py`, and `assumptions.py`.
- Acquisition. Created `Acquisition/MappedPackets.py`, `Acquisition/Hourly.py`,
  `Acquisition/FieldFeatures.py`, and `Acquisition/SpatialObservations.py`.
  Updated `Acquisition/AcquisitionController.py`, `Acquisition/Calibration.py`,
  `Acquisition/SensorData.py`, `Acquisition/SessionManager.py`, and `acquire.py`.
- Spatial reconstruction. Created `Spatial/EnergyFunctional.py`,
  `Spatial/Hamiltonian.py`, `Spatial/MonteCarlo.py`, `Spatial/NoiseGenerator.py`,
  `Spatial/SeedManager.py`, `Spatial/SpatialLoss.py`, `Spatial/Uncertainty.py`,
  and `Spatial/VariationalSolver.py`. Updated `Spatial/SpatialGrid.py`,
  `Spatial/SpatialModel.py`, and `Spatial/Validation.py`.
- Integrated model and artifacts. Created `Pipeline.py`, `CombinedLoss.py`,
  `SpatiotemporalModel.py`, `Utils/Diagnostics.py`, `Utils/Metrics.py`,
  `Utils/ModelLoader.py`, and `Utils/Serialization.py`. Updated
  `Utils/Checkpoints.py` and `train.py`.
- Evolution and verification. Updated `Evolution/Fitness.py`, `config.py`,
  `experiments.py`, `README.md`, and `BUILD_LOG.md`. Created
  `Examples/__init__.py`, `Examples/SyntheticPhase3.py`,
  `Tests/test_phase3_pipeline.py`, `Tests/test_phase3_spatial.py`,
  and `Tests/test_phase3_weather.py`.

Counts since `b3c0792` from Git numstat: 2879 additions and 88 deletions
across all files. Python accounted for 2,677 additions and 88 deletions.
Tracked Python grew from 4,333 lines to 6,922 across 112 files, a net gain of
2,589 lines.

### Verification and experiments

- `OPENBLAS_NUM_THREADS=1 python -W error::DeprecationWarning -m unittest discover -s Tests -q`
  passed 31 tests in the clean virtual environment. Coverage includes
  recursive and spatial finite differences, chronology with changed or absent
  held-out targets, unit conversion, ODE and water balance checks, deterministic
  posterior sampling, Gaussian variance, calibration, byte-preserving replay,
  close-hour and receive-cutoff behavior, and checkpoint round trips. The
  end-to-end test removes its historical source before two matching forecasts.
- Initial test runs exposed an incorrect weather-fixture assumption, a
  row-major seed order assertion, and deprecation errors with newer
  NumPy/pandas datetime arithmetic. The fixture and assertion now compare the
  intended quantities. Duration comparisons and shifts use explicit units.
- `python -m compileall -q .`, `python main.py demo`, and
  `python main.py demo --legacy` passed. The checkpoint format retains
  version 1 and version 2 inference compatibility.
- `python -m Examples.SyntheticPhase3 /tmp/<new-directory>` generated the
  physics-based synthetic history, acquired and replay-tested packets, trained
  a checkpoint, forecast from local readings and generated an independent field
  map. The historical dataset had 44/15/15 target rows and 39/10/10 complete
  six-step windows. Nine field intervals had all six features. The four-epoch
  smoke model had six-hour temperature RMSE of 11.276 °C, versus 6.818 °C for
  persistence and effectively zero for seasonal persistence. The smoke run
  verifies connectivity and does not establish forecast accuracy.
- The synthetic map used four fit sites and one independently labeled test
  site. Variational descent converged in 62 iterations and reduced energy from
  1.13725 to 1.11827. Across four Metropolis chains, post-burn-in acceptance
  ranged from 70.13% to 70.75%. At the single held-out site, absolute errors
  were 0.249 °C (variational), 0.052 °C (IDW) and 0.081 °C (kriging).
  The conditional interval covered the held-out label. One point gives no
  meaningful empirical calibration estimate.
- `python experiments.py <prepared-periodic-fixture.csv> --epochs 30`
  ran on the prior 240-hour synthetic periodic fixture. Both searches scored
  80 candidates, 1,600 candidate epochs and 1,536,000 candidate sample-steps
  before opening the test period. Six-hour temperature RMSE was 17.486 °C
  (fixed), 5.048 °C (classical GA), 5.323 °C (classical amplitude search) and
  7.571 °C (persistence). Actual parameter-step proxies differed because the
  searches sampled different widths.
- Repeating the search with `--candidate-parameter-step-limit 100000000`
  gave both methods a total upper bound of 8.0 billion parameter sample-steps.
  Actual proxies were 5.872 billion (classical) and 5.542 billion (amplitude).
  Six-hour RMSE was 6.820 and 7.613 °C, respectively. This single-seed fixture
  result does not establish an advantage across stations or seasons.

### Remaining work

- Verify an actual AZMET station export, including its headers, station
  metadata, timestamp convention, units and missing codes. Compare proxy ET
  and simulated wetness against independent observations where available.
- Verify AquaMap firmware and LoRa payloads, acquire live SEN0604, SHT45,
  PHYTOS31 and PA1010D streams, and measure actual calibration endpoints.
  The checked-in firmware only provides the existing air and soil text output.
- Validate temporal transfer and spatial reconstruction with real field
  observations, evaluate uncertainty calibration and train or validate joint
  losses only when independent paired field labels are available. GPU validation
  remains pending.
