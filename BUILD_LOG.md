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

## 2026-11-25 — Phase 4 extended development

Started at `75fed9718c38433148832b947b39efecb4251261`. The original suite
passed 31 tests. Verified `4f468ee604ab6b7b8e6f862bfc40770b60b20127` as the
parent of the first implementation commit (`8f11b2c`). Its tree contains
exactly 39 Python files plus AGENTS.md and PLANS.md, in addition to the
original firmware and ancillary files. Published it as `jupiter-foundation`
before editing the development branch. The final tested functional commit is
`064e73be9b75017ccf30cbd95a13019ea0a48b4d`. No merge into main.

### Files and counts

- Numerics and physics. Updated `Numerical/Differentiation.py`,
  `Numerical/ODESolver.py`, and `Physics/Leaf/LeafWetness.py`. Added
  `Numerical/Optimization.py` with Armijo L-BFGS.
- Historical data. Added `Data/StationAnalysis.py`, `station_study.py`,
  `scenario_study.py`, and `Examples/azmet/tucson-2019.json`. Updated
  `azmet.py` to record mapped station provenance.
- Forecasting. Added `Temporal/Direct.py`, `Temporal/DirectCheckpoint.py`,
  `Temporal/ForecastUncertainty.py`, `Temporal/TrainingRecovery.py`,
  `direct_forecast.py`, and `forecast_benchmark.py`. Updated
  `Temporal/RolloutTrainer.py`, `train.py`, and `forecast.py`.
- Spatial and integrated pipeline. Added `Spatial/Geostatistics.py`,
  `Spatial/RobustField.py`, and `multisite.py`. Updated
  `Spatial/Interpolation.py`, `Spatial/MonteCarlo.py`, `Pipeline.py`,
  and `SpatiotemporalModel.py`.
- Search and comparisons. Added `Evolution/Diagnostics.py`,
  `Evolution/Resumable.py`, and `Utils/Bootstrap.py`. Updated
  `Utils/Serialization.py` for cross-platform cleanup and `experiments.py` with
  random search, validation Pareto fronts, and
  generation recovery. Cache hits and physical evaluations are reported
  separately. Resume verifies the data, implementation, recipe and RNG.
- Acquisition. Added `Acquisition/Sequence.py` and `Acquisition/Health.py`.
  Updated `Acquisition/Hourly.py`, `Acquisition/MappedPackets.py`,
  `Acquisition/SerialReader.py`, and `acquire.py`.
- Local application. Added `Dashboard/Jobs.py`, `Dashboard/Server.py`,
  `Dashboard/__init__.py`, and three assets in `Dashboard/static/`
  (`index.html`, `app.js`, `style.css`). HTTP tests exercised the
  train-to-saved-forecast path through the existing CLI.
- Verification and docs. Added `.github/workflows/jupiter-tests.yml` and
  `Tests/test_phase4_acquisition.py`, `test_phase4_dashboard.py`,
  `test_phase4_experiments.py`, `test_phase4_multisite.py`,
  `test_phase4_numerical.py`, `test_phase4_spatial.py`,
  `test_phase4_station.py`, and `test_phase4_temporal.py` under `Tests/`.
  Updated README.md, PLANS.md, and BUILD_LOG.md.

Git numstat reports 2,498 Python insertions and 55 deletions. Tracked Python
grew from 6,922 lines in 112 files to 9,365 lines in 141 files, a net gain
of 2,443. The difference for all tracked files is 3094 insertions and
55 deletions.

### Reproduction and results

- `OPENBLAS_NUM_THREADS=1 python -W error::DeprecationWarning -m unittest discover -s Tests -q`
  passed 54 tests in the clean verification environment. This includes manual
  gate and spatial finite differences, exact CPU recovery with dropout,
  recovered searches matching uninterrupted searches, serial reconnect
  fixtures, strict GPS receipt cutoffs, challenge cases for dashboard request
  validation, and standalone model checkpoints. `node --check Dashboard/static/app.js`
  and `python -m compileall -q .` passed.
- The first Windows run found a temporary file still open during cleanup and
  doubled CSV newlines. After closing the handle on failure and supplying a
  single CSV newline convention, all four Linux and Windows jobs passed on
  Python 3.11 and 3.12 in [GitHub Actions](https://github.com/jaychandrawrites-ai/precision-agriculture-sensor-system/actions/runs/45723252444).
- The [AZMET 2019 Tucson archive](https://azmet.arizona.edu/azmet/data/0119rh.txt)
  contained 8,760 records with 18 fields in the
  [published raw format](https://azmet.arizona.edu/azmet/raw2003.htm).
  SHA-256 `bc4dc71cccf96feaf649336cdc0234d8638ded745ab8273bf5a6f638956c3399`.
  `python station_study.py tucson-study --station tucson-2019.txt Examples/azmet/tucson-2019.json`
  produced 8,737 complete feature rows after the 23-hour trailing ET warmup.
  The first full export attempt failed at a leaf ODE trial stage outside the
  supported temperature range. A 60-second maximum leaf step preserved the
  physical equations and allowed the run to finish. The Hargreaves proxy versus
  archived station-estimated ET₀ had 0.099 mm hourly RMSE and −0.032 mm bias on
  matched hours. The comparison preserved 1,098 negative station estimates.
- `python forecast_benchmark.py tucson-study/az01/hourly.csv tucson-benchmark.json --epochs 8 --seeds 42 43`
  evaluated 1,267 shared test origins and 610 distinct calibration origins.
  Temperature RMSE in °C was as follows.

| Method | 12 h | 24 h | 48 h |
| --- | ---: | ---: | ---: |
| Persistence | 9.729 | 3.901 | 5.156 |
| Seasonal persistence | 3.895 | 3.901 | 5.156 |
| Recursive | 9.864 | 14.179 | 18.718 |
| Gated direct | 7.219 | 4.182 | 5.509 |
| Horizon blend | 7.616 | 5.061 | 5.697 |

The gated direct model’s nominal 90% temperature interval covered 93.8%,
89.7%, and 89.9% at 12, 24, and 48 hours. Mean interval widths were 25.89,
14.09, and 17.82 °C, respectively. The period-derived OOD indicator flagged
410 of 1,267 origins. The results describe one archive, with overlapping
weather windows and separate marginal intervals. Seasonal persistence had
the lowest RMSE at all three reported horizons.

- Synthetic check. Generate the periodic fixture with
  `python -c "from main import synthetic_history; synthetic_history(1152).to_csv('periodic.csv',index=False)"`.
  Run `python forecast_benchmark.py periodic.csv periodic-results.json --epochs 8 --synthetic`.
  The direct model’s 12/24/48-hour temperature RMSE was 8.105/2.849/7.733 °C
  and the recursive model’s was 15.519/18.339/27.072 °C. Seasonal persistence
  was effectively exact on this repeating fixture.
- `python experiments.py periodic.csv --random-baseline --epochs 4 --population 3 --generations 2 --candidate-epochs 3`
  allocated 86,184 training sample-steps per search. Six-hour synthetic
  temperature RMSE was 20.491 °C (fixed and random), 21.765 °C (classical GA),
  and 14.182 °C (classical amplitude search). This small search was a budget
  and workflow check. Running the small parallel cached search twice with
  the same `--search-directory` reproduced all three generation reports.
- On the first 168 archive records, `scenario_study.py` estimated that changing
  crop coefficient from 1.0 to 0.6 reduced hourly ETa by 0.0265 mm on average.
  Changing the rain interception scenario from 0.05 to 0.01 reduced mean
  simulated LWI by 0.0444. These are model sensitivities using common forcing,
  with no observed ETa or LWI labels assumed.

### Remaining work

- Verify the actual AquaMap LoRa receiver and payload, calibration endpoints,
  and device behavior with hardware. Acquisition tests used simulated inputs.
- Broaden the temporal benchmark to independent years and stations. Improve
  seasonal baselines, test equal-compute architectural comparisons, and
  verify forecast reliability on independent periods. Spatial parameter
  uncertainty, adaptive grids, joint fitting on genuine paired field labels,
  and GPU consistency remain open.
- Dashboard verification used loopback HTTP integration and JavaScript syntax
  checking. Browser rendering and physical serial streaming remain to be
  verified.

## 2026-11-25 — Phase 5 integrated research

### Branch and scope

The previous development head was `38190fd65bf8e1f637bf2b5ad5daa23ab83cdf2e`.
Created `jupiter-clean-development` from `main` at `cbd2992`, squashed the
41 original foundation commits into `9c80c96`, and replayed the later
development commits through `758372355bb67e3d3258b2e34636da6ebb627179`.
The clean starting tree had zero differences from `38190fd`. Historical
`jupiter-foundation` at `4f468ee`, historical `jupiter-development` at
`38190fd`, and `main` were preserved. The final functional commit is
`6cff9f0b0777cced5c843ccac677bba61ceed029`. The firmware tree has zero
differences. Independent workers contributed data, spatial, framework,
evolution, acquisition and dashboard subsystems.

Tracked Python grew from 9,365 lines in 141 files to 14,906 lines in 188
files. Git recorded 5,753 Python insertions and 212 deletions, a net gain
of 5,541 lines. The all-file change counts for the current range are
7006 insertions and 346 deletions. Counts exclude external archives,
prepared datasets and machine-generated experiment artifacts.

### Implemented workflows and verification

- Data registry pins five annual releases. Yuma Valley 2016 yielded
  8,761 complete feature rows from 8,784 archive hours. Yuma 2018, Yuma 2019
  and Maricopa 2019 each yielded 8,737 from 8,760. The missing first 23
  feature rows reflect the causal ET warmup. The Tucson mapping remains in
  the catalog. The registry verifies source and prepared artifacts and
  reports seasonal drift and rolling-origin scores without pooling stations.
- The hourly FAO-56 implementation reproduces the published night and day
  example to 0.01 mm/hour. An optional Yuma 2019 comparison on 8,737 matched
  hours gave 0.09593 mm hourly RMS difference and +0.03049 mm FAO-minus-
  Hargreaves mean difference. The 2 m input wind height and 0.8 dark
  cloud ratio were explicitly declared comparison assumptions. This
  compares two estimates. The default six-feature pipeline was preserved.
- Eight stratified scenarios on the first 168 Yuma 2019 hours propagated
  declared crop-coefficient and rain-interception ranges. Spearman
  correlations between mean estimated ETa and crop coefficient, and
  between mean simulated LWI and rain interception, were both +1.0
  in this small modeled example. The other rank correlations reflect the
  finite stratified design. This does not establish field uncertainty.
- The experimental direct model shares the loss and checkpoint workflow
  across gated, GRU, causal convolution and attention variants. The existing
  derivative-based default remains available. Numerical additions include
  tensor adjoints, sparse operators and PCG, Dormand–Prince RK45,
  directional Hessian products, Adagrad and weighted gradient accumulation.
- Structured search compares fixed, random, genetic, adapted amplitude
  and hybrid candidates with checkpoint compatibility. The synthetic
  three-seed study used 24 candidates per searched method, at most 6,000
  parameters and a shared 16-million parameter-sample-step cap per candidate.
  Actual costs varied. At six hours, temperature RMSE for seeds 42/43/44
  was 6.717/9.524/10.535 °C for random, 15.466/10.317/7.939 °C for the
  GA, 5.479/6.535/7.171 °C for amplitude, and 5.479/10.559/11.072 °C
  for hybrid. The short fixed reference ran at 27.955/24.115/21.997 °C.
  This synthetic ranking is not evidence of general advantage.
- On the explicitly synthetic spatial experiment, 24 sites supplied fit data
  and 48 independently generated sites were held out. RMSE was 0.4483 for
  affine-drift kriging, 0.4934 for variational reconstruction, 0.5067 for
  the block posterior mean, 0.5211 for ordinary kriging, 0.6161 for nearest
  neighbor, 0.6449 for IDW and 0.8282 for RBF. Nominal 90% interval
  coverage was 47/48 with mean width 2.003. The short chains flagged
  19 and 24 of 72 tested cells under two settings. This is diagnostic.
- Acquisition now preserves verifiable raw archives and offers recovery into
  a new destination, with coverage and calibrated reading diagnostics.
  Multi-site forecasts can propagate aligned ensemble draws through IDW.
  Forecast maps stay labeled as model outputs. No paired field observations
  were available to validate joint inference.
- The loopback dashboard supports persistent jobs, moving from prepared
  datasets to model selection, session health, benchmark and search jobs,
  bounded previews and CSV exports. Actual loopback subprocess tests and
  JavaScript syntax checks passed. The managed browser relay timed out.
- The local full suite ran **116 tests**, with **114 passes and two CUDA
  skips**. Bytecode compilation, JavaScript syntax validation and the
  Phase 3 synthetic end-to-end demo passed. Cross-platform CI evidence is
  recorded below.
- All four GitHub Actions jobs passed at the final functional commit on
  Linux and Windows with Python 3.11 and 3.12.
  [Verification matrix](https://github.com/jaychandrawrites-ai/precision-agriculture-sensor-system/actions/runs/45757254102)

### Independent forecasting results

Yuma 2018 was split into fitting, selection and calibration before Yuma
2019 testing. The fixed-year Yuma and frozen Maricopa comparisons each
used 8,663 complete shared origins. There were 36 non-overlapping
calibration origins and 121 spaced Yuma verification origins.

| Archive and model | 1 h | 12 h | 24 h | 48 h | 72 h |
| --- | ---: | ---: | ---: | ---: | ---: |
| Yuma daily persistence | 2.802 | 2.812 | 2.800 | 3.470 | 3.821 |
| Yuma direct | 4.552 | 6.821 | 4.615 | 3.986 | 4.878 |
| Yuma validation blend | 2.842 | 2.812 | 2.800 | 3.428 | 3.831 |
| Maricopa daily persistence | 3.032 | 3.044 | 3.038 | 3.800 | 4.086 |
| Maricopa frozen Yuma blend | 3.132 | 3.038 | 3.038 | 3.779 | 4.081 |

Table entries are temperature RMSE in °C. Ordinary persistence gave
1.728 °C at one hour in Yuma and 1.691 °C in Maricopa. The unblended
daily-residual network also trailed daily persistence at the reported
leads. The Yuma blend's approximate adjusted 48-hour paired interval
was [−0.057, −0.028] °C for blend minus daily RMSE. The corresponding
Maricopa interval [−0.042, +0.002] °C included zero. The 72-hour intervals
included zero at both stations. This indicates limited, horizon-dependent
benefit and no clear long-range advantage. For Yuma, the blend's nominal
90% empirical interval covered 87.6% for 24-hour temperature and 91.7%
for all temperature leads together on the spaced sample. The reported
coverage does not imply guarantees across features, stations or weather
regimes.

The separate expanding-window Yuma study used 8,330 shared origins in
four test blocks. At 48 and 72 hours, harmonic regression scored
3.370 and 3.365 °C. Daily persistence scored 3.469 and 3.811 °C.
This study had a longer training span and a different rolling protocol.

### Reproduction

From the repository root on Linux, with outputs outside the checkout:

```sh
mkdir -p /tmp/jupiter5
curl -fL https://azmet.arizona.edu/azmet/data/0218rh.txt -o /tmp/jupiter5/yuma18.txt
curl -fL https://azmet.arizona.edu/azmet/data/0219rh.txt -o /tmp/jupiter5/yuma19.txt
curl -fL https://azmet.arizona.edu/azmet/data/0619rh.txt -o /tmp/jupiter5/maricopa19.txt
python station_study.py /tmp/jupiter5/y18 --station /tmp/jupiter5/yuma18.txt Examples/azmet/yuma-valley-2018.json
python station_study.py /tmp/jupiter5/y19 --station /tmp/jupiter5/yuma19.txt Examples/azmet/yuma-valley-2019.json
python station_study.py /tmp/jupiter5/m19 --station /tmp/jupiter5/maricopa19.txt Examples/azmet/maricopa-2019.json
OPENBLAS_NUM_THREADS=1 python seasonal_study.py /tmp/jupiter5/yuma-study.json /tmp/jupiter5/y18/az02/hourly.csv /tmp/jupiter5/y19/az02/hourly.csv --station az02 --selection-from 2018-07-01T07:00Z --calibration-from 2018-09-15T07:00Z --final-from 2019-01-01T07:00Z --horizon 72 --epochs 12 --seeds 42 43 44 --model-output /tmp/jupiter5/model
OPENBLAS_NUM_THREADS=1 python seasonal_transfer.py /tmp/jupiter5/m19/az06/hourly.csv /tmp/jupiter5/model/seasonal-ensemble.json /tmp/jupiter5/transfer.json --station az06
OPENBLAS_NUM_THREADS=1 python rolling_station_study.py /tmp/jupiter5/rolling.json /tmp/jupiter5/y18/az02/hourly.csv /tmp/jupiter5/y19/az02/hourly.csv --initial-hours 8760 --test-hours 2160 --horizon 72
python et_comparison.py /tmp/jupiter5/yuma19.txt Examples/azmet/yuma-valley-2019.json /tmp/jupiter5/et.json --wind-height 2 --dark-cloud-ratio .8
head -n 168 /tmp/jupiter5/yuma19.txt >/tmp/jupiter5/yuma-week.txt
printf '%s\n' '[{"path":"soil.crop_coefficient","lower":0.6,"upper":1.1},{"path":"scenario.rain_interception_fraction","lower":0.01,"upper":0.1}]' >/tmp/jupiter5/ranges.json
python physical_ensemble.py /tmp/jupiter5/yuma-week.txt Examples/azmet/yuma-valley-2019.json /tmp/jupiter5/ranges.json /tmp/jupiter5/physical.json --draws 8
OPENBLAS_NUM_THREADS=1 python spatial_benchmark.py --synthetic --output /tmp/jupiter5/spatial.json
python -c "from main import synthetic_history; synthetic_history(360).to_csv('/tmp/jupiter5/search-source.csv', index=False)"
for seed in 42 43 44; do
  OPENBLAS_NUM_THREADS=1 python experiments.py /tmp/jupiter5/search-source.csv --epochs 8 --population 6 --generations 4 --candidate-epochs 4 --candidate-parameter-step-limit 16000000 --max-parameters 6000 --extended-space --adaptive-amplitude --random-baseline --hybrid-baseline --synthetic --seed "$seed" >"/tmp/jupiter5/search-$seed.json"
done
OPENBLAS_NUM_THREADS=1 python -W error::DeprecationWarning -m unittest discover -s Tests -q
python -m compileall -q .
node --check Dashboard/static/app.js
OPENBLAS_NUM_THREADS=1 python -m Examples.SyntheticPhase3 /tmp/jupiter5/offline-demo
```

### Remaining priorities

Predeclare new independent stations and years. Compare further temporal
models against daily persistence and expanding harmonic fits with matched
compute. Verify the native AquaMap protocol and calibration with hardware.
Use genuine paired field measurements to validate spatial and joint
predictive intervals. Browser rendering, GPU consistency, and adaptive
grids remain open. See [PLANS.md](PLANS.md) for the next-run checklist.

### File inventory

Status relative to the verified clean starting tree:

```text
M .github/workflows/jupiter-tests.yml
A ARCHITECTURE.md
M Acquisition/AcquisitionController.py
A Acquisition/Archives.py
A Acquisition/CalibrationReport.py
A Acquisition/Coverage.py
M Acquisition/DataLogger.py
M Acquisition/MappedPackets.py
A Acquisition/Recovery.py
M Acquisition/SessionManager.py
A DATA_FORMAT.md
M Dashboard/Jobs.py
A Dashboard/Reports.py
M Dashboard/Server.py
M Dashboard/static/app.js
M Dashboard/static/index.html
M Dashboard/static/style.css
A Data/Registry.py
A Data/RollingOrigin.py
M Data/RolloutWindows.py
A Data/StationCatalog.py
A Data/Studies.py
A EXPERIMENT_PROTOCOL.md
M Evolution/Fitness.py
M Evolution/Genome.py
M Evolution/Resumable.py
A Examples/azmet/catalog.json
A Examples/azmet/maricopa-2019.json
M Examples/azmet/tucson-2019.json
A Examples/azmet/yuma-valley-2016.json
A Examples/azmet/yuma-valley-2018.json
A Examples/azmet/yuma-valley-2019.json
M Framework/Math_Utilities.py
M Framework/Optimizer.py
A Framework/SequenceLayers.py
A HARDWARE.md
A LIMITATIONS.md
A MATHEMATICS.md
M Numerical/Differentiation.py
A Numerical/LinearAlgebra.py
M Numerical/ODESolver.py
A PHYSICS.md
M PLANS.md
A Physics/Evapotranspiration/ETComparison.py
M Physics/Evapotranspiration/FAO56.py
A QuantumInspired/Adaptive.py
M QuantumInspired/QuantumGenome.py
M README.md
A Spatial/BlockSampler.py
A Spatial/GaussianField.py
M Spatial/Interpolation.py
A Spatial/Kriging.py
A Spatial/PosteriorStudy.py
M Spatial/SeedManager.py
M Spatial/SpatialModel.py
M Spatial/VariationalSolver.py
A Spatial/VariogramFit.py
A Temporal/BlockIntervals.py
M Temporal/Direct.py
M Temporal/DirectCheckpoint.py
M Temporal/Model.py
A Temporal/SeasonalCheckpoint.py
A Temporal/Seasonality.py
A Temporal/SequenceForecast.py
M Temporal/TrainingRecovery.py
A Tests/test_phase5_acquisition.py
A Tests/test_phase5_backend_equivalence.py
A Tests/test_phase5_dashboard.py
A Tests/test_phase5_evolution.py
A Tests/test_phase5_integrators.py
A Tests/test_phase5_linear_algebra.py
A Tests/test_phase5_optimizers.py
A Tests/test_phase5_physics.py
A Tests/test_phase5_registry.py
A Tests/test_phase5_rolling.py
A Tests/test_phase5_seasonal.py
A Tests/test_phase5_sequence_forecast.py
A Tests/test_phase5_sequence_layers.py
A Tests/test_phase5_spatial.py
A Tests/test_phase5_spatial_forecast.py
M Utils/Bootstrap.py
M Utils/Checkpoints.py
A VALIDATION.md
M acquire.py
A acquisition_archive.py
A dataset_registry.py
M direct_forecast.py
A et_comparison.py
M experiments.py
A physical_ensemble.py
A rolling_station_study.py
A seasonal_forecast.py
A seasonal_study.py
A seasonal_transfer.py
A spatial_benchmark.py
A spatial_forecast.py
M BUILD_LOG.md
```

## 2026-11-25 — Foundation and layout audit

Verified active `jupiter-clean-development` at Phase 5 head `a6da6c4`.
The consolidated foundation is `9c80c96`, whose parent is `main` at
`cbd2992`. Its tree matches the archived `4f468ee` tree. The archive
contains 41 original Jupiter 01–10 commits and remains in the ancestry of
historical `jupiter-development`. None of those 41 commit IDs or individual
subjects is in the active branch's ancestry. All 37 subsequent implementation
commits through historical `38190fd` replay with matching commit messages and
per-commit trees on the clean branch. No branch reconstruction was necessary.

Reviewed the tracked foundation and current source paths. Module placement
already matches the functional layout in ARCHITECTURE.md. Root-level
entry points, shared configuration, and integration interfaces remain in
place. The pre-existing empty Arduino-named marker is tracked in main and
was preserved alongside the PlatformIO firmware. No tracked cache or
temporary build artifacts were found.

Moved the duplicated latest-three-hour forecast eligibility logic into
`Data/ForecastHistory.py` for the recursive, direct, seasonal and field-session
paths. Added the offline `Pipeline.py check-session` report with optional
checkpoint physics compatibility. It requires no training or network
access and contains no coordinates. Added regressions for hourly gaps,
invalid feature values and the transition through the synthetic 26-hour
default warmup and forecast context. Checkpoint formats remain unchanged.

`OPENBLAS_NUM_THREADS=1 python -W error::DeprecationWarning -m unittest discover -s Tests -q`
ran 118 tests with 116 passing and two optional CUDA checks skipped.
Live hardware and calibration remain to be verified on the field equipment.

## 2026-11-25 — Phase 6 field workflow and scientific extensions

Started at `74b0229` on `jupiter-clean-development`. Pushed the AI Trace
milestone at `ef6805b` before continuing with optional modeling. The final
functional commit is `2cb6f9f`. The archival branches and original Arduino
firmware retain their previous trees. The cumulative work is tracked in
MASTER_REQUIREMENTS.md and DEVELOPMENT_STATUS.md.

- AI Trace records separate visits with SQLite, raw-first storage,
  restarts, durable recovery and portable checksummed archives. It supports
  five-location synthetic survey capture, recording stops with the receiver
  connected, revisits, serial interruption and reconnect, explicit manual
  coordinates, mapped CSV import and location/segment/survey exports.
- Observational maps retain a bounded observation window and individual
  sample timestamps. Forecast readiness and inference work independently
  on closed visits. Point forecast maps require the same issue and valid
  time and reject GPS fixes received after issuance. The synthetic demo
  mapped five forecast-eligible locations and produced separate observed and
  three-hour forecast maps. It provides software evidence only.
- Experimental LSTM and LSTM-attention models passed finite-difference and
  checkpoint tests. The baseline remains available. Optional physical
  sampling passed marginal-strata and positive-definite matrix checks with
  a declared correlation. Adaptive kriging passed cell-area, budget, mask
  and variance tests on synthetic coordinates.

```sh
OPENBLAS_NUM_THREADS=1 python -W error::DeprecationWarning -m unittest discover -s Tests -q
node --check Dashboard/static/trace.js
python -m compileall -q .
python -m Examples.SyntheticTrace /tmp/jupiter-trace-synthetic
python -m Dashboard.Server --workspace ./jupiter-field --allow-port YOUR_SERIAL_PORT
```

The full suite ran 131 tests with 129 passing and two optional CUDA checks
skipped. The AI Trace dashboard milestone passed all four Linux and Windows
CI jobs. [Milestone CI](https://github.com/jaychandrawrites-ai/precision-agriculture-sensor-system/actions/runs/45764199983)
Dashboard verification exercised actual loopback routes with a synthetic
serial connector and checked JavaScript syntax. Browser rendering and live
instrument behavior remain unverified.

Python grew to 16,867 tracked lines across 205 files, with 1,889 insertions
and 16 deletions. Native AquaMap verification, calibration with real
instruments, the specified newer Yuma export, broader sequence architecture
search and forecast-map uncertainty propagation remain open.

## 2026-11-25 — Final checklist checkpoint

Implemented and pushed four functional commits on `jupiter-clean-development`.

- `b203519` added aligned saved-model ensembles to AI Trace, a command to
  generate time-indexed uncertainty maps, and an offline five-site fixture.
  Changed `Acquisition/TraceAnalysis.py`, `trace_analysis.py` and
  `Examples/SyntheticTrace.py`. Added `Tests/test_trace_ensemble.py`.
- `04e7332` connected portable direct and sequence checkpoints to the
  shared forecasting entry point. Changed `Temporal/DirectCheckpoint.py`,
  `Utils/ModelLoader.py` and `Tests/test_phase5_sequence_forecast.py`.
- `8e14557` integrated fit-only adaptive kriging in `Pipeline.py` and
  `adaptive_map.py`. `Tests/test_adaptive_grid.py` checks that a held-out
  outlier cannot affect predictions.
- `df5915b` exported the shared interpolation weights and verified exact
  reconstruction of ensemble statistics. Changed `spatial_forecast.py`,
  `Acquisition/TraceAnalysis.py` and `Tests/test_trace_ensemble.py`.

Functional diff from `9f33690` added 220 Python lines and deleted 28.
The resulting tree contains 17,059 lines across 206 tracked Python files.
This documentation update added `FINAL_DEVELOPMENT_CHECKLIST.md` and
updated `DEVELOPMENT_STATUS.md` and this log.

Verification from the repository root:

```sh
python -W error::DeprecationWarning -m unittest discover -s Tests -q
node --check Dashboard/static/trace.js
python -m Examples.SyntheticTrace /tmp/jupiter-checklist-demo --ensemble
```

The full suite ran 133 tests, with 131 passes and two optional CUDA skips.
All four Linux and Windows jobs passed on Python 3.11 and 3.12 at
`df5915b`. A fresh single-branch clone and virtual environment installed
`requirements.txt` and ran the offline demo successfully.
[CI matrix](https://github.com/jaychandrawrites-ai/precision-agriculture-sensor-system/actions/runs/45767131538)

The demo exported two members across five synthetic sites, six hourly leads
and six features. The test reproduced the mean and spread on each 25 × 25
grid directly from the saved member values and interpolation weights.
Intervals summarize checkpoint disagreement. Their predictive coverage in a
real field is unknown. No new measured-field accuracy estimates were made.
There were no failures in the final CPU suite or the clean-install smoke test.

Open work and external dependencies are indexed by ID in the final checklist.
Browser interactions and physical receiver evaluation remain unverified.
The original firmware and historical branches were preserved.

## 2026-11-25 — Phase 7 software checkpoint

Work continued from `ec21e8e` on `jupiter-clean-development`.
The functional commits changed 63 files, with 3,598 Python lines added
and 174 deleted (3,424 net). Python totals reached 20,483 lines in 230 files.

- `e31c48e` integrated survey adaptive kriging, keeping shared geometry,
  exported trajectories, boundary flags and the IDW baseline.
- `16920c2` added the paired joint optimizer, a synthetic study, an exact
  CPU recovery test and finite differences for the combined gradient.
- `b14fdad` added trained ordered quantiles, validation-only convex weights
  and non-overlapping whole-trajectory residual draws.
- `d2b3d77` added receiver diagnostics, browser tests and research jobs.
  Chromium uncovered a checksum mismatch when the client reserialized
  archive numbers. Upload now preserves original bytes. The regression
  also verifies that a duplicate archive is refused and a fresh one restores.
- `07f2500` added saved-model physical scenarios, sensitivity reports,
  balance audits and independently paired observation tooling.
- `fe27472` added finite-volume adaptive variational reconstruction, field
  holes, covariance ensembles and held-out predictive checks.
- `cac9c0e` added portable sequence search candidates, matched-window
  evaluation, memory controls and checkpoint tests.
- `9422265` corrected deprecated pandas time conversions found by the
  warnings-as-errors suite and wired saved direct models into dashboard
  forecasts.

Reproduction from the repository root:

```sh
OPENBLAS_NUM_THREADS=1 python -W error::DeprecationWarning -m unittest discover -s Tests -q
JUPITER_TRACE_BROWSER=1 python -m unittest Tests.test_trace_browser -q
python -m Examples.SyntheticJoint /tmp/jupiter-joint.json --epochs 160 --seed 82
python quantile_forecast.py hourly.csv quantile.json quantile-report.json --epochs 3
```

The browser test uses Playwright with Chromium and an isolated synthetic
receiver. It covers desktop and narrow layouts, survey recording and revisit,
archive export and restore, readiness and ensemble maps.

The seed 82 joint study used 160 updates per model. Held-out site temperature
RMSE fell from 1.789 to 1.173 °C and relative-humidity RMSE fell from 10.033
to 7.053 percentage points. Leaf wetness RMSE rose from 0.0244 to 0.0299
and reference ET RMSE rose from 0.00769 to 0.02620 mm per interval.
This one synthetic fixture cannot establish field performance.

The three-epoch Yuma Valley 2019 quantile smoke study used 6,101 fitting
origins and 1,309 each for validation and testing. Nominal 90% temperature
coverage ranged from 95.8% at one hour to 84.2% at six hours. Simultaneous
coverage over all six leads and six features was 26.4%. Marginal quantile
training alone gave limited simultaneous coverage.

The final development checklist records unresolved software, external
validation inputs and merge approval. CUDA equivalence remains unverified.

The local full suite on `9422265` ran 153 tests with Chromium enabled.
151 passed and two optional CUDA tests skipped. Both dashboard scripts
also passed JavaScript syntax checks. [Cross-platform CI run](https://github.com/jaychandrawrites-ai/precision-agriculture-sensor-system/actions/runs/45770802014)

## 2026-11-26 — Phase 8 integration checkpoint

Started from `eefbb97` on `jupiter-clean-development`. Functional changes
through `0f11a03` added 1,914 lines and deleted 100 across 37 files. Python
accounted for 1,851 additions and 80 deletions, yielding 22,254 Python lines
in 243 files.

| Commit | Change |
| --- | --- |
| 1ee1eeb | Coherent covariance draws, aligned trajectory residuals and frozen variational meshes |
| bebd0c0 | AI Trace predictive map display |
| 3f59750 | Pareto ranking, crowding, compute objectives and recovered candidate artifacts |
| 5eae3a6 | Physical scenarios from closed visits and end-to-end synthetic survey example |
| c72ec1a | Quantile checkpoint inference and dashboard reports |
| ec12d9d | Reliability profiles and checkpoint horizon verification |
| 4ab2456 | Masked field exports, reliability checks and research console browser regression |
| b9dbf7f | Fix horizontal overflow observed at 390 px |
| 0f11a03 | Joint and marginal quantile calibration on separated blocks |

The full CPU suite ran 165 tests, including actual Chromium for AI Trace
and the research console. 163 passed and two CUDA tests skipped.
A fresh checkout installed requirements and generated the offline survey
exports. [Phase 8 CI matrix](https://github.com/jaychandrawrites-ai/precision-agriculture-sensor-system/actions/runs/45772897549)

The five-site synthetic survey exported 32 predictive draws across six
leads, 16 cells and six features. Shared calibration and spatial
dependence are modeled assumptions. Actual field coverage is unknown.

The Yuma Valley 2019 quantile reanalysis split the previously inspected
test period into 90 spaced calibration and 97 evaluation origins.
Simultaneous coverage was 24.7% raw, 32.0% with marginal expansion and
84.5% with simultaneous expansion at nominal 90%. The mean one-hour
temperature interval widened from 7.88 to 21.72 °C. Its interval score
increased from 8.88 to 21.72 °C. Independent station-year replication
is still needed.

A 72-hour reliability profile calibrated on 36 spaced Yuma 2018 origins
accepted 71 of 119 Maricopa 2019 origins. Accepted-origin temperature
RMSE was 3.64 °C at 72 hours versus 3.81 °C across all complete origins.
Accepted-origin error was higher at 1, 24 and 48 hours. This compares
different subsets of weather and gives no general reliability claim.

The three-seed synthetic architecture experiment used shared splits and
12 candidate evaluations per search method. Aggregate work varied from
44.8 to 60.9 million parameter-sample steps under equal per-candidate caps.
Six-hour RMSE was 6.15–8.19 °C for amplitude and 6.15–8.07 °C for hybrid
search. Daily persistence was exact on the deliberately periodic fixture.
Amplitude search is classical computation.

Reproduction:

```sh
python -m Dashboard.Server --workspace ./jupiter-dashboard
python -m Examples.SyntheticUncertainty /tmp/jupiter-uncertainty
python -m Examples.SyntheticJoint /tmp/jupiter-joint.json --epochs 160 --seed 82
JUPITER_TRACE_BROWSER=1 OPENBLAS_NUM_THREADS=1 python -W error::DeprecationWarning -m unittest discover -s Tests -q
python calibrate_quantiles.py hourly.csv quantile.json calibration.json --calibration-start 2019-11-07T14:00 --test-start 2019-12-04T00:00 --stride 7
```

External hardware, modern Yuma export, paired field observations, CUDA and
release approval remain outstanding. For independent experiments, freeze
the checkpoint and thresholds before acquiring a new station-year.
Predeclare a six-hour study with seven-hour spacing and a 72-hour transfer
study with 73-hour spacing. Compare coverage and width on the same origins.

## 2026-11-26 — Phase 9 survey milestone

Started at `86eed5b`. The requested runbook was absent in both the checkout
and remote branch history, so `EXECUTION_RUNBOOK.md` now records the resume
and verification procedure.

`2be8e9e` adds selectable IDW, fixed variational and adaptive variational
propagation of shared checkpoint, physical and residual trajectories.
The solver keeps one mesh and applies its weights to every draw. Covariance
scenarios remain available through kriging. Other operators explicitly
reject that combination. Cell-wise batched evaluation and summaries passed
chunk-size equivalence checks. Dense draw storage remains budget-limited.

`f986d24` exports observational conditioning diagnostics and the sample
variance decomposition within and between drawn covariance scenarios.
Thirteen focused spatial tests passed, including reproducibility and the
empirical total-variance identity. Measured field validation remains open.

The Chromium-enabled full suite at `2be8e9e` ran 166 tests, with 164
passes and two CUDA skips. The subsequent variance diagnostic passed
all 13 focused spatial tests at `f986d24`.

### Phase 9 continuation

- `6f0cec2` adds `trace_analysis.py ensemble-map --disk-backed`, which stages
  the predictive draw cube in a temporary memory-mapped NumPy file and
  streams it into the existing NPZ. Exact summaries are calculated in cell
  chunks sized to the working-memory budget. A regression verified that
  the mapped export matches dense output under a budget that rejected the
  dense cube. The CLI also ran successfully on the synthetic survey.
  Kriging still needs an in-memory covariance factorization.
- `fd6c85b` enables sequence search checkpoint selection in CSV and field
  forecast jobs. The loopback test trained a sequence search and served
  a saved random-search winner.
- `020f81e` displays Pareto objectives and verifies model selection and
  forecasting in Chromium. Verified R042 and R067.

The complete local suite at `fd6c85b` ran 168 tests with Chromium enabled.
166 passed and two CUDA tests skipped. The expanded Chromium search test
passed separately at `020f81e`. The checklist now records 37 verified,
42 partial, one unstarted and 16 externally blocked requirements.
