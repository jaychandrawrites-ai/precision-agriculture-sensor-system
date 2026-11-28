# AI validation and local deployment

| Requirement | Evidence and status |
| --- | --- |
| Variable duration and horizon | Adjacent hourly context and elapsed-time input utilities are available. Tested 24-hour synthetic inference. High-frequency model training remains open. |
| CSV and GPS | Single and batch import, explicit timezone, stationary GPS, and separate visits are implemented. Real-file import was verified previously. |
| Parallel acquisition | Simulated independent receivers and group controls are covered. Physical verification remains open. |
| Spatial mapping | Circular diffusion and explicitly non-simultaneous field composites are implemented and covered by existing tests. |
| Calibration and readiness | Explicit calibration and compatible hourly context required for field forecasts. No independently calibrated real forecast was attempted in this run. |
| Local training | Four CPU epochs on the prepared az43 export. Chronological rollout windows and daily persistence comparison. |
| Diagnostics page | New loopback page loads saved Matplotlib artifacts after restart. |
| Internal diagnostics | Saved weight distributions and global peak gradient and clipping counters are available. Per-layer activations, gradient norms and update magnitudes remain open. |
| Deployment | Linux loopback browser exercised. Windows and CUDA hardware execution still require verification. |

## Feature compatibility

Both pathways preserve the feature order below. Hourly bins are UTC
interval ends. The input scaler and target scaler are fitted on training
data. Missing or nonfinite features make a forecast window ineligible.
AquaMap GPS is used for mapping only.

| Feature and unit | Historical AZMET | AquaMap field |
| --- | --- | --- |
| Air temperature, °C | Mapped weather column | Hourly sensor aggregate |
| Relative humidity, % | Mapped weather column | Hourly sensor aggregate |
| Leaf wetness, 0–1 | Leaf energy and water balance estimate | Voltage mapped through measured wet/dry endpoints |
| Reference ET, mm per hour | Causal Hargreaves estimate from recent temperature | Same method and history requirement |
| Actual ET, mm per hour | Soil water bucket driven by declared forcing | Water stress estimated from calibrated soil moisture |
| Cotton heat stress, index | Modeled heat accumulation and recovery | Same index with field moisture-dependent canopy estimate |

The common units and equations do not establish interchangeability between
estimated station features and field features. The az43 export lacked
precipitation. Prepared data used a declared dry forcing scenario and
retained missingness flags. Field calibration must come from independent
measurement.

## Numerical audit scope

- `Data/Dataset.py`, `Data/Normalization.py` and `Data/RolloutWindows.py`
  construct chronological targets, exclude windows crossing a split or a
  missing hour, and fit scales on training data.
- `Temporal/Rollout.py` feeds projected six-feature predictions back
  through a causal derivative window. `Tests/test_rollout.py` checks
  manually propagated gradients against finite differences.
- `Framework/Layers.py` and `Framework/SequenceLayers.py` own parameter
  initialization, activations and sequence tensor contracts. Sequence-layer
  and sequence-evolution regressions check masks, dilation and checkpoint
  inference.
- `Temporal/RolloutTrainer.py` applies sample-weighted training loss,
  validation selection, cosine scheduling and gradient clipping.
  `Temporal/TrainingRecovery.py` persists the epoch boundary and RNG
  state. Timing and learning-rate traces are currently omitted for resumed
  runs because prior instrumentation history is unavailable.
- `Utils/Checkpoints.py` and `Utils/ModelLoader.py` preserve portable
  hourly inference. CPU/GPU agreement was not verified on hardware.

This review and the regression suite provide targeted checks, not an
exhaustive proof of numerical correctness.

## Reproduce locally

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe azmet.py prepare supplied-az43.csv Examples/azmet/yuma-valley-az43.json az43-prepared
.\.venv\Scripts\python.exe train.py az43-prepared/hourly.csv az43-model.json --epochs 4 --seed 42 --preparation-report az43-prepared/report.json --diagnostics az43-diagnostics
.\.venv\Scripts\python.exe -m Dashboard.Server --workspace "$env:TEMP\jupiter-dashboard"
```

To see saved diagnostics in the website, prepare or upload eligible
historical data and submit **Model training** from the console, then open
**AI diagnostics**. CLI training exports its artifacts to the chosen
directory independently of the dashboard job store.

## Supported time combinations

| Visit duration | Hourly checkpoint eligibility |
| --- | --- |
| 15 or 30 minutes | Requires a usable closed hourly field bin under the configured coverage rules and suitable prior context. A short visit alone cannot establish full-hour coverage. |
| About one hour | Can supply a closed field aggregate. Adjacent compatible history and calibration are still required. |
| Multiple hours | Can supply consecutive hourly aggregates if each has eligible observations and enough prior history for reference ET. |
| Arbitrary subhourly forecast | No trained native subhourly checkpoint is available. |

Direct, quantile and blend checkpoints enforce their saved output horizon.
Recursive checkpoints expose the software limit separately from validation.
The existing 24-lead synthetic test demonstrates inference mechanics.
It does not validate accuracy at that lead.

## Development experiment

The az43 chronological comparison used seeds 42 and 43, four epochs,
width 16 and six leads. Selection, calibration and reporting began
2025-07-01, 2025-10-01 and 2026-02-01 UTC. The archive overlaps
previously examined data, so this is development evidence.

| Method | Six-hour temperature RMSE (°C) |
| --- | ---: |
| Persistence | 7.552 |
| Daily persistence | 2.662 |
| Harmonic regression | 3.892 |
| Direct neural forecast | 4.401 |
| Daily residual | 3.041 |
| Validation-weighted blend | 2.665 |

The residual model used the daily anchor and improved over the direct
model; daily persistence remains preferable in this comparison. For
each feature and lead the study exports RMSE, MAE, bias, R² and skill.

```powershell
.\.venv\Scripts\python.exe seasonal_study.py seasonal-development.json az43-prepared/hourly.csv --selection-from 2025-07-01T00:00Z --calibration-from 2025-10-01T00:00Z --final-from 2026-02-01T00:00Z --station az43 --horizon 6 --width 16 --epochs 4 --seeds 42 43
```

## Persistent runs

```powershell
New-Item -ItemType Directory -Force training/az43-initial
.\.venv\Scripts\python.exe train.py az43-prepared/hourly.csv training/az43-initial/model.json --epochs 4 --seed 42 --diagnostic-every 1 --preparation-report az43-prepared/report.json --diagnostics training/az43-initial/diagnostics --resume-state training/az43-initial/resume.json > training/az43-initial/run.json
.\.venv\Scripts\python.exe -m Dashboard.Server --workspace "$env:LOCALAPPDATA\Jupiter" --import-training-run training/az43-initial
```

For a longer CPU run choose a new directory and specify `--epochs 30
--diagnostic-every 5` from the start. A resume requires the same settings.
Sampling records the first batch of selected epochs, before clipping
for gradients and after the update for inference activations. Histograms
use bounded samples. The imported checkpoint hash must match its report.
The dashboard copies it into its local job store and can reload it later
without retraining.

Real inference still requires independent leaf wet and dry voltage
endpoints, a soil calibration tied to reference volumetric measurements,
the ET solar offset and compatible adjacent hourly feature context.
Use manufacturer procedures or an approved lab protocol for calibration.
The five field records remain inference-only. An AZMET weather series
cannot provide measured soil moisture or leaf calibration for those visits.

## Hourly forecast contract and evolutionary deployment

Forecast requests are integers from 1 through 72, with exactly one hour
per step and six physical features. A shorter fixed-output checkpoint
advertises its cap. Recursive checkpoints separately report the maximum
evaluated horizon when known. A legacy recursive model remains callable
through 72, but its historical evaluation may cover only shorter leads.
One-hour field visits can supply an eligible hourly anchor. The derivative
model also needs two preceding consecutive state rows and the independently
configured reference ET history. Native sub-hour forecasting is outside
this specification.

Recursive search can save its final refits directly in dashboard training
run format, with checkpoint, prediction arrays, loss and internal layer
plots, generation plots and numerical statistics. See [HANDOVER.md](HANDOVER.md)
for commands and [MATHEMATICS.md](MATHEMATICS.md) for the algorithm.

The 72-hour seasonal comparison used the previously examined az43 archive,
with July 1 and October 1, 2025 and February 1, 2026 UTC boundaries and two
epochs each for seeds 42 and 43. Six-hour temperature RMSE (°C) was 7.559
for persistence, 2.651 for daily persistence, 3.901 for harmonic regression,
4.737 for direct AI, 3.573 for daily-residual AI and 2.651 for the validation
blend. At hour 72 the corresponding errors were 3.856, 3.856, 3.894, 4.200,
4.170 and 3.854. The small blend difference does not establish skill. The
study exports every feature and hourly lead in a metrics CSV.

The initial unbounded recursive search failed on az43 with a nonfinite raw
rollout state. The long-horizon increment bound addresses runaway proposed
states, and the Huber calculation now squares only the bounded quadratic
region. Independent calibrated field predictions and native Windows and
CUDA validation remain pending. No field CSV supplied training or
selection data.

### Recursive evolution on the 72-hour AZMET development split

All five refits used two CPU epochs, seed 42, and 21,726 training, 4,605
validation, and 4,606 final evaluation origins. Search used two candidates
per generation for two generations and one training epoch per candidate,
with a 2,200-parameter candidate ceiling and no caching. Every searched
method executed 6,257,088 model sample-steps. Parameter-weighted work
varied from 10.9 to 13.0 billion steps. Fixed architecture training
performed no candidate evaluations. The boundaries were 2025-10-27
15:00 UTC and 2026-05-10 11:00 UTC. This split differs from the seasonal
study above and overlaps previously examined historical observations.

| Method | Parameters | Validation selection score | Hour 1 RMSE °C | Hour 6 | Hour 24 | Hour 72 |
|---|---:|---:|---:|---:|---:|---:|
| Persistence | | | 1.574 | 7.106 | 2.346 | 3.453 |
| Daily persistence | | | 2.317 | 2.333 | 2.346 | 3.453 |
| Fixed recursive AI | 3,494 | | 7.385 | 49.555 | 90.439 | 90.300 |
| Random search | 806 | 7.7264 | 7.205 | 28.966 | 45.078 | 35.815 |
| Classical GA | 2,086 | 6.5016 | 8.456 | 50.778 | 90.387 | 90.300 |
| QIGA | 1,350 | 2.0294 | 6.716 | 20.990 | 47.833 | 80.521 |
| Hybrid | 1,782 | 6.1963 | 6.554 | 26.014 | 44.816 | 50.816 |

The QIGA best validation fitness improved from 6.5016 to 2.0294,
but the resulting forecast performed poorly in physical temperature
units. Daily persistence strongly outperformed all recursive refits
at the long leads. The numerical bound enabled a finite run. It did
not establish forecast accuracy, and the short refits do not identify
whether longer training can recover useful skill. This is one seed and
is insufficient for ranking search algorithms.

The recursive report contains all six features at all 72 leads and
baseline skill, candidate training time, and test-batch inference time.
The QIGA checkpoint SHA-256 was
`04b1bc83c738b3c6d77f3967b55875889f40967e33b9b3ee06ce42218ac55e23`.
The checkpoint produced 72 finite hourly rows in the CLI. Its imported
evolution and neural diagnostics rendered in Chromium.

### Acceptance status

| Requirement | Evidence and remaining work |
|---|---|
| 1–72-hour software contract | Integer API and loader limits verified. The elected QIGA checkpoint produced six values per lead. |
| Numerical and evolutionary checks | Finite-difference rollout tests cover the bounded output Jacobian. Fixed-seed frequency tests check amplitude sampling, rotation direction, and normalization. Generation and CPU recovery tests are in the regression suite. |
| Local training workflow | The Linux CLI exported, loaded, forecast from, and registered the winning architecture. PowerShell recipes are in the handover. Native Windows and physical CUDA remain unverified. |
| Independent scientific validation | 72-hour development evaluation completed. Forecast skill and generalization require a fresh chronological archive and a multi-seed study. |
| Real-field use | Independent leaf voltage endpoints, soil moisture reference calibration, a declared solar offset, and compatible hourly context remain required. Hardware acceptance remains open. |

## Stability follow-up and local training handoff

The previous delta rollout scaled each neural correction by the
variability of the state level. That scale is larger than typical hourly
changes for some features. Persistent directional corrections could
accumulate for 72 iterations. At hour 72 the QIGA model had a −72.742 °C
temperature bias, and the temperature projection changed 49.26% of its
forecasts. The short model also had no explicit hour-of-day signal.
The observed bias and projection rate directly demonstrate drift.
They do not isolate the effect of periodicity, training duration, or
architecture size.

For new training above 24 hours, the recursive step applies a fixed
12-hour anomaly half-life toward the training mean and learns a bounded
hourly residual. Each residual is scaled by the RMS of training-hour
changes after subtracting that relaxation. The previous state coefficient
and output-scale derivatives are included in manual backpropagation.
The loss continues to use the original targets. Legacy checkpoints
retain their saved dynamics. The two changes were tested together.

The before-and-after experiment held data, origin times, seed, and
search and refit epoch budgets constant. It used the same 4,606 AZMET
development origins as the previous comparison, so the results are
not a new blinded test. The architecture was selected again on
chronological validation. No AquaMap data entered fitting or selection.

| Temperature RMSE (°C) | Hour 1 | Hour 6 | Hour 24 | Hour 72 |
|---|---:|---:|---:|---:|
| QIGA before | 6.716 | 20.990 | 47.833 | 80.521 |
| QIGA after | 1.875 | 9.838 | 5.613 | 8.507 |
| Persistence | 1.574 | 7.106 | 2.346 | 3.453 |
| Daily persistence | 2.317 | 2.333 | 2.346 | 3.453 |

At hour 72, the revised QIGA model's temperature bias was −4.646 °C and
temperature projection changed 0 of 4,606 forecasts. This substantially
reduced drift. Diurnal accuracy and long-horizon skill remain limited.
The short search does not establish a ranking of search methods.
The project can proceed to local QIGA training without making baseline
superiority an acceptance criterion.

The revised QIGA checkpoint SHA-256 is
`33552f459f153da963dace30221acab4bc6a61e719f0413270ffd3bfd0cb35e6`.
The CLI forecast produced 72 finite hourly rows. The dashboard imported
the checkpoint and Chromium rendered its internal, evolution, and horizon
plots. A separate labeled synthetic smoke run used explicit UTC boundaries
and reproduced the checkpoint hash and generation records after recovery.

See [stability plots](Examples/forecast-stability/stability.png),
[six-feature RMSE](Examples/forecast-stability/feature-rmse.png), and
[underlying values](Examples/forecast-stability/stability.json).

## Spectral AZMET development evaluation

The spectral comparison used seed 42, 21,726 training origins, 4,605
validation origins, and 4,606 test origins. UTC boundaries were
2025-10-27 15:00 and 2026-05-10 11:00. This chronological period was
examined previously and is development evidence. Each searched method
used two generations of two fresh candidates with one epoch each, then
four refit epochs. The fixed model used four fit epochs and all seven
periods. AquaMap files were excluded.

| Method | Temperature RMSE at 1 h | 6 h | 24 h | 72 h | Mean trajectory correlation |
|---|---:|---:|---:|---:|---:|
| Stabilized recursive QIGA | 1.875 | 9.838 | 5.613 | 8.507 | 0.037 |
| Fixed Fourier | 1.581 | 7.112 | 2.349 | 3.459 | 0.003 |
| QIGA Fourier | 1.237 | 3.366 | 2.849 | 3.869 | 0.876 |
| QIGA Fourier with transient | 1.049 | 2.829 | 2.875 | 4.256 | 0.875 |
| QIGA Fourier with sparse inputs | 1.237 | 3.366 | 2.849 | 3.869 | 0.876 |
| Daily persistence | 2.317 | 2.333 | 2.346 | 3.453 | 0.904 |

RMSE is in °C. The evolved Fourier temperature cycle had amplitude MAE
1.524 °C and phase MAE 0.163 radians. The recursive reference had 5.344 °C
and 1.451 radians. The near-flat fixed model displays why matching RMSE
alone can miss poor trajectory shape. The damped model improved the
6-hour temperature result, with a higher 72-hour error. Sparse augmentation
recorded 17 events over four refit epochs and had no material effect in
this run. There was no raw temperature violation in either evolved
Fourier model. For QIGA Fourier, raw trajectories had 123,524 wetness
violations and 186,712 negative-ET rows out of 331,632 origin-lead
rows, so physical projection still materially affects other variables.

The study saves per-feature and per-lead metrics, an oracle least-squares
reconstruction comparison, raw physical violation counts, fit and search
diagnostics, and all evaluated predictions. Results do not establish
generalization to fresh AZMET periods or calibrated AquaMap visits.
See [Fourier components](Examples/spectral/qiga-fourier.png),
[damped components](Examples/spectral/qiga-laplace.png),
[all six RMSE curves](Examples/spectral/qiga-horizons.png), and
[pattern metrics and generation records](Examples/spectral/summary.json).

### Joint curve-fit experiment

With matched seed, search budget, four refit epochs, and the same
4,606 previously examined AZMET origins, the joint objective improved
Fourier temperature trajectory correlation from 0.876 to 0.889.
Six-hour and 72-hour RMSE went from 3.366 to 3.163 °C and 3.869 to
3.757 °C. For the transient arm they went from 2.829 to 2.693 °C and
4.256 to 4.078 °C. The Fourier slope RMSE fell from 1.067 to 1.049 °C
per hour. Daily persistence still achieved 0.904 correlation and
3.453 °C RMSE at 72 hours. The sparse arm recorded 17 refit events
and made no material difference. Actual ET correlation remained
undefined. These are one-seed development results, with no fresh test
period or AquaMap data used. [Curve comparison](Examples/spectral/curve-comparison.png).
