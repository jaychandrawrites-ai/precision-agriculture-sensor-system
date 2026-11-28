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
