# Jupiter architecture

Jupiter has separate inputs for training, field forecasting and spatial
validation. An archived station export supplies historical weather. A field
session supplies independent sensor readings. A measured spatial snapshot
supplies coordinates and held-out field labels. A saved forecast can be
mapped, with its origin preserved as a model forecast.

| Workflow | Entry point | Result |
| --- | --- | --- |
| Verify and register AZMET releases | `dataset_registry.py` | Source hashes, quality flags and prepared features |
| Train and reuse recursive models | `train.py`, `forecast.py` | Versioned inference and optional recovery checkpoints |
| Train experimental direct models | `direct_forecast.py` | Gated, GRU, causal convolution or attention direct forecasts |
| Run seasonal studies | `seasonal_study.py`, `seasonal_transfer.py` | Common-origin reports, calibration and source-station model transfer |
| Capture and verify a sensor session | `acquire.py`, `acquisition_archive.py` | Raw capture, parsed readings and an integrity manifest |
| Process sensor sessions | `Pipeline.py`, `multisite.py` | Hourly features and local forecasts with explicit receipt cutoffs |
| Reconstruct and evaluate maps | `Pipeline.py map`, `spatial_benchmark.py`, `spatial_forecast.py` | Measured-site comparisons or model-forecast maps |
| Run jobs locally | `python -m Dashboard.Server` | Dataset, training, forecast and comparison reports |

The legacy temporal model takes the last state and its two backward
derivatives as an 18-value input. Experimental sequence models take the last
three ordered six-feature observations. The seasonal variant also needs
eligible hours from the preceding cycle, or falls back to the last state and
reports the missing seasonal observation. Spatial mapping maintains its own
source identity and does not use GPS as a temporal feature.

Checkpoint-compatible preprocessing, station metadata and experimental
protocol details are in [DATA_FORMAT.md](DATA_FORMAT.md) and
[EXPERIMENT_PROTOCOL.md](EXPERIMENT_PROTOCOL.md).

## Source layout

`Framework/` and `Numerical/` contain the execution and numerical kernels.
`Physics/` is organized into `Evapotranspiration/`, `Soil/`, `Leaf/` and
`Crop/`, with shared physical bounds at the package root. `Data/` owns
preparation and forecast context validation. `Temporal/`, `Evolution/`,
`QuantumInspired/`, and `Spatial/` keep the forecasting, search and field
algorithms separate. `Acquisition/` handles hardware, archives and field
features. `Dashboard/` contains the local UI and service. `Utils/` holds
shared support, `Tests/` holds verification, and `Examples/` holds examples
and schema fixtures. The root contains CLI entry points, shared configuration,
and the integration interfaces for combined loss and spatiotemporal modeling.
The PlatformIO tree uses `src/`, `lib/`, `include/`, and `platformio.ini`.

## AI Trace storage boundary

AI Trace records survey, location, segment and packet relations in a local
SQLite workspace. Stopping a location finalizes only its segment. A
receiver failure interrupts the active segment. A revisit always has its
own identifier. Recorded bytes are committed before parser output, and
saved JSON survey archives carry an integrity checksum. The map layer
separates bounded-time measured composites from point forecasts aligned
to one issue and valid time.

## QSEN v3 local release

QSEN v3 (Quantum Spectral Evolutionary Network) is Jupiter's
quantum-inspired spectral and neural residual forecaster for classical
hardware. The [QSEN local guide](QSEN_LOCAL_GUIDE.md) is the current
entry point for installation, training, dashboard import and release
checks. Earlier commands and results describe reference experiments.
