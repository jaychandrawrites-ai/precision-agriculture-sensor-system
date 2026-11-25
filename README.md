# Precision Agriculture PlatformIO Project

Hardware:
- Arduino MKR WAN 1310 (ABX00029)
- Adafruit SHT45
- DFRobot DFR0845 isolated RS485-to-UART adapter
- DFRobot SEN0604 4-in-1 soil sensor

## Open in VS Code / PlatformIO

1. Install the PlatformIO IDE extension in Visual Studio Code.
2. Extract this project folder.
3. In PlatformIO, choose **Open Project** and select this folder.
4. Connect the MKR WAN 1310 by USB.
5. Build and upload.
6. Open the PlatformIO Serial Monitor. It is configured for 9600 baud.

PlatformIO will download the Arduino framework and required Adafruit libraries.

## SHT45 wiring

SHT45 -> MKR WAN 1310

- VIN -> 3V3
- GND -> GND
- SDA -> SDA
- SCL -> SCL

The SHT45 and DFR0845 logic side may share the 3.3 V and GND rails.

## DFR0845 logic-side wiring

DFR0845 -> MKR WAN 1310

- + -> 3V3
- - -> GND
- T (TX) -> MKR RX
- R (RX) -> MKR TX

The sketch uses `Serial1` for this hardware UART.

## SEN0604 RS485 side

SEN0604 -> DFR0845 sensor/RS485 side

- Brown (power +) -> 12V output
- Black (ground) -> GND
- Yellow (RS485 A) -> A
- Blue (RS485 B) -> B

The external supply for the DFR0845 isolated side connects to the DFR0845's
12V-IN/GND input according to the module documentation.

IMPORTANT: Never connect 12 V to the MKR WAN 1310's 3V3, TX, RX, SDA, SCL,
or other GPIO pins.

## Expected output

The serial monitor should show air temperature and humidity. If the SEN0604
is communicating, it will also show:

- Soil moisture
- Soil temperature
- Soil EC
- Soil pH

If the soil probe is disconnected or RS485 communication is not working,
the program reports:

`ERROR: No valid response from SEN0604.`

## Current SEN0604 assumptions

The code assumes the factory-default Modbus RTU configuration:
- Address: 1
- Baud: 9600
- 8 data bits
- No parity
- 1 stop bit

## Jupiter offline research pipeline

Use Python 3.10 or later. From the repository root:

```sh
python -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest discover -s Tests -v
python main.py demo
```

The demo uses an explicitly synthetic periodic series. Its metrics describe that
fixture only. The default model selects a checkpoint using six-step validation
loss and requires complete six-hour windows in the training and validation
splits. The report includes held-out one-step and per-horizon metrics.

### Prepare, train, and forecast

Supply an hourly CSV with these headers:

```csv
timestamp,air_temperature_c,relative_humidity_pct,lwi,eto_mm_interval,eta_mm_interval,cotton_heat_stress
```

Timestamps must be increasing and unique, with an explicit UTC offset. Use
right-end timestamps for intervals, millimeters per interval for both ET series,
and a 0–1 leaf wetness index. Training skips incomplete windows and windows
that cross missing hours. Keep the feature estimation methods consistent
between historical and field CSVs.

```sh
python train.py historical_hourly.csv temporal.json --epochs 100
python forecast.py field_hourly.csv temporal.json --steps 6 > predictions.csv
```

Forecast uses a saved model and the last three consecutive hourly field samples.
It needs no live AZMET input. The project still needs a verified AZMET column
mapping and measured field calibration before the prepared CSVs can be generated
from their original sources. See BUILD_LOG.md for the remaining work.

### Capture Arduino output or replay it offline

```sh
python acquire.py --port /dev/ttyACM0 session-01
python acquire.py --replay session-01/raw.jsonl session-replay
```

Use a fresh output directory. Stop capture with Ctrl+C. The capture saves the
original bytes and host UTC receive time per line in `raw.jsonl` and parsed
readings with quality flags in `readings.jsonl`. Replay of a raw JSONL preserves
the logged timestamps. A plain-text replay assigns new receive times. The
checked-in sketch provides air and soil fields. Leaf wetness and GPS require the
AquaMap firmware and calibration before an acquired session can feed forecasting.

### Compare research methods

`python experiments.py historical_hourly.csv` evaluates a fixed network and
the architectures selected by the classical and quantum-inspired searches.
The default searches each train 80 candidates for 20 epochs. Both use
validation scores for selection and evaluate their final models on the same
chronological test split. The amplitude-based method runs classically.

For spatial measurements, use `Spatial.SpatialGrid.local_coordinates` to
project GPS into local meters and `Spatial.Validation.evaluate_held_out`
to compare variational reconstruction, IDW and ordinary kriging at
separate measurement sites. Choose the kriging range with training data
or domain knowledge before scoring the held-out sites.

### Recursive forecast stability

The training CLI and synthetic demo use a smoothed delta model with six
steps of unrolled training. For the original one-step state model, pass
`--legacy` to either command. Saved version 1 checkpoints still load through
`forecast.py`. Version 2 stores the derivative and output settings needed to
reproduce the training-time recurrence.

```sh
python main.py demo
python rollout_study.py --seeds 42 43 44 --steps 6 > rollout-study.json
python rollout_study.py prepared_hourly.csv --seeds 42 43 44 --steps 24 > rollout-24h.json
```

The study compares both baselines and all model variants on the same complete
test windows for a requested horizon. Seasonal persistence repeats the last
known daily cycle and falls back to ordinary persistence if the seasonal
observation is missing. It uses a fixed 24-hour UTC lag. The report includes
window counts and dates, forecast metrics before and after physical projection,
projection counts, normalized input and step sizes, and training gradient
statistics. Inspect [BUILD_LOG.md](BUILD_LOG.md) for the synthetic results and
the drop in accuracy seen beyond six hours.

## Phase 3 historical, sensor and spatial workflow

Run the integrated example against explicitly synthetic weather, packets,
calibration values, and field sites. The command creates a new directory
containing inputs, configurations, a saved model, a forecast, and spatial maps.

```sh
python -m Examples.SyntheticPhase3 /tmp/jupiter-phase3-demo
```

### Historical station files

```sh
python azmet.py inspect station.csv
python azmet.py inspect station-hourly.dat --headerless
python azmet.py prepare station.csv station-mapping.json prepared-history
python Pipeline.py train-history station.csv station-mapping.json temporal.json
```

Use the export documentation to fill in the mapping. The example generates
`historical-mapping.json` with synthetic headers. Map air temperature, relative
humidity, shortwave radiation, wind speed and interval precipitation. Specify
source units, missing sentinels, timestamp convention and source timezone when
needed. Headerless files require explicit ordered names. Separate year,
day-of-year and hour columns require a clock encoding (`0-23` or `1-24`). Hour
24 advances the calendar date, and a start-labeled timestamp shifts by one
hour to become the interval end.

The output includes UTC hourly features, missing and imputation flags,
source SHA-256, and per-period quality and eligibility. Bound any optional
forward filling of temperature, humidity and wind. Missing hourly records and
fluxes remain visible. Historical leaf wetness comes from the leaf ODE, ET₀
from the trailing-temperature Hargreaves proxy, ETa from the soil scenario,
and cotton heat stress from the crop index. Irrigation needs either a source
mapping or an explicit scenario. The fixed solar offset places hourly intervals
on an approximate solar clock. It is independent of the input timestamp zone.
The optional approximate clear-sky longwave forcing is part of the leaf model.

Historical training fixes chronological boundaries before filtering out invalid
target windows and fits scaling on training only. A checkpoint records the
source hash, feature derivations and the assumptions used. Calibration and
reference ET warmup can reduce usable hours, so inspect the report before
interpreting the results.

### Field acquisition and forecasting

```sh
python acquire.py --port /dev/ttyACM0 --packet-schema packet-schema.json field-session
python acquire.py --replay field-session/raw.jsonl --packet-schema packet-schema.json field-replay
python Pipeline.py prepare-session field-session/readings.jsonl field-settings.json prepared-field
python Pipeline.py forecast-session field-session/readings.jsonl field-settings.json forecasts \
  --checkpoint temporal.json --steps 6
```

Use the receiver's documented JSON envelope to define canonical sensor fields,
units, and optional device, boot, packet, fix and observed-time fields. Omit
`--packet-schema` for the checked-in sketch's text format. The synthetic config
illustrates the mapping structure. Real AquaMap LoRa firmware and payload
format remain to be verified. Raw bytes and receive times are retained. Use
measured dry and wet PHYTOS31 endpoints and a documented soil VWC conversion
in field settings. A packet header, serial log or sensor model name alone
does not establish the calibration.

Forecasting reads the checkpoint and local session only. It uses the latest
three closed hourly intervals and the same causal ET history convention as
training. Field LWI comes from calibrated leaf readings, and field ETa uses
measured soil moisture. These differences from historical simulation remain
a source of domain shift. For an as-of replay, set `hourly.as_of` to an
offset-bearing timestamp. The aggregation discards late arrivals beyond that
cutoff and retains empty hours so stale samples cannot serve as current input.
GPS fixes are written separately from the feature matrix.

### Spatial reconstruction

```sh
python Pipeline.py extract-spatial field-session/readings.jsonl snapshot-settings.json snapshot \
  --also second-session/readings.jsonl
python Pipeline.py map snapshot/measurements.csv spatial-settings.json spatial-map
```

Snapshot settings specify a measured field, an offset-aware interval end,
a timestamp clock and a lookback window. Only readings with the requested
measurement and a GPS fix in the same packet become spatial observations.
Multiple distinct sites are required. Alternatively, provide a measured CSV
with `timestamp,latitude_deg,longitude_deg` and a measurement column. A
configured split column can label independent sites as `fit` or `test`.

Supply the grid in local east/north meters, an origin, and an optional kriging
range. Choose `descent` or `conjugate_gradient` for minimization. Optional
`boundary_values` are a north-by-east array with JSON null for unconstrained
cells. With a `boundary_weight` they act as soft constraints. Without a weight
they are fixed. Multiple observations assigned to one grid cell are averaged.

The algorithm is classical. The Metropolis maps describe uncertainty
conditional on the quadratic energy, temperature and supplied observations.
Inspect acceptance, split R-hat and effective sample sizes. The maps do not
establish calibrated field prediction intervals. Keep precise real GPS and
session files outside the public repository.

### Recursive architecture search

The default `experiments.py` command scores both searches with the improved
rollout objective on validation data, then evaluates the selected networks on
the untouched test period. Equal candidate counts and maximum epochs give
equal sample-step budgets. An optional `--candidate-parameter-step-limit`
also caps the parameter-weighted training work per candidate. The report
includes both budgets and actual work proxies because architecture size
changes arithmetic costs. `--legacy` uses the original one-step experiment.

## Phase 4 research workflows

### Verified AZMET archive

The supplied mapping is specific to the published Tucson 2019 raw hourly
archive. The archive stays outside the repository.

```sh
curl -fL https://azmet.arizona.edu/azmet/data/0119rh.txt -o tucson-2019.txt
python azmet.py inspect tucson-2019.txt --headerless
python station_study.py tucson-study --station tucson-2019.txt Examples/azmet/tucson-2019.json
python forecast_benchmark.py tucson-study/az01/hourly.csv tucson-benchmark.json --epochs 8 --seeds 42 43
```

Repeat `--station SOURCE MAPPING` for separate station series. Each mapping
needs a distinct station ID. The report contains monthly coverage and extremes,
source SHA-256, and an optional comparison of Hargreaves with the archive’s
station-estimated ET₀. Real station observations and derived or simulated
features keep their original provenance labels.

`scenario_study.py SOURCE MAPPING SCENARIOS OUTPUT` compares named physics
assumption or forcing overrides against the reference scenario on matched
hours. For example, use
`{"lower_crop_coefficient":{"assumptions":{"soil":{"crop_coefficient":0.6}}}}`.
Scenario output measures sensitivity to assumptions.

### Experimental forecasters and recovery

```sh
python direct_forecast.py train prepared.csv direct.json --horizon 24
python direct_forecast.py predict recent.csv direct.json direct.csv
python train.py prepared.csv recursive.json --epochs 30 --resume-state training-state.json
python experiments.py prepared.csv --random-baseline --search-directory search-state --workers 2
```

Repeat the same training command to resume a CPU run after a completed epoch.
It validates the original schedule, data and architecture and restores the
optimizer, shuffle stream and dropout RNG. Keep inference and recovery files
separate. Search resumes at a completed generation. Optional
`--cache-fitness` counts reused validation results separately from actual
candidate work. Leave caching disabled when matching actual sample-step
budgets. The amplitude-based search is classical.

The benchmark keeps test origins fixed across direct, recursive, hybrid and
persistence forecasts. It reserves separate validation periods for model
selection and marginal interval estimation. Historical coverage and OOD
flags are diagnostics. Weather-dependence and seasonal shift limit their
interpretation. The models have different output sizes and training window
lengths, so their forecast comparison is not an equal-compute model study.

### Spatial and multi-site workflows

Set `algorithm` to `robust_lbfgs` in a map configuration for pseudo-Huber
fidelity and anisotropic regularization. Optional `robust` keys are
`east_conductance`, `north_conductance`, and `fidelity_delta`.
Conductance may be specified per edge. Keep quadratic Metropolis sampling
disabled for this solver. For the quadratic model, `sampling.proposal` accepts
`normal` or `uniform`, and `sampling.adapt` tunes the scale only in burn-in.
The geostatistics module provides empirical variograms and comparisons with
Gaussian radial basis fits, leave-one-location-out folds, or spatial blocks.

`python multisite.py LOCATIONS CHECKPOINT OUTPUT --snapshot-at OFFSET_TIMESTAMP`
loads one checkpoint for independent sites. Each location supplies a local
`readings` path and its field `configuration`. Spatial snapshots require a
common origin and fresh GPS fixes received by that origin. These outputs are
labeled model forecasts. To map them, set `allow_forecast_snapshot` explicitly.
Held-out spatial validation requires independent measurements.

### Local dashboard and Windows setup

```sh
python -m Dashboard.Server --workspace /tmp/jupiter-dashboard
python Pipeline.py session-health field-session/readings.jsonl session-health.json
```

Open the printed `127.0.0.1` URL. The dashboard processes uploaded data on the
local machine and runs the existing CLI jobs. Start with `--allow-port PORT`
to enable live capture for a specific local serial port. Run directories can
contain sensor locations and calibration details, so keep them outside the
checkout. Training and search artifacts can also be produced with the standalone
commands if a browser is unavailable.

For Windows Command Prompt:

```bat
py -3 -m venv .venv
.venv\Scripts\activate.bat
python -m pip install -r requirements.txt
python -m unittest discover -s Tests -q
python -m Dashboard.Server --workspace "%TEMP%\jupiter-dashboard"
```

For PowerShell, use the environment’s Python directly so activation policies
do not affect setup:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m unittest discover -s Tests -q
.\.venv\Scripts\python.exe -m Dashboard.Server --workspace "$env:TEMP\jupiter-dashboard"
```
