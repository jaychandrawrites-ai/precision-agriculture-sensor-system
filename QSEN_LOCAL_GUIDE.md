# QSEN v3 local training and acceptance

QSEN means Quantum Spectral Evolutionary Network. QSEN v3 is the
quantum-inspired spectral and neural residual model running on classical
hardware. These commands target Python 3.12 and Windows PowerShell.
Linux equivalents were exercised. Native Windows and physical CUDA and
Arduino acceptance remain local checks.

## Install and launch the development model

For a fresh clone, run:

```powershell
git clone --branch jupiter-clean-development https://github.com/jaychandrawrites-ai/precision-agriculture-sensor-system.git
Set-Location precision-agriculture-sensor-system
py -3.12 -m venv .venv
$py = ".\.venv\Scripts\python.exe"
& $py -m pip install -r requirements.txt
& $py -m pip check
& $py -m Dashboard.Server --workspace "$env:LOCALAPPDATA\Jupiter" --import-training-run Examples/spectral/qsen-v3-development
```

For an existing clone, run `git switch jupiter-clean-development` and
`git pull --ff-only origin jupiter-clean-development` first. Open
http://127.0.0.1:8765/diagnostics or http://127.0.0.1:8765/trace.
Stop the server with Ctrl+C. Restart with the same workspace to retain
the registered checkpoint and surveys.

The bundled development checkpoint was trained on the previously
examined Yuma Valley archive. The model includes architecture, weights,
normalization and provenance. The diagnostic report stores training
settings, validation metrics, and checkpoint hash. The archive excludes
raw observations and large prediction arrays. Existing spectral plots
and the all-feature comparison are under `Examples/spectral`. Verify
the model hash with:

```powershell
Get-FileHash Examples/spectral/qsen-v3-development/model.json -Algorithm SHA256
```

Expected SHA-256 is
`87aaa7039a262e84f305aea382bfe7b3a590875579255326c63f3cdc1e012043`.

## Prepare eligible historical data

Use the supplied Yuma Valley hourly ETo CSV or obtain a compatible
station-43 export from the University of Arizona AZMET portal. Save it
as `local-data/az43.csv`. The verified export covers May 2023 through
November 2026. The mapping accepts other dates with the same hourly
schema and adequate contiguous data in each split. Other stations need
their own verified mapping and timezone. Check the header before reuse.
[AZMET portal](https://azmet.arizona.edu/)

The CSV mapping reads `date_year`, `date_doy`, `date_hour`, `temp_airC`,
`relative_humidity`, `sol_rad_total` and `wind_spd_mps`. The hour field
uses local 1–24 hour-ending HHMM. An available compatible `precip_total`
column provides precipitation in mm/hour. The verified export lacked
that column, and the mapping explicitly runs a zero precipitation and
zero irrigation scenario. This exhausted the modeled soil bucket and
made actual ET effectively constant over the earlier evaluation.
Temperature and humidity are measured. Leaf wetness, both ET features
and cotton heat stress are derived under the configured assumptions.
Blank and NA records and preparation gaps are tracked in the quality
outputs. Training uses valid contiguous windows and training-only scaling.

```powershell
New-Item -ItemType Directory -Force local-data
& $py azmet.py inspect local-data/az43.csv
& $py azmet.py prepare local-data/az43.csv Examples/azmet/yuma-valley-az43.json az43-prepared
```

Review `az43-prepared/report.json` and `flags.csv`. Keep AquaMap
files out of historical preparation, fitting, data normalization,
calibration fitting and candidate selection.

## Train and recover

Each recipe evaluates the spectral-only references and QSEN v3.
Start with SMOKE. Use the same command and directory to restore
completed evolutionary generations. A partial generation and an
interrupted final fit rerun from their configured seed. Change recipes
in a new directory.

```powershell
$env:OPENBLAS_NUM_THREADS = "1"
$recipes = @{
  SMOKE    = @(2,2,1,2,1)
  MODERATE = @(4,4,2,6,1)
  RESEARCH = @(8,8,4,16,2)
}
$tier = "SMOKE"
$seeds = if ($tier -eq "RESEARCH") { @(42,43,44) } else { @(42) }
$population,$generations,$candidateEpochs,$epochs,$jointEpochs = $recipes[$tier]
foreach ($seed in $seeds) {
  $run = "training/qsen-$tier-$seed"
  $elapsed = Measure-Command {
    & $py spectral_study.py az43-prepared/hourly.csv $run --preparation-report az43-prepared/report.json --hybrid --population $population --generations $generations --candidate-epochs $candidateEpochs --epochs $epochs --joint-epochs $jointEpochs --seed $seed --sparse-scale 0.0001
    if ($LASTEXITCODE -ne 0) { throw "QSEN training failed" }
  }
  $elapsed.TotalSeconds
}
```

Approximate search cost from the smoke run using population × generations
× candidate epochs, allowing for variation in model widths. Estimate final
fit separately from the recorded epoch times and the number of phases.
For CUDA, `--device cuda:0` requires a CuPy installation compatible with
your local driver. CPU is the verified default. Extra epochs may change
validation and test results in either direction.

## Formal evaluation and deployment

Freeze the model recipe, seeds and split before inspecting test outputs.
Choose two hourly UTC boundaries in your own archive. Leave at least
72 valid consecutive target hours in each split. Set these PowerShell
variables to your actual predeclared boundaries before running:

```powershell
$validationFrom = Read-Host "Frozen validation start in UTC (YYYY-MM-DDTHH:00Z)"
$testFrom = Read-Host "Frozen test start in UTC (YYYY-MM-DDTHH:00Z)"
& $py spectral_study.py az43-prepared/hourly.csv training/qsen-formal-42 --preparation-report az43-prepared/report.json --hybrid --population 4 --generations 4 --candidate-epochs 2 --epochs 6 --joint-epochs 1 --seed 42 --validation-from $validationFrom --test-from $testFrom
```

Use previously unused observations for the final split where available.
The study selects on validation and then scores the test windows. Record
all predetermined seeds and avoid adjusting the recipe after reading
test metrics. `report.json` stores hourly and trajectory metrics. Each
method directory stores the model, weights and optimizer diagnostics,
spectral plots, generation summaries, and numerical evaluation arrays.
Exact per-gene flip counts and full layer-level residual activations are
not currently recorded.

```powershell
$run = "training/qsen-SMOKE-42"
& $py forecast.py az43-prepared/hourly.csv "$run/qiga_hybrid/model.json" --steps 72 > "$run/forecast.csv"
& $py -m Dashboard.Server --workspace "$env:LOCALAPPDATA\Jupiter" --import-training-run "$run/qiga_hybrid"
```

Select QSEN v3 under Saved dashboard model in AI Trace. For recorded
data, use Preview CSV batch and then Import CSV batch, with an explicit
IANA timezone such as America/Phoenix when appropriate. For a physical
receiver, identify its COM port in Device Manager, close other serial
programs and restart the server with `--allow-port COM3` after replacing
COM3 with the correct port. Choose the AquaMap receiver protocol and
verify received packets before recording. Repeat the flag for a second
receiver if needed. Use compatible closed hourly context and independent
calibration before enabling field forecasts. Real coordinates and
archives should stay in your local workspace.

If a port is missing, check the cable, driver, port allow-list and
whether another process owns the port. If forecasting is ineligible,
check timestamps, adjacent hourly rows, ET history and calibration.
If registration fails, verify the checkpoint hash against the report.
If a search recipe differs, use a fresh output directory. If the
site is already running, stop the other server or select another
`--port`.

## Release checks before merging

- Run installation and the CPU smoke recipe on native Windows.
- Reopen the dashboard, select the registered model and inspect diagnostics.
- Confirm physical serial acquisition and independent calibration on the intended devices.
- Run the frozen formal AZMET evaluation and archive its provenance and results.
- Verify that the proposed merge contains the intended commits and excludes local data and generated run directories.
