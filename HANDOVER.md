# Jupiter handover

Jupiter trains custom NumPy/CuPy forecasters on chronological AZMET data
and uses saved checkpoints for field inference. AquaMap sends measurements
to the base station, which emits native serial CSV. The dashboard and AI
Trace are loopback applications. The active development branch is
`jupiter-clean-development`.

See [README.md](README.md) for install and launch instructions,
[ARCHITECTURE.md](ARCHITECTURE.md) for module boundaries,
[MATHEMATICS.md](MATHEMATICS.md) for equations,
[DATA_FORMAT.md](DATA_FORMAT.md) for feature contracts,
[HARDWARE.md](HARDWARE.md) for receiver details, and
[BUILD_LOG.md](BUILD_LOG.md) for verification and limitations.

Historical fitting, architecture search and scaling are restricted to AZMET.
Field files stay in the survey store for import, inference and mapping.
They must never inform model selection or calibration. Hourly checkpoints
need consecutive valid six-feature rows. A short visit can use a compatible,
adjacent historical prefix supplied in the field configuration. Use
`historical_context.match_visit: true` to select a visit-specific prefix
from the supplied timezone-aware, chronological rows. This does not infer
missing soil or crop inputs from weather-station temperature.

Each connected survey owns an independent receiver worker. Group start
records a shared group ID while keeping individual timestamps. Aligned CSV
exports retain all parsed rows and report UTC bin starts and receipt offsets.
Empty bins are implicit in this long-form export. Each survey still offers
its own original raw archive. Group operations report failures per receiver.

## Windows launch

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m Dashboard.Server --workspace "$env:TEMP\jupiter-dashboard"
```

For a physical acceptance run, use a separate local workspace and devices
already configured for use. Verify each authorized port independently,
check fresh packet counts and GPS readiness, start a visit, stop it and
compare the raw archive against the receiver output. For two receivers,
verify recording on both, disconnect one, and confirm that the other
continues. Export aligned data and check node IDs and timestamp offsets.
Only attempt a real forecast once measured calibration and compatible
adjacent history are supplied.

Release acceptance still requires real receiver checks, Windows execution,
and an authorized calibrated field forecasting case. For the next
development session, resume from the pushed development branch, retain the
field-data isolation rules, and prioritize a dashboard start–restart–reload
acceptance run with user-supplied calibration and compatible context.

See [AI_VALIDATION.md](AI_VALIDATION.md) for the feature compatibility
matrix, numerical audit scope and commands for training diagnostics.
Dashboard training saves plots and held-out predictions under its job
directory. Open /diagnostics after a completed model training job.

For persistent CLI training and dashboard registration see
[AI_VALIDATION.md](AI_VALIDATION.md). AI Trace displays the selected
checkpoint's hourly context and horizon constraints. Use a new run directory
when changing total epochs or layer sampling frequency.

## Recursive evolutionary training on Windows

From the repository root, using a fresh local output directory for each
recipe and retaining the previous configuration for exact resume:

```powershell
git switch jupiter-clean-development
git pull --ff-only origin jupiter-clean-development
py -3.12 -m venv .venv
$python = ".\.venv\Scripts\python.exe"
& $python -m pip install -r requirements.txt
& $python azmet.py prepare supplied-az43.csv Examples/azmet/yuma-valley-az43.json az43-prepared
$env:OPENBLAS_NUM_THREADS = "1"
New-Item -ItemType Directory -Force training/evolution-smoke
& $python experiments.py az43-prepared/hourly.csv --preparation-report az43-prepared/report.json --population 2 --generations 2 --candidate-epochs 1 --epochs 2 --horizon 6 --max-parameters 2200 --random-baseline --hybrid-baseline --adaptive-amplitude --search-directory training/evolution-smoke/search --export-selected training/evolution-smoke/models > training/evolution-smoke/report.json
& $python -m Dashboard.Server --workspace "$env:LOCALAPPDATA\Jupiter" --import-training-run training/evolution-smoke/models/quantum_inspired
```

Open http://127.0.0.1:8765/diagnostics to inspect the imported run and
http://127.0.0.1:8765/trace to select the saved forecast model. Use
independently measured calibration and compatible adjacent context before
running a field forecast. Native Windows execution remains to be verified.

For research, measure the smoke run first. A starting recipe is population
4, generations 4, candidate epochs 3, final epochs 12 and horizon 72.
Repeat with predeclared seeds (for example 42, 43 and 44) and separate
output folders. Use eligible historical data only, and retain an untouched
evaluation archive before choosing a recipe. Candidate budget equality
uses the same population, generations and epoch quota. Architectures can
consume different parameter-sample work, so use the measured work for
interpretation and configure `--candidate-parameter-step-limit` if
an explicit ceiling is required. Fixed architecture training uses no
search trials. Repeat an identical command to restore completed search
generations and CPU training epoch state.

This PowerShell loop runs the research recipe with independent output
directories. Adjust the budget after timing a local smoke run.

```powershell
foreach ($seed in 42,43,44) {
    $run = "training/evolution-72-$seed"
    New-Item -ItemType Directory -Force $run
    & $python experiments.py az43-prepared/hourly.csv --preparation-report az43-prepared/report.json --seed $seed --population 4 --generations 4 --candidate-epochs 3 --epochs 12 --horizon 72 --max-parameters 2200 --random-baseline --hybrid-baseline --adaptive-amplitude --search-directory "$run/search" --export-selected "$run/models" > "$run/report.json"
    if ($LASTEXITCODE -ne 0) { throw "Training failed for seed $seed" }
}
```

To reproduce the historical 72-hour development sweep, use the smoke
configuration in a fresh directory with `--horizon 72`. To verify a
saved winner and generate a cross-method plot:

```powershell
& $python forecast.py az43-prepared/hourly.csv training/evolution-72/models/quantum_inspired/model.json --steps 72 > training/evolution-72/forecast.csv
& $python research_figures.py evolution-forecast training/evolution-72/report.json training/evolution-72/temperature.png
```

## Local stability training recipes

Use the Windows setup above. These are CPU starting budgets. Run the
smoke recipe first and inspect the validation and stability diagnostics
before scheduling a longer search. Use a fresh directory for each recipe.

```powershell
$recipes = @{
    smoke    = @(2, 2, 1, 2, 6)
    moderate = @(4, 3, 2, 6, 72)
    research = @(6, 6, 4, 16, 72)
}
$tier = "smoke"
$seed = 42
$population, $generations, $candidateEpochs, $epochs, $horizon = $recipes[$tier]
$run = "training/stability-$tier-$seed"
New-Item -ItemType Directory -Force $run
Measure-Command {
    & $python experiments.py az43-prepared/hourly.csv --preparation-report az43-prepared/report.json --seed $seed --population $population --generations $generations --candidate-epochs $candidateEpochs --epochs $epochs --horizon $horizon --max-parameters 2200 --random-baseline --hybrid-baseline --adaptive-amplitude --search-directory "$run/search" --export-selected "$run/models" > "$run/report.json"
    if ($LASTEXITCODE -ne 0) { throw "Jupiter experiment failed" }
}
& $python -m Dashboard.Server --workspace "$env:LOCALAPPDATA\Jupiter" --import-training-run "$run/models/quantum_inspired"
```

Use the report's search and fit timings to estimate local cost. Search
work grows approximately with population × generations × candidate epochs
× horizon; refitting scales with final epochs × horizon. The six-hour
smoke run does not measure the overhead of the longer rollout tape. A
72-hour run with the smoke population and generation budget gives a
closer timing estimate. For exact recovery, repeat the same command in
the same directory with the same sources, software revision and settings.

For formal evaluation, predeclare hourly UTC boundaries and append
`--validation-from` and `--test-from` with those timestamps. Use only
training and validation periods to make decisions. Keep the test period
unexamined until settings are frozen. The existing comparison archive
is development evidence because its evaluation period was already used.

## Spectral training on Windows

After preparing AZMET and installing dependencies with the commands
above, use a new persistent run directory:

```powershell
$python = ".\.venv\Scripts\python.exe"
$env:OPENBLAS_NUM_THREADS = "1"
$run = "training/spectral-az43"
& $python spectral_study.py az43-prepared/hourly.csv $run --preparation-report az43-prepared/report.json --population 2 --generations 2 --candidate-epochs 1 --epochs 4 --sparse-scale 0.0001
& $python forecast.py az43-prepared/hourly.csv "$run/qiga_fourier/model.json" --steps 72 > "$run/forecast.csv"
& $python -m Dashboard.Server --workspace "$env:LOCALAPPDATA\Jupiter" --import-training-run "$run/qiga_fourier"
```

Open http://127.0.0.1:8765/diagnostics to inspect the spectral,
evolutionary and internal network plots. AI Trace derives the UTC
forecast origin from the most recent eligible closed hourly bin.
Existing field calibration, compatible adjacent history, and ET
requirements still apply. To compare with a saved recursive model,
pass its checkpoint to `--recursive-reference`. Use
`--validation-from` and `--test-from` to freeze both UTC boundaries.
Loss weights can be specified in a JSON object with the keys
`value`, `spectral`, `derivative`, `physical` and `regularization`.

Repeat the same command to recover completed search generations.
Final spectral weight fits restart from their seeded initialization.
The Windows commands were exercised with equivalent Linux invocations.
