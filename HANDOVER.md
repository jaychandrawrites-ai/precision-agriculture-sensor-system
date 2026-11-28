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
