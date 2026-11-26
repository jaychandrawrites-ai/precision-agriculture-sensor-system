# Development status

Status labels describe software and validation evidence separately.

| ID | Status | Evidence or next step |
| --- | --- | --- |
| R01 | Implemented and integrated | Active clean branch retains the audited foundation and replayed history |
| R02 | Implemented and integrated | SQLite surveys, visit isolation, checksummed archives, unit-mapped CSV and receiver replay |
| R03 | Implemented, live verification pending | Configurable parser, receipt/observation clock checks, GPS status and supply-your-own calibration |
| R04 | Blocked by external hardware/source | Obtain actual AquaMap field/base firmware or labeled receiver bytes, then verify native fields, units and timestamps |
| R05 | Implemented, field validation pending | Six-feature physical pipeline, dry-tail checks, ET comparison and constrained scenarios |
| R06 | Partially implemented | Five pinned releases and manual modern schema path. Obtain the actual May 2023–November 2026 Yuma export before mapping it |
| R07 | Partially implemented | Manual dense, GRU, conv, attention and numerical kernels. Manual LSTM and recurrent-attention train and load through the experimental direct model |
| R08 | Partially implemented | Portable models and exact CPU recovery. Verify CUDA and mixed precision on a suitable device |
| R09 | Implemented and integrated with extensions open | Independent visit checks, recursive/direct/seasonal models, explicit refusal when hours are missing |
| R10 | Partially implemented | Empirical intervals and block comparisons. More quality/transfer diagnostics require independent validation |
| R11 | Partially implemented | Structured classic/amplitude/hybrid search. Recurrent candidates and broader cost objectives remain |
| R12 | Partially implemented | Kriging, variational baselines and observed survey composites. Budgeted adaptive kriging is available; field comparison remains |
| R13 | Partially implemented | Conditional Gaussian/block sampling and mixing tests. Broader parameter uncertainty remains |
| R14 | Partially implemented | Survey outputs are segment-specific. Common issue/valid-time point maps are available; survey uncertainty and joint field verification remain |
| R15 | Partially implemented | Persistent dashboard and dedicated AI Trace controls. Physical receiver and visual browser validation remain |
| R16 | Partially implemented | Reproducible independent station studies and JSON/CSV. Publication figure coverage remains |
| R17 | Implemented and integrated | Milestone commits, test commands, build log and requirement inventory |

Prioritize real receiver verification and completing the survey-to-maps
workflow before expensive benchmark runs. Keep synthetic and observed
evidence labeled in every experiment report.

## Final checklist checkpoint

The [final development checklist](FINAL_DEVELOPMENT_CHECKLIST.md) tracks all
96 remaining requirements. It records 17 verified, 57 partially implemented,
two awaiting browser verification, four unstarted and 16 externally blocked.
See the build log for the verification commands and commit evidence.

AI Trace now exports shared ensemble trajectories across sites and forecast
leads. The field loader serves portable direct and sequence checkpoints.
Adaptive kriging is integrated into measured-site mapping. The five-site
offline demonstration verified software behavior using synthetic inputs.

## Phase 7 checkpoint

The final checklist records the latest evidence for saved physical scenarios,
sequence architecture search, adaptive variational reconstruction, paired
joint training and the browser-tested AI Trace workflow. Trained quantile
inference and survey adaptive kriging are available from the CLI. Remaining
application integration and field validation are indexed by requirement.

## Phase 8 checkpoint

The final checklist records 35 verified requirements, 44 partial, one
unstarted and 16 externally blocked. The combined survey export, quantile
console, constrained sequence search and calibration experiments are
documented in BUILD_LOG.md. Remaining work includes joint inference model
selection, candidate-isolated memory measurements, optional multi-device
execution, publication exports and independent field verification.

## Phase 9 survey milestone

Joint site draws now propagate through selectable reconstruction operators.
The checklist retains R056 as partial pending a disk-backed export path.
See EXECUTION_RUNBOOK.md for checkpoint and verification procedure.

The Phase 9 continuation adds disk-backed CLI staging for survey draws and
browser-verified Pareto viewing and reuse of sequence search winners.
R042 and R067 are verified. R056 remains partial because covariance
factorization is dense and the export cannot resume after interruption.

Verified R012, R013, R046, R049 and R075. The checklist now records 42
verified, 37 partial, one unstarted and 16 externally blocked.

R082 is verified through synthetic predictive mean and interval-width
figure exports. The current checklist has 43 verified requirements.
