# Jupiter execution runbook

Work on `jupiter-clean-development`. Start with `git status -sb`, fetch the
branch and inspect the latest commits. Read AGENTS.md, PLANS.md,
MASTER_REQUIREMENTS.md and FINAL_DEVELOPMENT_CHECKLIST.md. Keep the historical
branches, original firmware and main intact.

Select actionable checklist work, inspect its call sites and acceptance
criteria, and implement through the normal entry points. Run focused tests
before committing. Record commands, results, limitations and the affected
requirement IDs in BUILD_LOG.md. Update DEVELOPMENT_STATUS.md and mark a
requirement verified only when its acceptance evidence is available.

Run the full regression suite and an offline demo at each release checkpoint:

```sh
OPENBLAS_NUM_THREADS=1 python -W error::DeprecationWarning -m unittest discover -s Tests -q
python -m Examples.SyntheticUncertainty /tmp/jupiter-uncertainty
```

Set `JUPITER_TRACE_BROWSER=1` when Playwright and Chromium are available.
Push recoverable commits to `jupiter-clean-development` and check the Linux
and Windows CI matrix. Resume from the latest verified commit. Record
hardware, data and CUDA blockers and continue independent software work.

## Phase 9 Part 2 queue

| Deliverable | Requirement IDs | Next acceptance work |
| --- | --- | --- |
| C01 Resource-aware training | R014, R016, R017, R038 | Exact direct-training resume, candidate peak in selection, CUDA verification when devices are available |
| C02 Evolutionary experiments | R039, R041, R076 | Equal aggregate work and independent multi-seed comparisons |
| C03 Temporal reliability | R028, R029, R032, R033, R074, R078, R079 | Portable named blends and frozen independent evaluation |
| C04 Physical and spatial uncertainty | R021, R048, R056 | Separate measurement scenarios, supported parameter ranges and resumable exports |
| C05 Research application and figures | R061–R066, R068–R073, R080, R081, R083 | Complete report controls, publication exports and browser coverage |
| C06 Final integration | R085, R086, R092, R093, R095 | Master requirement crosswalk and reproducibility audit |

The final development checklist controls verification of individual IDs.
