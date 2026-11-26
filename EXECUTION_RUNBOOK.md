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
