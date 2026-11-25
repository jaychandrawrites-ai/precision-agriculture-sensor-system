# Jupiter build log

## 2026-11-25

Started from `jupiter-development` at `4f468ee`. Read AGENTS.md and
PLANS.md and audited the existing Python modules and Arduino sketch.

- **Phase 1:** Implemented a six-feature hourly CSV contract, physical quality
  flags, causal filters segmented at missing hours, consecutive lag windows,
  chronological splits and training-only scaling. Reused existing physics,
  resampling, imputation and derivative implementations. Real AZMET column
  inspection and historical feature derivation are pending.
- **Phase 2:** Implemented live serial capture and offline replay for the
  checked-in sketch, including byte-preserving raw logs, receive timestamps,
  invalid and duplicate field flags, and final partial-block handling. Device
  timestamps and packet IDs are absent in this sketch. AquaMap, GPS, PHYTOS31
  endpoint measurements and live hardware validation remain pending.
- **Phase 3:** Implemented dense regression with manual gradients, optional
  residual blocks, Huber/MSE, quality weighting, Adam/SGD, gradient clipping,
  scheduling, best-validation checkpoint selection and held-out one-step and
  recursive reporting against persistence.
- **Phases 4 and 5:** Implemented validation-only classical GA and a classical
  quantum-inspired amplitude-pair search with equal candidate budgets and a
  parameter-count penalty. Verified on small synthetic budgets. Full default
  budget comparisons and held-out winner evaluation are pending.
- **Phase 6:** Added local metric GPS projection, east/north grid, IDW baseline,
  and hard/soft variational reconstruction through the existing stable PDE
  solver. Kriging, held-out GPS experiments and Metropolis uncertainty remain
  pending.
- **Phases 7–9:** Connected the offline temporal training, saved model and
  independent recursive forecast commands. Added acquisition and demo entry
  points. The versioned JSON artifact saves model, normalization, illustrative
  assumptions, optimizer and scheduler snapshots, shuffle RNG and CPU backend
  RNG. Training resume and GPU RNG restoration are pending. Joint or field
  integration remains pending.
- **Phase 10:** Added tests for chronology, causal features, resampling, gaps,
  checkpoint round trip, projected residual gradients, raw replay, equal
  evolutionary budgets, normalized amplitudes and spatial seed fidelity.

### Verification

- `OPENBLAS_NUM_THREADS=1 python -m unittest discover -s Tests -v` passed 8
  tests under NumPy 2.3.5 and pandas 2.2.3.
- `OPENBLAS_NUM_THREADS=1 python main.py demo` trained on 165 examples,
  validated on 36, tested on 36, saved and reloaded the model, and emitted six
  recursive forecasts. The fixture gave 0.433 °C one-step temperature RMSE
  and 70.481 °C six-step temperature RMSE, with forecasts reaching physical
  bounds. Recursive stability and real-data performance remain open.
- `python -m compileall -q .` passed. Live serial hardware, CUDA and real
  dataset runs were unavailable for this verification.
