# Verification and interpretation

Run `python -m unittest discover -s Tests -q` on Windows or Linux. The suite
checks manual gradients, numerical references, analytical Gaussian moments,
data integrity, seasonality leakage, checkpoint recovery and local server
integration. Optional CUDA checks skip on hosts without CuPy/CUDA.

Forecast reports give RMSE, MAE, bias, R² and skill against persistence by
feature and horizon. Where defined, seasonal skill uses daily persistence as
the comparator. Empirical marginal intervals calibrate independently for
each feature and lead. Whole-trajectory intervals cover all leads for one
feature. Neither interval report claims simultaneous coverage across all
features. Spaced windows reduce direct target overlap. Weather dependence
and station transfer can still change coverage.

Spatial conclusions require distinct held-out sites. Conditional posterior
spread depends on the energy or covariance, fitted parameters and observed
sites. The forecast-map spread is propagated ensemble disagreement. These
quantities should not be interpreted as field-calibrated predictive error.
Review acceptance, effective sample size and split R-hat before using
posterior summaries.
