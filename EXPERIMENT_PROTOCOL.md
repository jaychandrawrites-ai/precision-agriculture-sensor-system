# Experiment protocol

Use explicit fit, model-selection, calibration and held-out periods. Fit
normalization, harmonic coefficients and neural weights on fit data only.
Select epochs and residual blend weights on the selection period. Freeze the
model before calibration and testing. Rolling-origin evaluation admits earlier
observations available at each origin and excludes future observations,
including when repeating the last known seasonal cycle.

The Yuma 2018 data supplied fitting, selection and calibration. Yuma 2019
supplied the fixed-year test. Maricopa 2019 was evaluated with the frozen Yuma
forecast and interval estimates. Its values did not tune model parameters,
blend weights or interval radii. For a clean follow-up, designate additional
years and stations before comparing new variants. Prior Tucson 2019 results
are exploratory context for changes developed in response to those results.

Forecast score reports use common complete windows. Paired RMSE comparisons
resample fixed UTC blocks and report adjusted intervals for the declared
temperature-lead family. Spatial evaluation separates site groups and
requires measured held-out labels. Synthetic field runs verify behavior and
reproducibility.

Match candidate counts and compute caps across evolutionary methods. Track
actual epochs, parameter-sample steps, model size and runtime separately.
Source checksums, settings and full results can be regenerated with the
commands in [BUILD_LOG.md](BUILD_LOG.md).
