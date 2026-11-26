# Data and artifact formats

Prepared weather CSVs contain an offset-aware hourly `timestamp` followed
by the six ordered columns from `config.FEATURES`. Timestamps identify
interval ends and are normalized to UTC. The AZMET raw archive requires a
station/year mapping, declared units, source checksum and timezone. The
catalog pins supported releases. For legacy `1–24` hours, hour 24 denotes
the next midnight. Registry validation verifies source and prepared artifact
hashes. Source records remain outside the checkout.

A sensor archive retains received bytes, offsets, receipt timestamps and
parsed measurements. Optional measurement times use an explicit offset.
GPS is kept separate from temporal features. Hourly bins close only when
permitted by the receipt cutoff. Missing or invalid values carry quality
flags. Forecasts require a recent physically valid history. The default
temperature-derived ET requires 24 consecutive causal temperature
observations before complete feature vectors are available.

Spatial CSVs identify latitude, longitude and observation time. Forecast
ensemble mapping requires the same sites in every `draw_id`, a common
origin and valid time, and `value_origin=model_forecast`. Data arrays use
`[north, east, feature]` in multi-feature ensemble maps and numeric-only
NPZ archives. Coordinates and raw sensor exports should remain local.
