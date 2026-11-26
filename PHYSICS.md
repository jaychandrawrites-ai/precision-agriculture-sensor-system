# Feature physics and units

| Feature | Units | Historical source |
| --- | --- | --- |
| Air temperature | °C | Station measurement |
| Relative humidity | % | Station measurement |
| Leaf wetness index | 0–1 | Coupled leaf energy and surface storage simulation |
| Reference ET | mm per hour | Trailing-temperature Hargreaves proxy in the default pipeline |
| Actual ET | mm per hour | Soil water bucket scenario and stress coefficient |
| Cotton heat stress | modeled index | Canopy temperature and heat accumulation/recovery model |

The leaf model computes radiation, convection and latent heat exchange,
then advances leaf temperature and surface water storage. Storage is bounded
between dry and saturated conditions. Field wetness uses an independently
calibrated sensor. Historical simulation and sensor observations use
different observation models.

The root-zone bucket tracks depletion in millimeters. Rain and irrigation
less runoff enter the bucket, drainage removes excess water, and crop ET
is limited by available water. The conservation residual is checked by
tests. Historical irrigation and soil properties must be declared as
scenario assumptions when measurements are unavailable.

The optional hourly FAO-56 comparison uses net radiation and soil heat in
MJ m⁻² per hour, wind at 2 m in m/s, and vapor and atmospheric pressure in
kPa. The nighttime cloud ratio is supplied explicitly. The comparison
leaves the default shared forecasting features unchanged. Agreement between
the two estimators does not establish accuracy against observed water flux.
[FAO Irrigation and Drainage Paper 56, Chapter 4](https://www.fao.org/4/x0490e/x0490e08.htm)
