# Hardware and offline replay

The original Arduino MKR WAN 1310 firmware remains in the repository. The
existing SHT45 and SEN0604 reading path is preserved. Jupiter also accepts
caller-mapped packet JSON with optional device, boot, sequence, measurement
time and GPS status. An actual AquaMap configuration must be derived from
its receiver format. The native protocol and live hardware remain unverified.

```sh
python acquire.py --help
python acquisition_archive.py --help
python Pipeline.py prepare-session --help
python Pipeline.py forecast-session --help
```

Capture and inference are independent processes. Verify raw integrity before
replay. Recovery writes to a new archive, preserving the original.
Per-device packet flags and coverage distinguish missing records, duplicates
and restarts. Use only supplied, measured soil and leaf calibration values.
GPS fixes must have been received by a spatial forecast's origin. Keep
serial settings, calibration records and exact field locations local.

## Recorded-session readiness

```sh
python Pipeline.py check-session readings.jsonl field-config.json readiness.json --checkpoint model.json
```

This offline check reports whether the latest closed hourly bins can supply a
forecast. It reports incomplete features, ET warmup, the latest closed bin
context, and whether the checkpoint's field physics metadata matches.
The default ET warmup plus the three-hour forecast context require at least
26 consecutive hourly temperatures. The report contains no GPS coordinates.
For a specific operational cutoff, set `hourly.as_of` in the field configuration.

## AI Trace multi-location field survey

Start the local dashboard using a workspace dedicated to the survey. Pass
the serial device path or COM port shown by your operating system.

```sh
python -m Dashboard.Server --workspace ./jupiter-field --allow-port YOUR_SERIAL_PORT
```

Open the loopback address printed by the server and choose **AI Trace**.

1. Create a survey and connect the receiver. Select the allowed port and
   provide a mapping when your receiver uses a verified JSON format.
2. Check receipt counts, decoded measurements and GPS status. Start a named
   recording. A valid manual coordinate pair is optional.
3. Select **Stop Location** to save the visit. The receiver stays connected.
   Use **Record New Location** or **Revisit Location** for the next visit.
4. Select **Finish Survey** when finished. Export the full JSON archive to
   retain the checksum, measurements, timestamps and raw packet bytes.
   CSV and raw packet exports are also available for individual visits.
5. Choose a map time and lookback window for an observed spatial composite.
   Run readiness using actual field calibration settings. Forecasts require
   an eligible visit and compatible saved model. Brief visits still support
   collection, export and mapping without any synthetic hourly history.

If the receiver disconnects, the current visit ends with an interruption
status. After reconnection, start a new visit. If the dashboard is interrupted,
restart it with the same workspace to recover saved visits. Keep survey
archives and exact field positions local.

The native AquaMap protocol still requires a verified firmware source or
receiver capture. A labeled software demonstration is available via
`python -m Examples.SyntheticTrace OUTPUT_DIRECTORY`.

Saved forecast reports can also be mapped at a common issue and valid time:

```sh
python trace_analysis.py --root ./jupiter-field/surveys forecast-map SURVEY_ID map.json --forecast-report forecasts.json --feature air_temperature_c --origin 2025-06-01T12:00Z --valid-at 2025-06-01T15:00Z
```

Use times present in your report. The command omits sites without eligible
coordinates and labels the result as a point forecast map.
