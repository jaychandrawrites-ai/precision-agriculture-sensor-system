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
