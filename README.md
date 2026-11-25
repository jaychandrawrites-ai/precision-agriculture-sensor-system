# Precision Agriculture PlatformIO Project

Hardware:
- Arduino MKR WAN 1310 (ABX00029)
- Adafruit SHT45
- DFRobot DFR0845 isolated RS485-to-UART adapter
- DFRobot SEN0604 4-in-1 soil sensor

## Open in VS Code / PlatformIO

1. Install the PlatformIO IDE extension in Visual Studio Code.
2. Extract this project folder.
3. In PlatformIO, choose **Open Project** and select this folder.
4. Connect the MKR WAN 1310 by USB.
5. Build and upload.
6. Open the PlatformIO Serial Monitor. It is configured for 9600 baud.

PlatformIO will download the Arduino framework and required Adafruit libraries.

## SHT45 wiring

SHT45 -> MKR WAN 1310

- VIN -> 3V3
- GND -> GND
- SDA -> SDA
- SCL -> SCL

The SHT45 and DFR0845 logic side may share the 3.3 V and GND rails.

## DFR0845 logic-side wiring

DFR0845 -> MKR WAN 1310

- + -> 3V3
- - -> GND
- T (TX) -> MKR RX
- R (RX) -> MKR TX

The sketch uses `Serial1` for this hardware UART.

## SEN0604 RS485 side

SEN0604 -> DFR0845 sensor/RS485 side

- Brown (power +) -> 12V output
- Black (ground) -> GND
- Yellow (RS485 A) -> A
- Blue (RS485 B) -> B

The external supply for the DFR0845 isolated side connects to the DFR0845's
12V-IN/GND input according to the module documentation.

IMPORTANT: Never connect 12 V to the MKR WAN 1310's 3V3, TX, RX, SDA, SCL,
or other GPIO pins.

## Expected output

The serial monitor should show air temperature and humidity. If the SEN0604
is communicating, it will also show:

- Soil moisture
- Soil temperature
- Soil EC
- Soil pH

If the soil probe is disconnected or RS485 communication is not working,
the program reports:

`ERROR: No valid response from SEN0604.`

## Current SEN0604 assumptions

The code assumes the factory-default Modbus RTU configuration:
- Address: 1
- Baud: 9600
- 8 data bits
- No parity
- 1 stop bit

## Jupiter offline research pipeline

Use Python 3.10 or later. From the repository root:

```sh
python -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest discover -s Tests -v
python main.py demo
```

The demo uses an explicitly synthetic periodic series. Its metrics describe that
fixture only. Training selects a checkpoint using one-step validation loss.
The report includes held-out one-step and per-horizon recursive metrics against
persistence. Recursive errors can grow substantially even with low one-step loss.

### Prepare, train, and forecast

Supply an hourly CSV with these headers:

```csv
timestamp,air_temperature_c,relative_humidity_pct,lwi,eto_mm_interval,eta_mm_interval,cotton_heat_stress
```

Timestamps must be increasing and unique, with an explicit UTC offset. Use
right-end timestamps for intervals, millimeters per interval for both ET series,
and a 0–1 leaf wetness index. Training skips incomplete windows and windows
that cross missing hours. Keep the feature estimation methods consistent
between historical and field CSVs.

```sh
python train.py historical_hourly.csv temporal.json --epochs 100
python forecast.py field_hourly.csv temporal.json --steps 6 > predictions.csv
```

Forecast uses a saved model and the last three consecutive hourly field samples.
It needs no live AZMET input. The project still needs a verified AZMET column
mapping and measured field calibration before the prepared CSVs can be generated
from their original sources. See BUILD_LOG.md for the remaining work.

### Capture Arduino output or replay it offline

```sh
python acquire.py --port /dev/ttyACM0 session-01
python acquire.py --replay session-01/raw.jsonl session-replay
```

Use a fresh output directory. Stop capture with Ctrl+C. The capture saves the
original bytes and host UTC receive time per line in `raw.jsonl` and parsed
readings with quality flags in `readings.jsonl`. Replay of a raw JSONL preserves
the logged timestamps. A plain-text replay assigns new receive times. The
checked-in sketch provides air and soil fields. Leaf wetness and GPS require the
AquaMap firmware and calibration before an acquired session can feed forecasting.

### Compare research methods

`python experiments.py historical_hourly.csv` evaluates a fixed network and
the architectures selected by the classical and quantum-inspired searches.
The default searches each train 80 candidates for 20 epochs. Both use
validation scores for selection and evaluate their final models on the same
chronological test split. The amplitude-based method runs classically.

For spatial measurements, use `Spatial.SpatialGrid.local_coordinates` to
project GPS into local meters and `Spatial.Validation.evaluate_held_out`
to compare variational reconstruction, IDW and ordinary kriging at
separate measurement sites. Choose the kriging range with training data
or domain knowledge before scoring the held-out sites.
