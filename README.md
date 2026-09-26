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
