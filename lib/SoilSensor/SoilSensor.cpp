#include "SoilSensor.h"


// =====================================================
// SEN0604 MODBUS COMMAND
// =====================================================
//
// Address:        1
// Function:       03
// Start register: 0000
// Registers:      4
//
// Reads:
//
// 0000 = moisture
// 0001 = temperature
// 0002 = EC
// 0003 = pH
//

static const uint8_t soilRequest[8] =
{
    0x01,
    0x03,
    0x00,
    0x00,
    0x00,
    0x04,
    0x44,
    0x09
};


// =====================================================
// CONSTRUCTOR
// =====================================================

SoilSensor::SoilSensor(HardwareSerial &serial)
    : _serial(serial)
{
}


// =====================================================
// START COMMUNICATION
// =====================================================

void SoilSensor::begin(uint32_t baud)
{
    _serial.begin(baud);
}


// =====================================================
// MODBUS CRC
// =====================================================

uint16_t SoilSensor::calculateCRC(
    const uint8_t *data,
    size_t length
)
{
    uint16_t crc = 0xFFFF;

    for (size_t pos = 0; pos < length; pos++)
    {
        crc ^= (uint16_t)data[pos];

        for (uint8_t i = 0; i < 8; i++)
        {
            if (crc & 0x0001)
            {
                crc >>= 1;
                crc ^= 0xA001;
            }
            else
            {
                crc >>= 1;
            }
        }
    }

    return crc;
}


// =====================================================
// READ SEN0604
// =====================================================

SoilData SoilSensor::read()
{
    SoilData data;

    data.valid = false;

    // Clear old bytes
    while (_serial.available())
    {
        _serial.read();
    }


    // Send request
    _serial.write(
        soilRequest,
        sizeof(soilRequest)
    );

    _serial.flush();


    // SEN0604 should return 13 bytes
    uint8_t response[13];

    size_t count = 0;

    unsigned long startTime = millis();


    // Wait up to one second
    while (
        millis() - startTime < 1000 &&
        count < sizeof(response)
    )
    {
        if (_serial.available())
        {
            response[count++] =
                (uint8_t)_serial.read();
        }
    }


    // Did we get 13 bytes?
    if (count != sizeof(response))
    {
        return data;
    }


    // Check Modbus response
    if (
        response[0] != 0x01 ||
        response[1] != 0x03 ||
        response[2] != 0x08
    )
    {
        return data;
    }


    // Check CRC
    uint16_t calculatedCRC =
        calculateCRC(response, 11);

    uint16_t receivedCRC =
        response[11] |
        ((uint16_t)response[12] << 8);


    if (calculatedCRC != receivedCRC)
    {
        return data;
    }


    // =================================================
    // EXTRACT VALUES
    // =================================================

    uint16_t rawMoisture =
        ((uint16_t)response[3] << 8) |
        response[4];


    int16_t rawTemperature =
        (int16_t)(
            ((uint16_t)response[5] << 8) |
            response[6]
        );


    uint16_t rawEC =
        ((uint16_t)response[7] << 8) |
        response[8];


    uint16_t rawPH =
        ((uint16_t)response[9] << 8) |
        response[10];


    // Convert raw values

    data.moisture =
        rawMoisture / 10.0f;

    data.temperature =
        rawTemperature / 10.0f;

    data.ec =
        rawEC;

    data.ph =
        rawPH / 10.0f;

    data.valid = true;


    return data;
}