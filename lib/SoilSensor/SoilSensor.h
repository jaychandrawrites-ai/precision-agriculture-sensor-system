#pragma once

#include <Arduino.h>


struct SoilData
{
    float moisture;
    float temperature;
    uint16_t ec;
    float ph;

    bool valid;
};


class SoilSensor
{
public:

    SoilSensor(HardwareSerial &serial);

    void begin(uint32_t baud = 9600);

    SoilData read();


private:

    HardwareSerial &_serial;

    uint16_t calculateCRC(
        const uint8_t *data,
        size_t length
    );
};