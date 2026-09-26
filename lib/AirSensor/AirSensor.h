#pragma once

#include <Arduino.h>
#include <Adafruit_SHT4x.h>


// Information returned by the SHT45
struct AirData
{
    float temperature;
    float humidity;

    bool valid;
};


class AirSensor
{
public:

    AirSensor();

    // Start the sensor
    bool begin();

    // Take a measurement
    AirData read();


private:

    Adafruit_SHT4x sht45;
};