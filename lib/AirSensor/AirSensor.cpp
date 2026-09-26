#include "AirSensor.h"


AirSensor::AirSensor()
{
}


// =====================================================
// START SENSOR
// =====================================================

bool AirSensor::begin()
{
    if (!sht45.begin())
    {
        return false;
    }

    sht45.setPrecision(SHT4X_HIGH_PRECISION);
    sht45.setHeater(SHT4X_NO_HEATER);

    return true;
}


// =====================================================
// READ SENSOR
// =====================================================

AirData AirSensor::read()
{
    AirData data;

    sensors_event_t humidity;
    sensors_event_t temperature;

    if (!sht45.getEvent(&humidity, &temperature))
    {
        data.valid = false;
        return data;
    }

    data.temperature = temperature.temperature;
    data.humidity = humidity.relative_humidity;

    data.valid = true;

    return data;
}