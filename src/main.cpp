#include <Arduino.h>

#include "config.h"
#include "AirSensor.h"
#include "SoilSensor.h"


// =====================================================
// CREATE OUR SENSORS
// =====================================================

AirSensor airSensor;

SoilSensor soilSensor(Serial1);


// =====================================================
// SETUP
// =====================================================

void setup()
{
    Serial.begin(SERIAL_BAUD);

    delay(2000);


    Serial.println();
    Serial.println(
        "======================================"
    );

    Serial.println(
        " PRECISION AGRICULTURE SENSOR SYSTEM"
    );

    Serial.println(
        "======================================"
    );


    // ---------------------------------------------
    // AIR SENSOR
    // ---------------------------------------------

    if (airSensor.begin())
    {
        Serial.println(
            "SHT45: OK"
        );
    }
    else
    {
        Serial.println(
            "SHT45: NOT FOUND"
        );
    }


    // ---------------------------------------------
    // SOIL SENSOR
    // ---------------------------------------------

    soilSensor.begin(SOIL_BAUD);

    Serial.println(
        "SEN0604 interface started."
    );


    Serial.println(
        "======================================"
    );
}


// =====================================================
// MAIN LOOP
// =====================================================

void loop()
{
    // Read sensors

    AirData air =
        airSensor.read();

    SoilData soil =
        soilSensor.read();


    Serial.println();
    Serial.println(
        "======================================"
    );

    Serial.println(
        "          SENSOR READINGS"
    );

    Serial.println(
        "======================================"
    );


    // =================================================
    // AIR
    // =================================================

    Serial.println("--- AIR ---");


    if (air.valid)
    {
        Serial.print(
            "Temperature: "
        );

        Serial.print(
            air.temperature,
            2
        );

        Serial.println(" C");


        Serial.print(
            "Humidity:    "
        );

        Serial.print(
            air.humidity,
            2
        );

        Serial.println(" %RH");
    }
    else
    {
        Serial.println(
            "ERROR: SHT45 read failed."
        );
    }


    // =================================================
    // SOIL
    // =================================================

    Serial.println();

    Serial.println("--- SOIL ---");


    if (soil.valid)
    {
        Serial.print(
            "Moisture:    "
        );

        Serial.print(
            soil.moisture,
            1
        );

        Serial.println(" %");


        Serial.print(
            "Temperature: "
        );

        Serial.print(
            soil.temperature,
            1
        );

        Serial.println(" C");


        Serial.print(
            "EC:          "
        );

        Serial.print(
            soil.ec
        );

        Serial.println(
            " uS/cm"
        );


        Serial.print(
            "pH:          "
        );

        Serial.println(
            soil.ph,
            1
        );
    }
    else
    {
        Serial.println(
            "ERROR: SEN0604 not responding."
        );
    }


    Serial.println(
        "======================================"
    );


    delay(SENSOR_INTERVAL_MS);
}