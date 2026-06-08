#ifndef IMU_H
#define IMU_H

#include <iostream>
#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <linux/i2c-dev.h>
#include <cstdint>
#include <cmath>
#include <chrono>
#include <thread>

class IMU {

    public : 
        struct Vec3 {
            double x;
            double y;
            double z;
        };

        IMU(bool &init);

        Vec3 readAccel();
        int readTemp();

        void printCalibration();
    
    private :

        void write8(uint8_t reg, uint8_t value);
        uint8_t read8(uint8_t reg);
        int16_t read16(uint8_t reg);
        Vec3 readVector(uint8_t reg, double scale);

        int i2c_fd = -1;

        const char* device = "/dev/i2c-7";
};

#endif
