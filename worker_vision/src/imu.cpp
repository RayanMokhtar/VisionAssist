#include "imu.h"

#define BNO055_ADDR 0x28

// Registres BNO055
#define BNO055_CHIP_ID_ADDR        0x00
#define BNO055_PAGE_ID_ADDR        0x07
#define BNO055_OPR_MODE_ADDR       0x3D
#define BNO055_PWR_MODE_ADDR       0x3E
#define BNO055_SYS_TRIGGER_ADDR    0x3F
#define BNO055_UNIT_SEL_ADDR       0x3B
#define BNO055_CALIB_STAT_ADDR     0x35

// Données
#define BNO055_ACCEL_DATA_X_LSB    0x08
#define BNO055_GYRO_DATA_X_LSB     0x14
#define BNO055_EULER_H_LSB         0x1A
#define BNO055_LINEAR_ACCEL_X_LSB  0x28
#define BNO055_GRAVITY_X_LSB       0x2E
#define BNO055_TEMP_ADDR           0x34

// Modes
#define OPERATION_MODE_CONFIG      0x00
#define OPERATION_MODE_NDOF        0x0C
#define POWER_MODE_NORMAL          0x00

void IMU::write8(uint8_t reg, uint8_t value)
{
    uint8_t buffer[2] = {reg, value};

    if (write(i2c_fd, buffer, 2) != 2) {
        std::cerr << "Erreur écriture registre 0x"
                  << std::hex << (int)reg << std::dec << std::endl;
    }
}

uint8_t IMU::read8(uint8_t reg)
{
    if (write(i2c_fd, &reg, 1) != 1) {
        std::cerr << "Erreur demande registre" << std::endl;
        return 0;
    }

    uint8_t value = 0;
    if (read(i2c_fd, &value, 1) != 1) {
        std::cerr << "Erreur lecture registre" << std::endl;
        return 0;
    }

    return value;
}

int16_t IMU::read16(uint8_t reg)
{
    uint8_t buffer[2];

    if (write(i2c_fd, &reg, 1) != 1) {
        std::cerr << "Erreur demande registre 16 bits" << std::endl;
        return 0;
    }

    if (read(i2c_fd, buffer, 2) != 2) {
        std::cerr << "Erreur lecture 16 bits" << std::endl;
        return 0;
    }

    return (int16_t)((buffer[1] << 8) | buffer[0]);
}

IMU::Vec3 IMU::readVector(uint8_t reg, double scale)
{
    int16_t x = read16(reg);
    int16_t y = read16(reg + 2);
    int16_t z = read16(reg + 4);

    return {
        x / scale,
        y / scale,
        z / scale
    };
}

IMU::IMU(bool &init) {
    i2c_fd = open(device, O_RDWR);

    if (i2c_fd < 0) {
        std::cerr << "Impossible d'ouvrir " << device << std::endl;
        init = false;
        return;
    }

    if (ioctl(i2c_fd, I2C_SLAVE, BNO055_ADDR) < 0) {
        std::cerr << "Impossible de sélectionner l'adresse I2C 0x28" << std::endl;
        close(i2c_fd);
        init = false;
        return;
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(650));

    uint8_t chipId = read8(BNO055_CHIP_ID_ADDR);

    if (chipId != 0xA0) {
        std::cerr << "BNO055 non détecté. CHIP_ID = 0x"
                  << std::hex << (int)chipId << std::dec << std::endl;
        init = false;
        return;
    }

    // Mode config
    write8(BNO055_OPR_MODE_ADDR, OPERATION_MODE_CONFIG);
    std::this_thread::sleep_for(std::chrono::milliseconds(25));

    // Page 0
    write8(BNO055_PAGE_ID_ADDR, 0x00);

    // Mode normal
    write8(BNO055_PWR_MODE_ADDR, POWER_MODE_NORMAL);
    std::this_thread::sleep_for(std::chrono::milliseconds(10));

    // Unités par défaut :
    // acceleration en m/s²
    // gyro en dps
    // euler en degrés
    write8(BNO055_UNIT_SEL_ADDR, 0x00);

    // Mode NDOF : accel + gyro + magneto fusionnés
    write8(BNO055_OPR_MODE_ADDR, OPERATION_MODE_NDOF);
    std::this_thread::sleep_for(std::chrono::milliseconds(50));

    init = true;
}

void IMU::printCalibration()
{
    uint8_t cal = read8(BNO055_CALIB_STAT_ADDR);

    int sys   = (cal >> 6) & 0x03;
    int gyro  = (cal >> 4) & 0x03;
    int accel = (cal >> 2) & 0x03;
    int mag   = cal & 0x03;

    std::cout << "Calibration: "
              << "sys=" << sys
              << " gyro=" << gyro
              << " accel=" << accel
              << " mag=" << mag
              << std::endl;
}

IMU::Vec3 IMU::readAccel()
{
    // Vec3 accel = readVector(BNO055_ACCEL_DATA_X_LSB, 100.0);
    Vec3 accel = readVector(BNO055_LINEAR_ACCEL_X_LSB, 100.0);

    return accel;
}

int IMU::readTemp()
{
    int temp = (int8_t)read8(BNO055_TEMP_ADDR);

    return temp;
}
