//
// Created by diego on 26/11/2025.
//

#include "CameraSensor.h"

CameraSensor::CameraSensor(int id) : cameraId(id) {
    log("Created CameraSensor");
}

CameraSensor::~CameraSensor() {
    log("Finished CameraSensor");
}

void CameraSensor::run() {
    log("Capturing camera frame " + std::to_string(cameraId));
}

std::string CameraSensor::name() const {
    return "CameraSensor" + std::to_string(cameraId);
}