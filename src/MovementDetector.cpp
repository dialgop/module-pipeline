//
// Created by diego on 26/11/2025.
//

#include "MovementDetector.h"

MovementDetector::MovementDetector() {
    log("Created MovementDetector");
}

MovementDetector::~MovementDetector() {
    log("Destroyed MovementDetector");
}

void MovementDetector::run() {
    log("Analyzing MovementDetector... Detected Movement!");
}

std::string MovementDetector::name() const {
    return "MovementDetector";
}