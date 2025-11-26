//
// Created by diego on 26/11/2025.
//

#include "ColorProcessor.h"

ColorProcessor::ColorProcessor() {
    log("Created ColorProcessor");
}

ColorProcessor::~ColorProcessor() {
    log("Terminated ColorProcessor");
}

void ColorProcessor::run() {
    log("ColorProcessor balance processing");
}

std::string ColorProcessor::name() const {
    return "ColorProcessor";
}