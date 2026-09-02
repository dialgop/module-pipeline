//
// Created by diego on 26/11/2025.
//

#include "ColorProcessor.h"

#include <opencv2/core.hpp>

ColorProcessor::ColorProcessor() {
    log("Created ColorProcessor");
}

ColorProcessor::~ColorProcessor() {
    log("Terminated ColorProcessor");
}

void ColorProcessor::run(FrameContext& ctx) {
    const cv::Scalar meanColor = cv::mean(ctx.frame, ctx.segmentedMask);
    log("Average color (BGR) = [" + std::to_string(meanColor[0]) + ", "
        + std::to_string(meanColor[1]) + ", " + std::to_string(meanColor[2]) + "]");
}

std::string ColorProcessor::name() const {
    return "ColorProcessor";
}