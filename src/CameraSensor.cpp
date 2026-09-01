//
// Created by diego on 26/11/2025.
//

#include "CameraSensor.h"

CameraSensor::CameraSensor(int cameraIndex)
    : capture(cameraIndex), sourceLabel("CameraSensor" + std::to_string(cameraIndex)) {
    if (!capture.isOpened()) {
        error("Failed to open camera index " + std::to_string(cameraIndex));
    }
    log("Created " + sourceLabel);
}

CameraSensor::CameraSensor(const std::string& videoPath)
    : capture(videoPath), sourceLabel("CameraSensor(" + videoPath + ")") {
    if (!capture.isOpened()) {
        error("Failed to open video file " + videoPath);
    }
    log("Created " + sourceLabel);
}

CameraSensor::~CameraSensor() {
    capture.release();
    log("Finished " + sourceLabel);
}

void CameraSensor::run(FrameContext& ctx) {
    capture >> ctx.frame;
    ctx.frameValid = !ctx.frame.empty();
}

std::string CameraSensor::name() const {
    return sourceLabel;
}