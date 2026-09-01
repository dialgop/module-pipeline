//
// Created by diego on 26/11/2025.
//

#include "CameraSensor.h"

// cv::VideoCapture's own constructor opens the device immediately - by the
// time this initializer list finishes, `capture` is already either open or
// failed. That's the RAII step; the isOpened() check below just reports it
// via our own logger instead of leaving the failure silent.
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
    // cv::VideoCapture's own destructor would release the device anyway;
    // this is just an explicit, loggable "we're done with it" step.
    capture.release();
    log("Finished " + sourceLabel);
}

void CameraSensor::run(FrameContext& ctx) {
    // operator>> decodes the next frame straight into ctx.frame, reusing
    // its existing buffer when the size/type already match instead of
    // reallocating every call.
    capture >> ctx.frame;
    ctx.frameValid = !ctx.frame.empty();
}

std::string CameraSensor::name() const {
    return sourceLabel;
}