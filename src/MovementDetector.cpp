//
// Created by diego on 26/11/2025.
//

#include "MovementDetector.h"

#include <opencv2/imgproc.hpp>

namespace {
    // Tunable magic numbers, kept together so they're easy to find and adjust.
    constexpr double kDiffThreshold = 25.0; // absdiff value above which a pixel counts as "changed"
    constexpr int kMinMotionPixels = 200;   // total changed pixels below which we call it noise, not motion
}

MovementDetector::MovementDetector() {
    log("Created MovementDetector");
}

MovementDetector::~MovementDetector() {
    log("Destroyed MovementDetector");
}

void MovementDetector::run(FrameContext& ctx) {
    log("Analyzing MovementDetector... Detected Movement!");
    cv::Mat gray;
    cv::cvtColor(ctx.frame, gray, cv::COLOR_BGR2GRAY);
    cv::GaussianBlur(gray, gray, cv::Size(21, 21), 0);

    if (prevGray.empty()) {
        // First frame ever seen - nothing to compare against yet.
        prevGray = gray;
        ctx.motionDetected = false;
        return;
    }

    cv::Mat diff;
    cv::absdiff(prevGray, gray, diff);

    cv::Mat thresh;
    cv::threshold(diff, thresh, kDiffThreshold, 255, cv::THRESH_BINARY);
    cv::dilate(thresh, thresh, cv::Mat(), cv::Point(-1, -1), 2);

    const int changedPixels = cv::countNonZero(thresh);
    ctx.motionDetected = changedPixels >= kMinMotionPixels;
    ctx.motionMask = thresh;
    ctx.motionRoi = cv::boundingRect(thresh);

    prevGray = gray;

    if (ctx.motionDetected) {
        log("Detected movement, changedPixels=" + std::to_string(changedPixels));
    } else {
        log("No movement detected");
    }
}

std::string MovementDetector::name() const {
    return "MovementDetector";
}
