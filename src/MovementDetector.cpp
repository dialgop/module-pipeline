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
    cv::Mat gray;
    cv::cvtColor(ctx.frame, gray, cv::COLOR_BGR2GRAY);
    // Blurring first means small lighting/sensor noise doesn't survive the
    // absdiff below as a false "changed" pixel.
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
    // Dilate to merge nearby changed-pixel blobs into fewer, larger regions
    // before we count them, so scattered single-pixel noise doesn't survive
    // as isolated specks.
    cv::dilate(thresh, thresh, cv::Mat(), cv::Point(-1, -1), 2);

    // We gate on *total* changed pixels rather than the single largest
    // contour: on real footage, motion often shows up as many small
    // scattered changed regions rather than one solid blob (see
    // README.md), so summing catches real motion that a single-contour
    // check would miss entirely.
    const int changedPixels = cv::countNonZero(thresh);
    ctx.motionDetected = changedPixels >= kMinMotionPixels;
    ctx.motionMask = thresh;
    // boundingRect() accepts a mask directly, treating its nonzero pixels
    // as a point set - this is the bounding box over every changed pixel,
    // not just one contour.
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
