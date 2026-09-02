//
// Created by diego on 02/09/2026.
//

#include "SegmentationProcessor.h"

#include <opencv2/imgproc.hpp>

SegmentationProcessor::SegmentationProcessor() {
    log("Created SegmentationProcessor");
}

SegmentationProcessor::~SegmentationProcessor() {
    log("Destroyed SegmentationProcessor");
}

void SegmentationProcessor::run(FrameContext& ctx) {
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(ctx.motionMask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    // Unlike MovementDetector (which sums *all* changed pixels to decide
    // "is there motion"), here we specifically want the single largest
    // contour: that's our best guess at "the moving object's shape",
    // rather than every scattered changed pixel.
    double bestArea = 0.0;
    int bestIdx = -1;
    for (size_t i = 0; i < contours.size(); ++i) {
        const double area = cv::contourArea(contours[i]);
        if (area > bestArea) {
            bestArea = area;
            bestIdx = static_cast<int>(i);
        }
    }

    cv::Mat mask = cv::Mat::zeros(ctx.motionMask.size(), CV_8UC1);
    if (bestIdx >= 0) {
        cv::drawContours(mask, contours, bestIdx, cv::Scalar(255), cv::FILLED);
    }

    // Morphological close (dilate then erode) fills small holes inside the
    // silhouette - e.g. a patch that didn't cross MovementDetector's diff
    // threshold - without changing the silhouette's overall size.
    cv::morphologyEx(mask, mask, cv::MORPH_CLOSE,
                      cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(9, 9)));

    ctx.segmentedMask = mask;

    if (bestIdx >= 0) {
        log("Segmented object, area=" + std::to_string(bestArea));
    } else {
        log("No contour found to segment");
    }
}

std::string SegmentationProcessor::name() const {
    return "SegmentationProcessor";
}
