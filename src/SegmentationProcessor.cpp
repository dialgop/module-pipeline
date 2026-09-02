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
    cv::Mat mask = cv::Mat::zeros(ctx.motionMask.size(), CV_8UC1);
    const cv::Rect frameBounds(0, 0, ctx.motionMask.cols, ctx.motionMask.rows);

    int segmentedCount = 0;
    for (const cv::Rect& personBox : ctx.detectedPeople) {
        const cv::Rect box = personBox & frameBounds;
        if (box.width <= 0 || box.height <= 0) continue;

        // Searching for the largest contour within just this person's crop
        // (instead of the whole frame) is what makes this correct for
        // several people at once: each gets their own contour search,
        // scoped to a region ObjectDetector already confirmed is them.
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(ctx.motionMask(box), contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        double bestArea = 0.0;
        int bestIdx = -1;
        for (size_t i = 0; i < contours.size(); ++i) {
            const double area = cv::contourArea(contours[i]);
            if (area > bestArea) {
                bestArea = area;
                bestIdx = static_cast<int>(i);
            }
        }
        if (bestIdx < 0) continue;

        // mask(box) is a view over the same pixel data as `mask`, offset to
        // this person's location - findContours returned points relative
        // to that same offset (since it ran on ctx.motionMask(box), not the
        // full mask), so drawing them here lands in the right place with no
        // manual coordinate translation needed.
        cv::Mat maskRoi = mask(box);
        cv::drawContours(maskRoi, contours, bestIdx, cv::Scalar(255), cv::FILLED);
        ++segmentedCount;
    }

    // Morphological close (dilate then erode) fills small holes inside each
    // silhouette - e.g. a patch that didn't cross MovementDetector's diff
    // threshold - without changing the silhouettes' overall size.
    cv::morphologyEx(mask, mask, cv::MORPH_CLOSE,
                      cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(9, 9)));

    ctx.segmentedMask = mask;

    if (segmentedCount > 0) {
        log("Segmented " + std::to_string(segmentedCount) + " person(s)");
    } else {
        log("No contour found to segment");
    }
}

std::string SegmentationProcessor::name() const {
    return "SegmentationProcessor";
}
