//
// Created by diego on 26/11/2025.
//

#include "MovementDetector.h"

#include <opencv2/imgproc.hpp>

namespace {
    // Tunable magic numbers, kept together so they're easy to find and adjust.
    constexpr double kDiffThreshold = 25.0; // absdiff value above which a pixel counts as "changed"
    constexpr int kMinMotionPixels = 200;   // a cluster's changed-pixel count below which we call it noise, not motion
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
    // A walking person's torso barely changes frame-to-frame - motion
    // concentrates at moving limb edges - so the raw mask fragments into
    // several small blobs per person (a hand here, a foot there) rather
    // than one solid shape. A large structuring element bridges those
    // gaps back into one cluster per person, without merging separate
    // people who are typically much further apart than this reaches.
    static const cv::Mat kDilateKernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(25, 25));
    cv::dilate(thresh, thresh, kDilateKernel);

    // connectedComponentsWithStats labels every separate blob of changed
    // pixels individually (label 0 is always the background), instead of
    // collapsing everything into one combined region the way a single
    // boundingRect(thresh) or "sum all changed pixels" check would. Each
    // real blob becomes its own candidate ROI - e.g. one per person - and
    // we filter out clusters too small to be real motion the same way the
    // old total-pixel gate filtered noise (see README.md), just applied
    // per-cluster instead of to the sum.
    cv::Mat labels, stats, centroids;
    const int numLabels = cv::connectedComponentsWithStats(thresh, labels, stats, centroids);

    std::vector<cv::Rect> regions;
    for (int label = 1; label < numLabels; ++label) {
        const int area = stats.at<int>(label, cv::CC_STAT_AREA);
        if (area < kMinMotionPixels) continue;
        regions.emplace_back(stats.at<int>(label, cv::CC_STAT_LEFT),
                              stats.at<int>(label, cv::CC_STAT_TOP),
                              stats.at<int>(label, cv::CC_STAT_WIDTH),
                              stats.at<int>(label, cv::CC_STAT_HEIGHT));
    }

    // One person can still fragment into several nearby regions (e.g. head
    // vs. legs) even after the dilation above - merge any regions whose
    // margin-expanded boxes overlap, repeatedly, until nothing more merges.
    constexpr int kMergeMargin = 40;
    bool merged = true;
    while (merged) {
        merged = false;
        for (size_t i = 0; i < regions.size() && !merged; ++i) {
            const cv::Rect expanded(regions[i].x - kMergeMargin, regions[i].y - kMergeMargin,
                                     regions[i].width + 2 * kMergeMargin, regions[i].height + 2 * kMergeMargin);
            for (size_t j = i + 1; j < regions.size(); ++j) {
                if ((expanded & regions[j]).area() > 0) {
                    regions[i] |= regions[j]; // cv::Rect::operator|= grows regions[i] to cover both
                    regions.erase(regions.begin() + static_cast<long>(j));
                    merged = true;
                    break;
                }
            }
        }
    }

    ctx.motionDetected = !regions.empty();
    ctx.motionMask = thresh;
    ctx.motionRegions = std::move(regions);

    prevGray = gray;

    if (ctx.motionDetected) {
        log("Detected movement, clusters=" + std::to_string(ctx.motionRegions.size()));
    } else {
        log("No movement detected");
    }
}

std::string MovementDetector::name() const {
    return "MovementDetector";
}
