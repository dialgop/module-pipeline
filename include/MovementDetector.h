//
// Created by diego on 26/11/2025.
//

#pragma once
#include <opencv2/core.hpp>

#include "IModule.h"
#include "LoggerBase.h"

/// Detects motion between consecutive frames via frame differencing:
/// grayscale + blur each frame, absdiff against the previous one, threshold
/// the result, then label each separate blob of changed pixels as its own
/// cluster. Reports whether at least one cluster is big enough to call
/// "motion", plus the diff mask and one bounding box per cluster, via the
/// shared FrameContext. See README.md for why detection is per-cluster
/// rather than one combined region or a single largest contour.
class MovementDetector : public IModule, LoggerBase {
public:
    MovementDetector();
    ~MovementDetector() override;

    /// Compares ctx.frame against the previous frame and sets
    /// ctx.motionDetected / ctx.motionMask / ctx.motionRegions. On the very
    /// first call there's no previous frame to compare against yet, so no
    /// motion is (or can be) reported.
    void run(FrameContext& ctx) override;

    [[nodiscard]] std::string name() const override;

private:
    cv::Mat prevGray; // grayscale, blurred previous frame - state that persists across run() calls
};
