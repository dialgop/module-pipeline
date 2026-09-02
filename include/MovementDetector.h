//
// Created by diego on 26/11/2025.
//

#pragma once
#include <opencv2/core.hpp>

#include "IModule.h"
#include "LoggerBase.h"

/// Detects motion between consecutive frames via frame differencing:
/// grayscale + blur each frame, absdiff against the previous one, threshold
/// the result. Reports whether enough pixels changed to call it "motion",
/// plus the diff mask and a bounding box over every changed pixel, via the
/// shared FrameContext. See README.md for why this counts total changed
/// pixels rather than looking for one large contour.
class MovementDetector : public IModule, LoggerBase {
public:
    MovementDetector();
    ~MovementDetector() override;

    /// Compares ctx.frame against the previous frame and sets
    /// ctx.motionDetected / ctx.motionMask / ctx.motionRoi. On the very
    /// first call there's no previous frame to compare against yet, so no
    /// motion is (or can be) reported.
    void run(FrameContext& ctx) override;

    [[nodiscard]] std::string name() const override;

private:
    cv::Mat prevGray; // grayscale, blurred previous frame - state that persists across run() calls
};
