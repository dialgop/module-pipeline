//
// Created by diego on 02/09/2026.
//

#pragma once
#include <opencv2/core.hpp>

#include "IModule.h"
#include "LoggerBase.h"

/// Refines MovementDetector's coarse motion mask into a precise silhouette
/// of the moving object: finds the largest contour in the mask, fills it
/// in, and cleans up small holes with a morphological close. Meant to run
/// only when ctx.motionDetected is true - that's enforced by the pipeline's
/// should_run gate in main.cpp, not by an internal check here (see
/// README.md).
class SegmentationProcessor : public IModule, LoggerBase {
public:
    SegmentationProcessor();
    ~SegmentationProcessor() override;

    /// Builds ctx.segmentedMask from ctx.motionMask: finds the largest
    /// contour, fills it solid, then morphologically closes small gaps.
    void run(FrameContext& ctx) override;

    [[nodiscard]] std::string name() const override;
};
