//
// Created by diego on 02/09/2026.
//

#pragma once
#include <opencv2/core.hpp>

#include "IModule.h"
#include "LoggerBase.h"

/// Builds a precise silhouette for each of ObjectDetector's confirmed
/// people: within each ctx.detectedPeople box, finds the largest contour in
/// MovementDetector's motion mask, fills it in, and cleans up small holes
/// with a morphological close. Segmenting per already-disambiguated person
/// box (rather than the whole frame's motion mask at once) is what lets
/// several people close together each get their own correct silhouette,
/// instead of one contour-search finding a single "biggest blob" across
/// everyone combined. Meant to run only when ctx.detectedPeople is
/// non-empty - that's enforced by the pipeline's should_run gate in
/// main.cpp, not by an internal check here (see README.md).
class SegmentationProcessor : public IModule, LoggerBase {
public:
    SegmentationProcessor();
    ~SegmentationProcessor() override;

    /// Builds ctx.segmentedMask by finding, per ctx.detectedPeople box, the
    /// largest contour within that box's crop of ctx.motionMask, filling it
    /// solid, then morphologically closing small gaps.
    void run(FrameContext& ctx) override;

    [[nodiscard]] std::string name() const override;
};
