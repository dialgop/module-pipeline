//
// Created by diego on 02/09/2026.
//

#pragma once


#include <opencv2/dnn.hpp>

#include "IModule.h"
#include "LoggerBase.h"

/// Builds a precise silhouette for each of ObjectDetector's confirmed
/// people using a real semantic segmentation model (PP-HumanSeg,
/// Apache-2.0 - see external/models/) run on each ctx.detectedPeople crop.
/// Unlike ObjectDetector's job, this doesn't need to tell multiple people
/// apart within one region - each crop is already (usually) one person, so
/// plain person-vs-background segmentation is enough; no anchor decoding
/// or NMS involved. Meant to run only when ctx.detectedPeople is
/// non-empty - that's enforced by the pipeline's should_run gate in
/// main.cpp, not by an internal check here (see README.md).
class SegmentationProcessor : public IModule, LoggerBase {
public:
    SegmentationProcessor();
    ~SegmentationProcessor() override;

    /// Builds ctx.segmentedMask: for each ctx.detectedPeople box, runs the
    /// segmentation model on that crop and writes its person-mask into the
    /// matching region of the shared full-frame canvas.
    void run(FrameContext& ctx) override;

    [[nodiscard]] std::string name() const override;

private:
    cv::dnn::Net net;
};