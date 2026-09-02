//
// Created by diego on 03/09/2026.
//

#pragma once
#include <opencv2/dnn.hpp>

#include "IModule.h"
#include "LoggerBase.h"

/// Runs a real, pretrained person detector (MobileNet-SSD, Caffe, MIT
/// license - see external/models/) on each of MovementDetector's candidate
/// motion regions, keeping only the ones that actually contain a person.
/// This is the "verify" stage of the motion -> detection -> (optional)
/// segmentation cascade: MovementDetector proposes cheap, coarse candidate
/// regions; ObjectDetector spends real model inference only on those
/// regions, never the whole frame.
class ObjectDetector : public IModule, LoggerBase {
public:
    ObjectDetector();
    ~ObjectDetector() override;

    /// For each region in ctx.motionRegions, crops ctx.frame (with some
    /// padding so a person isn't clipped at the region's edge), runs the
    /// detector, and records the full-frame bounding box of every
    /// detection classified as "person" above the confidence threshold
    /// into ctx.detectedPeople.
    void run(FrameContext& ctx) override;

    [[nodiscard]] std::string name() const override;

private:
    cv::dnn::Net net;
};
