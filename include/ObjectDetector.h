//
// Created by diego on 03/09/2026.
//

#pragma once
#include <opencv2/dnn.hpp>

#include "IModule.h"
#include "LoggerBase.h"

/// Runs a real, pretrained object detector (YOLOX, ONNX, Apache-2.0 - see
/// external/models/) on each of MovementDetector's candidate motion
/// regions, keeping only the detections actually classified as a person.
/// This is the "verify" stage of the motion -> detection -> segmentation
/// cascade: MovementDetector proposes cheap, coarse candidate regions;
/// ObjectDetector spends real model inference only on those regions, never
/// the whole frame.
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
    // YOLOX is anchor-free: instead of a fixed table of learned anchor
    // boxes (like the mediapipe model we couldn't use), each of the 8400
    // output rows just corresponds to one grid cell at one of 3 feature
    // map scales. grids[i]/strides[i] record which grid cell and stride
    // output row i came from, computed once in the constructor.
    std::vector<cv::Point2f> grids;
    std::vector<float> strides;
};