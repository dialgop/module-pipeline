//
// Created by diego on 03/09/2026.
//

#pragma once
#include <opencv2/dnn.hpp>

#include "IModule.h"
#include "LoggerBase.h"

/// Which pretrained model ObjectDetector runs. Both are real, verified-
/// compatible models with real, measured tradeoffs on this project's
/// video (park.mp4, ~858 frames) - see README.md for the full comparison:
///   - MobileNetSsd: Caffe, MIT license, ~90s total, one clean box per
///     person. Older (2017) architecture, general-purpose 20-class VOC
///     detector filtered to "person".
///   - Yolox: ONNX, Apache-2.0, ~810s total (~9x slower) - a modern,
///     usually more accurate architecture in general, but on this
///     project's per-motion-region crops it tends to split one tall
///     person into two boxes (upper/lower body), since a person spanning
///     most of a narrow letterboxed crop can score highly at multiple
///     grid scales without any single box covering all of them, and NMS
///     doesn't merge boxes that don't overlap enough.
enum class DetectorBackend {
    MobileNetSsd,
    Yolox,
};

/// Runs a real, pretrained object detector on each of MovementDetector's
/// candidate motion regions, keeping only the detections actually
/// classified as a person. This is the "verify" stage of the motion ->
/// detection -> segmentation cascade: MovementDetector proposes cheap,
/// coarse candidate regions; ObjectDetector spends real model inference
/// only on those regions, never the whole frame. Which model backs this
/// is chosen at construction time (see DetectorBackend) - main.cpp passes
/// it explicitly, so switching model is a one-line change, not a rebuild
/// of this class.
class ObjectDetector : public IModule, LoggerBase {
public:
    explicit ObjectDetector(DetectorBackend backend);
    ~ObjectDetector() override;

    /// For each region in ctx.motionRegions, crops ctx.frame (with some
    /// padding so a person isn't clipped at the region's edge), runs the
    /// selected backend's detector, and records the full-frame bounding
    /// box of every detection classified as "person" above the confidence
    /// threshold into ctx.detectedPeople.
    void run(FrameContext& ctx) override;

    [[nodiscard]] std::string name() const override;

private:
    DetectorBackend backend;
    cv::dnn::Net net;
    // Yolox-only: anchor-free, so instead of a fixed table of learned
    // anchor boxes, each of the 8400 output rows just corresponds to one
    // grid cell at one of 3 feature map scales. grids[i]/strides[i] record
    // which grid cell and stride output row i came from, computed once in
    // the constructor. Left empty when backend == MobileNetSsd.
    std::vector<cv::Point2f> grids;
    std::vector<float> strides;

    void runMobileNetSsd(FrameContext& ctx);
    void runYolox(FrameContext& ctx);
};