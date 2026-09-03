//
// Created by diego on 02/09/2026.
//

#include "SegmentationProcessor.h"

#include <opencv2/imgproc.hpp>

namespace {
    constexpr int kInputSize = 192;
}

SegmentationProcessor::SegmentationProcessor() {
    const std::string modelsDir = std::string(PROJECT_SOURCE_DIR) + "/external/models/";
    net = cv::dnn::readNet(modelsDir + "human_segmentation_pphumanseg_2023mar.onnx");
    if (net.empty()) {
        error("Failed to load PP-HumanSeg model");
    }
    log("Created SegmentationProcessor");
}

SegmentationProcessor::~SegmentationProcessor() {
    log("Destroyed SegmentationProcessor");
}

void SegmentationProcessor::run(FrameContext& ctx) {
    cv::Mat mask = cv::Mat::zeros(ctx.frame.size(), CV_8UC1);
    const cv::Rect frameBounds(0, 0, ctx.frame.cols, ctx.frame.rows);

    int segmentedCount = 0;
    for (const cv::Rect& personBox : ctx.detectedPeople) {
        const cv::Rect box = personBox & frameBounds;
        if (box.width <= 0 || box.height <= 0) continue;

        cv::Mat rgb;
        cv::cvtColor(ctx.frame(box), rgb, cv::COLOR_BGR2RGB);
        // Normalizes to [-1, 1]: (pixel/255 - 0.5) / 0.5, written here as
        // mean 127.5 / scale 1/127.5 - what the model was trained with.
        cv::Mat blob = cv::dnn::blobFromImage(rgb, 1.0 / 127.5, cv::Size(kInputSize, kInputSize),
                                               cv::Scalar(127.5, 127.5, 127.5), false);
        net.setInput(blob);
        cv::Mat out = net.forward(); // [1, 2, 192, 192]: channel 0 = background, channel 1 = person

        // Wrap the two channel planes directly (no copy) rather than
        // decide person-vs-background at 192x192: resize the raw scores
        // up to the crop's real size first (bilinear, so edges interpolate
        // smoothly), then compare - matching the reference implementation's
        // order (resize, then argmax) instead of upscaling an already-hard
        // binary decision with blocky nearest-neighbor edges.
        cv::Mat background(kInputSize, kInputSize, CV_32F, out.ptr<float>(0, 0));
        cv::Mat person(kInputSize, kInputSize, CV_32F, out.ptr<float>(0, 1));

        cv::Mat backgroundResized, personResized;
        cv::resize(background, backgroundResized, box.size(), 0, 0, cv::INTER_LINEAR);
        cv::resize(person, personResized, box.size(), 0, 0, cv::INTER_LINEAR);

        cv::Mat personMask = personResized > backgroundResized; // CV_8U, 255 where person wins

        // OR rather than copy: two detected people's boxes can overlap, and
        // this must not erase an earlier person's silhouette in that case.
        cv::Mat maskRoi = mask(box);
        cv::bitwise_or(maskRoi, personMask, maskRoi);
        ++segmentedCount;
    }

    ctx.segmentedMask = mask;

    if (segmentedCount > 0) {
        log("Segmented " + std::to_string(segmentedCount) + " person(s)");
    } else {
        log("No person to segment");
    }
}

std::string SegmentationProcessor::name() const {
    return "SegmentationProcessor";
}