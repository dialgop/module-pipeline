//
// Created by diego on 03/09/2026.
//

#include "ObjectDetector.h"

#include <cmath>
#include <opencv2/imgproc.hpp>

namespace {
    // YOLOX (object_detection_yolox_2022nov, opencv_zoo) expects a 640x640
    // input, BGR, raw 0-255 pixel values with NO normalization at all -
    // verified directly against the reference implementation's source
    // (its mean/std fields are declared but never actually applied).
    constexpr int kInputSize = 640;
    constexpr int kNumClasses = 80;
    constexpr int kPersonClassId = 0; // COCO class 0 = "person" (verified against the reference's class list)
    constexpr float kConfThreshold = 0.35f; // combined objectness*classScore threshold, matches the reference default
    constexpr float kNmsThreshold = 0.5f;
    constexpr float kPadValue = 114.0f; // YOLOX's standard letterbox padding color (mid-gray)
    // Extra margin around each motion cluster so a person isn't clipped
    // right at the cluster's edge before the detector ever sees them.
    constexpr int kCropPadding = 20;
}

ObjectDetector::ObjectDetector() {
    const std::string modelsDir = std::string(PROJECT_SOURCE_DIR) + "/external/models/";
    net = cv::dnn::readNet(modelsDir + "object_detection_yolox_2022nov.onnx");
    if (net.empty()) {
        error("Failed to load YOLOX model");
    }

    // Matches the reference's generateAnchors() exactly: per stride
    // (low to high), a size x size grid (size = 640/stride) traversed y
    // outer, x inner - this ordering must match how the model's own
    // feature maps were flattened during export, not just any square scan.
    for (const int stride : {8, 16, 32}) {
        const int size = kInputSize / stride;
        for (int y = 0; y < size; ++y) {
            for (int x = 0; x < size; ++x) {
                grids.emplace_back(static_cast<float>(x), static_cast<float>(y));
                strides.push_back(static_cast<float>(stride));
            }
        }
    }

    log("Created ObjectDetector");
}

ObjectDetector::~ObjectDetector() {
    log("Destroyed ObjectDetector");
}

void ObjectDetector::run(FrameContext& ctx) {
    ctx.detectedPeople.clear();

    for (const cv::Rect& region : ctx.motionRegions) {
        cv::Rect padded(region.x - kCropPadding, region.y - kCropPadding,
                         region.width + 2 * kCropPadding, region.height + 2 * kCropPadding);
        padded &= cv::Rect(0, 0, ctx.frame.cols, ctx.frame.rows);
        if (padded.width <= 0 || padded.height <= 0) continue;

        cv::Mat crop = ctx.frame(padded);

        // Letterbox: resize preserving aspect ratio, pad to 640x640 with
        // gray, top-left aligned (not centered) - matches the reference,
        // and means undoing it later is just a divide, no offset to track.
        const float ratio = std::min(static_cast<float>(kInputSize) / crop.rows,
                                      static_cast<float>(kInputSize) / crop.cols);
        cv::Mat resized;
        cv::resize(crop, resized, cv::Size(), ratio, ratio, cv::INTER_LINEAR);
        cv::Mat letterboxed(kInputSize, kInputSize, CV_8UC3, cv::Scalar(kPadValue, kPadValue, kPadValue));
        resized.copyTo(letterboxed(cv::Rect(0, 0, resized.cols, resized.rows)));

        cv::Mat blob = cv::dnn::blobFromImage(letterboxed, 1.0, cv::Size(), cv::Scalar(), false, false, CV_32F);
        net.setInput(blob);
        cv::Mat out = net.forward(); // [1, 8400, 85]: 4 box + 1 objectness + 80 class scores, per grid cell

        cv::Mat detections(out.size[1], out.size[2], CV_32F, out.ptr<float>());

        // Filter to person-class candidates *before* NMS, then run plain
        // (non-batched) NMS on just those - our installed OpenCV doesn't
        // have NMSBoxesBatched, but since we only ever want the "person"
        // class, per-class batching isn't needed: we don't care how a
        // person box overlaps a discarded, say, backpack box.
        std::vector<cv::Rect> personBoxes;
        std::vector<float> personScores;
        for (int i = 0; i < detections.rows; ++i) {
            const float* row = detections.ptr<float>(i);
            const float objectness = row[4];

            int bestClass = -1;
            float bestScore = 0.0f;
            for (int c = 0; c < kNumClasses; ++c) {
                const float score = objectness * row[5 + c];
                if (score > bestScore) {
                    bestScore = score;
                    bestClass = c;
                }
            }
            if (bestClass != kPersonClassId || bestScore < kConfThreshold) continue;

            // Decode: raw (cx,cy) are offsets from this row's grid cell,
            // in stride units; raw (w,h) are log-scale, also stride units.
            const float cx = (row[0] + grids[i].x) * strides[i];
            const float cy = (row[1] + grids[i].y) * strides[i];
            const float w = std::exp(row[2]) * strides[i];
            const float h = std::exp(row[3]) * strides[i];

            personBoxes.emplace_back(cv::Point(static_cast<int>(cx - w / 2), static_cast<int>(cy - h / 2)),
                                      cv::Size(static_cast<int>(w), static_cast<int>(h)));
            personScores.push_back(bestScore);
        }

        std::vector<int> keep;
        cv::dnn::NMSBoxes(personBoxes, personScores, kConfThreshold, kNmsThreshold, keep);

        for (const int idx : keep) {
            // Undo the letterbox (divide by ratio), then the crop offset.
            cv::Rect box(padded.x + static_cast<int>(personBoxes[idx].x / ratio),
                         padded.y + static_cast<int>(personBoxes[idx].y / ratio),
                         static_cast<int>(personBoxes[idx].width / ratio),
                         static_cast<int>(personBoxes[idx].height / ratio));
            box &= cv::Rect(0, 0, ctx.frame.cols, ctx.frame.rows);
            if (box.width > 0 && box.height > 0) {
                ctx.detectedPeople.push_back(box);
            }
        }
    }

    if (!ctx.detectedPeople.empty()) {
        log("Detected " + std::to_string(ctx.detectedPeople.size()) + " person(s)");
    } else {
        log("No person detected in candidate regions");
    }
}

std::string ObjectDetector::name() const {
    return "ObjectDetector";
}