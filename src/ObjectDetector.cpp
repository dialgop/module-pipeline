//
// Created by diego on 03/09/2026.
//

#include "ObjectDetector.h"

#include <cmath>
#include <opencv2/imgproc.hpp>

namespace {
    // MobileNet-SSD (chuanqi305/MobileNet-SSD, trained on Pascal VOC) was
    // trained on 300x300 crops, normalized the same way it saw during
    // training (mean-subtract 127.5, scale by 1/127.5, BGR - no channel
    // swap). "person" is class index 15 in the VOC label order the model
    // was trained with (verified against the repo's own demo script).
    constexpr int kSsdInputSize = 300;
    constexpr int kSsdPersonClassId = 15;
    constexpr float kSsdConfidenceThreshold = 0.5f;

    // YOLOX (object_detection_yolox_2022nov, opencv_zoo) expects a 640x640
    // input, BGR, raw 0-255 pixel values with NO normalization at all -
    // verified directly against the reference implementation's source
    // (its mean/std fields are declared but never actually applied).
    constexpr int kYoloxInputSize = 640;
    constexpr int kYoloxNumClasses = 80;
    constexpr int kYoloxPersonClassId = 0; // COCO class 0 = "person"
    constexpr float kYoloxConfThreshold = 0.35f;
    constexpr float kYoloxNmsThreshold = 0.5f;
    constexpr float kYoloxPadValue = 114.0f; // YOLOX's standard letterbox padding color (mid-gray)

    // Extra margin around each motion cluster so a person isn't clipped
    // right at the cluster's edge before the detector ever sees them.
    constexpr int kCropPadding = 20;
}

ObjectDetector::ObjectDetector(DetectorBackend backend) : backend(backend) {
    const std::string modelsDir = std::string(PROJECT_SOURCE_DIR) + "/external/models/";

    if (backend == DetectorBackend::MobileNetSsd) {
        net = cv::dnn::readNetFromCaffe(modelsDir + "mobilenet_ssd.prototxt",
                                         modelsDir + "mobilenet_ssd.caffemodel");
        if (net.empty()) {
            error("Failed to load MobileNet-SSD model");
        }
    } else {
        net = cv::dnn::readNet(modelsDir + "object_detection_yolox_2022nov.onnx");
        if (net.empty()) {
            error("Failed to load YOLOX model");
        }
        // Matches the reference's generateAnchors() exactly: per stride
        // (low to high), a size x size grid (size = 640/stride) traversed
        // y outer, x inner - this ordering must match how the model's own
        // feature maps were flattened during export, not just any square
        // scan.
        for (const int stride : {8, 16, 32}) {
            const int size = kYoloxInputSize / stride;
            for (int y = 0; y < size; ++y) {
                for (int x = 0; x < size; ++x) {
                    grids.emplace_back(static_cast<float>(x), static_cast<float>(y));
                    strides.push_back(static_cast<float>(stride));
                }
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

    if (backend == DetectorBackend::MobileNetSsd) {
        runMobileNetSsd(ctx);
    } else {
        runYolox(ctx);
    }

    if (!ctx.detectedPeople.empty()) {
        log("Detected " + std::to_string(ctx.detectedPeople.size()) + " person(s)");
    } else {
        log("No person detected in candidate regions");
    }
}

void ObjectDetector::runMobileNetSsd(FrameContext& ctx) {
    for (const cv::Rect& region : ctx.motionRegions) {
        cv::Rect padded(region.x - kCropPadding, region.y - kCropPadding,
                         region.width + 2 * kCropPadding, region.height + 2 * kCropPadding);
        padded &= cv::Rect(0, 0, ctx.frame.cols, ctx.frame.rows);
        if (padded.width <= 0 || padded.height <= 0) continue;

        cv::Mat crop = ctx.frame(padded);
        cv::Mat blob = cv::dnn::blobFromImage(crop, 1.0 / 127.5, cv::Size(kSsdInputSize, kSsdInputSize),
                                               cv::Scalar(127.5, 127.5, 127.5), false);
        net.setInput(blob);
        cv::Mat out = net.forward();

        // out has shape [1, 1, N, 7]; each row is
        // [imageId, classId, confidence, xmin, ymin, xmax, ymax], with the
        // box coordinates normalized to [0,1] relative to the crop.
        cv::Mat detections(out.size[2], out.size[3], CV_32F, out.ptr<float>());
        for (int i = 0; i < detections.rows; ++i) {
            const float confidence = detections.at<float>(i, 2);
            const auto classId = static_cast<int>(detections.at<float>(i, 1));
            if (confidence < kSsdConfidenceThreshold || classId != kSsdPersonClassId) continue;

            const int x1 = padded.x + static_cast<int>(detections.at<float>(i, 3) * padded.width);
            const int y1 = padded.y + static_cast<int>(detections.at<float>(i, 4) * padded.height);
            const int x2 = padded.x + static_cast<int>(detections.at<float>(i, 5) * padded.width);
            const int y2 = padded.y + static_cast<int>(detections.at<float>(i, 6) * padded.height);
            ctx.detectedPeople.emplace_back(cv::Point(x1, y1), cv::Point(x2, y2));
        }
    }
}

void ObjectDetector::runYolox(FrameContext& ctx) {
    for (const cv::Rect& region : ctx.motionRegions) {
        cv::Rect padded(region.x - kCropPadding, region.y - kCropPadding,
                         region.width + 2 * kCropPadding, region.height + 2 * kCropPadding);
        padded &= cv::Rect(0, 0, ctx.frame.cols, ctx.frame.rows);
        if (padded.width <= 0 || padded.height <= 0) continue;

        cv::Mat crop = ctx.frame(padded);

        // Letterbox: resize preserving aspect ratio, pad to 640x640 with
        // gray, top-left aligned (not centered) - matches the reference,
        // and means undoing it later is just a divide, no offset to track.
        const float ratio = std::min(static_cast<float>(kYoloxInputSize) / crop.rows,
                                      static_cast<float>(kYoloxInputSize) / crop.cols);
        cv::Mat resized;
        cv::resize(crop, resized, cv::Size(), ratio, ratio, cv::INTER_LINEAR);
        cv::Mat letterboxed(kYoloxInputSize, kYoloxInputSize, CV_8UC3,
                             cv::Scalar(kYoloxPadValue, kYoloxPadValue, kYoloxPadValue));
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
            for (int c = 0; c < kYoloxNumClasses; ++c) {
                const float score = objectness * row[5 + c];
                if (score > bestScore) {
                    bestScore = score;
                    bestClass = c;
                }
            }
            if (bestClass != kYoloxPersonClassId || bestScore < kYoloxConfThreshold) continue;

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
        cv::dnn::NMSBoxes(personBoxes, personScores, kYoloxConfThreshold, kYoloxNmsThreshold, keep);

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
}

std::string ObjectDetector::name() const {
    return "ObjectDetector";
}