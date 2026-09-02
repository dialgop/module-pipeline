//
// Created by diego on 03/09/2026.
//

#include "ObjectDetector.h"

namespace {
    // MobileNet-SSD (chuanqi305/MobileNet-SSD, trained on Pascal VOC) was
    // trained on 300x300 crops, normalized the same way it saw during
    // training (mean-subtract 127.5, scale by 1/127.5, BGR - no channel
    // swap). "person" is class index 15 in the VOC label order the model
    // was trained with (verified against the repo's own demo script).
    constexpr int kInputSize = 300;
    constexpr int kPersonClassId = 15;
    constexpr float kConfidenceThreshold = 0.5f;
    // Extra margin around each motion cluster so a person isn't clipped
    // right at the cluster's edge before the detector ever sees them.
    constexpr int kCropPadding = 20;
}

ObjectDetector::ObjectDetector() {
    const std::string modelsDir = std::string(PROJECT_SOURCE_DIR) + "/external/models/";
    net = cv::dnn::readNetFromCaffe(modelsDir + "mobilenet_ssd.prototxt",
                                     modelsDir + "mobilenet_ssd.caffemodel");
    if (net.empty()) {
        error("Failed to load MobileNet-SSD model");
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
        cv::Mat blob = cv::dnn::blobFromImage(crop, 1.0 / 127.5, cv::Size(kInputSize, kInputSize),
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
            if (confidence < kConfidenceThreshold || classId != kPersonClassId) continue;

            const int x1 = padded.x + static_cast<int>(detections.at<float>(i, 3) * padded.width);
            const int y1 = padded.y + static_cast<int>(detections.at<float>(i, 4) * padded.height);
            const int x2 = padded.x + static_cast<int>(detections.at<float>(i, 5) * padded.width);
            const int y2 = padded.y + static_cast<int>(detections.at<float>(i, 6) * padded.height);
            ctx.detectedPeople.emplace_back(cv::Point(x1, y1), cv::Point(x2, y2));
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
