//
// Created by diego on 01/09/2026.
//

#pragma once
#include <vector>

#include <opencv2/core.hpp>

struct FrameContext {
    cv::Mat frame;
    bool frameValid = false;

    bool motionDetected = false;
    cv::Mat motionMask;
    std::vector<cv::Rect> motionRegions; // one bounding box per distinct motion cluster

    std::vector<cv::Rect> detectedPeople; // motionRegions, verified to actually contain a person

    cv::Mat segmentedMask;
};
