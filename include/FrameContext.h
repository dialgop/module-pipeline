//
// Created by diego on 01/09/2026.
//

#pragma once
#include <opencv2/core.hpp>

struct FrameContext {
    cv::Mat frame;
    bool frameValid = false;

    bool motionDetected = false;
    cv::Mat motionMask;
    cv::Rect motionRoi;

    cv::Mat segmentedMask;
};
