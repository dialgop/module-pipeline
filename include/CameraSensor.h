//
// Created by diego on 26/11/2025.
//

#pragma once
#include <opencv2/videoio.hpp>

#include "IModule.h"
#include "LoggerBase.h"

class CameraSensor final : public IModule, public LoggerBase {
public:
    explicit CameraSensor(int cameraIndex);
    explicit CameraSensor(const std::string& videoPath);
    ~CameraSensor() override;

    void run(FrameContext& ctx) override;
    [[nodiscard]] std::string name() const override;

private:
    cv::VideoCapture capture; //actual OpenCV object representing the open device/file handl
    std::string sourceLabel; // replaces computing the name on the fly
};