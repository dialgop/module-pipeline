//
// Created by diego on 26/11/2025.
//

#pragma once
#include <opencv2/videoio.hpp>

#include "IModule.h"
#include "LoggerBase.h"

/// Captures frames from a live camera or a video file and writes them into
/// the pipeline's shared FrameContext. Wraps a cv::VideoCapture: RAII opens
/// the device/file as soon as a CameraSensor is constructed, and releases
/// it when the CameraSensor is destroyed.
class CameraSensor final : public IModule, public LoggerBase {
public:
    /// Opens a live camera by device index (e.g. 0 for the default webcam).
    explicit CameraSensor(int cameraIndex);

    /// Opens a video file instead of a live camera - useful for testing
    /// without a webcam attached (see external/example.mp4).
    explicit CameraSensor(const std::string& videoPath);

    ~CameraSensor() override;

    /// Grabs the next frame into ctx.frame and sets ctx.frameValid.
    /// If the capture never opened successfully, or the source is
    /// exhausted (e.g. end of a video file), ctx.frame comes back empty
    /// and ctx.frameValid is set to false - callers use that flag to know
    /// when to stop pulling frames.
    void run(FrameContext& ctx) override;

    [[nodiscard]] std::string name() const override;

private:
    cv::VideoCapture capture; // the open device/file handle (RAII: opened in the constructor)
    std::string sourceLabel;  // built once at construction so name() doesn't recompute it every call
};