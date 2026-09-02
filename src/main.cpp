#include <memory>
#include <iostream>

#include <opencv2/highgui.hpp>

#include "Pipeline.h"
#include "FrameContext.h"
#include "CameraSensor.h"
#include "MovementDetector.h"
#include "SegmentationProcessor.h"
#include "ColorProcessor.h"

int main() {
    Pipeline<IModule> pipeline;

    pipeline.add(std::make_shared<CameraSensor>(std::string(PROJECT_SOURCE_DIR) + "/external/example.mp4"));
    pipeline.add(std::make_shared<MovementDetector>());
    pipeline.add(std::make_shared<SegmentationProcessor>());
    pipeline.add(std::make_shared<ColorProcessor>());

    auto should_run = [](const std::shared_ptr<IModule>& m, FrameContext& ctx) {
        const std::string& moduleName = m->name();
        // Every stage after CameraSensor needs ctx.frameValid too: once the
        // video source is exhausted, CameraSensor sets frameValid=false and
        // empties ctx.frame within this same pipeline.run() call, and the
        // later stages must not touch that empty frame/stale data before
        // the caller gets a chance to check frameValid and stop the loop.
        if (moduleName == "MovementDetector") {
            return ctx.frameValid;
        }
        if (moduleName == "SegmentationProcessor") {
            return ctx.frameValid && ctx.motionDetected;
        }
        if (moduleName == "ColorProcessor") {
            return ctx.frameValid && !ctx.segmentedMask.empty();
        }
        return true; // CameraSensor: always attempt to capture the next frame
    };

    auto on_run = [](const std::shared_ptr<IModule>& m, FrameContext& ctx) {
        std::cout << "ON_RUN: " << m->name() << "\n";
    };

    FrameContext ctx;
    int frameNumber = 0;
    while (true) {
        pipeline.run(ctx, should_run, on_run);
        if (!ctx.frameValid) break;
        ++frameNumber;

        cv::imshow("module_pipe_project", ctx.frame);
        if (cv::waitKey(1) == 27) break; // Esc to exit
    }

    std::cout << "\nProcessed " << frameNumber << " frames" << std::endl;

    return 0;
}