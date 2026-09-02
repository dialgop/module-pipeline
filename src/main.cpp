#include <memory>
#include <iostream>

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "Pipeline.h"
#include "FrameContext.h"
#include "CameraSensor.h"
#include "MovementDetector.h"
#include "ObjectDetector.h"
#include "SegmentationProcessor.h"
#include "ColorProcessor.h"

int main() {
    Pipeline<IModule> pipeline;

    pipeline.add(std::make_shared<CameraSensor>(std::string(PROJECT_SOURCE_DIR) + "/external/park.mp4"));
    pipeline.add(std::make_shared<MovementDetector>());
    pipeline.add(std::make_shared<ObjectDetector>());
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
        if (moduleName == "ObjectDetector") {
            // Only worth spending real model inference when MovementDetector
            // actually found candidate regions to crop and check.
            return ctx.frameValid && ctx.motionDetected;
        }
        if (moduleName == "SegmentationProcessor") {
            // SegmentationProcessor now segments each ctx.detectedPeople
            // box, so it needs at least one to have anything to do.
            // Requiring ctx.motionDetected too matters for freshness, not
            // just correctness: ctx.detectedPeople is only repopulated when
            // ObjectDetector runs, which itself only happens when
            // motionDetected is true - without this check, a motion-free
            // tick would leave detectedPeople holding stale boxes from an
            // earlier tick, and this stage would wrongly run against them.
            return ctx.frameValid && ctx.motionDetected && !ctx.detectedPeople.empty();
        }
        if (moduleName == "ColorProcessor") {
            // Mirrors SegmentationProcessor's own gate exactly: ctx.segmentedMask
            // is only refreshed when SegmentationProcessor actually runs, so
            // ColorProcessor must require the same conditions - otherwise it
            // could sample a stale silhouette against an unrelated current frame.
            return ctx.frameValid && ctx.motionDetected && !ctx.detectedPeople.empty();
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

        // Draw on a display-only copy - ctx.frame itself gets fully
        // replaced by the next CameraSensor::run() call anyway, but this
        // keeps "working data" and "what we show the user" clearly separate.
        cv::Mat display = ctx.frame.clone();
        for (const cv::Rect& person : ctx.detectedPeople) {
            cv::rectangle(display, person, cv::Scalar(0, 0, 255), 2);
        }
        cv::imshow("module_pipe_project", display);
        if (cv::waitKey(1) == 27) break; // Esc to exit
    }

    std::cout << "\nProcessed " << frameNumber << " frames" << std::endl;

    return 0;
}