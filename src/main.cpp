#include <memory>
#include <vector>
#include <iostream>
#include <algorithm>

#include "Pipeline.h"
#include "FrameContext.h"
#include "CameraSensor.h"
#include "MovementDetector.h"
#include  "ColorProcessor.h"

int main() {
    Pipeline<IModule> pipeline;

    pipeline.add(std::make_shared<CameraSensor>(1));
    pipeline.add(std::make_shared<MovementDetector>());
    pipeline.add(std::make_shared<ColorProcessor>());

    auto should_run = [](const std::shared_ptr<IModule>& m, FrameContext& ctx) {
        return m->name().find("Detector") != std::string::npos;
    };

    auto on_run = [](const std::shared_ptr<IModule>& m, FrameContext& ctx) {
        std::cout << "ON_RUN: " << m->name() << "\n";
    };

    FrameContext ctx;
    int executed = pipeline.run(ctx, should_run, on_run);

    std::cout << "Executing pipeline: \n";

    std::cout << "\nTotal executed modules: " << executed << std::endl;

    return 0;
}
