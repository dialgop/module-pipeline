#include <memory>
#include <vector>
#include <iostream>
#include <algorithm>

#include "CameraSensor.h"
#include "MovementDetector.h"
#include  "ColorProcessor.h"

int main() {
    std::vector <std::shared_ptr<IModule>> pipeline;

    pipeline.push_back(std::make_shared<CameraSensor>(1));
    pipeline.push_back(std::make_shared<MovementDetector>());
    pipeline.push_back(std::make_shared<ColorProcessor>());

    auto should_run = [](const std::shared_ptr<IModule>& m) {
        return m->name().find("Detector") != std::string::npos;
    };

    auto make_counter = []() {
        int c=0;
        return [c]() mutable {return ++c;};
    };

    auto counter = make_counter();
    int executed = 0;

    std::cout << "Executing pipeline: \n";

    std::for_each(pipeline.begin(), pipeline.end(),
    [&](const std::shared_ptr<IModule>& m) {
        if (should_run(m)) {
            std::cout << "RUN: " << m->name() << "\n";
            m->run();
            executed = counter();
        } else {
            std::cout << "SKIP: " << m->name() << "\n";
        }
    }
);

    std::cout << "\nTotal executed modules: " << executed << std::endl;

    return 0;
}
