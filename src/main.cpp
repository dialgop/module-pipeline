#include <memory>
#include <vector>
#include <iostream>

#include "CameraSensor.h"
#include "MovementDetector.h"
#include  "ColorProcessor.h"

int main() {
    std::vector <std::shared_ptr<IModule>> pipeline;

    pipeline.push_back(std::make_shared<CameraSensor>(1));
    pipeline.push_back(std::make_shared<MovementDetector>());
    pipeline.push_back(std::make_shared<ColorProcessor>());

    for (auto& pipe : pipeline) {
        std::cout << "== Executing: " << pipe->name() << " ==" << std::endl;
        pipe->run();
        std::cout << std::endl;
    }
    return 0;
}
