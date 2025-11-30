#include <memory>
#include <vector>
#include <iostream>
#include <algorithm>

#include "CameraSensor.h"
#include "MovementDetector.h"
#include  "ColorProcessor.h"

int main() {
    std::vector <std::shared_ptr<IModule>> pipeline;

    int counter = 0;

    pipeline.push_back(std::make_shared<CameraSensor>(1));
    pipeline.push_back(std::make_shared<MovementDetector>());
    pipeline.push_back(std::make_shared<ColorProcessor>());

    std::cout << "Executing pipeline: \n";

    std::for_each(pipeline.begin(), pipeline.end(),
        [&counter](const std::shared_ptr<IModule>& module) {
            std::cout << module->name() << "\n";
            module->run();
            counter++;
        });

    std::cout << "\nTotal executed modules: " << counter << std::endl;

    return 0;
}
