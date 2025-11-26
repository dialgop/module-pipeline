//
// Created by diego on 26/11/2025.
//

#pragma once
#include "IModule.h"
#include "LoggerBase.h"

class MovementDetector : public IModule, LoggerBase {
public:
    MovementDetector();
    ~MovementDetector() override;

    void run() override;
    [[nodiscard]] std::string name() const override;
};