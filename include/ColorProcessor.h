//
// Created by diego on 26/11/2025.
//

#pragma once
#include "IModule.h"
#include "LoggerBase.h"

class ColorProcessor : public IModule, public LoggerBase {
public:
    ColorProcessor();
    ~ColorProcessor() override;

    void run() override;
    [[nodiscard]] std::string name() const override;
};