//
// Created by diego on 26/11/2025.
//

#pragma once
#include "IModule.h"
#include "LoggerBase.h"

class CameraSensor final : public IModule, public LoggerBase {
public:
    explicit CameraSensor(int id);
    ~CameraSensor() override;

    void run() override;
    [[nodiscard]] std::string name() const override;

private:
    int cameraId;
};