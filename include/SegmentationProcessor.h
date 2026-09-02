//
// Created by diego on 02/09/2026.
//

#pragma once
#include <opencv2/core.hpp>

#include "IModule.h"
#include "LoggerBase.h"

class SegmentationProcessor : public IModule, LoggerBase {
public:
    SegmentationProcessor();
    ~SegmentationProcessor() override;

    void run(FrameContext& ctx) override;

    [[nodiscard]] std::string name() const override;
};
