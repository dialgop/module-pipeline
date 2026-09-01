//
// Created by diego on 26/11/2025.
//

#pragma once
#include <string>

#include "FrameContext.h"

class IModule {
public:
    virtual ~IModule() = default;
    virtual void run(FrameContext& ctx) = 0;
    [[nodiscard]] virtual std::string name() const = 0;
};