//
// Created by diego on 26/11/2025.
//

#pragma once
#include <string>

class IModule {
public:
    virtual ~IModule() = default;
    virtual void run() = 0;
    [[nodiscard]] virtual std::string name() const = 0;
};