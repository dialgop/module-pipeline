//
// Created by diego on 21/01/2026.
//

#pragma once
#include <vector>
#include <memory>
#include <algorithm>
#include <iostream>

#include "FrameContext.h"

template<typename ModuleT>
class Pipeline {

public:
    void add(std::shared_ptr<ModuleT> m) {
        modules.push_back(std::move(m));
    }

    template<typename ShouldRun, typename OnRun>
    int run(FrameContext& ctx, ShouldRun should_run, OnRun on_run) {
        int executed = 0;

        std::for_each(modules.begin(), modules.end(),
            [&](const std::shared_ptr<ModuleT>& m) {
                if (should_run(m, ctx)) {
                    std::cout << "RUN: " << m->name() << "\n";
                    m->run(ctx);
                    on_run(m, ctx);
                    ++executed;
                } else {
                    std::cout << "SKIP: " << m->name() << "\n";
                }
            });
        return executed;
    }

private:
    std::vector<std::shared_ptr<ModuleT>> modules;
};