//
// Created by diego on 21/01/2026.
//

#pragma once
#include <vector>
#include <memory>
#include <algorithm>
#include <iostream>

template<typename ModuleT>
class Pipeline {

public:
    void add(std::shared_ptr<ModuleT> m) {
        modules.push_back(std::move(m));
    }

    template<typename ShouldRun, typename OnRun>
    int run(ShouldRun should_run, OnRun on_run) {
        int executed = 0;

        std::for_each(modules.begin(), modules.end(),
            [&](const std::shared_ptr<ModuleT>& m) {
                if (should_run(m)) {
                    std::cout << "RUN: " << m->name() << "\n";
                    m->run();
                    on_run(m);
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