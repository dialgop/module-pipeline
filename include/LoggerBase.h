//
// Created by diego on 26/11/2025.
//

#pragma once
#include <chrono>
#include <iostream>
#include <string>

class LoggerBase {
public:
    static void log(const std::string& msg) {
        auto now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
        std::cout << "[LOG " << std::put_time(std::localtime(&now), "%H:%M:%S")<< "] " << msg <<std::endl;
    }

    static void error(const std::string& msg) {
        auto now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
        std::cout << "[ERROR " << std::put_time(std::localtime(&now), "%H:%M:%S")<< "] " << msg <<std::endl;
    }
};