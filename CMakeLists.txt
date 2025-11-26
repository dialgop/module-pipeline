cmake_minimum_required(VERSION 3.28)
project(module_pipe_project)

set(CMAKE_CXX_STANDARD 20)

include_directories(include)

file(GLOB SOURCES src/*.cpp)

add_executable(module_pipeline ${SOURCES}
        include/IModule.h
        include/LoggerBase.h
        include/CameraSensor.h
        src/CameraSensor.cpp
        include/MovementDetector.h
        src/MovementDetector.cpp)
