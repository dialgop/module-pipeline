cmake_minimum_required(VERSION 3.28)
project(module_pipe_project)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Creating executable
add_executable(module_pipeline_project)

#Add includes
target_include_directories(module_pipeline_project
PRIVATE ${PROJECT_SOURCE_DIR}/include)

#Add cpp sources
target_sources(module_pipeline_project
PRIVATE src/main.cpp
        src/CameraSensor.cpp
        src/MovementDetector.cpp
        src/ColorProcessor.cpp
)