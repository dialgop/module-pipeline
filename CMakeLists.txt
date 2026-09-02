cmake_minimum_required(VERSION 3.28)
project(module_pipe_project)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)

# Creating executable
add_executable(module_pipeline_project)

#Add includes
target_include_directories(module_pipeline_project
PRIVATE ${PROJECT_SOURCE_DIR}/include ${OpenCV_INCLUDE_DIRS})

target_link_libraries(module_pipeline_project
PRIVATE ${OpenCV_LIBS})

# Lets main.cpp build an absolute path to external/example.mp4 regardless of
# the working directory the executable is launched from.
target_compile_definitions(module_pipeline_project
PRIVATE PROJECT_SOURCE_DIR="${PROJECT_SOURCE_DIR}")

#Add cpp sources
target_sources(module_pipeline_project
PRIVATE src/main.cpp
        src/CameraSensor.cpp
        src/MovementDetector.cpp
        src/ObjectDetector.cpp
        src/SegmentationProcessor.cpp
        src/ColorProcessor.cpp
)