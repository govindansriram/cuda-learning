cmake_minimum_required(VERSION 3.22)

set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")
set(CMAKE_CUDA_ARCHITECTURES 89)
project(learncuda LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find OpenCV package (ensure OpenCV is installed on your system)
find_package(OpenCV REQUIRED)

include_directories(${CUDA_INCLUDE_DIRS})

# Include OpenCV directories
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(learncuda main.cpp api.cu api.h)

set_target_properties(learncuda PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

target_link_libraries(learncuda PRIVATE ${CUDA_LIBRARIES} ${OpenCV_LIBS})