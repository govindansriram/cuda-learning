cmake_minimum_required(VERSION 3.22)

set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")
set(CMAKE_CUDA_ARCHITECTURES "89;89-virtual")
project(learncuda LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
find_package(CUDAToolkit REQUIRED)

# Find OpenCV package (ensure OpenCV is installed on your system)
find_package(OpenCV REQUIRED)

include_directories(CUDA::toolkit)

# Include OpenCV directories
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(learncuda main.cpp api.cu api.h buffer.cu ptx.cu)

set_target_properties(learncuda PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

target_link_libraries(learncuda PRIVATE CUDA::cudart ${OpenCV_LIBS})

target_compile_options(learncuda PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--generate-line-info --keep>
        $<$<COMPILE_LANGUAGE:CUDA>:--generate-code=arch=compute_89,code=sm_89>
        $<$<COMPILE_LANGUAGE:CUDA>:--generate-code=arch=compute_89,code=compute_89>
)