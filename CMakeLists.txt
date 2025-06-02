cmake_minimum_required(VERSION 3.22)

set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")
set(CMAKE_CUDA_ARCHITECTURES "89;89-virtual")
project(learncuda LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
find_package(CUDAToolkit REQUIRED)

# Find OpenCV package (ensure OpenCV is installed on your system)
find_package(OpenCV REQUIRED)

include(FetchContent)
FetchContent_Declare(
        googletest
        URL https://github.com/google/googletest/archive/b514bdc898e2951020cbdca1304b75f5950d1f59.zip
)

FetchContent_MakeAvailable(googletest)

include_directories(CUDA::toolkit)

# Include OpenCV directories
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(/home/sriram/Applications/cutlass/cutlass/include)
include_directories(/home/sriram/Applications/cutlass/cutlass/tools/util/include)

set(COMMON_SRC_SOURCES
        src/helpers.cuh
        src/ptx_helpers.cuh
)

set(COMMON_BENCH_SOURCES
        benchmark/bench_helpers.h
        benchmark/bench_helpers.cpp
)

set(COMMON_TEST_SOURCES
        test/test_helpers.cpp
        test/test_helpers.h
)

set(GEMM_SOURCES
        ${COMMON_SRC_SOURCES}
        src/gemm/gemm.cuh
)

set(CUTLASS_GEMM_SOURCES
        ${COMMON_SRC_SOURCES}
        src/gemm/cutlass_gemm.cuh
)

add_executable(
        benchmark_gemm
        ${GEMM_SOURCES}
        ${COMMON_BENCH_SOURCES}
        benchmark/gemm/bench_gemm_main.cpp
        benchmark/gemm/bench_gemm.cu
        benchmark/gemm/bench_gemm.h
)

add_executable(
        test_gemm
        ${GEMM_SOURCES}
        ${COMMON_TEST_SOURCES}
        test/gemm/test_gemm.cu
        test/gemm/test_gemm_main.cpp
        test/gemm/test_gemm.h
)

add_executable(
        test_cutlass_gemm
        ${CUTLASS_GEMM_SOURCES}
        ${COMMON_TEST_SOURCES}
        test/gemm/test_cutlass_gemm.cu
        test/gemm/test_cutlass_gemm.h
        test/gemm/test_cutlass_gemm_main.cpp
)

function(setup_cuda_target target_name)
    cmake_parse_arguments(
            ARG                        # Prefix for parsed args (can be empty)
            "WITH_GTEST"               # Optional single-value flags
            ""                         # Optional single-value args (none)
            ""                         # Optional multi-value args (none)
            ${ARGN}                    # Remaining arguments passed to the function
    )

    set_target_properties(${target_name} PROPERTIES
            CUDA_SEPARABLE_COMPILATION ON
    )

    if(ARG_WITH_GTEST)
        target_link_libraries(${target_name} PRIVATE CUDA::cudart ${OpenCV_LIBS} gtest GTest::gtest_main)
    else()
        target_link_libraries(${target_name} PRIVATE CUDA::cudart ${OpenCV_LIBS})
    endif()

    target_compile_options(${target_name} PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:--generate-line-info --keep>
            $<$<COMPILE_LANGUAGE:CUDA>:--generate-code=arch=compute_89,code=sm_89>
            $<$<COMPILE_LANGUAGE:CUDA>:--generate-code=arch=compute_89,code=compute_89>
    )
endfunction()

setup_cuda_target(benchmark_gemm)
setup_cuda_target(test_gemm WITH_GTEST)
setup_cuda_target(test_cutlass_gemm WITH_GTEST)
