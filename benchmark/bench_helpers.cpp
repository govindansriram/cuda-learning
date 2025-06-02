//
// Created by sriram on 5/27/25.
//
#include <cuda_runtime_api.h>
#include <iostream>
#include <random>
#include "bench_helpers.h"

void flush_l2_cache() {
    int dev_id{};
    int m_l2_size{};
    void *buffer;
    cudaGetDevice(&dev_id);
    cudaDeviceGetAttribute(&m_l2_size, cudaDevAttrL2CacheSize, dev_id);
    if (m_l2_size > 0) {
        cudaMalloc(&buffer, static_cast<std::size_t>(m_l2_size));
        int *m_l2_buffer = reinterpret_cast<int *>(buffer);
        cudaMemsetAsync(m_l2_buffer, 0, static_cast<std::size_t>(m_l2_size));
        cudaFree(m_l2_buffer);
    }

    // Check for errors in kernel execution
    if (const cudaError_t error = cudaGetLastError(); error != cudaSuccess)
        std::cerr << "CUDA error: " << cudaGetErrorString(error) << std::endl;
}

float generate_random_float(const float min_val, const float max_val) {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(min_val, max_val);
    return dis(gen);
}

void fill_matrix(
    float *matrix,
    const size_t M,
    const size_t N,
    const size_t stride,
    const float min_val,
    const float max_val) {
    for (size_t i{0}; i < M; ++i) {
        for (size_t j{0}; j < N; ++j) {
            matrix[i * stride + j] = generate_random_float(min_val, max_val);
        }
    }
}
