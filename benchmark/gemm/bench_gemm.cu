//
// Created by sriram on 5/27/25.
//

#include <iostream>
#include "../../src/gemm/gemm.cuh"
#include "../bench_helpers.h"

void time_2DBT() {
    auto host_A{new float[4096 * 4096]};
    auto host_B{new float[4096 * 4096]};

    float *dev_A;
    float *dev_B;
    float *dev_C;

    fill_matrix(host_A, 4096, 4096, 4096, -100.f, 100.f);
    fill_matrix(host_B, 4096, 4096, 4096, -100.f, 100.f);

    cudaMalloc(&dev_A, 4096 * 4096 * sizeof(float));
    cudaMalloc(&dev_B, 4096 * 4096 * sizeof(float));
    cudaMalloc(&dev_C, 4096 * 4096 * sizeof(float));

    cudaMemcpy(dev_A, host_A, 4096 * 4096 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_B, host_B, 4096 * 4096 * sizeof(float), cudaMemcpyHostToDevice);

    constexpr size_t TILE_SIZE_X{16};
    constexpr size_t TILE_SIZE_Y{16};
    constexpr size_t TILE_SIZE_K{16};

    constexpr size_t THREADS_PER_BLOCK{TILE_SIZE_X * TILE_SIZE_Y};

    static_assert(TILE_SIZE_K * TILE_SIZE_X % THREADS_PER_BLOCK == 0);
    static_assert(TILE_SIZE_K * TILE_SIZE_Y % THREADS_PER_BLOCK == 0);

    constexpr dim3 block_dim(TILE_SIZE_X, TILE_SIZE_Y);
    const dim3 grid_dim(ceil_div(4096, TILE_SIZE_X), ceil_div(4096, TILE_SIZE_Y));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    float total{0};

    for (int i{0}; i < 371; ++i) {
        float milliseconds = 0;
        cudaEventRecord(start);
        gemm_2DBT<
            TILE_SIZE_X,
            TILE_SIZE_Y,
            TILE_SIZE_K><<<grid_dim, block_dim>>>(
            dev_A,
            dev_B,
            dev_C,
            4096,
            4096,
            4096,
            4096,
            4096,
            4096
        );
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&milliseconds, start, stop);

        if (i > 185) total += milliseconds;
        flush_l2_cache();
    }

    std::cout << total / 185 << "\n";

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaFree(dev_A);
    cudaFree(dev_B);
    cudaFree(dev_C);

    delete []host_A;
    delete []host_B;

    flush_l2_cache();
}

void time_gemm_2DBT_2DWT_2DTT_vload() {
    auto host_A{new float[4096 * 4096]};
    auto host_B{new float[4096 * 4096]};

    float *dev_A;
    float *dev_B;
    float *dev_C;

    fill_matrix(host_A, 4096, 4096, 4096, -100.f, 100.f);
    fill_matrix(host_B, 4096, 4096, 4096, -100.f, 100.f);

    cudaMalloc(&dev_A, 4096 * 4096 * sizeof(float));
    cudaMalloc(&dev_B, 4096 * 4096 * sizeof(float));
    cudaMalloc(&dev_C, 4096 * 4096 * sizeof(float));

    cudaMemcpy(dev_A, host_A, 4096 * 4096 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_B, host_B, 4096 * 4096 * sizeof(float), cudaMemcpyHostToDevice);

    constexpr uint BLOCK_TILE_SIZE_X{128};
    constexpr uint BLOCK_TILE_SIZE_Y{128};
    constexpr uint BLOCK_TILE_SIZE_K{16};

    constexpr unsigned int WARP_TILE_SIZE_X{64};
    constexpr unsigned int WARP_TILE_SIZE_Y{64};

    constexpr size_t NUM_WARPS_PER_BLOCK_X{BLOCK_TILE_SIZE_X / WARP_TILE_SIZE_X};
    constexpr size_t NUM_WARPS_PER_BLOCK_Y{BLOCK_TILE_SIZE_Y / WARP_TILE_SIZE_Y};

    static_assert(BLOCK_TILE_SIZE_X % WARP_TILE_SIZE_X == 0);
    static_assert(BLOCK_TILE_SIZE_Y % WARP_TILE_SIZE_Y == 0);

    // The size of the internal register caches
    constexpr uint THREAD_TILE_SIZE_Y{8};
    constexpr uint THREAD_TILE_SIZE_X{8};

    constexpr unsigned int NUM_THREADS_PER_WARP_X{4};
    constexpr unsigned int NUM_THREADS_PER_WARP_Y{8};

    static_assert(NUM_THREADS_PER_WARP_X * NUM_THREADS_PER_WARP_Y == 32);

    // ensure each thread stores the same amount of data in their tiles
    static_assert(WARP_TILE_SIZE_X % (THREAD_TILE_SIZE_X * NUM_THREADS_PER_WARP_X) == 0);
    static_assert(WARP_TILE_SIZE_Y % (THREAD_TILE_SIZE_Y * NUM_THREADS_PER_WARP_Y) == 0);

    const dim3 grid_dim{
        ceil_div(4096, BLOCK_TILE_SIZE_X),
        ceil_div(4096, BLOCK_TILE_SIZE_Y)
    };

    constexpr size_t NUM_THREADS_PER_BLOCK{32 * NUM_WARPS_PER_BLOCK_X * NUM_WARPS_PER_BLOCK_Y};
    constexpr dim3 block_dim(NUM_THREADS_PER_BLOCK);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    float total{0};

    for (int i{0}; i < 371; ++i) {
        float milliseconds = 0;
        cudaEventRecord(start);

        gemm_2DBT_2DWT_2DTT_vload<
            float,
            BLOCK_TILE_SIZE_X,
            BLOCK_TILE_SIZE_Y,
            BLOCK_TILE_SIZE_K,
            WARP_TILE_SIZE_X,
            WARP_TILE_SIZE_Y,
            THREAD_TILE_SIZE_X,
            THREAD_TILE_SIZE_Y,
            NUM_THREADS_PER_WARP_X,
            NUM_THREADS_PER_WARP_Y><<<grid_dim, block_dim>>>(
            dev_A,
            dev_B,
            dev_C,
            1,
            1,
            4096,
            4096,
            4096,
            4096,
            4096,
            4096
        );

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&milliseconds, start, stop);

        if (i > 185) total += milliseconds;
        flush_l2_cache();
    }

    std::cout << total / 185 << "\n";

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaFree(dev_A);
    cudaFree(dev_B);
    cudaFree(dev_C);

    delete []host_A;
    delete []host_B;

    flush_l2_cache();
}

void time_gemm_2DBT_2DWT_2DTT_async() {
    auto host_A{new float[4096 * 4096]};
    auto host_B{new float[4096 * 4096]};

    float *dev_A;
    float *dev_B;
    float *dev_C;

    fill_matrix(host_A, 4096, 4096, 4096, -100.f, 100.f);
    fill_matrix(host_B, 4096, 4096, 4096, -100.f, 100.f);

    cudaMalloc(&dev_A, 4096 * 4096 * sizeof(float));
    cudaMalloc(&dev_B, 4096 * 4096 * sizeof(float));
    cudaMalloc(&dev_C, 4096 * 4096 * sizeof(float));

    cudaMemcpy(dev_A, host_A, 4096 * 4096 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_B, host_B, 4096 * 4096 * sizeof(float), cudaMemcpyHostToDevice);

    constexpr uint BLOCK_TILE_SIZE_X{128};
    constexpr uint BLOCK_TILE_SIZE_Y{128};
    constexpr uint BLOCK_TILE_SIZE_K{16};

    constexpr unsigned int WARP_TILE_SIZE_X{64};
    constexpr unsigned int WARP_TILE_SIZE_Y{64};

    constexpr size_t NUM_WARPS_PER_BLOCK_X{BLOCK_TILE_SIZE_X / WARP_TILE_SIZE_X};
    constexpr size_t NUM_WARPS_PER_BLOCK_Y{BLOCK_TILE_SIZE_Y / WARP_TILE_SIZE_Y};

    static_assert(BLOCK_TILE_SIZE_X % WARP_TILE_SIZE_X == 0);
    static_assert(BLOCK_TILE_SIZE_Y % WARP_TILE_SIZE_Y == 0);

    // The size of the internal register caches
    constexpr uint THREAD_TILE_SIZE_Y{8};
    constexpr uint THREAD_TILE_SIZE_X{8};

    constexpr unsigned int NUM_THREADS_PER_WARP_X{4};
    constexpr unsigned int NUM_THREADS_PER_WARP_Y{8};

    static_assert(NUM_THREADS_PER_WARP_X * NUM_THREADS_PER_WARP_Y == 32);

    // ensure each thread stores the same amount of data in their tiles
    static_assert(WARP_TILE_SIZE_X % (THREAD_TILE_SIZE_X * NUM_THREADS_PER_WARP_X) == 0);
    static_assert(WARP_TILE_SIZE_Y % (THREAD_TILE_SIZE_Y * NUM_THREADS_PER_WARP_Y) == 0);

    const dim3 grid_dim{
        ceil_div(4096, BLOCK_TILE_SIZE_X),
        ceil_div(4096, BLOCK_TILE_SIZE_Y)
    };

    constexpr size_t NUM_THREADS_PER_BLOCK{32 * NUM_WARPS_PER_BLOCK_X * NUM_WARPS_PER_BLOCK_Y};
    constexpr dim3 block_dim(NUM_THREADS_PER_BLOCK);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    float total{0};

    for (int i{0}; i < 371; ++i) {
        float milliseconds = 0;
        cudaEventRecord(start);

        gemm_2DBT_2DWT_2DTT_async_load<
            float,
            BLOCK_TILE_SIZE_X,
            BLOCK_TILE_SIZE_Y,
            BLOCK_TILE_SIZE_K,
            WARP_TILE_SIZE_X,
            WARP_TILE_SIZE_Y,
            THREAD_TILE_SIZE_X,
            THREAD_TILE_SIZE_Y,
            NUM_THREADS_PER_WARP_X,
            NUM_THREADS_PER_WARP_Y><<<grid_dim, block_dim>>>(
            dev_A,
            dev_B,
            dev_C,
            1,
            1,
            4096,
            4096,
            4096,
            4096,
            4096,
            4096
        );

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&milliseconds, start, stop);

        if (i > 185) total += milliseconds;
        flush_l2_cache();
    }

    std::cout << total / 185 << "\n";

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaFree(dev_A);
    cudaFree(dev_B);
    cudaFree(dev_C);

    delete []host_A;
    delete []host_B;

    flush_l2_cache();
}