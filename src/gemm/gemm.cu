//
// Created by sriram on 5/13/25.
//

#include "../../api.h"
#include <random>
#include "../ptx_helpers.cuh"
#include "../helpers.cuh"




// testing section
//---------------------------------------------------------------------------------------------------------------------

void test_gemm_2DBT() {
    // const auto data{new float[10 * 8]};
    //
    // print_matrix(data, 10, 8, 8);

    auto host_A{new float[211 * 35]};
    auto host_B{new float[35 * 68]};
    auto host_C{new float[211 * 68]};
    auto host_C_2{new float[211 * 68]};

    float *dev_A;
    float *dev_B;
    float *dev_C;

    fill_matrix_w(host_A, 211, 35, 35, -100, 100);
    fill_matrix_w(host_B, 35, 68, 68, -100, 100);
    fill_matrix_w(host_C, 211, 68, 68, 0, 0);

    cudaMalloc(&dev_A, 211 * 35 * sizeof(float));
    cudaMalloc(&dev_B, 35 * 68 * sizeof(float));
    cudaMalloc(&dev_C, 211 * 68 * sizeof(float));

    cudaMemcpy(dev_A, host_A, 211 * 35 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_B, host_B, 35 * 68 * sizeof(float), cudaMemcpyHostToDevice);

    cpu_matmul_naive(
        host_A,
        host_B,
        host_C,
        211,
        68,
        35,
        35,
        68,
        68
    );

    constexpr size_t TILE_SIZE_X{16};
    constexpr size_t TILE_SIZE_Y{16};
    constexpr size_t TILE_SIZE_K{16};

    constexpr size_t THREADS_PER_BLOCK{TILE_SIZE_X * TILE_SIZE_Y};

    static_assert(TILE_SIZE_K * TILE_SIZE_X % THREADS_PER_BLOCK == 0);
    static_assert(TILE_SIZE_K * TILE_SIZE_Y % THREADS_PER_BLOCK == 0);

    constexpr dim3 block_dim(TILE_SIZE_X, TILE_SIZE_Y);
    const dim3 grid_dim(ceil_div(68, TILE_SIZE_X), ceil_div(211, TILE_SIZE_Y));

    gemm_2DBT<
        TILE_SIZE_X,
        TILE_SIZE_Y,
        TILE_SIZE_K><<<grid_dim, block_dim>>>(
        dev_A,
        dev_B,
        dev_C,
        211,
        68,
        35,
        35,
        68,
        68
    );

    cudaDeviceSynchronize();

    // Check for errors in kernel execution
    if (const cudaError_t error = cudaGetLastError(); error != cudaSuccess)
        std::cerr << "CUDA error: " << cudaGetErrorString(error) << std::endl;

    cudaMemcpy(host_C_2, dev_C, 211 * 68 * sizeof(float), cudaMemcpyDeviceToHost);

    test_equivalency(host_C, host_C_2, 211, 68, 68);

    cudaFree(dev_A);
    cudaFree(dev_B);
    cudaFree(dev_C);

    delete []host_A;
    delete []host_B;
    delete []host_C;
    delete []host_C_2;
}

void test_gemm_2DBT_async() {
    // const auto data{new float[10 * 8]};
    //
    // print_matrix(data, 10, 8, 8);

    auto host_A{new float[211 * 35]};
    auto host_B{new float[35 * 68]};
    auto host_C{new float[211 * 68]};
    auto host_C_2{new float[211 * 68]};

    float *dev_A;
    float *dev_B;
    float *dev_C;

    fill_matrix_w(host_A, 211, 35, 35, -100, 100);
    fill_matrix_w(host_B, 35, 68, 68, -100, 100);
    fill_matrix_w(host_C, 211, 68, 68, 0, 0);

    cudaMalloc(&dev_A, 211 * 35 * sizeof(float));
    cudaMalloc(&dev_B, 35 * 68 * sizeof(float));
    cudaMalloc(&dev_C, 211 * 68 * sizeof(float));

    cudaMemcpy(dev_A, host_A, 211 * 35 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_B, host_B, 35 * 68 * sizeof(float), cudaMemcpyHostToDevice);

    cpu_matmul_naive(
        host_A,
        host_B,
        host_C,
        211,
        68,
        35,
        35,
        68,
        68
    );

    constexpr size_t TILE_SIZE_X{16};
    constexpr size_t TILE_SIZE_Y{16};
    constexpr size_t TILE_SIZE_K{16};

    constexpr size_t THREADS_PER_BLOCK{TILE_SIZE_X * TILE_SIZE_Y};

    static_assert(TILE_SIZE_K * TILE_SIZE_X % THREADS_PER_BLOCK == 0);
    static_assert(TILE_SIZE_K * TILE_SIZE_Y % THREADS_PER_BLOCK == 0);

    constexpr dim3 block_dim(TILE_SIZE_X, TILE_SIZE_Y);
    const dim3 grid_dim(ceil_div(68, TILE_SIZE_X), ceil_div(211, TILE_SIZE_Y));

    gemm_2DBT_async<
        TILE_SIZE_X,
        TILE_SIZE_Y,
        TILE_SIZE_K><<<grid_dim, block_dim>>>(
        dev_A,
        dev_B,
        dev_C,
        211,
        68,
        35,
        35,
        68,
        68
    );

    cudaDeviceSynchronize();

    // Check for errors in kernel execution
    if (const cudaError_t error = cudaGetLastError(); error != cudaSuccess)
        std::cerr << "CUDA error: " << cudaGetErrorString(error) << std::endl;

    cudaMemcpy(host_C_2, dev_C, 211 * 68 * sizeof(float), cudaMemcpyDeviceToHost);

    test_equivalency(host_C, host_C_2, 211, 68, 68);

    cudaFree(dev_A);
    cudaFree(dev_B);
    cudaFree(dev_C);

    delete []host_A;
    delete []host_B;
    delete []host_C;
    delete []host_C_2;
}

void test_gemm_2DBT_2DWT_2DTT_vload() {
    auto host_A{new float[211 * 32]};
    auto host_B{new float[32 * 64]};
    auto host_C{new float[211 * 64]};
    auto host_C_2{new float[211 * 64]};

    float *dev_A;
    float *dev_B;
    float *dev_C;

    fill_matrix_w(host_A, 211, 32, 32, -100, 100);
    fill_matrix_w(host_B, 32, 64, 64, -100, 100);
    fill_matrix_w(host_C, 211, 64, 64, 0, 0);

    cudaMalloc(&dev_A, 211 * 32 * sizeof(float));
    cudaMalloc(&dev_B, 32 * 64 * sizeof(float));
    cudaMalloc(&dev_C, 211 * 64 * sizeof(float));

    cudaMemcpy(dev_A, host_A, 211 * 32 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_B, host_B, 32 * 64 * sizeof(float), cudaMemcpyHostToDevice);

    cpu_matmul_naive(
        host_A,
        host_B,
        host_C,
        211,
        64,
        32,
        32,
        64,
        64
    );

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
        ceil_div(64, BLOCK_TILE_SIZE_X),
        ceil_div(211, BLOCK_TILE_SIZE_Y)
    };

    constexpr size_t NUM_THREADS_PER_BLOCK{32 * NUM_WARPS_PER_BLOCK_X * NUM_WARPS_PER_BLOCK_Y};
    constexpr dim3 block_dim(NUM_THREADS_PER_BLOCK);

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
        211,
        64,
        32,
        32,
        64,
        64
    );

    cudaDeviceSynchronize();

    // Check for errors in kernel execution
    if (const cudaError_t error = cudaGetLastError(); error != cudaSuccess)
        std::cerr << "CUDA error: " << cudaGetErrorString(error) << std::endl;

    cudaMemcpy(host_C_2, dev_C, 211 * 64 * sizeof(float), cudaMemcpyDeviceToHost);

    test_equivalency(host_C, host_C_2, 211, 64, 64);

    cudaFree(dev_A);
    cudaFree(dev_B);
    cudaFree(dev_C);

    delete []host_A;
    delete []host_B;
    delete []host_C;
    delete []host_C_2;
}

void test_gemm_2DBT_2DWT_2DTT_async() {
    // const auto data{new float[10 * 8]};
    //
    // print_matrix(data, 10, 8, 8);

    constexpr size_t m{211};
    constexpr size_t n{64};
    constexpr size_t k{32};

    auto host_A{new float[m * k]};
    auto host_B{new float[k * n]};
    auto host_C{new float[m * n]};
    auto host_C_2{new float[m * n]};

    float *dev_A;
    float *dev_B;
    float *dev_C;

    fill_matrix_w(host_A, m, k, k, -100, 100);
    fill_matrix_w(host_B, k, n, n, -100, 100);
    fill_matrix_w(host_C, m, n, n, 0, 0);

    cudaMalloc(&dev_A, m * k * sizeof(float));
    cudaMalloc(&dev_B, k * n * sizeof(float));
    cudaMalloc(&dev_C, m * n * sizeof(float));

    cudaMemcpy(dev_A, host_A, m * k * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_B, host_B, k * n * sizeof(float), cudaMemcpyHostToDevice);

    cpu_matmul_naive(
        host_A,
        host_B,
        host_C,
        m,
        n,
        k,
        k,
        n,
        n
    );

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
        ceil_div(n, BLOCK_TILE_SIZE_X),
        ceil_div(m, BLOCK_TILE_SIZE_Y)
    };

    constexpr size_t NUM_THREADS_PER_BLOCK{32 * NUM_WARPS_PER_BLOCK_X * NUM_WARPS_PER_BLOCK_Y};
    constexpr dim3 block_dim(NUM_THREADS_PER_BLOCK);

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
        m,
        n,
        k,
        k,
        n,
        n
    );

    cudaDeviceSynchronize();

    // Check for errors in kernel execution
    if (const cudaError_t error = cudaGetLastError(); error != cudaSuccess)
        std::cerr << "CUDA error: " << cudaGetErrorString(error) << std::endl;

    cudaMemcpy(host_C_2, dev_C, m * n * sizeof(float), cudaMemcpyDeviceToHost);

    test_equivalency(host_C, host_C_2, m, n, n);

    cudaFree(dev_A);
    cudaFree(dev_B);
    cudaFree(dev_C);

    delete []host_A;
    delete []host_B;
    delete []host_C;
    delete []host_C_2;
}

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

void time_gemm_2DBT_async() {
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
        gemm_2DBT_async<
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


void run_double_buffer_test() {
    // test_regular_shared_mem();
    // test_double_buffer_gemm();
    // time_db_gemm_memory();
    // test_gemm_2DBT_async();
    // time_2DBT();
    // time_gemm_2DBT_async();
    time_gemm_2DBT_2DWT_2DTT_vload();
    time_gemm_2DBT_2DWT_2DTT_async();
    // test_gemm_2DBT_2DWT_2DTT_vload();
    // test_gemm_2DBT_2DWT_2DTT_async();
    // time_double_buffer();
}
