//
// Created by sriram on 5/13/25.
//

#include "api.h"
#include <cooperative_groups.h>
#include <cuda/barrier>
#include <random>

using barrier = cuda::barrier<cuda::thread_scope_block>;

__device__ __host__ __forceinline__ size_t ceil_div(const size_t top, const size_t bottom) {
    return (top + (bottom - 1)) / bottom;
}

template<
    size_t TILE_SIZE_X,
    size_t TILE_SIZE_Y,
    size_t TILE_SIZE_K,
    size_t PRODUCER_THREADS_PER_BLOCK
>
__device__ __forceinline__ void load_to_shared(
    const size_t iter,
    float A_buffer[TILE_SIZE_Y][TILE_SIZE_K],
    float B_buffer[TILE_SIZE_K][TILE_SIZE_X],
    float *mat_A,
    float *mat_B,
    const size_t M,
    const size_t N,
    const size_t K,
    const size_t thread_linear_idx,
    const size_t A_stride,
    const size_t B_stride) {
    constexpr size_t A_iterations{TILE_SIZE_Y * TILE_SIZE_K / PRODUCER_THREADS_PER_BLOCK};

    for (size_t A_iter{0}; A_iter < A_iterations; ++A_iter) {
        size_t A_buffer_row{(thread_linear_idx + A_iter * PRODUCER_THREADS_PER_BLOCK) / TILE_SIZE_K};
        size_t A_buffer_column{(thread_linear_idx + A_iter * PRODUCER_THREADS_PER_BLOCK) % TILE_SIZE_K};

        float value{0};

        const size_t A_row{blockIdx.y * TILE_SIZE_Y + A_buffer_row};
        const size_t A_column{A_buffer_column + iter * TILE_SIZE_K};

        if (A_row < M && A_column < K)
            value = mat_A[A_row * A_stride + A_column];

        A_buffer[A_buffer_row][A_buffer_column] = value;
    }

    constexpr size_t B_iterations{TILE_SIZE_K * TILE_SIZE_X / PRODUCER_THREADS_PER_BLOCK};

    for (size_t B_iter{0}; B_iter < B_iterations; ++B_iter) {
        size_t B_buffer_row{(thread_linear_idx + B_iter * PRODUCER_THREADS_PER_BLOCK) / TILE_SIZE_X};
        size_t B_buffer_column{(thread_linear_idx + B_iter * PRODUCER_THREADS_PER_BLOCK) % TILE_SIZE_X};

        float value{0};

        const size_t B_row{B_buffer_row + iter * TILE_SIZE_K};
        const size_t B_column{blockIdx.x * TILE_SIZE_X + B_buffer_column};

        if (B_row < K && B_column < N)
            value = mat_B[B_row * B_stride + B_column];

        B_buffer[B_buffer_row][B_buffer_column] = value;
    }
}

template<size_t TILE_SIZE_X, size_t TILE_SIZE_Y, size_t TILE_SIZE_K>
__device__ __forceinline__ void consumer(
    const size_t iterations,
    barrier *ready,
    barrier *filled,
    float A_buffer[2][TILE_SIZE_Y][TILE_SIZE_K],
    float B_buffer[2][TILE_SIZE_K][TILE_SIZE_X],
    float *mat_C,
    const size_t thread_linear_idx,
    const size_t C_stride,
    const size_t M,
    const size_t N) {
    // signal we are ready for the initial shared memory filling
    barrier::arrival_token token1{ready[0].arrive()};
    barrier::arrival_token token2{ready[1].arrive()};

    float partial{0};

    size_t row{thread_linear_idx / TILE_SIZE_X};
    size_t column{thread_linear_idx % TILE_SIZE_X};

    const size_t C_row{TILE_SIZE_Y * blockIdx.y + row};
    const size_t C_column{blockIdx.x * TILE_SIZE_X + column};

    for (size_t iter{0}; iter < iterations; ++iter) {
        // alternate the buffers being used
        const size_t selected_buffer{iter % 2};

        // wait for that buffer to be ready
        filled[selected_buffer].arrive_and_wait();

        // consumption

        for (size_t k{0}; k < TILE_SIZE_K; ++k)
            partial += A_buffer[selected_buffer][row][k] * B_buffer[selected_buffer][k][column];

        // buffer is ready to be filled again
        barrier::arrival_token token{ready[selected_buffer].arrive()};
    }

    if (C_row < M && C_column < N) {
        mat_C[C_row * C_stride + C_column] = partial;
    }
}

template<
    size_t TILE_SIZE_X,
    size_t TILE_SIZE_Y,
    size_t TILE_SIZE_K,
    size_t PRODUCER_THREADS_PER_BLOCK
>
__device__ __forceinline__ void producer(
    const size_t iterations,
    barrier *ready,
    barrier *filled,
    float A_buffer[2][TILE_SIZE_Y][TILE_SIZE_K],
    float B_buffer[2][TILE_SIZE_K][TILE_SIZE_X],
    float *mat_A,
    float *mat_B,
    const size_t M,
    const size_t N,
    const size_t K,
    const size_t thread_linear_idx,
    const size_t A_stride,
    const size_t B_stride) {
    for (size_t iter{0}; iter < iterations; ++iter) {
        const size_t selected_buffer{iter % 2};
        ready[selected_buffer].arrive_and_wait();

        // fill shared memory
        load_to_shared<TILE_SIZE_X, TILE_SIZE_Y, TILE_SIZE_K, PRODUCER_THREADS_PER_BLOCK>(
            iter,
            A_buffer[selected_buffer],
            B_buffer[selected_buffer],
            mat_A,
            mat_B,
            M,
            N,
            K,
            thread_linear_idx,
            A_stride,
            B_stride
        );

        // wait for that buffer to be ready
        barrier::arrival_token token{filled[selected_buffer].arrive()};
    }
}


template<
    size_t CONSUMER_WARPS,
    size_t PRODUCER_WARPS,
    size_t WARP_SIZE = 32,
    size_t TILE_SIZE_X,
    size_t TILE_SIZE_Y,
    size_t TILE_SIZE_K>
__global__ void producer_consumer_pattern(
    float *mat_A,
    float *mat_B,
    float *mat_C,
    const size_t M,
    const size_t N,
    const size_t K,
    const size_t A_stride,
    const size_t B_stride,
    const size_t C_stride) {

    // CONSUMER_WARPS will handle processing the data
    // PRODUCER_WARPS will handle writing the data to memory

    // CONSUMER WARP signal if the array is ready to be filled
    // PRODUCER WARPS signal if the array is ready to be consumed

    // double buffer
    __shared__ float A_buffer[2][TILE_SIZE_Y][TILE_SIZE_K];
    __shared__ float B_buffer[2][TILE_SIZE_K][TILE_SIZE_X];
    __shared__ barrier bar[4];

    auto block = cooperative_groups::this_thread_block();
    constexpr size_t producer_threads_per_block{WARP_SIZE * PRODUCER_WARPS};
    constexpr size_t consumer_threads_per_block{WARP_SIZE * CONSUMER_WARPS};

    const size_t total_iters{ceil_div(K, TILE_SIZE_K)};

    // initialization
    if (block.thread_rank() == 0) {
        // tracks if a buffer is ready to be filled

        // printf("%d, %lu \n", block.size(), producer_threads_per_block + consumer_threads_per_block);
        init(&bar[0], producer_threads_per_block + consumer_threads_per_block);
        init(&bar[1], producer_threads_per_block + consumer_threads_per_block);

        // tracks if a buffer is ready to be consumed
        init(&bar[2], producer_threads_per_block + consumer_threads_per_block);
        init(&bar[3], producer_threads_per_block + consumer_threads_per_block);
    }

    block.sync();

    if (block.thread_rank() < producer_threads_per_block) {
        producer<TILE_SIZE_X,
            TILE_SIZE_Y, TILE_SIZE_K,
            producer_threads_per_block>(
            total_iters,
            bar,
            bar + 2,
            A_buffer,
            B_buffer,
            mat_A,
            mat_B,
            M,
            N,
            K,
            block.thread_rank(),
            A_stride,
            B_stride);
    } else {
        const size_t thread_linear_idx{block.thread_rank() - producer_threads_per_block};
        consumer<TILE_SIZE_X,
            TILE_SIZE_Y, TILE_SIZE_K>(
            total_iters,
            bar,
            bar + 2,
            A_buffer,
            B_buffer,
            mat_C,
            thread_linear_idx,
            C_stride,
            M,
            N
        );
    }
}


// testing section
//---------------------------------------------------------------------------------------------------------------------

template<
    size_t TILE_SIZE_X,
    size_t TILE_SIZE_Y,
    size_t TILE_SIZE_K>
__global__ void shared_block_tile_regular(
    float *mat_A,
    float *mat_B,
    float *mat_C,
    const size_t M,
    const size_t N,
    const size_t K,
    const size_t A_stride,
    const size_t B_stride,
    const size_t C_stride) {
    __shared__ float A_buffer[TILE_SIZE_Y][TILE_SIZE_K];
    __shared__ float B_buffer[TILE_SIZE_K][TILE_SIZE_X];

    const size_t total_iters{ceil_div(K, TILE_SIZE_K)};
    const size_t thread_linear_idx{TILE_SIZE_X * threadIdx.y + threadIdx.x};

    constexpr size_t total_threads{TILE_SIZE_X * TILE_SIZE_Y};

    const size_t C_col{TILE_SIZE_X * blockIdx.x + threadIdx.x};
    const size_t C_row{TILE_SIZE_Y * blockIdx.y + threadIdx.y};

    float partial{0.f};

    for (size_t iter{0}; iter < total_iters; ++iter) {
        load_to_shared<TILE_SIZE_X, TILE_SIZE_Y, TILE_SIZE_K, total_threads>(
            iter,
            A_buffer,
            B_buffer,
            mat_A,
            mat_B,
            M,
            N,
            K,
            thread_linear_idx,
            A_stride,
            B_stride);

        __syncthreads();

        for (size_t k{0}; k < TILE_SIZE_K; ++k) {
            partial += A_buffer[threadIdx.y][k] * B_buffer[k][threadIdx.x];
        }
        __syncthreads();
    }

    if (C_col < N && C_row < M)
        mat_C[C_stride * C_row + C_col] = partial;
}

void cpu_matmul_naive(
    const float *mat_A,
    const float *mat_B,
    float *mat_C,
    const size_t M,
    const size_t N,
    const size_t K,
    const size_t A_stride,
    const size_t B_stride,
    const size_t C_stride) {
    for (size_t i{0}; i < M; ++i) {
        for (size_t j{0}; j < N; ++j) {
            for (size_t k{0}; k < K; ++k) {
                mat_C[i * C_stride + j] +=
                        mat_A[A_stride * i + k] * mat_B[B_stride * k + j];
            }
        }
    }
}

void test_equivalency(
    const float * expected,
    const float * result,
    const size_t M,
    const size_t N,
    const size_t stride) {

    for (size_t i{0}; i < M; ++i) {
        for (size_t j{0}; j < N; ++j) {
            assert(expected[i * stride + j] == result[i * stride + j]);
        }
    }
}

float generate_random_float(const float min_val, const float max_val) {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(min_val, max_val);
    return dis(gen);
}

int generate_random_int(const int min_val, const int max_val) {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::uniform_int_distribution dis(min_val, max_val);
    return dis(gen);
}

void fill_matrix(
    float * matrix,
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

void fill_matrix_w(
    float * matrix,
    const size_t M,
    const size_t N,
    const size_t stride,
    const int min_val,
    const int max_val) {

    for (size_t i{0}; i < M; ++i) {
        for (size_t j{0}; j < N; ++j) {
            matrix[i * stride + j] = static_cast<float>(generate_random_int(min_val, max_val));
        }
    }
}


void print_matrix(
    const float * matrix,
    const size_t M,
    const size_t N,
    const size_t stride) {

    for (size_t i{0}; i < M; ++i) {
        std::cout << "[";
        for (size_t j{0}; j < N; ++j) {
            std::cout << matrix[i * stride + j] << " ";
        }
        std::cout << "]\n";
    }
}

void test_regular_shared_mem() {
    // const auto data{new float[10 * 8]};
    //
    // print_matrix(data, 10, 8, 8);

    auto host_A{new float[211 * 35]};
    auto host_B{new float[35 * 68]};
    auto host_C{new float[211 * 68]};
    auto host_C_2{new float[211 * 68]};

    float * dev_A;
    float * dev_B;
    float * dev_C;

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

    shared_block_tile_regular<
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

void test_double_buffer() {
    // const auto data{new float[10 * 8]};
    //
    // print_matrix(data, 10, 8, 8);

    auto host_A{new float[211 * 35]};
    auto host_B{new float[35 * 68]};
    auto host_C{new float[211 * 68]};
    auto host_C_2{new float[211 * 68]};

    float * dev_A;
    float * dev_B;
    float * dev_C;

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
    constexpr size_t CONSUMER_WARPS{16 * 16 / 32};
    constexpr size_t PRODUCER_WARPS{4};

    constexpr size_t THREADS_PER_BLOCK{TILE_SIZE_X * TILE_SIZE_Y};

    static_assert(TILE_SIZE_K * TILE_SIZE_X % THREADS_PER_BLOCK == 0);
    static_assert(TILE_SIZE_K * TILE_SIZE_Y % THREADS_PER_BLOCK == 0);

    constexpr dim3 block_dim(PRODUCER_WARPS * 32 + CONSUMER_WARPS * 32);
    const dim3 grid_dim(ceil_div(68, TILE_SIZE_X), ceil_div(211, TILE_SIZE_Y));

    producer_consumer_pattern<
        CONSUMER_WARPS,
        PRODUCER_WARPS,
        32,
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

void time_shared_memory() {
    auto host_A{new float[4096 * 4096]};
    auto host_B{new float[4096 * 4096]};

    float * dev_A;
    float * dev_B;
    float * dev_C;

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

    cudaEventRecord(start);
    shared_block_tile_regular<
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

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    std::cout << milliseconds << "\n";

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaFree(dev_A);
    cudaFree(dev_B);
    cudaFree(dev_C);

    delete []host_A;
    delete []host_B;
}

void time_double_buffer() {
    auto host_A{new float[4096 * 4096]};
    auto host_B{new float[4096 * 4096]};

    float * dev_A;
    float * dev_B;
    float * dev_C;

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
    constexpr size_t CONSUMER_WARPS{16 * 16 / 32};
    constexpr size_t PRODUCER_WARPS{16 * 16 / 32};

    constexpr size_t THREADS_PER_BLOCK{TILE_SIZE_X * TILE_SIZE_Y};

    static_assert(TILE_SIZE_K * TILE_SIZE_X % THREADS_PER_BLOCK == 0);
    static_assert(TILE_SIZE_K * TILE_SIZE_Y % THREADS_PER_BLOCK == 0);

    constexpr dim3 block_dim(PRODUCER_WARPS * 32 + CONSUMER_WARPS * 32);
    const dim3 grid_dim(ceil_div(4096, TILE_SIZE_X), ceil_div(4096, TILE_SIZE_Y));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    producer_consumer_pattern<
        CONSUMER_WARPS,
        PRODUCER_WARPS,
        32,
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

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    std::cout << milliseconds;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaFree(dev_A);
    cudaFree(dev_B);
    cudaFree(dev_C);

    delete []host_A;
    delete []host_B;
}


void run_double_buffer_test() {
    test_regular_shared_mem();
    test_double_buffer();
    time_shared_memory();
    time_double_buffer();
}
