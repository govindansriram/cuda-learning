//
// Created by sriram on 5/13/25.
//

#include "api.h"
#include <cooperative_groups.h>
#include <cuda/barrier>
#include <cuda/pipeline>
#include <random>

using barrier = cuda::barrier<cuda::thread_scope_block>;

__device__ __host__ __forceinline__ size_t ceil_div(const size_t top, const size_t bottom) {
    return (top + (bottom - 1)) / bottom;
}


template<
    typename T,
    typename VECTOR_TYPE = int4,
    size_t BLOCK_TILE_SIZE_X,
    size_t BLOCK_TILE_SIZE_Y,
    size_t BLOCK_TILE_SIZE_K,
    size_t THREADS_PER_BLOCK,
    size_t BLOCK_TILE_SKEW_SIZE_X = 0,
    size_t BLOCK_TILE_SKEW_SIZE_Y = 0
>
__device__ void load_data_to_shared_memory_transposed_vectorized(
    const T *matrix_one,
    const T *matrix_two,
    const size_t stride_one,
    const size_t stride_two,
    T one_shared[BLOCK_TILE_SIZE_K][BLOCK_TILE_SIZE_Y + BLOCK_TILE_SKEW_SIZE_Y],
    T two_shared[BLOCK_TILE_SIZE_K][BLOCK_TILE_SIZE_X + BLOCK_TILE_SKEW_SIZE_X],
    const size_t mat_one_rows,
    const size_t mat_two_columns,
    const size_t shared,
    const uint iteration,
    const uint thread_linear_idx,
    VECTOR_TYPE v0
) {
    constexpr size_t units_per_vector{sizeof(VECTOR_TYPE) / sizeof(T)};
    static_assert(sizeof(VECTOR_TYPE) % sizeof(T) == 0);

    // ensure there will be an even amount of vectorized loads
    static_assert(BLOCK_TILE_SIZE_X % units_per_vector == 0);
    static_assert(BLOCK_TILE_SIZE_K % units_per_vector == 0);

#ifndef BENCHMARK
    // ensures leading dimensions are padded to handle additional reads
    assert(stride_one % units_per_vector == 0);
    assert(stride_two % units_per_vector == 0);
#endif

    // We need to make sure the data alignment is correct.
    static_assert((BLOCK_TILE_SIZE_Y) * sizeof(T) % sizeof(VECTOR_TYPE) == 0U);
    static_assert((BLOCK_TILE_SIZE_X) * sizeof(T) % sizeof(VECTOR_TYPE) == 0U);

    static_assert((BLOCK_TILE_SIZE_Y + BLOCK_TILE_SKEW_SIZE_Y) * sizeof(T) % sizeof(VECTOR_TYPE) == 0U);
    static_assert((BLOCK_TILE_SIZE_X + BLOCK_TILE_SKEW_SIZE_X) * sizeof(T) % sizeof(VECTOR_TYPE) == 0U);

    // scaling the load number down to account for the vectorized size
    constexpr size_t VEC_BLOCK_TILE_SIZE_X{BLOCK_TILE_SIZE_X / units_per_vector};
    constexpr size_t VEC_BLOCK_TILE_SIZE_K{BLOCK_TILE_SIZE_K / units_per_vector};

    // determines how many vectorized loads are performed per thread
    constexpr size_t one_iterations{
        ceil_div(BLOCK_TILE_SIZE_Y * VEC_BLOCK_TILE_SIZE_K, THREADS_PER_BLOCK)
    };

    // load into matrix one
#pragma unroll
    for (size_t one_iter{0}; one_iter < one_iterations; ++one_iter) {
        const size_t one_shared_row{(thread_linear_idx + one_iter * THREADS_PER_BLOCK) / VEC_BLOCK_TILE_SIZE_K};
        const size_t one_shared_column{
            (thread_linear_idx + one_iter * THREADS_PER_BLOCK) % VEC_BLOCK_TILE_SIZE_K * units_per_vector
        };

        const size_t mat_one_row{blockIdx.y * BLOCK_TILE_SIZE_Y + one_shared_row};
        const size_t mat_one_column{iteration * BLOCK_TILE_SIZE_K + one_shared_column};

        VECTOR_TYPE mat_one_row_vector_vals{v0};

        // if in bounds we save the data to the temp register value mat_one_row_vector_vals
        if (mat_one_row < mat_one_rows && mat_one_column < shared) {
            const VECTOR_TYPE *mat_one_vec_ptr{
                reinterpret_cast<const VECTOR_TYPE *>(matrix_one + (mat_one_row * stride_one) + mat_one_column)
            };
            mat_one_row_vector_vals = *mat_one_vec_ptr;
        }

        // Transposed store of the data back into shared memory
        if (one_shared_row < BLOCK_TILE_SIZE_Y && one_shared_column < BLOCK_TILE_SIZE_K) {
            for (size_t i{0}; i < units_per_vector; ++i) {
                one_shared[one_shared_column + i][one_shared_row] =
                        reinterpret_cast<const T *>(&mat_one_row_vector_vals)[i];
            }
        }
    }

    constexpr size_t two_iterations{ceil_div(BLOCK_TILE_SIZE_K * VEC_BLOCK_TILE_SIZE_X, THREADS_PER_BLOCK)};

    // load into matrix two
#pragma unroll
    for (size_t two_iter{0}; two_iter < two_iterations; ++two_iter) {
        const size_t two_shared_row{(thread_linear_idx + two_iter * THREADS_PER_BLOCK) / VEC_BLOCK_TILE_SIZE_X};

        const size_t two_shared_column{
            (thread_linear_idx + two_iter * THREADS_PER_BLOCK) % VEC_BLOCK_TILE_SIZE_X * units_per_vector
        };

        const size_t mat_two_row{iteration * BLOCK_TILE_SIZE_K + two_shared_row};
        const size_t mat_two_column{blockIdx.x * BLOCK_TILE_SIZE_X + two_shared_column};

        VECTOR_TYPE mat_two_row_vector_vals{v0};

        // if in bounds we save the data to the temp register value mat_two_row_vector_vals
        if (mat_two_row < shared && mat_two_column < mat_two_columns) {
            const VECTOR_TYPE *mat_two_vec_ptr{
                reinterpret_cast<const VECTOR_TYPE *>(matrix_two + (mat_two_row * stride_two) + mat_two_column)
            };

            mat_two_row_vector_vals = *mat_two_vec_ptr;
        }

        if (two_shared_row < BLOCK_TILE_SIZE_K && two_shared_column < BLOCK_TILE_SIZE_X) {
            *reinterpret_cast<VECTOR_TYPE *>(&two_shared[two_shared_row][two_shared_column]) =
                    mat_two_row_vector_vals;
        }
    }
}

// TODO see if vectorized copies are better (16 bytes at at time)
// only possible for B matrices
template<
    typename T,
    size_t BLOCK_TILE_SIZE_X,
    size_t BLOCK_TILE_SIZE_K,
    size_t THREADS_PER_BLOCK,
    size_t BLOCK_TILE_SKEW_X = 0
>
__device__ __forceinline__ void load_data_to_shared_matrix_B_async(
    T B_shared[BLOCK_TILE_SIZE_K][BLOCK_TILE_SIZE_X + BLOCK_TILE_SKEW_X],
    const T *B_matrix,
    const size_t k,
    const size_t n,
    const size_t leading_dimension,
    const uint iteration,
    const uint thread_linear_idx,
    cuda::pipeline<cuda::thread_scope_thread> &B_shared_pipeline
) {
    // ensures even amount of copies per thread
    static_assert(BLOCK_TILE_SIZE_K * BLOCK_TILE_SIZE_X % THREADS_PER_BLOCK == 0);

    // determines how many copies are performed per thread
    constexpr size_t copy_iterations{
        ceil_div(BLOCK_TILE_SIZE_K * BLOCK_TILE_SIZE_X, THREADS_PER_BLOCK)
    };

#pragma unroll
    for (size_t copy_iter{0}; copy_iter < copy_iterations; ++copy_iter) {
        const size_t shared_row{
            (thread_linear_idx + copy_iter * THREADS_PER_BLOCK) / BLOCK_TILE_SIZE_X
        };

        const size_t shared_column{
            (thread_linear_idx + copy_iter * THREADS_PER_BLOCK) % BLOCK_TILE_SIZE_X
        };

        const size_t B_row{iteration * BLOCK_TILE_SIZE_K + shared_row};
        const size_t B_column{blockIdx.x * BLOCK_TILE_SIZE_X + shared_column};

        T value{static_cast<T>(0)};
        T *p_value{&value};

        if (B_row < k && B_column < n)
            p_value = B_matrix + (leading_dimension * B_row + B_column);

        B_shared_pipeline.producer_acquire();

        cuda::memcpy_async(
            &B_shared[shared_row][shared_column],
            p_value,
            sizeof(T),
            B_shared_pipeline
        );

        B_shared_pipeline.producer_commit();
    }
}

template<
    typename T,
    size_t BLOCK_TILE_SIZE_Y,
    size_t BLOCK_TILE_SIZE_K,
    size_t THREADS_PER_BLOCK,
    size_t BLOCK_TILE_SKEW_Y = 0
>
__device__ __forceinline__ void load_data_to_shared_matrix_A_transposed_async(
    T A_shared_T[BLOCK_TILE_SIZE_K][BLOCK_TILE_SIZE_Y + BLOCK_TILE_SKEW_Y],
    const T *A_matrix,
    const size_t k,
    const size_t m,
    const size_t leading_dimension,
    const uint iteration,
    const uint thread_linear_idx,
    cuda::pipeline<cuda::thread_scope_thread> &A_shared_pipeline
) {
    // ensures even amount of copies per thread
    static_assert(BLOCK_TILE_SIZE_K * BLOCK_TILE_SIZE_Y % THREADS_PER_BLOCK == 0);

    // determines how many copies are performed per thread
    constexpr size_t copy_iterations{
        ceil_div(BLOCK_TILE_SIZE_K * BLOCK_TILE_SIZE_Y, THREADS_PER_BLOCK)
    };

#pragma unroll
    for (size_t copy_iter{0}; copy_iter < copy_iterations; ++copy_iter) {
        const size_t shared_row{
            (thread_linear_idx + copy_iter * THREADS_PER_BLOCK) / BLOCK_TILE_SIZE_K
        };

        const size_t shared_column{
            (thread_linear_idx + copy_iter * THREADS_PER_BLOCK) % BLOCK_TILE_SIZE_K
        };

        const size_t A_row{blockIdx.y * BLOCK_TILE_SIZE_Y + shared_row};
        const size_t A_column{iteration * BLOCK_TILE_SIZE_K + shared_column};

        T value{static_cast<T>(0)};
        T *p_value{&value};

        if (A_row < m && A_column < k)
            p_value = A_matrix + (leading_dimension * A_row + A_column);

        A_shared_pipeline.producer_acquire();

        cuda::memcpy_async(
            &A_shared_T[shared_column][shared_row],
            p_value,
            sizeof(T),
            A_shared_pipeline
        );

        A_shared_pipeline.producer_commit();
    }
}

template<
    typename T,
    size_t BLOCK_TILE_SIZE_Y,
    size_t BLOCK_TILE_SIZE_K,
    size_t THREADS_PER_BLOCK,
    size_t BLOCK_TILE_SKEW_Y = 0
>
__device__ __forceinline__ void load_data_to_shared_async() {

}

__device__ __host__ __forceinline__ constexpr bool is_power_of_two(const size_t x) {
    /**
     * first checks if the number is not 0 we do this premptive check since 0 is not a power of 2
     * yet would pass the next check
     *
     * (x & (x - 1)) == 0 this checks if the number is a power of 2
     *
     * all powers of 2 have exactly one 1 bit e.g.:
     * 1 0 0 0 = 1
     * 0 1 0 0 = 2
     * 0 0 1 0 = 4
     * 0 0 0 1 = 8
     *
     * the number before of 2 has to have all 1 bits excpet at the last position
     *
     * 0 0 0 0 = 0
     * 1 0 0 0 = 1
     * 1 1 0 0 = 3
     * 1 1 1 0 = 7
     *
     * so we know a number is a multiple of 2 if the bit wise and of the previous current number == 0
     */
    return x != 0 && (x & (x - 1)) == 0;
}

__device__ __host__ __forceinline__ constexpr size_t next_power_of_two(size_t x) {
    /**
     * We know that 1 - (a power of 2) is all 1 bits so to find the next power of 2
     * our goal is to fill all bits before the highest set bit to 1 and then add
     * 1 to it giving us our next power of 2.
     *
     * We also start off by subtracting 1 to handle the case where the number
     * is already a power of 2
     **/


    if (x == 0) return 1;
    --x;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    x |= x >> 32;
    return x + 1;
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


template<
    size_t TILE_SIZE_X,
    size_t TILE_SIZE_Y,
    size_t TILE_SIZE_K,
    size_t A_BUFFER_SIZE,
    size_t B_BUFFER_SIZE,
    size_t THREADS_PER_BLOCK
>
__device__ __forceinline__ void load_to_shared_double_buffer(
    const size_t iter,
    float A_buffer[A_BUFFER_SIZE],
    float B_buffer[B_BUFFER_SIZE],
    const float *mat_A,
    const float *mat_B,
    const size_t M,
    const size_t N,
    const size_t K,
    const size_t thread_linear_idx,
    const size_t A_stride,
    const size_t B_stride) {
    constexpr size_t A_iterations{TILE_SIZE_Y * TILE_SIZE_K / THREADS_PER_BLOCK};

    for (size_t A_iter{0}; A_iter < A_iterations; ++A_iter) {
        size_t A_buffer_row{(thread_linear_idx + A_iter * THREADS_PER_BLOCK) / TILE_SIZE_K};
        size_t A_buffer_column{(thread_linear_idx + A_iter * THREADS_PER_BLOCK) % TILE_SIZE_K};

        float value{0};

        const size_t A_row{blockIdx.y * TILE_SIZE_Y + A_buffer_row};
        const size_t A_column{A_buffer_column + iter * TILE_SIZE_K};

        if (A_row < M && A_column < K)
            value = mat_A[A_row * A_stride + A_column];

        A_buffer[A_buffer_row * TILE_SIZE_K + A_buffer_column] = value;
    }

    constexpr size_t B_iterations{TILE_SIZE_K * TILE_SIZE_X / THREADS_PER_BLOCK};

    for (size_t B_iter{0}; B_iter < B_iterations; ++B_iter) {
        size_t B_buffer_row{(thread_linear_idx + B_iter * THREADS_PER_BLOCK) / TILE_SIZE_X};
        size_t B_buffer_column{(thread_linear_idx + B_iter * THREADS_PER_BLOCK) % TILE_SIZE_X};

        float value{0};

        const size_t B_row{B_buffer_row + iter * TILE_SIZE_K};
        const size_t B_column{blockIdx.x * TILE_SIZE_X + B_buffer_column};

        if (B_row < K && B_column < N)
            value = mat_B[B_row * B_stride + B_column];

        B_buffer[B_buffer_row * TILE_SIZE_X + B_buffer_column] = value;
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

template<size_t TILE_SIZE_X, size_t TILE_SIZE_K>
__device__ __forceinline__ void accumulate(
    float &partial,
    const uintptr_t A_block_load_addr,
    const uintptr_t B_block_load_addr) {
    const auto A_load_ptr{reinterpret_cast<float *>(A_block_load_addr)};
    const auto B_load_ptr{reinterpret_cast<float *>(B_block_load_addr)};

    for (size_t k{0}; k < TILE_SIZE_K; ++k)
        partial += A_load_ptr[threadIdx.y * TILE_SIZE_K + k] * B_load_ptr[k * TILE_SIZE_X + threadIdx.x];
}

template<
    size_t THREADS_PER_BLOCK,
    size_t TILE_SIZE_X,
    size_t TILE_SIZE_Y,
    size_t TILE_SIZE_K>
__global__ void gemm_double_buffering(
    float *mat_A,
    float *mat_B,
    float *mat_C,
    const size_t M,
    const size_t N,
    const size_t K,
    const size_t A_stride,
    const size_t B_stride,
    const size_t C_stride) {
    // implementation derived from https://salykova.github.io/sgemm-gpu section 5.1

    constexpr size_t A_buffer_size{next_power_of_two(2 * TILE_SIZE_Y * TILE_SIZE_K * sizeof(float))};
    constexpr size_t B_buffer_size{next_power_of_two(2 * TILE_SIZE_K * TILE_SIZE_X * sizeof(float))};

    __shared__ float __align__(A_buffer_size) A_buffer[A_buffer_size / sizeof(float)];
    __shared__ float __align__(B_buffer_size) B_buffer[B_buffer_size / sizeof(float)];

    const size_t iterations{ceil_div(K, TILE_SIZE_K) - 1};

    const size_t thread_linear_idx{threadIdx.y * TILE_SIZE_X + threadIdx.x};
    const size_t C_row{blockIdx.y * TILE_SIZE_Y + threadIdx.y};
    const size_t C_column{blockIdx.x * TILE_SIZE_X + threadIdx.x};

    load_to_shared_double_buffer<
        TILE_SIZE_X,
        TILE_SIZE_Y,
        TILE_SIZE_K,
        A_buffer_size,
        B_buffer_size,
        THREADS_PER_BLOCK>(
        0,
        A_buffer,
        B_buffer,
        mat_A,
        mat_B,
        M,
        N,
        K,
        thread_linear_idx,
        A_stride,
        B_stride
    );

    __syncthreads();

    auto A_block_store_addr = reinterpret_cast<uintptr_t>(A_buffer);
    auto A_block_load_addr = reinterpret_cast<uintptr_t>(A_buffer);

    auto B_block_store_addr = reinterpret_cast<uintptr_t>(B_buffer);
    auto B_block_load_addr = reinterpret_cast<uintptr_t>(B_buffer);

    constexpr uintptr_t flip_bits_A{A_buffer_size / 2};
    constexpr uintptr_t flip_bits_B{A_buffer_size / 2};

    A_block_store_addr ^= flip_bits_A;
    B_block_store_addr ^= flip_bits_B;

    float partial{0};

    for (size_t iter{0}; iter < iterations; ++iter) {
        auto A_store_ptr{reinterpret_cast<float *>(A_block_store_addr)};
        auto B_store_ptr{reinterpret_cast<float *>(B_block_store_addr)};

        load_to_shared_double_buffer<
            TILE_SIZE_X,
            TILE_SIZE_Y,
            TILE_SIZE_K,
            A_buffer_size,
            B_buffer_size,
            THREADS_PER_BLOCK>(
            iter + 1,
            A_store_ptr,
            B_store_ptr,
            mat_A,
            mat_B,
            M,
            N,
            K,
            thread_linear_idx,
            A_stride,
            B_stride
        );

        accumulate<TILE_SIZE_X, TILE_SIZE_K>(partial, A_block_load_addr, B_block_load_addr);

        A_block_load_addr ^= flip_bits_A;
        A_block_store_addr ^= flip_bits_A;

        B_block_load_addr ^= flip_bits_B;
        B_block_store_addr ^= flip_bits_B;

        __syncthreads();
    }

    accumulate<TILE_SIZE_X, TILE_SIZE_K>(partial, A_block_load_addr, B_block_load_addr);

    if (C_row < M && C_column < N) {
        mat_C[C_row * C_stride + C_column] = partial;
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
    const float *expected,
    const float *result,
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

void fill_matrix_w(
    float *matrix,
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
    const float *matrix,
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

void test_double_buffer_gemm() {
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

    gemm_double_buffering<
        THREADS_PER_BLOCK,
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

void test_regular_shared_mem() {
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

void time_shared_memory() {
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

void time_db_gemm_memory() {
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

    constexpr size_t TILE_SIZE_X{32};
    constexpr size_t TILE_SIZE_Y{32};
    constexpr size_t TILE_SIZE_K{32};

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
        gemm_double_buffering<
            THREADS_PER_BLOCK,
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
}

void time_double_buffer() {
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
    // test_regular_shared_mem();
    // test_double_buffer_gemm();
    time_db_gemm_memory();
    time_shared_memory();
    // time_double_buffer();
}
