//
// Created by sriram on 5/27/25.
//

#ifndef GEMM_CUH
#define GEMM_CUH

#include "../ptx_helpers.cuh"
#include "../helpers.cuh"
#include <cstdint>

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
    size_t TILE_SIZE_K>
__global__ void gemm_2DBT(
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

// #ifndef BENCHMARK
//     // ensures leading dimensions are padded to handle additional reads
//     assert(stride_one % units_per_vector == 0);
//     assert(stride_two % units_per_vector == 0);
// #endif

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

template<
    typename T,
    size_t BLOCK_TILE_SIZE_X,
    size_t BLOCK_TILE_SIZE_Y,
    size_t BLOCK_TILE_SIZE_K,
    size_t WARP_TILE_SIZE_X,
    size_t WARP_TILE_SIZE_Y,
    size_t THREAD_TILE_SIZE_X,
    size_t THREAD_TILE_SIZE_Y,
    size_t NUM_THREADS_PER_WARP_X,
    size_t NUM_THREADS_PER_WARP_Y
>
__global__ void gemm_2DBT_2DWT_2DTT_vload(
    const T *matrix_one,
    const T *matrix_two,
    T *matrix_dest,
    const T alpha,
    const T beta,
    const size_t mat_one_rows,
    const size_t mat_two_columns,
    const size_t shared,
    const size_t row_stride_one,
    const size_t row_stride_two,
    const size_t row_stride_dest) {
    __shared__ T mat_one_thread_block_tile_transposed[BLOCK_TILE_SIZE_K][BLOCK_TILE_SIZE_Y];
    __shared__ T mat_two_thread_block_tile[BLOCK_TILE_SIZE_K][BLOCK_TILE_SIZE_X];

    // One Warp TILE will be of size WARP_TILE_SIZE_X x WARP_TILE_SIZE_Y
    // One Warp will be responsible for each Warp block, ideally multiple warp blocks
    // will be able to fit in one regular block allowing multiple warps to exist per
    // block

    // EACH block computes BLOCK_TILE_SIZE_X * BLOCK_TILE_SIZE_Y outputs of C

    // max threads per warp is 32, so we ensure that the warp block also complies
    // with this.
    static_assert(NUM_THREADS_PER_WARP_X * NUM_THREADS_PER_WARP_Y == 32);

    // We need to figure out a couple of things,
    // 1) we need to figure out how many WARP Tiles will be present in
    // the x and y direction similar as to calculating how many blocks will
    // be in the grid for a GPU launch we are doing the same but making a block
    // the grid and having our WARP TILE Be the new block
    //
    // 2) This is needed to calculate the total amount of THREADS per block in a
    // constant way
    constexpr size_t NUM_WARPS_PER_BLOCK_X{BLOCK_TILE_SIZE_X / WARP_TILE_SIZE_X};
    static_assert(BLOCK_TILE_SIZE_X % WARP_TILE_SIZE_X == 0);

    // repeat for y dimension
    constexpr size_t NUM_WARPS_PER_BLOCK_Y{BLOCK_TILE_SIZE_Y / WARP_TILE_SIZE_Y};
    static_assert(BLOCK_TILE_SIZE_Y % WARP_TILE_SIZE_Y == 0);

    // so total amount of warp tiles in a block would be
    // NUM_WARPS_PER_BLOCK_X * NUM_WARPS_PER_BLOCK_Y

    // In the previous implementation each thread had 2 register caches
    //
    // one cache cached several values in the y dimension from matrix one, total
    // elements are THREAD_TILE_SIZE_Y
    //
    // two cache cached several values in the x dimension from matrix two, total
    // elements are THREAD_TILE_SIZE_X
    //
    // In the end these values were reused for multiplication computing a total
    // of THREAD_TILE_SIZE_Y x THREAD_TILE_SIZE_X elements.
    //
    // Now that we are bounding warps to WARP_TILE sizes the amount of data being
    // computed by each thread may go up. So we know need to adjust the amount
    // of values being cached and computed to reflect this.
    //
    // We ideally want to keep our thread tile sizes consistent so we
    // instead add an extra dimension to each cache
    constexpr size_t NUM_CACHES_PER_WARP_X{
        WARP_TILE_SIZE_X / (THREAD_TILE_SIZE_X * NUM_THREADS_PER_WARP_X)
    };

    // repeat for y TILE cache
    constexpr size_t NUM_CACHES_PER_WARP_Y{
        WARP_TILE_SIZE_Y / (THREAD_TILE_SIZE_Y * NUM_THREADS_PER_WARP_Y)
    };

    static_assert(WARP_TILE_SIZE_X % (THREAD_TILE_SIZE_X * NUM_THREADS_PER_WARP_X) == 0);
    static_assert(WARP_TILE_SIZE_Y % (THREAD_TILE_SIZE_Y * NUM_THREADS_PER_WARP_Y) == 0);

    // Now we create the caches with the extra dimension
    T one_cache[NUM_CACHES_PER_WARP_Y][THREAD_TILE_SIZE_Y] = {static_cast<T>(0)};
    T two_cache[NUM_CACHES_PER_WARP_X][THREAD_TILE_SIZE_X] = {static_cast<T>(0)};

    // since we have more caches we will have more intermediates (values computed per thread)
    // as well, so we add extra dimensions here as well reflecting this
    T intermediates[NUM_CACHES_PER_WARP_Y][NUM_CACHES_PER_WARP_X][THREAD_TILE_SIZE_Y][
        THREAD_TILE_SIZE_X] = {static_cast<T>(0)};

    // now we can also easily calculate the total threads per block, needed for loading data
    constexpr size_t THREADS_PER_BLOCK{NUM_WARPS_PER_BLOCK_X * NUM_WARPS_PER_BLOCK_Y * 32};

    // this kernel should be launched with a 1d block so the linear dimension is just the threadidx.x
    const size_t thread_linear_idx{threadIdx.x};

    // the linear idx of the warp in the thread block
    const size_t warp_linear_idx{thread_linear_idx / warpSize};

    // Now lets figure out what warp that linear idx maps too (x, y)
    const size_t warp_row_idx{warp_linear_idx / NUM_WARPS_PER_BLOCK_X};
    const size_t warp_col_idx{warp_linear_idx % NUM_WARPS_PER_BLOCK_X};

    // figure out what row and column we are in the warp
    const size_t thread_linear_idx_in_warp{thread_linear_idx % warpSize};
    const size_t thread_idx_in_warp_row{thread_linear_idx_in_warp / NUM_THREADS_PER_WARP_X};
    const size_t thread_idx_in_warp_column{thread_linear_idx_in_warp % NUM_THREADS_PER_WARP_X};

    constexpr size_t units_per_vector{sizeof(int4) / sizeof(T)};

    // ensure int4 can be event split up by the base TYPE necessary for conversion
    static_assert(sizeof(int4) % sizeof(T) == 0);

    // we will store data along these dimensions for vectorized storage they need to be divisible
    static_assert(BLOCK_TILE_SIZE_K % units_per_vector == 0);
    static_assert(BLOCK_TILE_SIZE_X % units_per_vector == 0);

    static_assert(THREAD_TILE_SIZE_X % units_per_vector == 0);
    static_assert(THREAD_TILE_SIZE_Y % units_per_vector == 0);

    // This determines how many vectorized loads we need to perform to fill one tile
    constexpr size_t vectorized_thread_tile_size_x{THREAD_TILE_SIZE_X / units_per_vector};
    constexpr size_t vectorized_thread_tile_size_y{THREAD_TILE_SIZE_Y / units_per_vector};

    const size_t total_iters{ceil_div(shared, BLOCK_TILE_SIZE_K)};

    for (size_t iter{0}; iter < total_iters; ++iter) {
        load_data_to_shared_memory_transposed_vectorized<
            T, int4,
            BLOCK_TILE_SIZE_X,
            BLOCK_TILE_SIZE_Y,
            BLOCK_TILE_SIZE_K,
            THREADS_PER_BLOCK
        >(
            matrix_one,
            matrix_two,
            row_stride_one,
            row_stride_two,
            mat_one_thread_block_tile_transposed,
            mat_two_thread_block_tile,
            mat_one_rows,
            mat_two_columns,
            shared,
            iter,
            thread_linear_idx,
            int4{0, 0, 0, 0}
        );

        __syncthreads();

        // #pragma unroll
        for (size_t k{0}; k < BLOCK_TILE_SIZE_K; ++k) {
            // we need to start filling the one matrix cache
#pragma unroll
            for (size_t y_cache_idx{0}; y_cache_idx < NUM_CACHES_PER_WARP_Y; ++y_cache_idx) {
                // Here we calculate the row in the shared block based on the warp coordinates
                // and the thread coordinates

                // To calculate the row we first multiply the warp block y coordinate by the
                // Warp y dimension scale on the grid scale this is equivalent to doing blockIdx.y * blockDim.y
                // Next based on what cache we are in we need to skip that many rows. We do this by multiplying the
                // y_cache_idx by (WARP_TILE_SIZE_Y / NUM_CACHES_PER_WARP_Y) this value is equivalent too
                // (THREAD_TILE_SIZE_Y * NUM_THREADS_PER_WARP_Y) which is how many distinct rows are processed per
                // one cache fill of warp tile. Finally, we add the row that this thread is part of in the warp.

                // each thread loads TILE_SIZE_Y
                // assuming this configuration NUM_THREADS_PER_WARP_X = 4, and NUM_THREADS_PER_WARP_Y = 8
                // we can assume this load pattern
                // Threads [0 to 3] load rows [0 to 7], Threads [4 to 7] load rows [8 to 15] ...
                // Threads [28 to 31] load rows [54 to 63], this would result in a bank conflict for each
                // new warp_row and a broadcast for all threads in warp row, but luckily
                // the shared memory is transposed resulting in only broadcasts
                const size_t one_shared_row_idx{
                    warp_row_idx * WARP_TILE_SIZE_Y +
                    y_cache_idx * (WARP_TILE_SIZE_Y / NUM_CACHES_PER_WARP_Y) +
                    thread_idx_in_warp_row * THREAD_TILE_SIZE_Y
                };

                const auto one_shared_ptr{
                    reinterpret_cast<int4 *>(&mat_one_thread_block_tile_transposed[k][one_shared_row_idx])
                };

                auto tile_ptr{
                    reinterpret_cast<int4 *>(&one_cache[y_cache_idx])
                };

                // load into register cache one[y_cache_idx] with vectorized loads
                // #pragma unroll
                for (size_t vy_iter{0}; vy_iter < vectorized_thread_tile_size_y; ++vy_iter)
                    tile_ptr[vy_iter] = one_shared_ptr[vy_iter];
            }

#pragma unroll
            for (size_t x_cache_id{0}; x_cache_id < NUM_CACHES_PER_WARP_X; ++x_cache_id) {
                const size_t two_shared_col_idx{
                    warp_col_idx * WARP_TILE_SIZE_X +
                    x_cache_id * (WARP_TILE_SIZE_X / NUM_CACHES_PER_WARP_X) +
                    thread_idx_in_warp_column * THREAD_TILE_SIZE_X
                };

                const auto two_shared_ptr{
                    reinterpret_cast<int4 *>(&mat_two_thread_block_tile[k][two_shared_col_idx])
                };

                auto tile_ptr{
                    reinterpret_cast<int4 *>(&two_cache[x_cache_id])
                };

                // #pragma unroll
                for (size_t vx_iter{0}; vx_iter < vectorized_thread_tile_size_x; ++vx_iter)
                    tile_ptr[vx_iter] = two_shared_ptr[vx_iter];
            }

            // compute intermediates
#pragma unroll
            for (size_t y_cache_idx{0}; y_cache_idx < NUM_CACHES_PER_WARP_Y; ++y_cache_idx) {
                // #pragma unroll
                for (size_t x_cache_idx{0}; x_cache_idx < NUM_CACHES_PER_WARP_X; ++x_cache_idx) {
#pragma unroll
                    for (size_t one_cache_idx{0}; one_cache_idx < THREAD_TILE_SIZE_Y; ++one_cache_idx) {
                        T one_cache_value{one_cache[y_cache_idx][one_cache_idx]};
                        // #pragma unroll
                        for (size_t two_cache_index{0}; two_cache_index < THREAD_TILE_SIZE_X; ++two_cache_index) {
                            intermediates[y_cache_idx][x_cache_idx][one_cache_idx][two_cache_index] +=
                                    one_cache_value * two_cache[x_cache_idx][two_cache_index];
                        }
                    }
                }
            }
        }
        __syncthreads();
    }

    // vectorized store back into the dest matrix
#pragma unroll
    for (size_t y_cache_idx{0}; y_cache_idx < NUM_CACHES_PER_WARP_Y; ++y_cache_idx) {
#pragma unroll
        for (size_t x_cache_idx{0}; x_cache_idx < NUM_CACHES_PER_WARP_X; ++x_cache_idx) {
            // #pragma unroll
            for (size_t one_cache_idx{0}; one_cache_idx < THREAD_TILE_SIZE_Y; ++one_cache_idx) {
                const size_t dest_row{
                    BLOCK_TILE_SIZE_Y * blockIdx.y +
                    warp_row_idx * WARP_TILE_SIZE_Y +
                    y_cache_idx * (WARP_TILE_SIZE_Y / NUM_CACHES_PER_WARP_Y) +
                    thread_idx_in_warp_row * THREAD_TILE_SIZE_Y + one_cache_idx
                };

                const size_t dest_column{
                    BLOCK_TILE_SIZE_X * blockIdx.x +
                    warp_col_idx * WARP_TILE_SIZE_X +
                    x_cache_idx * (WARP_TILE_SIZE_X / NUM_CACHES_PER_WARP_X) +
                    thread_idx_in_warp_column * THREAD_TILE_SIZE_X
                };

                auto dest_ptr{&matrix_dest[dest_row * row_stride_dest + dest_column]};
                T *tile_ptr{&intermediates[y_cache_idx][x_cache_idx][one_cache_idx][0]};

                // #pragma unroll
                for (size_t two_cache_vec_idx{0}; two_cache_vec_idx < vectorized_thread_tile_size_x; ++
                     two_cache_vec_idx) {
                    if (dest_row < mat_one_rows && (
                            dest_column + two_cache_vec_idx * units_per_vector < mat_two_columns)) {
                        // #pragma unroll
                        for (size_t tile_idx{0}; tile_idx < units_per_vector; ++tile_idx) {
                            tile_ptr[tile_idx] = tile_ptr[tile_idx] * alpha + dest_ptr[tile_idx] * beta;
                        }

                        reinterpret_cast<int4 *>(dest_ptr)[two_cache_vec_idx] =
                                reinterpret_cast<int4 *>(tile_ptr)[two_cache_vec_idx];
                    }
                }
            }
        }
    }
}

__device__ char ZEROS[16];

// TODO see if vectorized copies are better (16 bytes at at time)
// only possible for B matrices
template<
    typename T,
    size_t BLOCK_TILE_SIZE_X,
    size_t BLOCK_TILE_SIZE_K,
    size_t THREADS_PER_BLOCK,
    size_t BLOCK_TILE_SKEW_X = 0,
    size_t LOAD_BYTES = 4
>
__device__ __forceinline__ void load_data_to_shared_matrix_B_async(
    T B_shared[BLOCK_TILE_SIZE_K][BLOCK_TILE_SIZE_X + BLOCK_TILE_SKEW_X],
    const T *B_matrix,
    const size_t k,
    const size_t n,
    const size_t leading_dimension,
    const uint iteration,
    const uint thread_linear_idx
) {
    // ensures even amount of copies per thread
    static_assert(BLOCK_TILE_SIZE_K * BLOCK_TILE_SIZE_X % THREADS_PER_BLOCK == 0);

    constexpr size_t units_per_load(LOAD_BYTES / sizeof(T));

    static_assert(BLOCK_TILE_SIZE_K * BLOCK_TILE_SIZE_X % units_per_load == 0);

    // determines how many copies are performed per thread
    constexpr size_t copy_iterations{
        ceil_div(BLOCK_TILE_SIZE_K * BLOCK_TILE_SIZE_X / units_per_load, THREADS_PER_BLOCK)
    };

    constexpr size_t vec_block_tile_size_x{BLOCK_TILE_SIZE_X / units_per_load};

    // assert(leading_dimension % units_per_load == 0);

#pragma unroll
    for (size_t copy_iter{0}; copy_iter < copy_iterations; ++copy_iter) {
        const size_t shared_row{
            (thread_linear_idx + copy_iter * THREADS_PER_BLOCK) / vec_block_tile_size_x
        };

        const size_t shared_column{
            (thread_linear_idx + copy_iter * THREADS_PER_BLOCK) % vec_block_tile_size_x * units_per_load
        };

        const size_t B_row{iteration * BLOCK_TILE_SIZE_K + shared_row};
        const size_t B_column{blockIdx.x * BLOCK_TILE_SIZE_X + shared_column};

        constexpr T value[units_per_load] = {0};
        const T *p_value;

        if (B_row < k && B_column < n)
            p_value = B_matrix + (leading_dimension * B_row + B_column);
        else
            p_value = reinterpret_cast<const T *>(ZEROS);

        CP_ASYNC_LARGE(
            &B_shared[shared_row][shared_column],
            p_value
        );

    }
}


template<
    typename T,
    size_t BLOCK_TILE_SIZE_Y,
    size_t BLOCK_TILE_SIZE_K,
    size_t THREADS_PER_BLOCK,
    size_t BLOCK_TILE_SKEW_K = 0,
    size_t LOAD_BYTES = 4
>
__device__ __forceinline__ void load_data_to_shared_matrix_A_async(
    T A_shared[BLOCK_TILE_SIZE_Y][BLOCK_TILE_SIZE_K + BLOCK_TILE_SKEW_K],
    const T *A_matrix,
    const size_t m,
    const size_t k,
    const size_t leading_dimension,
    const uint iteration,
    const uint thread_linear_idx
) {
    // ensures even amount of copies per thread
    static_assert(BLOCK_TILE_SIZE_K * BLOCK_TILE_SIZE_Y % THREADS_PER_BLOCK == 0);

    constexpr size_t units_per_load(LOAD_BYTES / sizeof(T));

    // determines how many copies are performed per thread
    constexpr size_t copy_iterations{
        ceil_div(BLOCK_TILE_SIZE_K * BLOCK_TILE_SIZE_Y / units_per_load, THREADS_PER_BLOCK)
    };

    constexpr size_t vectorized_block_tile_size_k{BLOCK_TILE_SIZE_K / units_per_load};

#pragma unroll
    for (size_t copy_iter{0}; copy_iter < copy_iterations; ++copy_iter) {
        const size_t shared_row{
            (thread_linear_idx + copy_iter * THREADS_PER_BLOCK) / vectorized_block_tile_size_k
        };

        const size_t shared_column{
            (thread_linear_idx + copy_iter * THREADS_PER_BLOCK) % vectorized_block_tile_size_k * units_per_load
        };

        const size_t A_row{BLOCK_TILE_SIZE_Y * blockIdx.y + shared_row};
        const size_t A_column{iteration * BLOCK_TILE_SIZE_K + shared_column};

        T value[units_per_load];
        const T *p_value;

        if (A_row < m && A_column < k)
            p_value = A_matrix + (leading_dimension * A_row + A_column);
        else
            p_value = reinterpret_cast<const T *>(ZEROS);

        CP_ASYNC_LARGE(&A_shared[shared_row][shared_column], p_value);
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
    const uint thread_linear_idx
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
        const T *p_value;

        if (A_row < m && A_column < k)
            p_value = A_matrix + (leading_dimension * A_row + A_column);
        else
            p_value = ZEROS;

        CP_ASYNC_SMALL(&A_shared_T[shared_column][shared_row], p_value, 4);
    }
}

template<
    bool TRANSPOSED_A,
    size_t BLOCK_TILE_SIZE_Y,
    size_t BLOCK_TILE_SIZE_K,
    size_t Y_SKEW,
    size_t K_SKEW,
    bool IS_ROW>
constexpr uint A_shared_dim() {
    if constexpr (TRANSPOSED_A) {
        if constexpr (IS_ROW)
            return BLOCK_TILE_SIZE_K + K_SKEW;
        else
            return BLOCK_TILE_SIZE_Y + Y_SKEW;
    } else {
        if constexpr (IS_ROW)
            return BLOCK_TILE_SIZE_Y + Y_SKEW;
        else return BLOCK_TILE_SIZE_K + K_SKEW;
    }
}

template<
    typename T,
    size_t BLOCK_TILE_SIZE_X,
    size_t BLOCK_TILE_SIZE_Y,
    size_t BLOCK_TILE_SIZE_K,
    size_t THREADS_PER_BLOCK,
    size_t BLOCK_TILE_SKEW_X = 0,
    size_t BLOCK_TILE_SKEW_Y = 0,
    size_t BLOCK_TILE_SKEW_K = 0,
    bool TRANSPOSE_A = true,
    size_t LOAD_A_BYTES=4,
    size_t LOAD_B_BYTES=4
>
__device__ void load_data_to_shared_async(
    T A_shared[A_shared_dim<
        TRANSPOSE_A,
        BLOCK_TILE_SIZE_Y,
        BLOCK_TILE_SIZE_K,
        BLOCK_TILE_SKEW_Y,
        BLOCK_TILE_SKEW_K,
        true>()]
    [A_shared_dim<
        TRANSPOSE_A,
        BLOCK_TILE_SIZE_Y,
        BLOCK_TILE_SIZE_K,
        BLOCK_TILE_SKEW_Y,
        BLOCK_TILE_SKEW_K,
        false>()],
    T B_shared[BLOCK_TILE_SIZE_K][BLOCK_TILE_SIZE_X + BLOCK_TILE_SKEW_X],
    const T *A_matrix,
    const T *B_matrix,
    const size_t k,
    const size_t m,
    const size_t n,
    const size_t leading_dimension_A,
    const size_t leading_dimension_B,
    const uint iteration,
    const uint thread_linear_idx
) {
    if constexpr (TRANSPOSE_A) {
        load_data_to_shared_matrix_A_transposed_async<
            T, BLOCK_TILE_SIZE_Y, BLOCK_TILE_SIZE_K,
            THREADS_PER_BLOCK, BLOCK_TILE_SKEW_Y
        >(
            A_shared,
            A_matrix,
            k,
            m,
            leading_dimension_A,
            iteration,
            thread_linear_idx);
    } else {
        load_data_to_shared_matrix_A_async<
            T, BLOCK_TILE_SIZE_Y, BLOCK_TILE_SIZE_K,
            THREADS_PER_BLOCK, BLOCK_TILE_SKEW_Y,
            LOAD_A_BYTES
        >(
            A_shared,
            A_matrix,
            m,
            k,
            leading_dimension_A,
            iteration,
            thread_linear_idx
        );
    }
    CP_ASYNC_COMMIT_GROUP;
    load_data_to_shared_matrix_B_async<
        T, BLOCK_TILE_SIZE_X, BLOCK_TILE_SIZE_K,
        THREADS_PER_BLOCK, BLOCK_TILE_SKEW_X,
        LOAD_B_BYTES
    >(
        B_shared,
        B_matrix,
        k,
        n,
        leading_dimension_B,
        iteration,
        thread_linear_idx
    );
    CP_ASYNC_COMMIT_GROUP;
}

template<
    typename T,
    size_t BLOCK_TILE_SIZE_X,
    size_t BLOCK_TILE_SIZE_Y,
    size_t BLOCK_TILE_SIZE_K,
    size_t WARP_TILE_SIZE_X,
    size_t WARP_TILE_SIZE_Y,
    size_t THREAD_TILE_SIZE_X,
    size_t THREAD_TILE_SIZE_Y,
    size_t NUM_THREADS_PER_WARP_X,
    size_t NUM_THREADS_PER_WARP_Y,
    size_t STAGES = 2
>
__global__ void gemm_2DBT_2DWT_2DTT_async_load(
    const T *matrix_A,
    const T *matrix_B,
    T *matrix_C,
    const T alpha,
    const T beta,
    const size_t m,
    const size_t n,
    const size_t k,
    const size_t leading_dim_A,
    const size_t leading_dim_B,
    const size_t leading_dim_C) {
    // two buffer present compute loading overlap
    __shared__ T shared_A[STAGES][BLOCK_TILE_SIZE_Y][BLOCK_TILE_SIZE_K];
    __shared__ T shared_B[STAGES][BLOCK_TILE_SIZE_K][BLOCK_TILE_SIZE_X];

    // One Warp TILE will be of size WARP_TILE_SIZE_X x WARP_TILE_SIZE_Y
    // One Warp will be responsible for each Warp block, ideally multiple warp blocks
    // will be able to fit in one regular block allowing multiple warps to exist per
    // block

    // EACH block computes BLOCK_TILE_SIZE_X * BLOCK_TILE_SIZE_Y outputs of C

    // max threads per warp is 32, so we ensure that the warp block also complies
    // with this.
    static_assert(NUM_THREADS_PER_WARP_X * NUM_THREADS_PER_WARP_Y == 32);

    // We need to figure out a couple of things,
    // 1) we need to figure out how many WARP Tiles will be present in
    // the x and y direction similar as to calculating how many blocks will
    // be in the grid for a GPU launch we are doing the same but making a block
    // the grid and having our WARP TILE Be the new block
    //
    // 2) This is needed to calculate the total amount of THREADS per block in a
    // constant way
    constexpr size_t NUM_WARPS_PER_BLOCK_X{BLOCK_TILE_SIZE_X / WARP_TILE_SIZE_X};
    static_assert(BLOCK_TILE_SIZE_X % WARP_TILE_SIZE_X == 0);

    // repeat for y dimension
    constexpr size_t NUM_WARPS_PER_BLOCK_Y{BLOCK_TILE_SIZE_Y / WARP_TILE_SIZE_Y};
    static_assert(BLOCK_TILE_SIZE_Y % WARP_TILE_SIZE_Y == 0);

    // so total amount of warp tiles in a block would be
    // NUM_WARPS_PER_BLOCK_X * NUM_WARPS_PER_BLOCK_Y

    // In the previous implementation each thread had 2 register caches
    //
    // one cache cached several values in the y dimension from matrix one, total
    // elements are THREAD_TILE_SIZE_Y
    //
    // two cache cached several values in the x dimension from matrix two, total
    // elements are THREAD_TILE_SIZE_X
    //
    // In the end these values were reused for multiplication computing a total
    // of THREAD_TILE_SIZE_Y x THREAD_TILE_SIZE_X elements.
    //
    // Now that we are bounding warps to WARP_TILE sizes the amount of data being
    // computed by each thread may go up. So we know need to adjust the amount
    // of values being cached and computed to reflect this.
    //
    // We ideally want to keep our thread tile sizes consistent so we
    // instead add an extra dimension to each cache
    constexpr size_t NUM_CACHES_PER_WARP_X{
        WARP_TILE_SIZE_X / (THREAD_TILE_SIZE_X * NUM_THREADS_PER_WARP_X)
    };

    // repeat for y TILE cache
    constexpr size_t NUM_CACHES_PER_WARP_Y{
        WARP_TILE_SIZE_Y / (THREAD_TILE_SIZE_Y * NUM_THREADS_PER_WARP_Y)
    };

    static_assert(WARP_TILE_SIZE_X % (THREAD_TILE_SIZE_X * NUM_THREADS_PER_WARP_X) == 0);
    static_assert(WARP_TILE_SIZE_Y % (THREAD_TILE_SIZE_Y * NUM_THREADS_PER_WARP_Y) == 0);

    // Now we create the caches with the extra dimension
    T one_cache[NUM_CACHES_PER_WARP_Y][THREAD_TILE_SIZE_Y] = {static_cast<T>(0)};
    T two_cache[NUM_CACHES_PER_WARP_X][THREAD_TILE_SIZE_X] = {static_cast<T>(0)};

    // since we have more caches we will have more intermediates (values computed per thread)
    // as well, so we add extra dimensions here as well reflecting this
    T intermediates[NUM_CACHES_PER_WARP_Y][NUM_CACHES_PER_WARP_X][THREAD_TILE_SIZE_Y][
        THREAD_TILE_SIZE_X] = {static_cast<T>(0)};

    // now we can also easily calculate the total threads per block, needed for loading data
    constexpr size_t THREADS_PER_BLOCK{NUM_WARPS_PER_BLOCK_X * NUM_WARPS_PER_BLOCK_Y * 32};

    // this kernel should be launched with a 1d block so the linear dimension is just the threadidx.x
    const size_t thread_linear_idx{threadIdx.x};

    // the linear idx of the warp in the thread block
    const size_t warp_linear_idx{thread_linear_idx / 32};

    // Now lets figure out what warp that linear idx maps too (x, y)
    const size_t warp_row_idx{warp_linear_idx / NUM_WARPS_PER_BLOCK_X};
    const size_t warp_col_idx{warp_linear_idx % NUM_WARPS_PER_BLOCK_X};

    // figure out what row and column we are in the warp
    const size_t thread_linear_idx_in_warp{thread_linear_idx % warpSize};
    const size_t thread_idx_in_warp_row{thread_linear_idx_in_warp / NUM_THREADS_PER_WARP_X};
    const size_t thread_idx_in_warp_column{thread_linear_idx_in_warp % NUM_THREADS_PER_WARP_X};

    constexpr size_t units_per_vector{sizeof(int4) / sizeof(T)};

    // ensure int4 can be event split up by the base TYPE necessary for conversion
    static_assert(sizeof(int4) % sizeof(T) == 0);

    // we will store data along these dimensions for vectorized storage they need to be divisible
    static_assert(BLOCK_TILE_SIZE_K % units_per_vector == 0);
    static_assert(BLOCK_TILE_SIZE_X % units_per_vector == 0);

    static_assert(THREAD_TILE_SIZE_X % units_per_vector == 0);
    static_assert(THREAD_TILE_SIZE_Y % units_per_vector == 0);

    // This determines how many vectorized loads we need to perform to fill one tile
    constexpr size_t vectorized_thread_tile_size_x{THREAD_TILE_SIZE_X / units_per_vector};

    const size_t total_iters{ceil_div(k, BLOCK_TILE_SIZE_K)};

    // preload both buffers
    for (size_t stage{0}; stage < STAGES; ++stage) {
        load_data_to_shared_async<
            T, BLOCK_TILE_SIZE_X,
            BLOCK_TILE_SIZE_Y, BLOCK_TILE_SIZE_K,
            THREADS_PER_BLOCK, 0, 0, 0, false, 16, 16
        >(
            shared_A[stage],
            shared_B[stage],
            matrix_A,
            matrix_B,
            k,
            m,
            n,
            leading_dim_A,
            leading_dim_B,
            stage,
            thread_linear_idx
        );
    }

    size_t stage{0};

    for (size_t iter{0}; iter < total_iters; ++iter) {

        CP_ASYNC_WAIT(2);
        // print_matrix<BLOCK_TILE_SIZE_K, BLOCK_TILE_SIZE_X>(shared_B[stage]);
        __syncthreads();

        // #pragma unroll
        for (size_t kk{0}; kk < BLOCK_TILE_SIZE_K; ++kk) {
            // we need to start filling the one matrix cache
#pragma unroll
            for (size_t y_cache_idx{0}; y_cache_idx < NUM_CACHES_PER_WARP_Y; ++y_cache_idx) {
                // Here we calculate the row in the shared block based on the warp coordinates
                // and the thread coordinates

                // To calculate the row we first multiply the warp block y coordinate by the
                // Warp y dimension scale on the grid scale this is equivalent to doing blockIdx.y * blockDim.y
                // Next based on what cache we are in we need to skip that many rows. We do this by multiplying the
                // y_cache_idx by (WARP_TILE_SIZE_Y / NUM_CACHES_PER_WARP_Y) this value is equivalent too
                // (THREAD_TILE_SIZE_Y * NUM_THREADS_PER_WARP_Y) which is how many distinct rows are processed per
                // one cache fill of warp tile. Finally, we add the row that this thread is part of in the warp.

                // each thread loads TILE_SIZE_Y
                // assuming this configuration NUM_THREADS_PER_WARP_X = 4, and NUM_THREADS_PER_WARP_Y = 8
                // we can assume this load pattern
                // Threads [0 to 3] load rows [0 to 7], Threads [4 to 7] load rows [8 to 15] ...
                // Threads [28 to 31] load rows [54 to 63], this would result in a bank conflict for each
                // new warp_row and a broadcast for all threads in warp row, but luckily
                // the shared memory is transposed resulting in only broadcasts
                const size_t one_shared_row_idx{
                    warp_row_idx * WARP_TILE_SIZE_Y +
                    y_cache_idx * (WARP_TILE_SIZE_Y / NUM_CACHES_PER_WARP_Y) +
                    thread_idx_in_warp_row * THREAD_TILE_SIZE_Y
                };

                // load into register cache one[y_cache_idx] with vectorized loads
                // #pragma unroll
                for (size_t y_idx{0}; y_idx < THREAD_TILE_SIZE_Y; ++y_idx)
                    one_cache[y_cache_idx][y_idx] = shared_A[stage][one_shared_row_idx + y_idx][kk];
            }

#pragma unroll
            for (size_t x_cache_id{0}; x_cache_id < NUM_CACHES_PER_WARP_X; ++x_cache_id) {
                const size_t two_shared_col_idx{
                    warp_col_idx * WARP_TILE_SIZE_X +
                    x_cache_id * (WARP_TILE_SIZE_X / NUM_CACHES_PER_WARP_X) +
                    thread_idx_in_warp_column * THREAD_TILE_SIZE_X
                };

                const auto two_shared_ptr{
                    reinterpret_cast<int4 *>(&shared_B[stage][kk][two_shared_col_idx])
                };

                auto tile_ptr{
                    reinterpret_cast<int4 *>(&two_cache[x_cache_id])
                };

                // #pragma unroll
                for (size_t vx_iter{0}; vx_iter < vectorized_thread_tile_size_x; ++vx_iter)
                    tile_ptr[vx_iter] = two_shared_ptr[vx_iter];
            }

            // compute intermediates
#pragma unroll
            for (size_t y_cache_idx{0}; y_cache_idx < NUM_CACHES_PER_WARP_Y; ++y_cache_idx) {
                // #pragma unroll
                for (size_t x_cache_idx{0}; x_cache_idx < NUM_CACHES_PER_WARP_X; ++x_cache_idx) {
#pragma unroll
                    for (size_t one_cache_idx{0}; one_cache_idx < THREAD_TILE_SIZE_Y; ++one_cache_idx) {
                        T one_cache_value{one_cache[y_cache_idx][one_cache_idx]};
                        // #pragma unroll
                        for (size_t two_cache_index{0}; two_cache_index < THREAD_TILE_SIZE_X; ++two_cache_index) {
                            intermediates[y_cache_idx][x_cache_idx][one_cache_idx][two_cache_index] +=
                                    one_cache_value * two_cache[x_cache_idx][two_cache_index];
                        }
                    }
                }
            }
        }
        __syncthreads();

        if (iter < total_iters - 2) {
            load_data_to_shared_async<
                T, BLOCK_TILE_SIZE_X,
                BLOCK_TILE_SIZE_Y, BLOCK_TILE_SIZE_K,
                THREADS_PER_BLOCK, 0, 0, 0, false, 16, 16
            >(
                shared_A[stage],
                shared_B[stage],
                matrix_A,
                matrix_B,
                k,
                m,
                n,
                leading_dim_A,
                leading_dim_B,
                iter + 2,
                thread_linear_idx
            );
        }

        stage = (stage + 1) % STAGES;
    }

    // vectorized store back into the dest matrix
#pragma unroll
    for (size_t y_cache_idx{0}; y_cache_idx < NUM_CACHES_PER_WARP_Y; ++y_cache_idx) {
#pragma unroll
        for (size_t x_cache_idx{0}; x_cache_idx < NUM_CACHES_PER_WARP_X; ++x_cache_idx) {
            // #pragma unroll
            for (size_t one_cache_idx{0}; one_cache_idx < THREAD_TILE_SIZE_Y; ++one_cache_idx) {
                const size_t dest_row{
                    BLOCK_TILE_SIZE_Y * blockIdx.y +
                    warp_row_idx * WARP_TILE_SIZE_Y +
                    y_cache_idx * (WARP_TILE_SIZE_Y / NUM_CACHES_PER_WARP_Y) +
                    thread_idx_in_warp_row * THREAD_TILE_SIZE_Y + one_cache_idx
                };

                const size_t dest_column{
                    BLOCK_TILE_SIZE_X * blockIdx.x +
                    warp_col_idx * WARP_TILE_SIZE_X +
                    x_cache_idx * (WARP_TILE_SIZE_X / NUM_CACHES_PER_WARP_X) +
                    thread_idx_in_warp_column * THREAD_TILE_SIZE_X
                };

                auto dest_ptr{&matrix_C[dest_row * leading_dim_C + dest_column]};
                T *tile_ptr{&intermediates[y_cache_idx][x_cache_idx][one_cache_idx][0]};

                // #pragma unroll
                for (size_t two_cache_vec_idx{0}; two_cache_vec_idx < vectorized_thread_tile_size_x; ++
                     two_cache_vec_idx) {
                    if (dest_row < m && (
                            dest_column + two_cache_vec_idx * units_per_vector < n)) {
                        // #pragma unroll
                        for (size_t tile_idx{0}; tile_idx < units_per_vector; ++tile_idx) {
                            tile_ptr[tile_idx] = tile_ptr[tile_idx] * alpha + dest_ptr[tile_idx] * beta;
                        }

                        reinterpret_cast<int4 *>(dest_ptr)[two_cache_vec_idx] =
                                reinterpret_cast<int4 *>(tile_ptr)[two_cache_vec_idx];
                    }
                }
            }
        }
    }
}


#endif //GEMM_CUH
