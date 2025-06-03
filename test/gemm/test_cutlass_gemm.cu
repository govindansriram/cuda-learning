//
// Created by sriram on 5/27/25.
//

#include "../../src/gemm/cutlass_gemm.cuh"
#include "../test_helpers.h"
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>


// Regular implementation
template<
    typename A_GLOBAL_LAYOUT,
    typename A_GLOBAL_ENGINE,
    typename A_SHARED_LAYOUT,
    typename A_SHARED_ENGINE,
    typename B_GLOBAL_LAYOUT,
    typename B_GLOBAL_ENGINE,
    typename B_SHARED_LAYOUT,
    typename B_SHARED_ENGINE,
    typename THREAD_LAYOUT
>
CUTE_DEVICE void load_to_shared(
    const cute::Tensor<A_SHARED_ENGINE, A_SHARED_LAYOUT> &shared_A,
    const cute::Tensor<B_SHARED_ENGINE, B_SHARED_LAYOUT> &shared_B,
    const cute::Tensor<A_GLOBAL_ENGINE, A_GLOBAL_LAYOUT> &global_A,
    const cute::Tensor<B_GLOBAL_ENGINE, B_GLOBAL_LAYOUT> &global_B,
    const THREAD_LAYOUT &thread_layout
) {
    using namespace cute;

    constexpr size_t smem_length_A{cosize_v<A_SHARED_LAYOUT>};
    constexpr size_t smem_length_B{cosize_v<B_SHARED_LAYOUT>};
    constexpr size_t thread_length{cosize_v<THREAD_LAYOUT>};

    static_assert(smem_length_A % thread_length == 0);
    static_assert(smem_length_B % thread_length == 0);

    static_assert(size<0>(A_GLOBAL_LAYOUT{}) == size<0>(A_SHARED_LAYOUT{}));
    static_assert(size<1>(A_GLOBAL_LAYOUT{}) == size<1>(A_SHARED_LAYOUT{}));
    static_assert(size<0>(B_GLOBAL_LAYOUT{}) == size<0>(B_SHARED_LAYOUT{}));
    static_assert(size<1>(B_GLOBAL_LAYOUT{}) == size<1>(B_SHARED_LAYOUT{}));

    constexpr size_t A_loads_per_thread{smem_length_A / thread_length};
    constexpr size_t B_loads_per_thread{smem_length_B / thread_length};

    constexpr auto tv_layout_A{
        make_layout(
            make_shape(make_shape(size<1>(A_GLOBAL_LAYOUT{}),
                                  size<0>(A_GLOBAL_LAYOUT{}) / A_loads_per_thread), A_loads_per_thread),
            make_stride(make_stride(size<0>(A_GLOBAL_LAYOUT{}), A_loads_per_thread), _1{}))
    };

    constexpr auto tv_layout_B{
        make_layout(
            make_shape(
                make_shape(size<1>(B_GLOBAL_LAYOUT{}), size<0>(B_GLOBAL_LAYOUT{}) / B_loads_per_thread),
                B_loads_per_thread),
            make_stride(make_stride(size<0>(B_GLOBAL_LAYOUT{}), B_loads_per_thread), _1{}))
    };

    Tensor shared_A_tv{composition(shared_A, tv_layout_A)};
    Tensor shared_B_tv{composition(shared_B, tv_layout_B)};
    const Tensor global_A_tv{composition(global_A, tv_layout_A)};
    const Tensor global_B_tv{composition(global_B, tv_layout_B)};

    const Tensor global_A_value{global_A_tv(threadIdx.x, _)};
    Tensor shared_A_value{shared_A_tv(threadIdx.x, _)};

    const Tensor global_B_value{global_B_tv(threadIdx.x, _)};
    Tensor shared_B_value{shared_B_tv(threadIdx.x, _)};

    copy(global_A_value, shared_A_value);
    copy(global_B_value, shared_B_value);
}


template<
    typename A_GLOBAL_LAYOUT,
    typename A_GLOBAL_ENGINE,
    typename A_SHARED_T_LAYOUT,
    typename A_SHARED_T_ENGINE,
    typename B_GLOBAL_LAYOUT,
    typename B_GLOBAL_ENGINE,
    typename B_SHARED_LAYOUT,
    typename B_SHARED_ENGINE,
    typename THREAD_LAYOUT
>
CUTE_DEVICE void load_to_shared_transposed(
    const cute::Tensor<A_SHARED_T_ENGINE, A_SHARED_T_LAYOUT> &shared_A_transposed,
    const cute::Tensor<B_SHARED_ENGINE, B_SHARED_LAYOUT> &shared_B,
    const cute::Tensor<A_GLOBAL_ENGINE, A_GLOBAL_LAYOUT> &global_A,
    const cute::Tensor<B_GLOBAL_ENGINE, B_GLOBAL_LAYOUT> &global_B,
    const THREAD_LAYOUT &thread_layout
) {
    using namespace cute;

    constexpr size_t smem_length_A{cosize_v<A_SHARED_T_LAYOUT>};
    constexpr size_t smem_length_B{cosize_v<B_SHARED_LAYOUT>};
    constexpr size_t thread_length{cosize_v<THREAD_LAYOUT>};

    static_assert(smem_length_A % thread_length == 0);
    static_assert(smem_length_B % thread_length == 0);
    static_assert(size<0>(A_GLOBAL_LAYOUT{}) == size<1>(A_SHARED_T_LAYOUT{}));
    static_assert(size<1>(A_GLOBAL_LAYOUT{}) == size<0>(A_SHARED_T_LAYOUT{}));
    static_assert(size<0>(B_GLOBAL_LAYOUT{}) == size<0>(B_SHARED_LAYOUT{}));
    static_assert(size<1>(B_GLOBAL_LAYOUT{}) == size<1>(B_SHARED_LAYOUT{}));

    constexpr size_t A_loads_per_thread{smem_length_A / thread_length};
    constexpr size_t B_loads_per_thread{smem_length_B / thread_length};

    constexpr auto tv_layout_A{
        make_layout(
            make_shape(make_shape(size<1>(A_GLOBAL_LAYOUT{}),
                                  size<0>(A_GLOBAL_LAYOUT{}) / A_loads_per_thread), A_loads_per_thread),
            make_stride(make_stride(size<0>(A_GLOBAL_LAYOUT{}), A_loads_per_thread), _1{}))
    };

    constexpr auto tv_layout_B{
        make_layout(
            make_shape(
                make_shape(size<1>(B_GLOBAL_LAYOUT{}), size<0>(B_GLOBAL_LAYOUT{}) / B_loads_per_thread),
                B_loads_per_thread),
            make_stride(make_stride(size<0>(B_GLOBAL_LAYOUT{}), B_loads_per_thread), _1{}))
    };

    Tensor shared_A_tv{composition(shared_A_transposed, tv_layout_A)};
    Tensor shared_B_tv{composition(shared_B, tv_layout_B)};
    const Tensor global_A_tv{composition(global_A, tv_layout_A)};
    const Tensor global_B_tv{composition(global_B, tv_layout_B)};

    const Tensor global_A_value{global_A_tv(threadIdx.x, _)};
    Tensor shared_A_value{shared_A_tv(threadIdx.x, _)};

    const Tensor global_B_value{global_B_tv(threadIdx.x, _)};
    Tensor shared_B_value{shared_B_tv(threadIdx.x, _)};

    copy(global_A_value, shared_A_value);
    copy(global_B_value, shared_B_value);
}

template<
    typename T,
    typename A_GLOBAL_LAYOUT,
    typename A_SHARED_LAYOUT,
    typename B_GLOBAL_LAYOUT,
    typename B_SHARED_LAYOUT,
    typename C_GLOBAL_LAYOUT,
    typename THREAD_LAYOUT
>
__global__ static void gemm_2DBT(
    const T *gmem_A,
    const T *gmem_B,
    T *gmem_C,
    const A_GLOBAL_LAYOUT gmem_layout_A,
    const B_GLOBAL_LAYOUT gmem_layout_B,
    const C_GLOBAL_LAYOUT gmem_layout_C,
    const A_SHARED_LAYOUT smem_layout_A,
    const B_SHARED_LAYOUT smem_layout_B,
    const THREAD_LAYOUT thread_layout,
    const T alpha,
    const T beta
) {
    using namespace cute;
    static_assert(gmem_layout_A.rank == 2);
    static_assert(gmem_layout_B.rank == 2);
    static_assert(gmem_layout_C.rank == 2);
    static_assert(smem_layout_A.rank == 2);
    static_assert(smem_layout_B.rank == 2);

    constexpr size_t BLOCK_TILE_SIZE_K{size<1>(smem_layout_A)};
    static_assert(BLOCK_TILE_SIZE_K == size<0>(smem_layout_B));

    extern __shared__ T shared_memory[];
    constexpr size_t smem_length_A{cosize_v<A_SHARED_LAYOUT>};

    // helps with deciding what copy algorithm to use
    smem_ptr pShared_A{make_smem_ptr(shared_memory)};
    smem_ptr pShared_B{make_smem_ptr(&shared_memory[smem_length_A])};
    gmem_ptr pGlobal_A{make_gmem_ptr(gmem_A)};
    gmem_ptr pGlobal_B{make_gmem_ptr(gmem_B)};
    gmem_ptr pGlobal_C{make_gmem_ptr(gmem_C)};

    Tensor shared_A{make_tensor(pShared_A, smem_layout_A)};
    Tensor shared_B{make_tensor(pShared_B, smem_layout_B)};
    Tensor global_A{make_tensor(pGlobal_A, gmem_layout_A)};
    Tensor global_B{make_tensor(pGlobal_B, gmem_layout_B)};
    Tensor global_C{make_tensor(pGlobal_C, gmem_layout_C)};

    const size_t total_iters{ceil_div(size<1>(gmem_layout_A), BLOCK_TILE_SIZE_K)};

    Tensor gA_tiled{zipped_divide(global_A, smem_layout_A.shape())};
    Tensor gB_tiled{zipped_divide(global_B, smem_layout_B.shape())};
    Tensor gC_tiled{zipped_divide(global_C, thread_layout.shape())};

    Tensor tile_C{gC_tiled(make_coord(_, _), make_coord(blockIdx.y, blockIdx.x))};

    T partial{0};

    auto coords{idx2crd(threadIdx.x, thread_layout.shape(), thread_layout.stride())};
    const size_t row{coords.first_};
    const size_t col{coords.rest_.first_};

    for (size_t iter{0}; iter < total_iters; ++iter) {
        Tensor tile_A{gA_tiled(make_coord(_, _), make_coord(blockIdx.y, iter))};
        Tensor tile_B{gB_tiled(make_coord(_, _), make_coord(iter, blockIdx.x))};

        // load to shared
        load_to_shared(shared_A, shared_B, tile_A, tile_B, thread_layout);
        __syncthreads();

        Tensor slice_A{shared_A(make_coord(row, _))};
        Tensor slice_B{shared_B(make_coord(_, col))};

#pragma unroll
        for (size_t kk{0}; kk < BLOCK_TILE_SIZE_K; ++kk) {
            partial += slice_A(kk) * slice_B(kk);
        }
        __syncthreads();
    }

    tile_C(make_coord(row, col)) = tile_C(make_coord(row, col)) * beta + partial * alpha;
}

void test_cute_gemm_2DBT() {
    using namespace cute;

    constexpr size_t M{128};
    constexpr size_t N{64};
    constexpr size_t K{256};

    constexpr size_t BLOCK_TILE_SIZE_Y{16};
    constexpr size_t BLOCK_TILE_SIZE_X{16};
    constexpr size_t BLOCK_TILE_SIZE_K{16};

    static_assert((M * K) % (BLOCK_TILE_SIZE_Y * BLOCK_TILE_SIZE_K) == 0);
    static_assert((N * K) % (BLOCK_TILE_SIZE_K * BLOCK_TILE_SIZE_X) == 0);

    thrust::host_vector<float> host_matrixA(M * K);
    thrust::host_vector<float> host_matrixB(K * N);
    thrust::host_vector<float> host_matrixC(M * N);

    fill_matrix_w(host_matrixA.data(), M, K, K, -100, 100);
    fill_matrix_w(host_matrixB.data(), K, N, N, -100, 100);

    for (size_t i{0}; i < N * M; ++i) host_matrixC[i] = 0.f;

    thrust::device_vector<float> device_matrixA{host_matrixA};
    thrust::device_vector<float> device_matrixB{host_matrixB};
    thrust::device_vector<float> device_matrixC{host_matrixC};

    const Layout gmem_A_lo{make_layout(make_shape(M, K), LayoutRight{})};
    const Layout gmem_B_lo{make_layout(make_shape(K, N), LayoutRight{})};
    const Layout gmem_C_lo{make_layout(make_shape(M, N), LayoutRight{})};

    // print2D_tensor(make_tensor(host_matrixA.data(), gmem_A_lo));

    constexpr Layout smem_A_lo{
        make_layout(make_shape(Int<BLOCK_TILE_SIZE_Y>{}, Int<BLOCK_TILE_SIZE_K>{}), LayoutRight{})
    };
    constexpr Layout smem_B_lo{
        make_layout(make_shape(Int<BLOCK_TILE_SIZE_K>{}, Int<BLOCK_TILE_SIZE_X>{}), LayoutRight{})
    };
    constexpr Layout thread_lo{
        make_layout(make_shape(Int<16>{}, Int<16>{}), LayoutRight{})
    };

    constexpr size_t shared_mem_size{
        (BLOCK_TILE_SIZE_Y * BLOCK_TILE_SIZE_K) + (BLOCK_TILE_SIZE_K * BLOCK_TILE_SIZE_X) * sizeof(float)
    };

    dim3 grid_dim{
        ceil_div(N, BLOCK_TILE_SIZE_X),
        ceil_div(M, BLOCK_TILE_SIZE_Y)
    };

    dim3 block_dim{
        BLOCK_TILE_SIZE_X * BLOCK_TILE_SIZE_Y
    };

    gemm_2DBT<<<grid_dim, block_dim, shared_mem_size>>>(
        device_matrixA.data().get(),
        device_matrixB.data().get(),
        device_matrixC.data().get(),
        gmem_A_lo,
        gmem_B_lo,
        gmem_C_lo,
        smem_A_lo,
        smem_B_lo,
        thread_lo,
        1.f,
        1.f);

    thrust::host_vector<float> host_matrixC2{device_matrixC};
    cpu_matmul_naive(host_matrixA.data(), host_matrixB.data(), host_matrixC.data(), M, N, K, K, N, N);
    test_equivalency(host_matrixC.data(), host_matrixC2.data(), M, N, N);
}

template<typename TensorLayout, typename TensorEngine>
void print_ten(cute::Tensor<TensorEngine, TensorLayout> tens) {
    // cute::print_layout(tens.layout());

    for (size_t m{0}; m < cute::size<0>(tens); ++m) {
        std::cout << "[ ";
        for (size_t n{0}; n < cute::size<1>(tens); ++n) {
            std::cout << tens(cute::make_coord(m, n)) << " ";
        }
        std::cout << "]\n";
    }
}

template<
    typename T,
    typename A_GLOBAL_LAYOUT,
    typename B_GLOBAL_LAYOUT,
    typename C_GLOBAL_LAYOUT,
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
__global__ static void gemm_2DBT_2DWT_2DTT_vloadT(
    const T *gmem_A,
    const T *gmem_B,
    T *gmem_C,
    const A_GLOBAL_LAYOUT gmem_layout_A,
    const B_GLOBAL_LAYOUT gmem_layout_B,
    const C_GLOBAL_LAYOUT gmem_layout_C,
    const T alpha,
    const T beta
) {
    using namespace cute;

    constexpr size_t NUM_WARPS_X{BLOCK_TILE_SIZE_X / WARP_TILE_SIZE_X};
    constexpr size_t NUM_WARPS_Y{BLOCK_TILE_SIZE_Y / WARP_TILE_SIZE_Y};

    constexpr size_t NUM_CACHES_PER_WARP_X{WARP_TILE_SIZE_X / (THREAD_TILE_SIZE_X * NUM_THREADS_PER_WARP_X)};
    constexpr size_t NUM_CACHES_PER_WARP_Y{WARP_TILE_SIZE_Y / (THREAD_TILE_SIZE_Y * NUM_THREADS_PER_WARP_Y)};

    static_assert(gmem_layout_A.rank == 2);
    static_assert(gmem_layout_B.rank == 2);
    static_assert(gmem_layout_C.rank == 2);

    static_assert(size<1>(A_GLOBAL_LAYOUT{}) == size<0>(B_GLOBAL_LAYOUT{}));
    // TODO check c as well

    // Divides each thread into a warp block where further values can then be derivec
    constexpr Layout thread_layout{
        make_layout(
            make_shape(
                make_shape(Int<NUM_THREADS_PER_WARP_Y>{}, Int<NUM_WARPS_Y>{}),
                make_shape(Int<NUM_THREADS_PER_WARP_X>{}, Int<NUM_WARPS_X>{})),
            make_stride(
                make_stride(Int<NUM_THREADS_PER_WARP_X>{}, Int<NUM_WARPS_X * 32>{}),
                make_stride(_1{}, _32{})
            )
        )
    };

    constexpr Layout smem_A_T_layout{
        make_layout(make_shape(Int<BLOCK_TILE_SIZE_K>{}, Int<BLOCK_TILE_SIZE_Y>{}), LayoutRight{})
    };
    constexpr Layout smem_B_layout{
        make_layout(make_shape(Int<BLOCK_TILE_SIZE_K>{}, Int<BLOCK_TILE_SIZE_X>{}), LayoutRight{})
    };

    constexpr auto warp_tile{
        make_shape(Int<WARP_TILE_SIZE_Y>{}, Int<WARP_TILE_SIZE_X>{})
    };

    constexpr auto A_block_tiler{
        make_shape(Int<BLOCK_TILE_SIZE_Y>{}, Int<BLOCK_TILE_SIZE_K>{})
    };

    constexpr auto C_block_tiler{
        make_shape(Int<BLOCK_TILE_SIZE_Y>{}, Int<BLOCK_TILE_SIZE_X>{})
    };

    constexpr auto warp_slice_tile_A{
        make_layout(
            make_shape(
                make_shape(Int<NUM_THREADS_PER_WARP_Y>{}, Int<NUM_CACHES_PER_WARP_Y>{}),
                make_shape(Int<THREAD_TILE_SIZE_Y>{})
            ),
            make_stride(
                make_stride(Int<THREAD_TILE_SIZE_Y>{}, Int<THREAD_TILE_SIZE_Y * NUM_THREADS_PER_WARP_Y>{}),
                make_stride(_1{})
            )
        )
    };

    constexpr auto warp_slice_tile_B{
        make_layout(
            make_shape(
                make_shape(Int<NUM_THREADS_PER_WARP_X>{}, Int<NUM_CACHES_PER_WARP_X>{}),
                Int<THREAD_TILE_SIZE_X>{}
            ),
            make_stride(
                make_stride(Int<THREAD_TILE_SIZE_X>{}, Int<THREAD_TILE_SIZE_X * NUM_THREADS_PER_WARP_X>{}),
                _1{}
            )
        )
    };

    // This layout maps an entire warp tile to all 32 threads it assigns the threads
    // THREAD_TILE_SIZE_Y * THREAD_TILE_SIZE_X, matrices since that is what each thread is supposed to compute
    // however in the scenario where threads are responsible to compute multiple tiles (NUM_CACHES_PER_WARP) is greater
    // than 1 then we also group those matrices near each other for easy indexing
    //
    // the layout is organized as a high dimensional matrix. The first dimension is THREAD_TILE_SIZE_Y, as this is the
    // row length of the matrices, this gets repeated NUM_CACHES_PER_WARP_X * NUM_CACHES_PER_WARP_Y times since that
    // rows of matrices are being computed, the next coord pair correlate the cluster of matrices to the appropriate
    // warp idx pair
    // Finally the last idx specifies the length of each row which is THREAD_TILE_SIZE_X
    // To access the first cache matrix for warp index 0,2 this is the command
    // tv(make_coord(make_coord(_, make_coord(make_coord(0, 0), make_coord(0, 2))), _))
    // TODO: Add visual description
    constexpr Layout warp_tv_layout{
        make_layout(
            make_shape(
                make_shape(
                    Int<THREAD_TILE_SIZE_Y>{},
                    make_shape(
                        make_shape(Int<NUM_CACHES_PER_WARP_Y>{}, Int<NUM_CACHES_PER_WARP_X>{}),
                        make_shape(Int<NUM_THREADS_PER_WARP_Y>{}, Int<NUM_THREADS_PER_WARP_X>{})
                    )
                ),
                Int<THREAD_TILE_SIZE_X>{}
            ),
            make_stride(
                make_stride(
                    _1{},
                    make_stride(
                        make_stride(
                            Int<THREAD_TILE_SIZE_Y * NUM_THREADS_PER_WARP_Y>{},
                            Int<THREAD_TILE_SIZE_X * NUM_THREADS_PER_WARP_X * WARP_TILE_SIZE_Y>{}
                        ),
                        make_stride(Int<THREAD_TILE_SIZE_Y>{}, Int<THREAD_TILE_SIZE_X * WARP_TILE_SIZE_Y>{}))),
                Int<WARP_TILE_SIZE_Y>{}
            )
        )
    };

    extern __shared__ T shared_memory[];
    constexpr size_t smem_length_A{cosize_v<decltype(smem_A_T_layout)>};

    // helps with deciding what copy algorithm to use
    smem_ptr pShared_A{make_smem_ptr(shared_memory)};
    smem_ptr pShared_B{make_smem_ptr(&shared_memory[smem_length_A])};
    gmem_ptr pGlobal_A{make_gmem_ptr(gmem_A)};
    gmem_ptr pGlobal_B{make_gmem_ptr(gmem_B)};
    gmem_ptr pGlobal_C{make_gmem_ptr(gmem_C)};

    Tensor shared_A{make_tensor(pShared_A, smem_A_T_layout)};
    Tensor shared_B{make_tensor(pShared_B, smem_B_layout)};
    Tensor global_A{make_tensor(pGlobal_A, gmem_layout_A)};
    Tensor global_B{make_tensor(pGlobal_B, gmem_layout_B)};
    Tensor global_C{make_tensor(pGlobal_C, gmem_layout_C)};

    const size_t total_iters{ceil_div(size<1>(gmem_layout_A), BLOCK_TILE_SIZE_K)};

    const auto warp_coord{idx2crd(threadIdx.x, thread_layout.shape(), thread_layout.stride())};
    const size_t row_in_warp{warp_coord.first_.first_}; // index of the row in the warp_tile
    const size_t warp_y_idx{warp_coord.first_.rest_.first_}; // index of the warp_tile in the block
    const size_t col_in_warp{warp_coord.rest_.first_.first_}; // index of the col in the warp_tile
    const size_t warp_x_idx{warp_coord.rest_.first_.rest_.first_}; // index of the warp_tile in the block


     Tensor block_tile_C{local_tile(global_C, C_block_tiler, make_coord(blockIdx.y, blockIdx.x))};
     Tensor warp_tile_C{local_tile(block_tile_C, warp_tile, make_coord(warp_y_idx, warp_x_idx))};
     Tensor tv_warp_tile_C{composition(warp_tile_C, warp_tv_layout)};
     Tensor C_value{
         tv_warp_tile_C(make_coord(make_coord(_, make_coord(make_coord(_, _), make_coord(row_in_warp, col_in_warp))), _))
     };

     Tensor rmem_A_cache{
         make_tensor<T>(
             make_shape(Int<NUM_CACHES_PER_WARP_Y>(), Int<THREAD_TILE_SIZE_Y>()),
             LayoutRight{}
         )
     };

     Tensor rmem_B_cache{
         make_tensor<T>(
             make_shape(Int<NUM_CACHES_PER_WARP_X>(), Int<THREAD_TILE_SIZE_X>()),
             LayoutRight{}
         )
     };

     Tensor rmem_cache_intermediates{
         make_tensor<T>(
             make_shape(
                 make_shape(
                     Int<THREAD_TILE_SIZE_Y>{},
                     make_shape(Int<NUM_CACHES_PER_WARP_Y>{}, Int<NUM_CACHES_PER_WARP_X>{})
                 ),
                 Int<THREAD_TILE_SIZE_X>{}
             ),
             make_stride(
                 make_stride(
                     Int<THREAD_TILE_SIZE_X * NUM_CACHES_PER_WARP_X>{},
                     make_stride(Int<THREAD_TILE_SIZE_X * NUM_CACHES_PER_WARP_X * THREAD_TILE_SIZE_Y>{},
                                 Int<THREAD_TILE_SIZE_X>{})
                 ),
                 _1{}
             )
         )
     };

     for (size_t iter{0}; iter < total_iters; ++iter) {
         // same as zipped divide w/ less code
         Tensor tile_A{local_tile(global_A, A_block_tiler, make_coord(blockIdx.y, iter))};
         Tensor tile_B{local_tile(global_B, smem_B_layout.shape(), make_coord(iter, blockIdx.x))};

         // load to shared
         // TODO change
         // load_to_shared(shared_A, shared_B, tile_A, tile_B, thread_layout);
         __syncthreads();

         Tensor warp_tile_A{
             local_tile(shared_A, make_shape(Int<BLOCK_TILE_SIZE_K>{}, Int<WARP_TILE_SIZE_Y>{}),
                        make_coord(0, warp_y_idx))
         };
         Tensor warp_tile_B{
             local_tile(shared_B, make_shape(Int<BLOCK_TILE_SIZE_K>{}, Int<WARP_TILE_SIZE_X>{}),
                        make_coord(0, warp_x_idx))
         };

#pragma unroll
        for (size_t kk{0}; kk < BLOCK_TILE_SIZE_K; ++kk) {
            Tensor slice_warp_tile_A{warp_tile_A(make_coord(kk, _))};
            Tensor slice_warp_tile_B{warp_tile_B(make_coord(kk, _))};

            Tensor slice_warp_A_tv{composition(slice_warp_tile_A, warp_slice_tile_A)};
            Tensor slice_warp_B_tv{composition(slice_warp_tile_B, warp_slice_tile_B)};

            copy(rmem_A_cache, slice_warp_A_tv(make_coord(make_coord(row_in_warp, _), _)));
            copy(rmem_B_cache, slice_warp_B_tv(make_coord(make_coord(col_in_warp, _), _)));

            for (size_t rmem_cache_idxA{0}; rmem_cache_idxA < NUM_CACHES_PER_WARP_Y; ++rmem_cache_idxA) {
                for (size_t rmem_cache_idxB{0}; rmem_cache_idxB < NUM_CACHES_PER_WARP_X; ++rmem_cache_idxB) {
                    Tensor A_cache_slice{rmem_A_cache(make_coord(_, rmem_cache_idxA))};
                    Tensor B_cache_slice{rmem_B_cache(make_coord(_, rmem_cache_idxB))};
                    Tensor partials{
                        rmem_cache_intermediates(make_coord(make_coord(_, make_coord(rmem_cache_idxA, rmem_cache_idxB)), _))
                    };

                    for (size_t i{0}; i < THREAD_TILE_SIZE_Y; ++i) {
                        T acs{A_cache_slice(i)};
                        for (size_t j{0}; j < THREAD_TILE_SIZE_X; ++j) {
                            partials(make_coord(i, j)) = acs * B_cache_slice(j);
                        }
                    }
                }
            }
        }
         __syncthreads();
     }

     axpby(alpha, rmem_cache_intermediates, beta, C_value);
}


void test_cute_gemm_2DBT_2DWT_2DTT_vloadT() {
    using namespace cute;

    constexpr size_t M{128};
    constexpr size_t N{128};
    constexpr size_t K{256};

    constexpr size_t BLOCK_TILE_SIZE_X{128};
    constexpr size_t BLOCK_TILE_SIZE_Y{128};
    constexpr size_t BLOCK_TILE_SIZE_K{16};

    constexpr size_t WARP_TILE_SIZE_X{32};
    constexpr size_t WARP_TILE_SIZE_Y{64};

    static_assert((M * K) % (BLOCK_TILE_SIZE_Y * BLOCK_TILE_SIZE_K) == 0);
    static_assert((N * K) % (BLOCK_TILE_SIZE_K * BLOCK_TILE_SIZE_X) == 0);

    constexpr size_t NUM_WARPS_X{BLOCK_TILE_SIZE_X / WARP_TILE_SIZE_X};
    constexpr size_t NUM_WARPS_Y{BLOCK_TILE_SIZE_Y / WARP_TILE_SIZE_Y};

    static_assert(BLOCK_TILE_SIZE_X % WARP_TILE_SIZE_X == 0);
    static_assert(BLOCK_TILE_SIZE_Y % WARP_TILE_SIZE_Y == 0);

    constexpr size_t THREAD_TILE_SIZE_X{8};
    constexpr size_t THREAD_TILE_SIZE_Y{8};

    constexpr size_t NUM_THREADS_PER_WARP_X{4};
    constexpr size_t NUM_THREADS_PER_WARP_Y{8};

    static_assert(NUM_THREADS_PER_WARP_X * NUM_THREADS_PER_WARP_Y == 32);

    static_assert(WARP_TILE_SIZE_X % (THREAD_TILE_SIZE_X * NUM_THREADS_PER_WARP_X) == 0);
    static_assert(WARP_TILE_SIZE_Y % (THREAD_TILE_SIZE_Y * NUM_THREADS_PER_WARP_Y) == 0);

    constexpr size_t THREADS_PER_BLOCK{32 * NUM_WARPS_X * NUM_WARPS_Y};

    thrust::host_vector<float> host_matrixA(M * K);
    thrust::host_vector<float> host_matrixB(K * N);
    thrust::host_vector<float> host_matrixC(M * N);

    thrust::device_vector<float> d_matrixA{host_matrixA};
    thrust::device_vector<float> d_matrixB{host_matrixB};
    thrust::device_vector<float> d_matrixC{host_matrixC};

    fill_matrix_w(host_matrixA.data(), M, K, K, -100, 100);
    fill_matrix_w(host_matrixB.data(), K, N, N, -100, 100);
    for (size_t i{0}; i < N * M; ++i) host_matrixC[i] = 0.f;

    dim3 grid_dim{
        ceil_div(N, BLOCK_TILE_SIZE_X),
        ceil_div(M, BLOCK_TILE_SIZE_Y)
    };

    gemm_2DBT_2DWT_2DTT_vloadT<
        float,
        Layout<Shape<Int<M>, Int<K>>, Stride<Int<K>, _1>>,
        Layout<Shape<Int<K>, Int<N>>, Stride<Int<N>, _1>>,
        Layout<Shape<Int<M>, Int<N>>, Stride<Int<N>, _1>>,
        BLOCK_TILE_SIZE_X,
        BLOCK_TILE_SIZE_Y,
        BLOCK_TILE_SIZE_K,
        WARP_TILE_SIZE_X,
        WARP_TILE_SIZE_Y,
        THREAD_TILE_SIZE_X,
        THREAD_TILE_SIZE_Y,
        NUM_THREADS_PER_WARP_X,
        NUM_THREADS_PER_WARP_Y
        ><<<grid_dim, THREADS_PER_BLOCK>>>(
            d_matrixA.data().get(),
            d_matrixB.data().get(),
            d_matrixC.data().get(),
            make_layout(make_shape(Int<M>{}, Int<K>{}), LayoutRight{}),
            make_layout(make_shape(Int<K>{}, Int<N>{}), LayoutRight{}),
            make_layout(make_shape(Int<M>{}, Int<N>{}), LayoutRight{}),
            1.f,
            1.f
        );
}
