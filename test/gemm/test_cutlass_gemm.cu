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

template<
    typename T,
    typename A_GLOBAL_LAYOUT,
    typename A_SHARED_LAYOUT,
    typename B_GLOBAL_LAYOUT,
    typename B_SHARED_LAYOUT,
    typename C_GLOBAL_LAYOUT,
    typename THREAD_LAYOUT,
    typename WARP_LAYOUT
>
__global__ static void gemm_2DBT_2DWT_2DTT_vloadT(
    const T *gmem_A,
    const T *gmem_B,
    T *gmem_C,
    const A_GLOBAL_LAYOUT gmem_layout_A,
    const B_GLOBAL_LAYOUT gmem_layout_B,
    const C_GLOBAL_LAYOUT gmem_layout_C,
    const A_SHARED_LAYOUT smem_layout_A_T,
    const B_SHARED_LAYOUT smem_layout_B,
    const THREAD_LAYOUT thread_layout,
    const WARP_LAYOUT warp_layout,
    const T alpha,
    const T beta
) {
    using namespace cute;
    static_assert(gmem_layout_A.rank == 2);
    static_assert(gmem_layout_B.rank == 2);
    static_assert(gmem_layout_C.rank == 2);
    static_assert(smem_layout_A_T.rank == 2);
    static_assert(smem_layout_B.rank == 2);

    constexpr size_t BLOCK_TILE_SIZE_K{size<0>(smem_layout_A_T)};
    static_assert(BLOCK_TILE_SIZE_K == size<0>(smem_layout_B));

    extern __shared__ T shared_memory[];
    constexpr size_t smem_length_A{cosize_v<A_SHARED_LAYOUT>};

    // helps with deciding what copy algorithm to use
    smem_ptr pShared_A{make_smem_ptr(shared_memory)};
    smem_ptr pShared_B{make_smem_ptr(&shared_memory[smem_length_A])};
    gmem_ptr pGlobal_A{make_gmem_ptr(gmem_A)};
    gmem_ptr pGlobal_B{make_gmem_ptr(gmem_B)};
    gmem_ptr pGlobal_C{make_gmem_ptr(gmem_C)};

    Tensor shared_A{make_tensor(pShared_A, smem_layout_A_T)};
    Tensor shared_B{make_tensor(pShared_B, smem_layout_B)};
    Tensor global_A{make_tensor(pGlobal_A, gmem_layout_A)};
    Tensor global_B{make_tensor(pGlobal_B, gmem_layout_B)};
    Tensor global_C{make_tensor(pGlobal_C, gmem_layout_C)};

    const size_t total_iters{ceil_div(size<1>(gmem_layout_A), BLOCK_TILE_SIZE_K)};

    const auto warp_coord{idx2crd(threadIdx.x, thread_layout.shape(), thread_layout.stride())};

    const size_t row_in_warp{warp_coord.first_};
    const size_t warp_y_idx{warp_coord.rest_.first_};

    const size_t col_in_warp{warp_coord.rest_.rest_.first_};
    const size_t warp_x_idx{warp_coord.rest_.rest_.rest_.first_};

    const auto smem_A{
        make_shape(Int<size<1>(A_SHARED_LAYOUT{})>{}, Int<size<0>(A_SHARED_LAYOUT{})>{})
    };

    Tensor block_tile_C{local_tile(global_C, thread_layout.shape(), make_coord(blockIdx.y, blockIdx.x))};
    Tensor warp_tile_C{local_tile(block_tile_C, warp_layout.shape(), make_coord(warp_y_idx, warp_x_idx))};

    T partial{0};

    auto coords{idx2crd(threadIdx.x, thread_layout.shape(), thread_layout.stride())};
    const size_t row{coords.first_};
    const size_t col{coords.rest_.first_};

    for (size_t iter{0}; iter < total_iters; ++iter) {
        // same as zipped divide w/ less code
        Tensor tile_A{local_tile(global_A, smem_A, make_coord(blockIdx.y, iter))};
        Tensor tile_B{local_tile(global_B, smem_layout_B.shape(), make_coord(iter, blockIdx.x))};

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

    block_tile_C(make_coord(row, col)) = block_tile_C(make_coord(row, col)) * beta + partial * alpha;
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

    constexpr size_t NUM_CACHES_PER_WARP_A{WARP_TILE_SIZE_X / (THREAD_TILE_SIZE_X * NUM_THREADS_PER_WARP_X)};
    constexpr size_t NUM_CACHES_PER_WARP_B{WARP_TILE_SIZE_Y / (THREAD_TILE_SIZE_Y * NUM_THREADS_PER_WARP_Y)};

    constexpr size_t THREADS_PER_BLOCK{32 * NUM_WARPS_X * NUM_WARPS_Y};

    constexpr Layout register_cache_A_lo{
        make_shape(Int<NUM_CACHES_PER_WARP_A>{}, Int<THREAD_TILE_SIZE_X>{}),
        LayoutRight{}
    };

    constexpr Layout register_cache_B_lo{
        make_shape(Int<NUM_CACHES_PER_WARP_B>{}, Int<THREAD_TILE_SIZE_X>{}),
        LayoutRight{}
    };

    thrust::host_vector<float> host_matrixA(M * K);
    thrust::host_vector<float> host_matrixB(K * N);
    thrust::host_vector<float> host_matrixC(M * N);

    fill_matrix_w(host_matrixA.data(), M, K, K, -100, 100);
    fill_matrix_w(host_matrixB.data(), K, N, N, -100, 100);
    for (size_t i{0}; i < N * M; ++i) host_matrixC[i] = 0.f;

    const Layout gmem_A_lo{make_layout(make_shape(M, K), LayoutRight{})};
    const Layout gmem_B_lo{make_layout(make_shape(K, N), LayoutRight{})};
    const Layout gmem_C_lo{make_layout(make_shape(M, N), LayoutRight{})};

    constexpr Layout smem_A_lo{
        make_layout(make_shape(Int<BLOCK_TILE_SIZE_K>{}, Int<BLOCK_TILE_SIZE_Y>{}), LayoutRight{})
    };
    constexpr Layout smem_B_lo{
        make_layout(make_shape(Int<BLOCK_TILE_SIZE_K>{}, Int<BLOCK_TILE_SIZE_X>{}), LayoutRight{})
    };

    constexpr Layout thread_lo{
        make_layout(
            make_shape(
                make_shape(Int<NUM_THREADS_PER_WARP_Y>{}, Int<BLOCK_TILE_SIZE_Y / WARP_TILE_SIZE_Y>{}),
                make_shape(Int<NUM_THREADS_PER_WARP_X>{}, Int<BLOCK_TILE_SIZE_X / WARP_TILE_SIZE_X>{})),
            make_stride(
                make_stride(Int<NUM_THREADS_PER_WARP_X>{}, Int<BLOCK_TILE_SIZE_X / WARP_TILE_SIZE_X * 32>{}),
                make_stride(_1{}, _32{})
            )
        )
    };

    constexpr size_t cache_y{2};
    constexpr size_t cache_x{2};

    constexpr Layout warp_layout{
        make_layout(
            make_shape(_128{}, _64{}),
            LayoutRight{}
        )
    };

    // constexpr Layout resp_layout{
    //     make_layout(
    //         make_shape(
    //             make_shape(make_shape(make_shape(NUM_THREADS_PER_WARP_Y, NUM_THREADS_PER_WARP_X), make_shape(cache_y, cache_x)), THREAD_TILE_SIZE_Y),
    //             make_shape(THREAD_TILE_SIZE_X)
    //         ),
    //         make_stride(
    //             make_stride(make_stride(make_stride(THREAD_TILE_SIZE_Y *  _64{}, THREAD_TILE_SIZE_X), make_stride(THREAD_TILE_SIZE_Y * NUM_THREADS_PER_WARP_Y * _64{}, THREAD_TILE_SIZE_X * NUM_THREADS_PER_WARP_X)), _64{}),
    //             make_stride(_1{})
    //         )
    //     )
    // };

    constexpr Layout tv_layout{
        make_layout(
            make_shape(
                make_shape(THREAD_TILE_SIZE_Y, make_shape(make_shape(cache_y, cache_x),
                               make_shape(make_shape(NUM_THREADS_PER_WARP_Y, NUM_THREADS_PER_WARP_X)))
                ),
                make_shape(THREAD_TILE_SIZE_X)
            ),
            make_stride(
                make_stride(_1{}, make_stride(make_stride(THREAD_TILE_SIZE_Y * NUM_THREADS_PER_WARP_Y, THREAD_TILE_SIZE_X * _128{}),
                    make_stride(make_stride(THREAD_TILE_SIZE_Y, THREAD_TILE_SIZE_X * _128{})))
                    ),
                make_stride(_128{})
            )
        )
    };

    print_layout(warp_layout(make_coord(0, make_coord())));

    auto tv{composition(warp_layout, tv_layout)};

    print_layout(tv_layout);

    std::cout << "\n";

    print_layout(tv);
    // print(idx2crd(255, thread_lo.shape(), thread_lo.stride()));
}
