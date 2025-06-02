//
// Created by sriram on 5/27/25.
//

#include "../../src/gemm/cutlass_gemm.cuh"
#include "../test_helpers.h"
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

// taken directly from here
// https://github.com/NVIDIA/cutlass/blob/main/examples/cute/tutorial/sgemm_sm80.cu#L584

// A is not transposed, B is transposed

// these all assume column major
void f16_gemm_NT_column_major(int argc, char **argv) {
    constexpr size_t m{5120};
    constexpr size_t n{5120};
    constexpr size_t k{4096};
    constexpr bool transpose_A{false}; // A is column major
    constexpr bool transpose_B{true}; // B is transposed now it's row major

    using type_A = cute::half_t;
    using type_B = cute::half_t;
    using type_C = cute::half_t;
    using type_scalars = cute::half_t;

    auto alpha{static_cast<type_scalars>(1.0f)};
    auto beta{static_cast<type_scalars>(0.0f)};

    thrust::host_vector<type_A> h_A(m * k);
    thrust::host_vector<type_B> h_B(n * k);
    thrust::host_vector<type_C> h_C(m * n);

    fill_matrix_w<type_A>(h_A.data(), m, k, k, 0, 100);
    fill_matrix_w<type_B>(h_B.data(), n, k, k, 0, 100);
    fill_matrix_w<type_C>(h_C.data(), m, n, n, 0, 0);

    thrust::device_vector<type_A> d_A = h_A;
    thrust::device_vector<type_B> d_B = h_B;
    thrust::device_vector<type_C> d_C = h_C;

    size_t ldA{m}; // column major
    size_t ldB{n}; // row major
    size_t ldC{m}; // column major

    CUTE_CHECK_LAST();
}

template<typename Shape, typename Stride>
void print2D(cute::Layout<Shape, Stride> const &layout) {
    std::cout << "shape: (" << cute::size<0>(layout) << " ";
    std::cout << cute::size<1>(layout) << ")" << std::endl;

    for (size_t m{0}; m < cute::size<0>(layout); ++m) {
        for (size_t n{0}; n < cute::size<1>(layout); ++n) {
            std::cout << layout(m, n) << " ";
        }
        std::cout << std::endl;
    }
}

template<typename Shape, typename Stride>
void print_info(cute::Layout<Shape, Stride> const &layout) {
    // a mode correlates to a dimension but it may compromise of multiple elements such as a tuple

    std::cout << "shape : stride" << std::endl;
    print(layout);
    std::cout << std::endl;
    std::cout << "rank (# of modes): " << rank(layout) << std::endl;
    std::cout << "depth (# of nested tuples): " << depth(layout) << std::endl;
    std::cout << "size (total # of elements): " << size(layout) << std::endl;;
    std::cout << std::endl;
}


void test_layout() {
    cute::Layout<cute::Int<8> > s8{
        cute::make_layout(cute::_8{})
    };

    const int d8_num{8};
    cute::Layout<cute::Shape<int> > d8{
        cute::make_layout(cute::make_shape(d8_num))
    };

    cute::Layout<cute::Shape<int, int>, cute::Stride<int, int> > matrix{
        cute::make_layout(
            cute::make_shape(5, 4),
            cute::make_stride(4, 1)
        )
    };

    cute::Layout<cute::Shape<cute::_5, cute::_4> > matrix2{
        cute::make_layout(cute::make_shape(cute::_5{}, cute::_4{})) // column wise
    };

    cute::Layout<cute::Shape<int, int, int>, cute::Stride<cute::_1, int, int> > tensor1{
        cute::make_layout(
            cute::make_shape(3, 4, 2),
            cute::LayoutLeft{}
        )
    };

    cute::Layout<cute::Shape<int, cute::Shape<int, int> > > tensor2{
        cute::make_layout(
            cute::make_shape(3, cute::make_shape(4, 2))
        )
    };

    // same as size 2

    cute::Layout<cute::Shape<int, cute::Shape<int, int> >, cute::Stride<int, cute::Shape<int, cute::_1> > >
            tensor3_right{
                cute::make_layout(
                    cute::make_shape(3, cute::make_shape(4, 2)),
                    cute::LayoutRight{}
                )
            };

    // vector layout standard layout
    print_info(s8);

    // vector layout depth 1
    print_info(d8);

    // matrix layout (row wise)
    print_info(matrix);

    // matrix layout (column wise)
    print_info(matrix2);

    // tensor (column wise)
    print_info(tensor1);

    // tensor (depth 2, rank 2) column wise
    print_info(tensor2);

    // tensor (depth 2, rank 2) row wise
    print_info(tensor3_right);

    std::cout << "--------------------------------------------" << std::endl;

    // row wise matrix coordinate to memory index
    print2D(matrix);
    std::cout << "--------------------------------------------" << std::endl;

    // column wise coordiante to memory index
    print2D(matrix2);
    std::cout << "--------------------------------------------" << std::endl;

    // tensor with rank 2 can be used here as well even if total modes are 3
    print2D(tensor3_right);

    // when used for coordinates the shape becomes reduced
    // so (3,(4,2)) becomes (3, 8); however the stride (8,(2,_1))
    // stays the same
    // assume the matrices look like this
    // 0 1   8  9    16 17
    // 2 3   10 11   18 19
    // 4 5   12 13   20 21
    // 6 7   14 15   22 23
    //
    // the 8 in the stride means that all corresponding positions in each matrix must be 8 elements
    // away from each other. The 2 means that the first 4 (from shape) elements must be 2 elemnts away from each other
    // the 1 means that those 2 arrays must be 1 away the stride then becomes
    // [0, 2, 4, 6][1, 3, 5, 7] -> 0, 2, 3, 4, 1, 3, 5, 7
    cute::print_latex(tensor3_right);

    // coalescing
    /**
     * This is essentially equivalent to a flatten operation where you try to reduce the dimensionality of a tensor
     * to as low as possible. This only works with static (not dynamic) sizes
     *
     * Flattening can happen in a number of scenarios
     * - if a stride starts with 1 it can generally be flattened since it signals that the data is contigous
     * - if a shape contains a mode with size 1, it can generally be removed, as it has no effect on the actual layout
     *  of the data
     *  - finally if the If the second mode’s stride is the product of the first mode’s size and stride,
     *  then they can be combined as this signals that those dimensions are contigous
     */

    auto layout = cute::Layout<cute::Shape<cute::_3, cute::Shape<cute::_4, cute::_2> > >{};
    auto result = cute::coalesce(layout);

    // auto layout2 = cute::make_shape(
    //     cute::make_shape(cute::_3{}, cute::_4{}, cute::_2{}),
    //     cute::LayoutRight{}
    // );

    auto layout2 = cute::make_layout(
        cute::make_shape(cute::_3{}, cute::_1{}, cute::_4{}, cute::_2{}),
        cute::LayoutRight{}
    );

    auto result2 = cute::coalesce(layout2);

    // ends
    print_info(result);
    print_info(result2);
}

template<typename Engine, typename Layout>
void print_tensor_local(cute::Tensor<Engine, Layout> tens) {
    for (size_t m{0}; m < cute::size<0>(tens.layout()); ++m) {
        std::cout << "[ ";
        for (size_t n{0}; n < cute::size<1>(tens.layout()); ++n) {
            std::cout << tens(m, n) << " ";
        }
        std::cout << "]" << std::endl;
    }
}

// shared_gemm (rename)
void thread_value_partitioning() {
    using namespace cute;

    thrust::host_vector<float> host_matrix(16 * 16);
    std::cout << host_matrix.capacity() << std::endl;
    // auto tv_layout = Layout<Shape <Shape <_2,_4>,Shape <_2, _2>>,
    //                     Stride<Stride<_8,_1>,Stride<_4,_16>>>{}; // (8,4)

    for (size_t i{0}; i < 256; ++i) host_matrix[i] = static_cast<float>(i);

    auto layout{
        make_layout(
            make_shape(_16{}, _16{}),
            LayoutRight{}
        )
    };

    auto tensor{make_tensor(host_matrix.data(), layout)};
    //
    print_layout(layout);

    // print_layout(tensor.layout());

    // auto tiler{
    //     make_tile(
    //         make_layout(make_shape(_4{}, _4{}))
    //     )
    // };

    // auto logical_layout{logical_divide(layout, tiler)};

    // auto zd{zipped_divide(tensor, tiler)};

    // print(zd);

    auto tiler2 = Shape<_4, _4>{}; // (_4,_8)

    Tensor tiled_a = zipped_divide(tensor, tiler2); // ((_4,_8),(2,3))

    print(tiled_a.layout());

    std::cout << std::endl;

    for (size_t i{0}; i < 4; ++i) {
        for (size_t j{0}; j < 4; ++j) {
            // Tensor cta = tiled_a(make_coord(_, _), make_coord(i, j)); // (_4,_8)
            print_tensor_local(tiled_a(make_coord(_, _), make_coord(i, j)));
            std::cout << std::endl;
        }
    }

    // print_tensor_local(cta);


    // auto value{cta(0, 0)};
    //
    // print_layout(cta.layout());
    //
    // std::cout << value << std::endl;

    // print_layout(other);

    // print_info(other);

    // auto tiler = Shape<_4,_4>{};
    //
    // Tensor tiled_a = zipped_divide(global_tens, tiler);
    //
    // Tensor cta_a = tiled_a(make_coord(_,_), make_coord(0, 1));

    // print_layout(layout(cta_a));


    // print_layout(tv_layout);
    // print_latex(tv_layout);
}

template <class Shape, class Stride>
CUTE_HOST_DEVICE void print1D(cute::Layout<Shape,Stride> const& layout)
{
    for (int i = 0; i < size(layout); ++i) printf("%3d  ", layout(i));
}

template <class ENGINE, class LAYOUT>
CUTE_HOST_DEVICE void print2D_tensor(const cute::Tensor<ENGINE, LAYOUT> &tensor) {
    for (int i = 0; i < cute::size<0>(tensor); ++i) {
        for (int j = 0; j < cute::size<1>(tensor); ++j) {
            printf("%3f  ", tensor(cute::make_coord(i, j)));
        }
        printf("\n");
    }
}

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

    constexpr size_t A_loads_per_thread{smem_length_A / thread_length};
    constexpr size_t B_loads_per_thread{smem_length_B / thread_length};

    // constexpr auto tv_layout_A{
    //     make_layout(
    //         make_shape(Int<cosize_v<THREAD_LAYOUT>>{}, Int<A_loads_per_thread>{}),
    //         make_stride(_1{}, Int<cosize_v<THREAD_LAYOUT>>{})
    //         )
    // };

    // constexpr auto tv_layout_A{
    //     make_layout(
    //         make_shape(make_shape(_16{}, _2{}), _8{}),
    //         make_shape(make_stride(_16{}, _8{}), _1{})
    //         // make_stride(Int<cosize_v<THREAD_LAYOUT>>{}, _1{})
    //         )
    // };

    // TODO make it completely constexpr

    // constexpr size_t global_size_A{size<0>(A_GLOBAL_LAYOUT{})};

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

    // if (threadIdx.x == 16 && blockIdx.x + blockIdx.y == 0) {
    //     print_layout(tv_layout_B.layout());
    //     printf("\n\n");
    //     print_layout(B_SHARED_LAYOUT{});
    //     printf("\n\n");
    //     print_layout(shared_B_tv.layout());
    // }

    const Tensor global_A_value{global_A_tv(threadIdx.x, _)};
    Tensor shared_A_value{shared_A_tv(threadIdx.x, _)};

    const Tensor global_B_value{global_B_tv(threadIdx.x, _)};
    Tensor shared_B_value{shared_B_tv(threadIdx.x, _)};

    copy(global_A_value, shared_A_value);
    copy(global_B_value, shared_B_value);

    // if (threadIdx.x == 16 && blockIdx.x + blockIdx.y == 0) {
    //     // print(global_B);
    //     // print_layout(global_B.layout());
    //     // print(global_B_tv);
    //     printf("\n");
    //     print_layout(global_A.layout());
    //     print(global_A_value);
    //     printf("%f", shared_A_value(0));
    // }

    // copy(global_B_tv(threadIdx.x, _), shared_B_tv(threadIdx.x, _));
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
    const T * gmem_A,
    const T * gmem_B,
    T * gmem_C,
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
        Tensor tile_A{gA_tiled(make_coord(_, _), make_coord(iter, blockIdx.y))};
        Tensor tile_B{gB_tiled(make_coord(_, _), make_coord(iter, blockIdx.x))};

        // load to shared
        load_to_shared(shared_A, shared_B, tile_A, tile_B, thread_layout);

        __syncthreads();

        if (threadIdx.x == 0 && blockIdx.x + blockIdx.y == 0) {
            print2D_tensor(shared_B);
            printf("\n");

            print2D_tensor(shared_A);
        }

        Tensor slice_A{shared_A(make_coord(_, row))};
        Tensor slice_B{shared_B(make_coord(col, _))};

// #pragma unroll
//         for (size_t kk{0}; kk < BLOCK_TILE_SIZE_K; ++kk) {
//             partial += slice_A(kk) * slice_B(kk);
//         }
//
//         __syncthreads();
         break;
    }

    // tile_C(make_coord(col, row)) = tile_C(make_coord(col, col)) * beta + partial * alpha;
}

void test_others() {
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

    for (size_t i{0}; i < M * K; ++i) host_matrixA[i] = static_cast<float>(i);
    for (size_t i{0}; i < N * K; ++i) host_matrixB[i] = static_cast<float>(i);
    for (size_t i{0}; i < N * M; ++i) host_matrixC[i] = 0.f;

    thrust::device_vector<float> device_matrixA{host_matrixA};
    thrust::device_vector<float> device_matrixB{host_matrixB};
    thrust::device_vector<float> device_matrixC{host_matrixC};

    const Layout gmem_A_lo{make_layout(make_shape(M, K), LayoutRight{})};
    const Layout gmem_B_lo{make_layout(make_shape(K, N), LayoutRight{})};
    const Layout gmem_C_lo{make_layout(make_shape(M, N), LayoutRight{})};

    // print2D_tensor(make_tensor(host_matrixA.data(), gmem_A_lo));

    constexpr Layout smem_A_lo{make_layout(make_shape(Int<BLOCK_TILE_SIZE_Y>{}, Int<BLOCK_TILE_SIZE_K>{}), LayoutRight{})};
    constexpr Layout smem_B_lo{make_layout(make_shape(Int<BLOCK_TILE_SIZE_K>{}, Int<BLOCK_TILE_SIZE_X>{}), LayoutRight{})};
    constexpr Layout thread_lo{
        make_layout(make_shape(Int<4>{}, Int<8>{}), LayoutRight{})};

    constexpr size_t shared_mem_size{
        (BLOCK_TILE_SIZE_Y * BLOCK_TILE_SIZE_K) + (BLOCK_TILE_SIZE_K * BLOCK_TILE_SIZE_X) * sizeof(float)
    };

    dim3 grid_dim{
        ceil_div(N, BLOCK_TILE_SIZE_X),
        ceil_div(M, BLOCK_TILE_SIZE_Y)
    };

    // dim3 block_dim{
    //     BLOCK_TILE_SIZE_X * BLOCK_TILE_SIZE_Y
    // };

    dim3 block_dim{
        32
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
}
