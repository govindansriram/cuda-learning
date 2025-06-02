//
// Created by sriram on 5/27/25.
//

#ifndef CUTLASS_GEMM_H
#define CUTLASS_GEMM_H

#include <cute/tensor.hpp>

template<
    typename TYPE_MATRIX_A,
    typename TYPE_MATRIX_B,
    typename TYPE_MATRIX_C,
    typename TYPE_ALPHA,
    typename TYPE_BETA>
void gemm_nt(
    const size_t m,
    const size_t n,
    const size_t k,
    const TYPE_ALPHA alpha,
    const TYPE_BETA beta,
    const TYPE_MATRIX_A * matrix_A,
    const TYPE_MATRIX_B * matrix_B,
    TYPE_MATRIX_C * matrix_C,
    const size_t ldA,
    const size_t ldB,
    const size_t ldC
) {

    cute::Shape<size_t, size_t, size_t> shape{cute::make_shape(m, n, k)};
    cute::Stride<cute::Int<1>, size_t> dimension_A{cute::make_stride(cute::Int<1>{}, ldA)};
    cute::Stride<cute::Int<1>, size_t> dimension_B{cute::make_stride(cute::Int<1>{}, ldB)};
    cute::Stride<cute::Int<1>, size_t> dimension_C{cute::make_stride(cute::Int<1>{}, ldC)};

    // Block (CTA) Tile Size

    const cute::Int<128> block_M;
    const cute::Int<128> block_N;
    const cute::Int<8> block_K;
    cute::Shape<cute::Int<128>, cute::Int<128>, cute::Int<8>> tiler{cute::make_shape(block_M, block_N, block_K)};

    // staging pipeline used to overlap computation and data processing
    const cute::Int<3> block_pipeline;

    cute::Layout<cute::Shape<cute::Int<128>, cute::Int<8>, cute::Int<3>>> shared_A_layout{
        cute::make_layout(cute::make_shape(block_M, block_K, block_pipeline))
    };

    cute::Layout<cute::Shape<cute::Int<128>, cute::Int<8>, cute::Int<3>>> shared_B_layout{
        cute::make_layout(cute::make_shape(block_N, block_K, block_pipeline))
    };

    cute::Layout<cute::Shape<cute::Int<128>, cute::Int<128>>> shared_C_layout{
        cute::make_layout(cute::make_shape(block_N, block_M))
    };

    // cute::make_tiled_copy()

}

#endif //CUTLASS_GEMM_H
