//
// Created by sriram on 5/27/25.
//


#include "bench_gemm.h"

int main() {
    // time_2DBT();
    time_gemm_2DBT_2DWT_2DTT_vload();
    time_gemm_2DBT_2DWT_2DTT_async();
    time_gemm_2DBT_2DWT_2DTT_async_A_transposed();
}
