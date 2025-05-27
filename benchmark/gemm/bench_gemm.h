//
// Created by sriram on 5/27/25.
//

#ifndef BENCH_GEMM_H
#define BENCH_GEMM_H

void time_2DBT();
void time_gemm_2DBT_2DWT_2DTT_vload();
void time_gemm_2DBT_2DWT_2DTT_async();
void time_gemm_2DBT_2DWT_2DTT_async_A_transposed();

#endif //BENCH_GEMM_H
