//
// Created by sriram on 5/27/25.
//

#ifndef BENCH_HELPERS_CUH
#define BENCH_HELPERS_CUH

void flush_l2_cache();

void fill_matrix(
    float *matrix,
    size_t M,
    size_t N,
    size_t stride,
    float min_val,
    float max_val);

#endif //BENCH_HELPERS_CUH
