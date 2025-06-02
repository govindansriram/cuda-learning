//
// Created by sriram on 5/27/25.
//

#ifndef TEST_HELPERS_H
#define TEST_HELPERS_H

void cpu_matmul_naive(
    const float *mat_A,
    const float *mat_B,
    float *mat_C,
    size_t M,
    size_t N,
    size_t K,
    size_t A_stride,
    size_t B_stride,
    size_t C_stride);

void test_equivalency(
    const float *expected,
    const float *result,
    size_t M,
    size_t N,
    size_t stride);

int generate_random_int(int min_val, int max_val);

template <typename T>
void fill_matrix_w(
    T * matrix,
    const size_t M,
    const size_t N,
    const size_t stride,
    const int min_val,
    const int max_val) {

    for (size_t i{0}; i < M; ++i) {
        for (size_t j{0}; j < N; ++j) {
            matrix[i * stride + j] = static_cast<T>(generate_random_int(min_val, max_val));
        }
    }
}

template<>
void fill_matrix_w<float>(
    float *matrix,
    size_t M,
    size_t N,
    size_t stride,
    int min_val,
    int max_val);


void print_matrix(
    const float *matrix,
    size_t M,
    size_t N,
    size_t stride);

#endif //TEST_HELPERS_H
