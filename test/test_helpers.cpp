//
// Created by sriram on 5/27/25.
//

#include <random>
#include <gtest/gtest.h>
#include "test_helpers.h"

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
            if (expected[i * stride + j] != result[i * stride + j]) {
                std::cout << "row: " << i << " " << "column: " << j << std::endl;
            }
            ASSERT_EQ(expected[i * stride + j], result[i * stride + j]);
        }
    }
}

int generate_random_int(const int min_val, const int max_val) {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::uniform_int_distribution dis(min_val, max_val);
    return dis(gen);
}


template<>
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