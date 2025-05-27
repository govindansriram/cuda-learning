//
// Created by sriram on 5/27/25.
//

#include <gtest/gtest.h>
#include "test_gemm.h"


TEST(G_2D_BT, GEMM) {
    test_gemm_2DBT();
}

TEST(G_2DBT_2DWT_2DTT_VLOAD, GEMM) {
    test_gemm_2DBT();
}

TEST(G_2DBT_2DWT_2DTT_ASYNC, GEMM) {
    test_gemm_2DBT_2DWT_2DTT_async();
}
