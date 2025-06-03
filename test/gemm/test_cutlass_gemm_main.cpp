//
// Created by sriram on 5/28/25.
//

#include "test_cutlass_gemm.h"
#include <gtest/gtest.h>


TEST(CUTE, cute_gemm_2DBT) {
    test_cute_gemm_2DBT();
}

TEST(CUTE, test_cute_gemm_2DBT_2DWT_2DTT_vloadT) {
    test_cute_gemm_2DBT_2DWT_2DTT_vloadT();
}
