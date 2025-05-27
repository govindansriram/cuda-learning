//
// Created by sriram on 5/27/25.
//

#ifndef HELPERS_CUH
#define HELPERS_CUH

__device__ __host__ __forceinline__ constexpr size_t ceil_div(const size_t top, const size_t bottom) {
    return (top + (bottom - 1)) / bottom;
}

template<size_t M, size_t N>
__device__ void print_matrix(float matrix[M][N]) {
    if (blockIdx.x + blockIdx.y == 0 && threadIdx.x + threadIdx.y == 0) {
        printf("\n");
        for (size_t m{0}; m < M; ++m) {
            printf("[");
            for (size_t n{0}; n < N; ++n) {
                printf("%f ", matrix[m][n]);
            }
            printf("]\n");
        }
    }
}

#define CUDA_0_EXPR(expr)                                                \
{                                                                        \
    if (threadIdx.x + threadIdx.y == 0 && blockIdx.x + blockIdx.y == 0){ \
        expr                                                             \
    }                                                                    \
}

__device__ __host__ __forceinline__ constexpr bool is_power_of_two(const size_t x) {
    /**
     * first checks if the number is not 0 we do this premptive check since 0 is not a power of 2
     * yet would pass the next check
     *
     * (x & (x - 1)) == 0 this checks if the number is a power of 2
     *
     * all powers of 2 have exactly one 1 bit e.g.:
     * 1 0 0 0 = 1
     * 0 1 0 0 = 2
     * 0 0 1 0 = 4
     * 0 0 0 1 = 8
     *
     * the number before of 2 has to have all 1 bits excpet at the last position
     *
     * 0 0 0 0 = 0
     * 1 0 0 0 = 1
     * 1 1 0 0 = 3
     * 1 1 1 0 = 7
     *
     * so we know a number is a multiple of 2 if the bit wise and of the previous current number == 0
     */
    return x != 0 && (x & (x - 1)) == 0;
}

__device__ __host__ __forceinline__ constexpr size_t next_power_of_two(size_t x) {
    /**
     * We know that 1 - (a power of 2) is all 1 bits so to find the next power of 2
     * our goal is to fill all bits before the highest set bit to 1 and then add
     * 1 to it giving us our next power of 2.
     *
     * We also start off by subtracting 1 to handle the case where the number
     * is already a power of 2
     **/


    if (x == 0) return 1;
    --x;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    x |= x >> 32;
    return x + 1;
}

#endif //HELPERS_CUH
