//
// Created by sriram on 5/25/25.
//

#ifndef PTX_HELPERS_CUH
#define PTX_HELPERS_CUH

#define CONVERT_SHARED_PTR_TO_UINT(addr, shared_P)                                 \
    {                                                                              \
        asm volatile("cvta.to.shared.u64 %0, %1;\n": "=l"(addr) : "l"(shared_P));  \
    }                                                                              \

// taken from
// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html?highlight=cp%2520async#data-movement-and-conversion-instructions-cp-async-commit-group
// and
// https://github.com/NVIDIA/cutlass/blob/main/include/cutlass/arch/memory_sm80.h


# define CP_ASYNC_SMALL(shared_P, global_P, amt)                                                         \
    {                                                                                                    \
        static_assert(amt == 4 || amt == 8);                                                             \
        uintptr_t addr;                                                                                  \
        CONVERT_SHARED_PTR_TO_UINT(addr, shared_P)                                                       \
        asm volatile("cp.async.ca.shared.global %0, %1, %2;\n" :: "l"(addr), "l"(global_P), "n"(amt));   \
    }                                                                                                    \

# define CP_ASYNC_LARGE(shared_P, global_P)                                                              \
    {                                                                                                    \
        uintptr_t addr;                                                                                  \
        CONVERT_SHARED_PTR_TO_UINT(addr, shared_P)                                                       \
        asm volatile("cp.async.ca.shared.global %0, %1, %2;\n" :: "l"(addr), "l"(global_P), "n"(16));    \
    }                                                                                                    \

#define CP_ASYNC_COMMIT_GROUP                            \
    {                                                    \
        asm volatile("cp.async.commit_group;\n" ::);     \
    }                                                    \


#define CP_ASYNC_WAIT(N)                                     \
    {                                                        \
        asm volatile("cp.async.wait_group %0;\n" ::"n"(N));  \
    }                                                        \


#define CP_ASYNC_WAIT_ALL {asm volatile("cp.async.wait_all;\n" ::)};


#endif //PTX_HELPERS_CUH
