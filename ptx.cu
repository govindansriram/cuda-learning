
#include "api.h"



__global__ void global_load(float * data) {

    /**
     * much more in depth breakdown:
     * https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-ld
     **/

    float reg; // single float register
    float* gmem_ptr{data}; // pointer to global memory

    /**
     * here is the base instruction brackets mean the argument is technically optional, this is a small
     * subset of possibilities
     *
     * ld{.weak}{.ss}.type d, [a];
     *
     * the weak qualifier means a load instruction with no synchronization, essentially the effects of this load
     * are only present to other threads after a sync
     *
     * .ss is the state space, this could be constant (const) memory, global, shared, etc.
     * type is the datatype eg: f32, f64, s32 (signed 32 bit int), u32( unsigned 32 bit int)
     *
     * d is the register being loaded too,
     * a is the source address operand
     */

    /**
     * this is inline assembly for the load operation
     * we mark the operation as volatile so the compiler does not optimize it away
     * we signal the variables for the operation using %0 and %1
     * We supply those variables after the instruction, seperated by colons
     *
     * The first argument "=f"(reg), means the register is a float, the = means the variable is being written too
     * l signals the data is a long variable (an address is a unsigned long int)
     */
    asm volatile("ld.weak.global.f32 %0, [%1];" : "=f"(reg) : "l"(gmem_ptr + 2));

    printf("the register value is %f \n", reg);
}


void test_global_load() {
    constexpr float h_data[4]{1.f, 2.f, 3.f, 4.f};

    float * d_data;

    cudaMalloc(&d_data, 16);

    cudaMemcpy(d_data, h_data, 16, cudaMemcpyHostToDevice);

    global_load<<<1, 1>>>(d_data);

    cudaDeviceSynchronize();

    // Check for errors in kernel execution
    if (const cudaError_t error = cudaGetLastError(); error != cudaSuccess)
        std::cerr << "CUDA error: " << cudaGetErrorString(error) << std::endl;

    cudaFree(d_data);
}

