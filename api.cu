#include <iostream>
#include "api.h"
#include <iomanip>  // For std::fixed and std::setprecision


__global__ void vector_add_kernel(const float *d_a, const float *d_b, float *d_c, size_t const elements) {
    unsigned int current_thread = blockIdx.x * blockDim.x + threadIdx.x;
    if (current_thread < elements) d_c[current_thread] = d_a[current_thread] + d_b[current_thread];
}

__global__ void picture_kernel(const float *d_pin, float *d_pout, int const n, int const m) {
    unsigned int current_row{blockIdx.y * blockDim.y + threadIdx.y};
    unsigned int current_col{blockIdx.x * blockDim.x + threadIdx.x};

    if (current_row < m && current_col < n) {
        d_pout[current_row * n + current_col] = d_pin[current_row * n + current_col] * 2;
    }
}

__global__ void transpose_square_matrix_kernel(unsigned char *d_pin, int const n) {
    unsigned int const current_row{blockIdx.y * blockDim.y + threadIdx.y};
    unsigned int const current_col{blockIdx.x * blockDim.x + threadIdx.x};

    if ((current_row < n && current_col < n) && (current_col >= current_row)) {
        unsigned char const temp = d_pin[current_row * n + current_col];
        d_pin[current_row * n + current_col] = d_pin[current_col * n + current_row];
        d_pin[current_col * n + current_row] = temp;
    }
}

__global__ void transpose_matrix_kernel(const unsigned char *d_pin, unsigned char *d_pout, int const n, int const m) {
    unsigned int const current_row{blockIdx.y * blockDim.y + threadIdx.y};
    unsigned int const current_col{blockIdx.x * blockDim.x + threadIdx.x};

    if (current_row < m && current_col < n) {
        d_pout[current_col * m + current_row] = d_pin[current_row * n + current_col];
    }
}

__global__ void transpose_cv_image_kernel(const unsigned char *d_pin, unsigned char *d_pout, int const columns, int const rows, int const channels, int const in_skip, int const out_skip) {
    unsigned int const current_row{blockIdx.y * blockDim.y + threadIdx.y};
    unsigned int const current_col{blockIdx.x * blockDim.x + threadIdx.x};

    if (current_row < rows && current_col < columns) {
        for (size_t i{0}; i < channels; ++i) d_pout[(current_col * out_skip) + (current_row * channels) + i] = d_pin[(current_row * in_skip) + (current_col * channels) + i];
    }
}

cv::Mat transpose_opencv_image(cv::Mat const &image) {
    cv::Mat transposed(image.cols, image.rows, image.type());

    unsigned char *dsource_ptr;
    unsigned char *ddest_Ptr;

    cudaMalloc(&dsource_ptr, image.step * image.rows);
    cudaMemcpy(dsource_ptr, image.data, image.step * image.rows, cudaMemcpyHostToDevice);
    cudaMalloc(&ddest_Ptr, transposed.step * transposed.rows);

    constexpr dim3 block_dim(16, 16);
    dim3 grid_dim(
        std::ceil(static_cast<float>(image.cols) / block_dim.x),
        std::ceil(static_cast<float>(image.rows) / block_dim.x)
        );

    transpose_cv_image_kernel<<<grid_dim, block_dim>>>(dsource_ptr, ddest_Ptr, image.cols, image.rows, image.channels(), image.step, transposed.step);

    cudaDeviceSynchronize(); // Force flush of printf buffer
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(error) << std::endl;
    }

    cudaMemcpy(transposed.data, ddest_Ptr, transposed.step * transposed.rows, cudaMemcpyDeviceToHost);

    cudaFree(dsource_ptr);
    cudaFree(ddest_Ptr);

    return transposed;
}

void transpose_wrapper(unsigned char *data, size_t const rows, size_t const columns) {
    // size_t const rows = mat.rows * 3;
    // size_t const columns = mat.cols * 3;
    // unsigned char * underlying_data = mat.data();

    unsigned char *dsource_ptr;
    unsigned char *ddest_Ptr;
    cudaMalloc(&dsource_ptr, rows * columns);
    cudaMemcpy(dsource_ptr, data, rows * columns, cudaMemcpyHostToDevice);

    constexpr dim3 block_dim(16, 16);
    dim3 grid_dim(
        static_cast<unsigned int>(std::ceil(static_cast<float>(columns) / block_dim.x)),
        static_cast<unsigned int>(std::ceil(static_cast<float>(rows) / block_dim.y))
    );

    if (rows == columns) {
        transpose_square_matrix_kernel<<<grid_dim, block_dim>>>(dsource_ptr, static_cast<int>(columns));
    }else {
        cudaMalloc(&ddest_Ptr, rows * columns);
        transpose_matrix_kernel<<<grid_dim, block_dim>>>(dsource_ptr, ddest_Ptr,
            static_cast<int>(columns), static_cast<int>(rows));
    }

    cudaDeviceSynchronize(); // Force flush of printf buffer
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(error) << std::endl;
    }

    if (rows != columns) {
        cudaMemcpy(data, ddest_Ptr, rows * columns, cudaMemcpyDeviceToHost);
        cudaFree(ddest_Ptr);
    }else {
        cudaMemcpy(data, dsource_ptr, rows * columns, cudaMemcpyDeviceToHost);
    }
    cudaFree(dsource_ptr);
}

void picture_kernel_wrapper() {
    cv::Mat const img{cv::imread("/home/sriram/Pictures/maxresdefault.jpg")};
    std::cout << img.rows << std::endl;
    std::cout << img.cols << std::endl;
    std::cout << img.channels() << std::endl;

    constexpr size_t row_max{2234};
    constexpr size_t col_max{2258};

    constexpr dim3 block_dim(16, 16);
    constexpr dim3 grid_dim(
        static_cast<unsigned int>(std::ceil(static_cast<float>(col_max) / block_dim.x)),
        static_cast<unsigned int>(std::ceil(static_cast<float>(row_max) / block_dim.y))
    );

    std::cout << "block_dim x: " << block_dim.x << " " << "block_dim y: " << block_dim.y << std::endl;
    std::cout << "grid_dim x: " << grid_dim.x << " " << "grid_dim y: " << grid_dim.y << std::endl;

    constexpr size_t total_len{row_max * col_max};

    auto *h_source = new float[total_len];
    auto *h_dest = new float[total_len];

    PseudoRNG<float> gen;

    for (int i = 0; i < total_len; ++i) h_source[i] = static_cast<float>(i) / 100;

    float *d_source;
    float *d_dest;

    cudaMalloc(&d_source, total_len * sizeof(float));
    cudaMalloc(&d_dest, total_len * sizeof(float));
    cudaMemcpy(d_source, h_source, total_len * sizeof(float), cudaMemcpyHostToDevice);

    picture_kernel<<<grid_dim, block_dim>>>(d_source, d_dest, col_max, row_max);

    cudaDeviceSynchronize();

    // Check for errors in kernel execution
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(error) << std::endl;
    } else {
        std::cout << "CUDA kernel execution completed successfully." << std::endl;
    }

    cudaMemcpy(h_dest, d_dest, total_len * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_source);
    cudaFree(d_dest);

    // for (int i = 0; i < total_len; ++i) std::cout << std::fixed << std::setprecision(2) << h_dest[i] << "\n";

    delete[] h_source;
    delete[] h_dest;
}


void vadd_wrapper() {
    constexpr size_t arr_max{5};

    // Launch the CUDA kernel with 1 block and 10 threads
    constexpr float h_a[arr_max]{1, 2, 3, 4, 5};
    constexpr float h_b[arr_max]{10, 20, 30, 40, 0};
    float h_c[arr_max]{};

    constexpr size_t size{arr_max * sizeof(float)};

    float *d_c;
    float *d_a;
    float *d_b;

    cudaMalloc(reinterpret_cast<void **>(&d_c), size);
    cudaMalloc(reinterpret_cast<void **>(&d_a), size);
    cudaMalloc(reinterpret_cast<void **>(&d_b), size);

    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

    vector_add_kernel<<<1, 30>>>(d_a, d_b, d_c, arr_max);

    // Wait for all threads to complete before ending the program
    cudaDeviceSynchronize();

    // Check for errors in kernel execution
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(error) << std::endl;
    } else {
        std::cout << "CUDA kernel execution completed successfully." << std::endl;
    }

    cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);
    cudaFree(d_c);
    cudaFree(d_a);
    cudaFree(d_b);

    for (float const &i: h_c) {
        std::cout << std::fixed << std::setprecision(2) << i << std::endl;
    }
}
