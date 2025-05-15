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

// first mat, m x n; second mat n x m final mat
__global__ void mat_mul(const float * mat1, const float * mat2, float * dest_mat, size_t const mat1_rows, size_t const shared_dim, size_t const mat2_columns) {
    unsigned int const column{blockDim.x * blockIdx.x + threadIdx.x};
    unsigned int const row{blockDim.y * blockIdx.y + threadIdx.y};

    if (column < mat2_columns && row < mat1_rows) {
        float sum{0};

        for (size_t i = 0; i < shared_dim; ++i) {
            sum += mat1[row * shared_dim + i] * mat2[i * mat2_columns + column];
        }

        dest_mat[mat2_columns * row + column] = sum;
    }
}

// requires matrices perfectly divisible by TILE_WIDTH
#define TILE_WIDTH 2
__global__ void tiled_mat_mul(const float * mat1, const float * mat2, float * dest_mat, const size_t mat_1_width) {

    const unsigned int curr_col{threadIdx.x};
    const unsigned int curr_row = {threadIdx.y};
    const unsigned int dest_column{blockIdx.x * TILE_WIDTH + curr_col};
    const unsigned int dest_row{blockIdx.y * TILE_WIDTH + curr_row};

    const unsigned int mat_2_width{gridDim.x * TILE_WIDTH};

    __shared__ float row_arr[TILE_WIDTH][TILE_WIDTH];
    __shared__ float column_arr[TILE_WIDTH][TILE_WIDTH];

    float partial{0};

    for (int m = 0; m < mat_1_width / TILE_WIDTH; ++m) {
        row_arr[curr_row][curr_col] = mat1[mat_1_width * dest_row + m * TILE_WIDTH + curr_col];
        column_arr[curr_row][curr_col] = mat2[(TILE_WIDTH * m + curr_row) * mat_2_width + dest_column];
        __syncthreads();

        for (int i{0}; i < TILE_WIDTH; ++i) {
            partial += row_arr[curr_row][i] * column_arr[i][curr_col];
        }
        __syncthreads();
    }

    dest_mat[dest_row * mat_2_width + dest_column] = partial;
}

__global__ void gemv(const float * matrix, const float * vector, float * result, const int result_len, const int mat_cols) {

    const unsigned int curr_col{threadIdx.x};
    const unsigned int dest_col{blockIdx.x * TILE_WIDTH + curr_col};

    __shared__ float vector_arr[TILE_WIDTH];

    float partial{0};

    for (int i{0}; i < mat_cols / TILE_WIDTH; ++i) {
        vector_arr[curr_col] = vector[i * TILE_WIDTH + curr_col];
        __syncthreads();

        for (size_t k = 0; k < TILE_WIDTH; ++k) partial += vector_arr[k] * matrix[mat_cols * dest_col + i * TILE_WIDTH + k];
        __syncthreads();
    }

    result[dest_col] += partial;
}

void tiled_gemv(const float * matrix, const float * vector, float * dest_result, size_t const result_len, size_t const vector_len) {

}

void tiled_matmul(const float * mat1, const float * mat2, float * dest_mat, size_t const mat1_rows, size_t const shared_dim, size_t const mat2_columns) {

    float * d_mat1;
    float * d_mat2;
    float * d_dest;

    cudaMalloc(&d_mat1, sizeof(float) * mat1_rows * shared_dim);
    cudaMalloc(&d_mat2, sizeof(float) * mat2_columns * shared_dim);
    cudaMalloc(&d_dest, sizeof(float) * mat1_rows * mat2_columns);

    cudaMemcpy(d_mat1, mat1, sizeof(float) * mat1_rows * shared_dim,  cudaMemcpyHostToDevice);
    cudaMemcpy(d_mat2, mat2, sizeof(float) * mat2_columns * shared_dim,  cudaMemcpyHostToDevice);
    cudaMemcpy(d_dest, dest_mat, sizeof(float) * mat2_columns * mat1_rows,  cudaMemcpyHostToDevice);

    constexpr dim3 block_dim(TILE_WIDTH, TILE_WIDTH);
    dim3 grid_dim(
        mat2_columns / TILE_WIDTH, mat1_rows / TILE_WIDTH);

    tiled_mat_mul<<<grid_dim, block_dim>>>(d_mat1, d_mat2, d_dest, mat1_rows);

    cudaDeviceSynchronize(); // Force flush of printf buffer
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(error) << std::endl;
    }

    cudaMemcpy(dest_mat, d_dest, sizeof(float) * mat2_columns * mat1_rows,  cudaMemcpyDeviceToHost);

    cudaFree(d_mat1);
    cudaFree(d_mat2);
    cudaFree(d_dest);
}

void matmul(const float * mat1, const float * mat2, float * dest_mat, size_t const mat1_rows, size_t const shared_dim, size_t const mat2_columns) {

    float * d_mat1;
    float * d_mat2;
    float * d_dest;

    cudaMalloc(&d_mat1, sizeof(float) * mat1_rows * shared_dim);
    cudaMalloc(&d_mat2, sizeof(float) * mat2_columns * shared_dim);
    cudaMalloc(&d_dest, sizeof(float) * mat1_rows * mat2_columns);

    cudaMemcpy(d_mat1, mat1, sizeof(float) * mat1_rows * shared_dim,  cudaMemcpyHostToDevice);
    cudaMemcpy(d_mat2, mat2, sizeof(float) * mat2_columns * shared_dim,  cudaMemcpyHostToDevice);
    cudaMemcpy(d_dest, dest_mat, sizeof(float) * mat2_columns * mat1_rows,  cudaMemcpyHostToDevice);

    constexpr dim3 block_dim(16, 16);
    dim3 grid_dim(
        std::ceil(static_cast<float>(mat2_columns) / block_dim.x),
        std::ceil(static_cast<float>(mat1_rows) / block_dim.y)
        );

    mat_mul<<<grid_dim, block_dim>>>(d_mat1, d_mat2, d_dest, mat1_rows, shared_dim, mat2_columns);

    cudaDeviceSynchronize(); // Force flush of printf buffer
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(error) << std::endl;
    }

    cudaMemcpy(dest_mat, d_dest, sizeof(float) * mat2_columns * mat1_rows,  cudaMemcpyDeviceToHost);

    cudaFree(d_mat1);
    cudaFree(d_mat2);
    cudaFree(d_dest);
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

void get_device_settings() {
    int dev_count;
    cudaGetDeviceCount( &dev_count);

    // std::cout << dev_count;
    cudaDeviceProp dev_prop;
    for (int i = 0; i < dev_count; ++i) {
        cudaGetDeviceProperties( &dev_prop, i);
        std::cout << "warp size: " <<  dev_prop.warpSize << std::endl;
        std::cout << "max threads per block: " <<  dev_prop.maxThreadsPerBlock << std::endl;
        std::cout << "shared memory per block: " <<  dev_prop.sharedMemPerBlock << std::endl;
        std::cout << "shared memory per multiprocessor: " <<  dev_prop.sharedMemPerMultiprocessor << std::endl;
        std::cout << "total global memory: " <<  dev_prop.totalGlobalMem << std::endl;
        std::cout << "multiprocessor count: " <<  dev_prop.multiProcessorCount << std::endl;
        std::cout << "threads per multiprocessor: " <<  dev_prop.maxThreadsPerMultiProcessor << std::endl;
        std::cout << "device name: " <<  dev_prop.name << std::endl;
        std::cout << "version: " << dev_prop.major << "." << dev_prop.minor << std::endl;
        // decide if device has sufficient resources and capabilities
    }
}
