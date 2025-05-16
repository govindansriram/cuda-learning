//
// Created by sriram on 3/1/25.
//


#include "api.h"
#include "cuda_runtime.h"
#include <opencv2/opencv.hpp>

int main() {

    // run_double_buffer_test();

    test_global_load();

    // int dev_count;
    // cudaGetDeviceCount( &dev_count);
    //
    // // std::cout << dev_count;
    // cudaDeviceProp dev_prop;
    // for (int i = 0; i < dev_count; ++i) {
    //     cudaGetDeviceProperties( &dev_prop, i);
    //     std::cout << "warp size: " <<  dev_prop.warpSize << std::endl;
    //     std::cout << "max threads per block: " <<  dev_prop.maxThreadsPerBlock << std::endl;
    //     std::cout << "shared memory per block: " <<  dev_prop.sharedMemPerBlock << std::endl;
    //     std::cout << "shared memory per multiprocessor: " <<  dev_prop.sharedMemPerMultiprocessor << std::endl;
    //     std::cout << "total global memory: " <<  dev_prop.totalGlobalMem << std::endl;
    //     std::cout << "multiprocessor count: " <<  dev_prop.multiProcessorCount << std::endl;
    //     std::cout << "threads per multiprocessor: " <<  dev_prop.maxThreadsPerMultiProcessor << std::endl;
    //     std::cout << "device name: " <<  dev_prop.name << std::endl;
    //     std::cout << "version: " << dev_prop.major << "." << dev_prop.minor << std::endl;
    //     // decide if device has sufficient resources and capabilities
    // }
    //
    // // get_device_settings();
    //
    // float mat1[] {
    //     1, 2, 3, 4,
    //     5, 6, 7, 8,
    //     9, 10, 11, 12,
    //     13, 14, 15, 16
    // };
    //
    // float mat2[]{
    //     1, 2,
    //     3, 4,
    //     5, 6,
    //     7, 8
    // };
    //
    // float dest[8]{};
    //
    // tiled_matmul(mat1, mat2, dest, 4, 4, 2);
    //
    // for (float const &num: dest) {
    //     std::cout << num << std::endl;
    // }

    // cv::Mat const img{cv::imread("/home/sriram/Pictures/maxresdefault.jpg")};
    // std::cout << img.rows << std::endl;
    // std::cout << img.cols << std::endl;
    // std::cout << img.step << std::endl;

    // cv::imshow("display the image", img);
    // cv::waitKey(0);  // 0 means wait indefinitely until a key is pressed

    // cv::Mat const ti{transpose_opencv_image(img)};

    // cv::imshow("display the image", ti);
    // cv::waitKey(0);  // 0 means wait indefinitely until a key is pressed
    // // Step 4: Destroy all windows
    // cv::destroyAllWindows();

//     size_t const rows = 3;
//     size_t const columns = 3;
//     unsigned char underlying_data[]{
//         1, 2, 3,
//         4, 5, 6,
//         7, 8, 9,
//         10, 11, 12};
//
//     transpose_wrapper(underlying_data, rows, columns);
//
//     for (unsigned char num: underlying_data) {
//         std::cout << static_cast<int>(num) << std::endl;
//     }
//
//     return 0;
}