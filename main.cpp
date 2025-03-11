//
// Created by sriram on 3/1/25.
//


#include "api.h"
#include <opencv2/opencv.hpp>

int main() {
    cv::Mat const img{cv::imread("/home/sriram/Pictures/maxresdefault.jpg")};
    std::cout << img.rows << std::endl;
    std::cout << img.cols << std::endl;
    std::cout << img.step << std::endl;

    // cv::imshow("display the image", img);
    // cv::waitKey(0);  // 0 means wait indefinitely until a key is pressed

    cv::Mat const ti{transpose_opencv_image(img)};

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