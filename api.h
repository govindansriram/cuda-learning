//
// Created by sriram on 3/4/25.
//

#ifndef API_H
#define API_H

#include <cstddef>
#include <stdexcept>
#include <cmath>
#include <opencv2/opencv.hpp>

template<typename dtype>
struct scale_factors {
    static constexpr size_t scale = 0;
};

template<>
struct scale_factors <int>{
    static constexpr size_t scale = 10000;
};

template<>
struct scale_factors <float>{
    static constexpr size_t scale = 1;
};

template <typename Dtype>
class PseudoRNG {
    size_t scale_factor{1};

public:

    PseudoRNG() {
        if (size_t const &type_factor = scale_factors<Dtype>::scale) scale_factor *= type_factor;
        else throw std::runtime_error("unsupported type");
    }

    Dtype pseudo_random_number(int const num) {
        switch (num % 4) {
            case 0:
                return static_cast<Dtype>(std::tan(num) * scale_factor);
            case 1:
                return static_cast<Dtype>(std::cos(num) * scale_factor);
            case 2:
                return static_cast<Dtype>(std::sin(num) * scale_factor);
            case 3:
                return static_cast<Dtype>(std::exp(num) * scale_factor);
            default:
                throw std::runtime_error("invalid remainder");
        }
    }
};


void vadd_wrapper();
void picture_kernel_wrapper();
void transpose_wrapper(unsigned char * data, size_t rows, size_t columns);
cv::Mat transpose_opencv_image(cv::Mat const &image);

#endif //API_H
