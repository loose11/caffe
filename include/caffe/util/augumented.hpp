#ifndef CAFFE_UTIL_AUGUMENTED_H_
#define CAFFE_UTIL_AUGUMENTED_H_

#include <vector>
#include <string>

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

namespace caffe {
	
template <typename Dtype>
std::vector<int> aug_load_bounding_box(std::string image_filename);

template <typename Dtype>
std::vector<cv::Mat> aug_create_rotated_images(cv::Mat source, std::vector<int>, int num_rotations, int min_angle, int max_angle);

}

#endif  // CAFFE_UTIL_AUGUMENTED_H_