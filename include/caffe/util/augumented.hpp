#ifndef CAFFE_UTIL_AUGUMENTED_H_
#define CAFFE_UTIL_AUGUMENTED_H_

#include <vector>
#include <string>

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

namespace caffe {

std::vector<int>  aug_load_bounding_box(std::string image_path, int position);
std::vector<int> aug_load_labels(std::string image_path);
std::vector<cv::Mat> aug_create_rotated_images(cv::Mat source, std::vector<int>, int num_rotations, int min_angle, int max_angle);
std::string get_ref_box(std::string image_path);
std::string create_raw_name(std::string image_path);
cv::Mat resize_image(cv::Mat cv_img_origin, int width, int height);
}
#endif  // CAFFE_UTIL_AUGUMENTED_H_