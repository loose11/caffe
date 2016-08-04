#include "caffe/util/augumented.hpp"

namespace caffe {
	
// Loads bounding boxes from file and returns it as an vector.
template <typename Dtype>
std::vector<int> aug_load_bounding_box(std::string image_filename){
		return NULL;
}
	
// Rotates a image based on the min and max angle and there parameters.
template <typename Dtype>
std::vector<cv::Mat> aug_create_rotated_images(cv::Mat source, std::vector<int>, int num_rotations, int min_angle, int max_angle){
		return NULL;
}



}