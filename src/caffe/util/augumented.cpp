#include "caffe/util/augumented.hpp"

#include <iostream>
#include <stdio.h>
#include <fstream>
#include <string>
#include <iterator>
#include <algorithm>
#include <sstream>
#include <vector>
#include <math.h>
#include <cstdlib>

namespace caffe {
	
// Loads bounding boxes from file and returns it as an vector.
template <typename Dtype>
std::vector< std::vector<int> > aug_load_bounding_box(std::string image_filename){
	std::vector< std::vector<int> > boundingBoxes;
	std::vector<int> boundingBox;

	std::ifstream file(image_filename);
	std::string str;
	std::string word;
	while (std::getline(file, str)){
		std::istringstream iss(str, std::istringstream::in);
		while( iss >> word ){
			if(boundingBox.size() == 4){
				break;
			}
			// atoi instead of stoi because, compiler dont support.
			boundingBox.push_back(std::atoi(word.c_str()));
		}
		boundingBoxes.push_back(boundingBox);
		boundingBox.clear();
	}

	return boundingBoxes;
}
	
// Rotates a image based on the min and max angle and there parameters.
template <typename Dtype>
std::vector<cv::Mat> aug_create_rotated_images(cv::Mat source, std::vector<int>, int num_rotations, int min_angle, int max_angle){
		return NULL;
}
}