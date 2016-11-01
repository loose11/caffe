#include "caffe/util/augmented.hpp"
#include "caffe/common.hpp"
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
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_int_distribution.hpp>
#include <boost/math/distributions/normal.hpp>

namespace caffe
{

// Loads a specific bounding box from file and returns it as an vector.
std::vector<int> aug_load_bounding_box(std::string image_path, int position)
{
    std::vector<int> boundingBox;

    std::ifstream file(image_path.c_str());
    std::string str;
    std::string word;

    int current_line_ = 0;
    while (std::getline(file, str))
    {
	if (current_line_ == position)
	{
	    std::istringstream iss(str, std::istringstream::in);
	    while (iss >> word)
	    {
		if (boundingBox.size() == 4)
		{
		    break;
		}
		// atoi instead of stoi because, compiler dont support.
		boundingBox.push_back(std::atoi(word.c_str()));
	    }
	}
	current_line_++;
    }

    return boundingBox;
}

// Return the labels for each bounding box
std::vector<int> aug_load_labels(std::string image_path)
{
    std::vector<int> labels;

    std::ifstream file(image_path.c_str());
    std::string str;
    std::string word;

    while (std::getline(file, str))
    {
	std::istringstream iss(str, std::istringstream::in);
	int counter = 1;

	while (iss >> word)
	{
	    counter++;
	    // Skip the bounding boxes
	    if (counter == 6)
	    {
		// atoi instead of stoi because, compiler dont support.
		labels.push_back(std::atoi(word.c_str()));
		break;
	    }
	}
    }

    return labels;
}

// Rotates a image based on the min and max angle and there parameters.

std::vector<cv::Mat> aug_create_rotated_images(cv::Mat source, std::vector<int> boundingBox, int num_rotations, float angle)
{

    std::vector<cv::Mat> images;
    //float randomNum = static_cast <float> (rand()) / (static_cast <float> (RAND_MAX));
    //angle = min_angle + randomNum * (max_angle - min_angle);
    //LOG(WARNING) << "!!!Angle: " << angle;
    int ltX = boundingBox.at(0);
    int ltY = boundingBox.at(1);
    int rbX = boundingBox.at(2);
    int rbY = boundingBox.at(3);
    cv::Point2f midi = (cv::Point2f(ltX, ltY) + cv::Point2f(rbX, rbY)) * 0.5;

    cv::Mat M, rotated, cropped;
    cv::RotatedRect rRect = cv::RotatedRect(midi, cv::Size2i(boundingBox.at(2) - boundingBox.at(0), boundingBox.at(3) - boundingBox.at(1)), angle);

    cv::Size rect_size = rRect.size;

    M = cv::getRotationMatrix2D(rRect.center, angle, 1.0);
    cv::warpAffine(source, rotated, M, source.size(), cv::INTER_CUBIC);
    cv::getRectSubPix(rotated, rect_size, rRect.center, cropped);

    images.push_back(cropped);

    return images;
}

// Calculate the file path of the reference file
std::string get_ref_box(std::string image_path)
{

    size_t found;
    found = image_path.find_last_of("/\\");

    std::string folder = image_path.substr(0, found);
    std::string filename = image_path.substr(found + 1);
    size_t lastindex = filename.find_last_of(".");
    std::string rawname = filename.substr(0, lastindex);
    std::string ref_box_file = image_path.substr(0, found) + '/' + rawname + ".ref_boxes.txt";

    return ref_box_file;
}

// Create raw name out of the image_path
std::string create_raw_name(std::string image_path)
{
    size_t found;
    found = image_path.find_last_of("/\\");

    std::string folder = image_path.substr(0, found);
    std::string filename = image_path.substr(found + 1);
    size_t lastindex = filename.find_last_of(".");
    std::string rawname = filename.substr(0, lastindex);

    return rawname;
}

// Resizes images with specificed width and height
cv::Mat resize_image(cv::Mat cv_img_origin, int width, int height)
{
    cv::Mat cv_img;
    cv::resize(cv_img_origin, cv_img, cv::Size(width, height));
    return cv_img;
}

cv::Mat translate_image(cv::Mat cv_img, int offset_x, int offset_y){
	cv::Mat trans_mat = (cv::Mat_<double>(2,3) << 1, 0, offset_x, 0, 1, offset_y);
	cv::warpAffine(cv_img, cv_image, trans_mat, img.size());
	return trans_mat;
}
}
