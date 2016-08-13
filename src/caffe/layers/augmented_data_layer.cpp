#include "stdint.h"

#include "caffe/layers/augmented_data_layer.hpp"

namespace caffe {

template <typename Dtype>
AUGUMENTEDDataLayer<Dtype>::~AUGUMENTEDDataLayer<Dtype>() { }

// Load data and label from HDF5 filename into the class property blobs.
template <typename Dtype>
void AUGUMENTEDDataLayer<Dtype>::LoadImageFileData(const char* filename) {
	DLOG(INFO) << "Loading image file: " << filename;
	// Loading the image
	cv::Mat image_file = cv::imread(filename);
	if (!image_file.data) {
		LOG(FATAL) << "Failed opening image file: " << filename;
	}

	int top_size = this->layer_param_.top_size();
	image_blobs_.resize(top_size);

	for(int i = 0; i < top_size; ++i){
		image_blobs_[i] = shared_ptr<Blob<Dtype> >(new Blob<Dtype>());
		// 1 have to the batch size
		image_blobs_[i].get()->Reshape(1, image_file.channels(), image_file.rows, image_file.cols);
	}
	
	CHECK_GE(image_blobs_[0]->num_axes(), 1) << "Input must have at least 1 axis.";
	const int num = image_blobs_[0]->shape(0);
	for(int i = 1; i < top_size; ++i){
			CHECK_EQ(image_blobs_[i]->shape(0), num);
	}
	
	// Default to identity permutation.
	data_permutation_.clear();
	data_permutation_.resize(image_blobs_[0]->shape(0));
	for (int i = 0; i < image_blobs_[0]->shape(0); i++)
		data_permutation_[i] = i;

	DLOG(INFO) << "Successully loaded " << image_blobs_[0]->shape(0) << " rows";

}

template <typename Dtype>
void AUGUMENTEDDataLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
	AugumentedDataParameter const& aug_param = this->layer_param_.augumented_param();
	string const& source = aug_param.source();
	LOG(INFO) << "Loading images from file list: " << source;
	const int batch_size = aug_param.batch_size();
	/*this->num_rotations_img = aug_param.num_rotations_img();
	this->min_rotation_angle = aug_param.min_rotation_angle();
	this->max_rotation_angle = aug_param.max_rotation_angle();*/
	
	// Read all filenames from list into a vector
	aug_filenames_.clear();
	std::ifstream source_file(source.c_str());
	if(source_file.is_open()){
		std::string line;
		while(source_file >> line){
			aug_filenames_.push_back(line);
		}
	}else{
		LOG(FATAL) << "Failed to open source file: " << source;
	}
	source_file.close();
	num_files_ = aug_filenames_.size();
	current_file_ = 0;
	LOG(INFO) << "Number of image files: " << num_files_;
	CHECK_GE(num_files_, 1) << "Must have at least 1 image filename listed in "
    << source;
	
	file_permutation_.clear();
	file_permutation_.resize(num_files_);
	// Default to identity permutation.
	for (int i = 0; i < num_files_; i++) {
		file_permutation_[i] = i;
	}
	LOG(INFO) << "Loading files...";
	LoadImageFileData(aug_filenames_[file_permutation_[current_file_]].c_str());
	current_row_ = 0;
	
	// Reshape blobs
	const int top_size = this->layer_param_.top_size();
	vector<int> top_shape;
	for(int i = 0; i < top_size; ++i){
			top_shape.resize(image_blobs_[i]->num_axes());
			top_shape[0] = batch_size;
			for(int j = 1; j < top_shape.size(); ++j){
				top_shape[j] = image_blobs_[i]->shape(j);
			}
			top[i]->Reshape(top_shape);
	}

}

template <typename Dtype>
void AUGUMENTEDDataLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
	const int batch_size = this->layer_param_.augumented_param().batch_size();
	// TODO current_row ist falsch, oder?
	for(int i = 0; i < batch_size; ++i, ++current_row_){
		// If we reached the batch_siez
		if(current_row_ == image_blobs_[0]->shape(0)){
			if(num_files_ > 1){
				++current_file_;
				if(current_file_ == num_files_){
						current_file_ = 0;
						DLOG(INFO) << "Looping around to first image.";
				}
				LoadImageFileData(aug_filenames_[file_permutation_[current_file_]].c_str());
			}
			current_row_ = 0;
		}
		for(int j = 0; j < this->layer_param_.top_size(); ++j){
			int data_dim = top[j]->count() / top[j]->shape(0);
			DLOG(INFO) << "Dimension: " << data_dim;
			caffe_copy(data_dim,
				&image_blobs_[j]->cpu_data()[data_permutation_[current_row_]
					* data_dim], &top[j]->mutable_cpu_data()[i * data_dim]);
		}
	}
	
}

#ifdef CPU_ONLY
STUB_GPU_FORWARD(AUGUMENTEDDataLayer, Forward);
#endif

INSTANTIATE_CLASS(AUGUMENTEDDataLayer);
REGISTER_LAYER_CLASS(AUGUMENTEDData);

}  // namespace caffe
