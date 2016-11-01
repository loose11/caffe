#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>

#include <fstream>  // NOLINT(readability/streams)
#include <iostream>  // NOLINT(readability/streams)
#include <string>
#include <utility>
#include <vector>

#include "caffe/data_transformer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/layers/image_data_layer.hpp"
#include "caffe/layers/augmented_data_layer.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"
#include "caffe/util/augmented.hpp"

using namespace cv;
using namespace std;

namespace caffe {

template <typename Dtype>
AugmentedDataLayer<Dtype>::~AugmentedDataLayer<Dtype>() {
  this->StopInternalThread();
}

template <typename Dtype>
void AugmentedDataLayer<Dtype>::GenerateBox(string line, int position){
    /*
    * Generating the path to the file of the bounding box description,
    * and load the postion related bounding box.
    */
    std::string ref_box_file = get_ref_box(line);
    bounding_box = aug_load_bounding_box(ref_box_file, position);
}

template <typename Dtype>
void AugmentedDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  
  // Get the preferred output size of the data.
  const int new_height = this->layer_param_.image_data_param().new_height();
  const int new_width  = this->layer_param_.image_data_param().new_width();
  string root_folder = this->layer_param_.image_data_param().root_folder();

  // Load only the num_rotations, because here is it only needed to infer the blob shape
  AugmentedDataParameter aug_data_param = this->layer_param_.augmented_param();
  const int num_rotations_img = aug_data_param.num_rotations_img();

  const float variance = aug_data_param.variance();
  const float expectation = aug_data_param.expectation();

  CHECK(variance >= 0) << "Variance could not be smaller than 0."

  CHECK((new_height == 0 && new_width == 0) ||
      (new_height > 0 && new_width > 0)) << "Current implementation requires "
      "new_height and new_width to be set at the same time.";
  // Read the file with filenames and labels
  const string& source = this->layer_param_.image_data_param().source();
  LOG(INFO) << "Opening file " << source;
  std::ifstream infile(source.c_str());
  string line;
  size_t pos;

  while (std::getline(infile, line)) {
    pos = line.find_last_of(' ');

    // Get all labels of a specific image corresponding to the bounding boxes
    labels = aug_load_labels(get_ref_box(line));

    for(int i = 0; i < labels.size(); i++){
      for(int j = 0; j < num_rotations_img; j++){
              lines_.push_back(std::make_pair(line.substr(0, pos), labels.at(i)));
      }
    }
  }

  CHECK(!lines_.empty()) << "File is empty";

  LOG(INFO) << "A total of " << lines_.size() << " images.";

  lines_id_ = 0;

  // Read the first image, and use it to initialize the top blob.
  this->GenerateBox(lines_[lines_id_].first, 0);
  labels = aug_load_labels(get_ref_box(lines_[lines_id_].first));
  cv::Mat cv_img_origin = cv::imread(root_folder + lines_[lines_id_].first, CV_LOAD_IMAGE_COLOR);
  std::vector<cv::Mat> augmented_images = aug_create_rotated_images(cv_img_origin, bounding_box, num_rotations_img, 1.);

  cv::Mat resized_image = resize_image(augmented_images.at(0), new_width, new_height);

  CHECK(resized_image.data) << "Could not load " << lines_[lines_id_].first;

  // Use data_transformer to infer the expected blob shape from a cv_image.
  vector<int> top_shape = this->data_transformer_->InferBlobShape(resized_image);
  this->transformed_data_.Reshape(top_shape);
  // Reshape prefetch_data and top[0] according to the batch_size.
  const int batch_size = this->layer_param_.image_data_param().batch_size();
  CHECK_GT(batch_size, 0) << "Positive batch size required";
  top_shape[0] = batch_size;
  for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
    this->prefetch_[i].data_.Reshape(top_shape);
  }
  top[0]->Reshape(top_shape);

  LOG(INFO) << "output data size: " << top[0]->num() << ","
      << top[0]->channels() << "," << top[0]->height() << ","
      << top[0]->width();
  // label
  vector<int> label_shape(1, batch_size);
  top[1]->Reshape(label_shape);
  for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
    this->prefetch_[i].label_.Reshape(label_shape);
  }
}


// This function is called on prefetch thread
template <typename Dtype>
void AugmentedDataLayer<Dtype>::load_batch(Batch<Dtype>* batch) {
  CPUTimer batch_timer;
  batch_timer.Start();
  double read_time = 0;
  double trans_time = 0;
  CPUTimer timer;
  CHECK(batch->data_.count());
  CHECK(this->transformed_data_.count());
  ImageDataParameter image_data_param = this->layer_param_.image_data_param();
  const int batch_size = image_data_param.batch_size();
  const int new_height = image_data_param.new_height();
  const int new_width = image_data_param.new_width();
  std::string root_folder = image_data_param.root_folder();

  AugmentedDataParameter aug_data_param = this->layer_param_.augmented_param();
  const int num_rotations_img = aug_data_param.num_rotations_img();
  const int min_rotation_angle = aug_data_param.min_rotation_angle();
  const int max_rotation_angle = aug_data_param.max_rotation_angle();
  std::string output_directory = aug_data_param.output_directory();

  // Get the standard deviation for the variance and the mean
  const float s_deviation = aug_data_param.deviation();
  const float mean = aug_data_param.mean();
  const int max_translation = aug_data_param.max_translation();

  // Reshape according to the first image of each batch
  // on single input batches allows for inputs of varying dimension.
  cv::Mat cv_img = cv::imread(root_folder + lines_[lines_id_].first, CV_LOAD_IMAGE_COLOR);
  // Resize only for shape construction
  cv_img = resize_image(cv_img, new_width, new_height);
  CHECK(cv_img.data) << "Could not load " << lines_[lines_id_].first;
  // Use data_transformer to infer the expected blob shape from a cv_img.
  vector<int> top_shape = this->data_transformer_->InferBlobShape(cv_img);
  this->transformed_data_.Reshape(top_shape);
  // Reshape batch according to the batch_size.
  top_shape[0] = batch_size;
  batch->data_.Reshape(top_shape);

  Dtype* prefetch_data = batch->data_.mutable_cpu_data();
  Dtype* prefetch_label = batch->label_.mutable_cpu_data();

  // datum scales
  const int lines_size = lines_.size();
  int box_position = 0;
  int rotations = num_rotations_img;

  // Distribution over the rotation_angle for the later randomization
  boost::random::mt19937 							generator(time(0));
  boost::random::uniform_int_distribution<>  dist(min_rotation_angle, max_rotation_angle);

  for (int item_id = 0; item_id < batch_size; ++item_id) {

    float angle = dist(generator);
    //LOG(INFO) << "Angle: " << angle << " file: " << lines_[lines_id_].first;

    if(lines_id_+1 < lines_.size() && lines_[lines_id_].first == lines_[lines_id_+1].first){
      // consider the rotation in the lines_ structure, because of multiplication
      rotations--;
      if(rotations == 0){
        box_position++;
      }

    }else{
      // Reached a new file, reset all.
      box_position = 0;
      rotations = num_rotations_img;
    }

    // get a blob
    timer.Start();
    CHECK_GT(lines_size, lines_id_);
    cv::Mat cv_img = cv::imread(root_folder + lines_[lines_id_].first, CV_LOAD_IMAGE_COLOR);
    CHECK(cv_img.data) << "Could not load " << lines_[lines_id_].first;
    read_time += timer.MicroSeconds();
    timer.Start();
    // Apply transformations (mirror, crop...) to the image

    cv_img = translate_image(cv_img, 10, 0);
    if(!output_directory.empty()){
      char buffer[300];
      sprintf(buffer, "translate_%s%s_%d_%d.png", output_directory.c_str(), create_raw_name(lines_[lines_id_].first).c_str(), lines_[lines_id_].second, item_id);
      std::string path = buffer;
      cv::imwrite(path, cv_img);
    }

    // Going to find the bounding boxes, which discribe parts of the image
    this->GenerateBox(lines_[lines_id_].first, box_position);
    std::vector<cv::Mat> augmented_images = aug_create_rotated_images(cv_img, bounding_box, num_rotations_img, angle);
    // We take only the first, due a correct seed we get different images
    cv_img = resize_image(augmented_images.at(0), new_width, new_height);

    // Comment in if you want to save the images
    if(!output_directory.empty()){
      char buffer[300];
      sprintf(buffer, "%s%s_%d_%d.png", output_directory.c_str(), create_raw_name(lines_[lines_id_].first).c_str(), lines_[lines_id_].second, item_id);
      std::string path = buffer;
      cv::imwrite(path, resize_image(augmented_images.at(0), new_width, new_height));
    }


    // Send data to upper level
    int offset = batch->data_.offset(item_id);
    this->transformed_data_.set_cpu_data(prefetch_data + offset);
    this->data_transformer_->Transform(cv_img, &(this->transformed_data_));
    trans_time += timer.MicroSeconds();

    prefetch_label[item_id] = lines_[lines_id_].second;
    // go to the next iter
    lines_id_++;
    if (lines_id_ >= lines_size) {
      // We have reached the end. Restart from the first.
      DLOG(INFO) << "Restarting data prefetching from start.";
      lines_id_ = 0;
    }
  }
  batch_timer.Stop();
  DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
  DLOG(INFO) << "     Read time: " << read_time / 1000 << " ms.";
  DLOG(INFO) << "Transform time: " << trans_time / 1000 << " ms.";
}

INSTANTIATE_CLASS(AugmentedDataLayer);
REGISTER_LAYER_CLASS(AugmentedData);

}  // namespace caffe
#endif  // USE_OPENCV
