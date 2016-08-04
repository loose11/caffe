#ifndef CAFFE_AUGUMENTED_DATA_LAYERS_HPP_
#define CAFFE_AUGUMENTED_DATA_LAYERS_HPP_

#include <vector>

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <stdio.h>
#include <fstream>
#include <string>
#include <iterator>
#include <algorithm>
#include <sstream>
#include <vector>
#include <math.h>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/base_data_layer.hpp"

namespace caffe {

/**
 * @brief Provides data to the Net with augumented data input.
 *
 */
template <typename Dtype>
class AUGUMENTEDDataLayer : public Layer<Dtype> {
 public:
  explicit AUGUMENTEDDataLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual ~AUGUMENTEDDataLayer();
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  // Data layers should be shared by multiple solvers in parallel
  virtual inline bool ShareInParallel() const { return true; }
  // Data layers have no bottoms, so reshaping is trivial.
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {}

  virtual inline const char* type() const { return "AUGUMENTEDData"; }
  virtual inline int ExactNumBottomBlobs() const { return 0; }
  virtual inline int MinTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  //virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    //  const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {}
  //virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
    //  const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {}
  virtual void LoadImageFileData(const char* filename);

  std::vector<std::string> aug_filenames_;
  unsigned int num_files_;
  unsigned int current_file_;
  unsigned int current_row_;
  std::vector<shared_ptr<Blob<Dtype> > > image_blobs_;
  std::vector<unsigned int> data_permutation_;
  std::vector<unsigned int> file_permutation_;
};

}  // namespace caffe

#endif  // CAFFE_AUGUMENTED_DATA_LAYERS_HPP_
