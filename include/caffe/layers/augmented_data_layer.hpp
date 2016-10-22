#ifndef CAFFE_AUGMENTED_DATA_LAYER_HPP_
#define CAFFE_AUGMENTED_DATA_LAYER_HPP_

#include <string>
#include <utility>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/data_transformer.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/augmented.hpp"


namespace caffe {

/**
 * @brief Provides data to the Net from image files.
 *
 *
 */
template <typename Dtype>
class AugmentedDataLayer : public BasePrefetchingDataLayer<Dtype> {
 public:
  explicit AugmentedDataLayer(const LayerParameter& param)
      : BasePrefetchingDataLayer<Dtype>(param) {}
  virtual ~AugmentedDataLayer();
  virtual void GenerateBox(std::string line, int position);
  virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "AugmentedData"; }
  virtual inline int ExactNumBottomBlobs() const { return 0; }
  virtual inline int ExactNumTopBlobs() const { return 2; }

 protected:
  shared_ptr<Caffe::RNG> prefetch_rng_;
  virtual void load_batch(Batch<Dtype>* batch);

  vector<std::pair<std::string, int> > lines_;
  int lines_id_;

  std::vector<int> bounding_box;
  std::vector<int> labels;
};


}  // namespace caffe

#endif  // CAFFE_AUGMENTED_DATA_LAYER_HPP_
