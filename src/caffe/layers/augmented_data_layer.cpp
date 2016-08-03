#include "stdint.h"

#include "caffe/layers/augmented_data_layer.hpp"

namespace caffe {

template <typename Dtype>
AUGUMENTEDDataLayer<Dtype>::~AUGUMENTEDDataLayer<Dtype>() { }

// Load data and label from HDF5 filename into the class property blobs.
template <typename Dtype>
void AUGUMENTEDDataLayer<Dtype>::LoadHDF5FileData(const char* filename) {
  /*DLOG(INFO) << "Loading HDF5 file: " << filename;
  hid_t file_id = H5Fopen(filename, H5F_ACC_RDONLY, H5P_DEFAULT);
  if (file_id < 0) {
    LOG(FATAL) << "Failed opening HDF5 file: " << filename;
  }

  int top_size = this->layer_param_.top_size();
  hdf_blobs_.resize(top_size);

  const int MIN_DATA_DIM = 1;
  const int MAX_DATA_DIM = INT_MAX;

  for (int i = 0; i < top_size; ++i) {
    hdf_blobs_[i] = shared_ptr<Blob<Dtype> >(new Blob<Dtype>());
    hdf5_load_nd_dataset(file_id, this->layer_param_.top(i).c_str(),
        MIN_DATA_DIM, MAX_DATA_DIM, hdf_blobs_[i].get());
  }

  herr_t status = H5Fclose(file_id);
  CHECK_GE(status, 0) << "Failed to close HDF5 file: " << filename;

  // MinTopBlobs==1 guarantees at least one top blob
  CHECK_GE(hdf_blobs_[0]->num_axes(), 1) << "Input must have at least 1 axis.";
  const int num = hdf_blobs_[0]->shape(0);
  for (int i = 1; i < top_size; ++i) {
    CHECK_EQ(hdf_blobs_[i]->shape(0), num);
  }
  // Default to identity permutation.
  data_permutation_.clear();
  data_permutation_.resize(hdf_blobs_[0]->shape(0));
  for (int i = 0; i < hdf_blobs_[0]->shape(0); i++)
    data_permutation_[i] = i;

  // Shuffle if needed.
  if (this->layer_param_.hdf5_data_param().shuffle()) {
    std::random_shuffle(data_permutation_.begin(), data_permutation_.end());
    DLOG(INFO) << "Successully loaded " << hdf_blobs_[0]->shape(0)
               << " rows (shuffled)";
  } else {
    DLOG(INFO) << "Successully loaded " << hdf_blobs_[0]->shape(0) << " rows";
  }*/
}

template <typename Dtype>
void AUGUMENTEDDataLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
	AugumentedDataParameter const& aug_param = this->layer_param_.augumented_param();
	string const& source = aug_param.source();
	uint32 const batch_size = aug_param.batch_size();
	uint32 const num_rotations_img = aug_param.num_rotations_img();
	unit32 const min_rotation_angle = aug_param.min_rotation_angle();
	unit32 const max_rotation_angle = aug_param.max_rotation_angle();
}

template <typename Dtype>
void AUGUMENTEDDataLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  
}

#ifdef CPU_ONLY
STUB_GPU_FORWARD(AUGUMENTEDDataLayer, Forward);
#endif

INSTANTIATE_CLASS(AUGUMENTEDDataLayer);
REGISTER_LAYER_CLASS(AUGUMENTEDData);

}  // namespace caffe
