# Uni Augsburg Augmentation Data Layer
##(by Matthias Lieb)

# Installation
Checkout the release branch and use the normal setup workflow of the caffe framework.

# Usage
Through the constraints of this project, you can configure your Augmentation Layer pretty easy. 
There is one constraint in the Augmentation Layer definition, because you need to specifiy a file
where the absolute paths of the images are saved. 

See definition:

filelist_abs.txt
```
/data/raid_ssd/Datasets/FlickrLogos_v3/train/000001/000001517.png
/data/raid_ssd/Datasets/FlickrLogos_v3/train/000001/000001030.png
/data/raid_ssd/Datasets/FlickrLogos_v3/train/000001/000001657.png
```

000001517.ref_boxes.txt
```
109 168 215 286 4 -1 _m00
```

After the correct file structure, you can configure the Layer. 

```
layer {
  name: "data"
  type: "AugmentedData"
  top: "data"
  top: "label"
  image_data_param {
    source: "/data/raid_ssd/Datasets/FlickrLogos_v3/test/filelist_abs.txt"
    batch_size: 100
    new_height: 128
    new_width: 128
  }
  augmented_param {
    num_rotations_img: 10
    min_rotation_angle: 0
    max_rotation_angle: 5
    output_directory: "/home/liebmatt/images/"
    mean: 0
    deviation: 1
    max_translation: 100
  }
}
````


__num_rotations_img__ : number of augmentations with rotations
__min_roation_angle__ & __max_rotation_angle__ : range of the random rotations
__output_directory__ : if you want to save the augmented data, you can set a output_directory

##Translation with gaussian distribution
__mean__ & __deviation__ : standard definition for the distribution. deviation is to evaluate as the variance
__max_translation__ : restrict the maximum Translation





# Caffe

[![Build Status](https://travis-ci.org/BVLC/caffe.svg?branch=master)](https://travis-ci.org/BVLC/caffe)
[![License](https://img.shields.io/badge/license-BSD-blue.svg)](LICENSE)

Caffe is a deep learning framework made with expression, speed, and modularity in mind.
It is developed by the Berkeley Vision and Learning Center ([BVLC](http://bvlc.eecs.berkeley.edu)) and community contributors.

Check out the [project site](http://caffe.berkeleyvision.org) for all the details like

- [DIY Deep Learning for Vision with Caffe](https://docs.google.com/presentation/d/1UeKXVgRvvxg9OUdh_UiC5G71UMscNPlvArsWER41PsU/edit#slide=id.p)
- [Tutorial Documentation](http://caffe.berkeleyvision.org/tutorial/)
- [BVLC reference models](http://caffe.berkeleyvision.org/model_zoo.html) and the [community model zoo](https://github.com/BVLC/caffe/wiki/Model-Zoo)
- [Installation instructions](http://caffe.berkeleyvision.org/installation.html)

and step-by-step examples.

[![Join the chat at https://gitter.im/BVLC/caffe](https://badges.gitter.im/Join%20Chat.svg)](https://gitter.im/BVLC/caffe?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)

Please join the [caffe-users group](https://groups.google.com/forum/#!forum/caffe-users) or [gitter chat](https://gitter.im/BVLC/caffe) to ask questions and talk about methods and models.
Framework development discussions and thorough bug reports are collected on [Issues](https://github.com/BVLC/caffe/issues).

Happy brewing!

## License and Citation

Caffe is released under the [BSD 2-Clause license](https://github.com/BVLC/caffe/blob/master/LICENSE).
The BVLC reference models are released for unrestricted use.

Please cite Caffe in your publications if it helps your research:

    @article{jia2014caffe,
      Author = {Jia, Yangqing and Shelhamer, Evan and Donahue, Jeff and Karayev, Sergey and Long, Jonathan and Girshick, Ross and Guadarrama, Sergio and Darrell, Trevor},
      Journal = {arXiv preprint arXiv:1408.5093},
      Title = {Caffe: Convolutional Architecture for Fast Feature Embedding},
      Year = {2014}
    }
