# Faster-RCNN

This is an experimental Tensorflow implementation of Faster RCNN - a convnet for object detection with a region proposal network.
For details about R-CNN please refer to the paper [Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks](http://arxiv.org/pdf/1506.01497v3.pdf) by Shaoqing Ren, Kaiming He, Ross Girshick, Jian Sun.


## Requirements:

* Tensorflow
* Numpy
* Opencv (>= 3.0)
* Scipy
* Image
* Matplotlib
* Pyyaml
* g++ (< 5.0)

## Requirements: hardware

### Installation

Build Lib
```bash
cd lib
make
```

If you receive an error related to "nsync_cv.h", replace it with the following code:

```c++
/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.
  
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
  
http://www.apache.org/licenses/LICENSE-2.0
  
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
  
#ifndef TENSORFLOW_PLATFORM_DEFAULT_MUTEX_H_
#define TENSORFLOW_PLATFORM_DEFAULT_MUTEX_H_
  
// IWYU pragma: private, include "third_party/tensorflow/core/platform/mutex.h"
// IWYU pragma: friend third_party/tensorflow/core/platform/mutex.h
  
#include <chrono>
#include <condition_variable>
#include <mutex>
#include "tensorflow/core/platform/thread_annotations.h"
namespace tensorflow {
  
#undef mutex_lock
  
enum LinkerInitialized { LINKER_INITIALIZED };
  
// A class that wraps around the std::mutex implementation, only adding an
// additional LinkerInitialized constructor interface.
class LOCKABLE mutex : public std::mutex {
public:
mutex() {}
// The default implementation of std::mutex is safe to use after the linker
// initializations
explicit mutex(LinkerInitialized x) {}
  
void lock() ACQUIRE() { std::mutex::lock(); }
bool try_lock() EXCLUSIVE_TRYLOCK_FUNCTION(true) {
return std::mutex::try_lock();
};
void unlock() RELEASE() { std::mutex::unlock(); }
};
  
class SCOPED_LOCKABLE mutex_lock : public std::unique_lock<std::mutex> {
public:
mutex_lock(class mutex& m) ACQUIRE(m) : std::unique_lock<std::mutex>(m) {}
mutex_lock(class mutex& m, std::try_to_lock_t t) ACQUIRE(m)
: std::unique_lock<std::mutex>(m, t) {}
mutex_lock(mutex_lock&& ml) noexcept
: std::unique_lock<std::mutex>(std::move(ml)) {}
~mutex_lock() RELEASE() {}
};
  
// Catch bug where variable name is omitted, e.g. mutex_lock (mu);
#define mutex_lock(x) static_assert(0, "mutex_lock_decl_missing_var_name");
  
using std::condition_variable;
  
inline ConditionResult WaitForMilliseconds(mutex_lock* mu,
condition_variable* cv, int64 ms) {
std::cv_status s = cv->wait_for(*mu, std::chrono::milliseconds(ms));
return (s == std::cv_status::timeout) ? kCond_Timeout : kCond_MaybeNotified;
}
  
}  // namespace tensorflow
  
#endif  // TENSORFLOW_PLATFORM_DEFAULT_MUTEX_H_
```

When you launch the script, if you receive an error like:
```bash
Faster-RCNN_TF/tools/../lib/roi_pooling_layer/roi_pooling.so: undefined symbol: _Z22ROIPoolBackwardLaucherPKffiiiiiiiS0_PfPKiRKN5Eigen9GpuDeviceE
```

You have to use g++ < 5.0 in the file "make.sh"

If you have this error:
```Shell
ImportError: No module named gpu_nms
```
Please edit the file <i>setup.py</i> inside the <i>lib</i> folder and change:
```Shell
cudaconfig = {'home':home, 'nvcc':nvcc,
'include': pjoin(home, 'include'),
'lib64': pjoin(home, 'lib')}
```
with:
```Shell
cudaconfig = {'home':home, 'nvcc':nvcc,
'include': pjoin(home, 'include'),
'lib64': pjoin(home, 'lib64')}
```

If your compiler does not find CUDA, change the line <i>CUDA_PATH</i> inside the file make.sh  inside the <i>lib</i> folder with your own path as well as the <i>ARCH</i> line with your own architecture.

## Training/Testing

Download the training, validation, test data and VOCdevkit

```Shell
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCdevkit_08-Jun-2007.tar
```

Extract all of these tars into one directory named `VOCdevkit`

```Shell
tar xvf VOCtrainval_06-Nov-2007.tar
tar xvf VOCtest_06-Nov-2007.tar
tar xvf VOCdevkit_08-Jun-2007.tar
```

Create symlinks for the PASCAL VOC dataset

```Shell
cd $FRCN_ROOT/data
ln -s $VOCdevkit VOCdevkit2007
```

Download pre-trained  model: <a href="https://drive.google.com/open?id=1tSECsv2gnwo-S-xXR7VKANoAvY11MO8p" target="_blank">here</a>.

Move the model to the folder <i>data/pretrain_model</i>.

To run the training/testing you have to run the script faster_rcnn_checkout.sh from the root directiory Faster-RCNN
```bash
./experiments/scripts/faster_rcnn_pascal_voc.sh $DEVICE $DEVICE_ID VGG16 checkout
```
Example:
```bash
./experiments/scripts/faster_rcnn_pascal_voc.sh gpu 0 VGG16 checkout
```
As default the exposure and the flipping are set to True, if you want to disable/enable them, you have to edit the file config.py inside the folder lib/fast_rcnn and set the variables to disable to false/true:
```bash

...

# Use horizontally-flipped images during training?
__C.TRAIN.USE_FLIPPED = True

# Use changing in exposure
__C.TRAIN.USE_EXPOSURE = True

...

```

<strong>!!!!!IMPORTANT: </strong> if you test several configuration please delete each time the folder "cache" inside "data"

## Demo

To run the demo
```Shell
cd $FRCN_ROOT
python ./tools/demo.py --model model_path
```

<strong>IMPORTANT: </strong> I suggest to train again the network starting from this model.
   
   






