# KCF tracker in Python

Python implementation of
> [High-Speed Tracking with Kernelized Correlation Filters](http://www.robots.ox.ac.uk/~joao/publications/henriques_tpami2015.pdf)<br>
> J. F. Henriques, R. Caseiro, P. Martins, J. Batista<br>
> TPAMI 2015

It is fix version for python 3 of [KCFpy](https://github.com/uoip/KCFpy).


It is translated from [KCFcpp](https://github.com/joaofaro/KCFcpp) (Authors: Joao Faro, Christian Bailer, Joao F. Henriques), a C++ implementation of Kernelized Correlation Filters. Find more references and code of KCF at http://www.robots.ox.ac.uk/~joao/circulant/

To install dependencies run requirements.sh script.

# KCF Tracker without cnn

## Requirements 
- Python 3
- NumPy
- Numba (needed if you want to use the hog feature)
- OpenCV (ensure that you can `import cv2` in python)

## Execute these lines
Download the sources and execute
```shell
git clone https://github.com/Sshanu/KCFpy.git
cd KCFpy
python run_updated.py
```
It will open the default camera of your computer, you can also open a video
```shell
python run_updated.py -inv test.avi  
```

# KCF Tracker with cnn

## Theano version

### Requirements 
- Python 3
- NumPy
- Numba (needed if you want to use the hog feature)
- OpenCV (ensure that you can `import cv2` in python)
- Theano
- Pickle
- Lasagne

### For KCF Tracker with Vggnet 16
Download pretrained weights from: https://s3.amazonaws.com/lasagne/recipes/pretrained/imagenet/vgg16.pkl
and save in new_vgg16 folder.
Then execute
```shell
python run_cnn.py -inv test.avi -opt Output_Folder -mo cnn
```
For Webcam
```shell
python run_cnn.py -opt Output_Folder -mo cnn
```

For using GPU to compute conv layer :
change line 14 of vgg16.py in new_vgg16 folder
```
from lasagne.layers import Conv2DLayer as ConvLayer
```
to
```
from lasagne.layers.dnn import Conv2DDNNLayer as ConvLayer
```

## Caffe version

### Requirements
- Python 2.7
- NumPy
- OpenCV (ensure that you can `import cv2` in python)
- Caffe

Download the Vgg16 caffe model from: http://www.robots.ox.ac.uk/~vgg/software/very_deep/caffe/VGG_ILSVRC_16_layers.caffemodel
and save in vgg16 folder.
Edit the ground_truth.txt file with original coordinates of image.
Save images in 'img'folder.
Then execute these:
``` shell
python2.7 caffe_run.py -inp img -mo cnn
```


