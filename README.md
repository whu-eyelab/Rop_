# Introduction
We develop a robust intelligent system based on deep learning to automatically classify the severity of retinopathy of prematurity from fundus images and detect the stage of retinopathy of prematurity and presence of plus disease, in order to enable automated diagnosis and further referral or observation. A 101-layer convolutional neural network (ResNet) and a faster region-based convolutional neural network (Faster-RCNN) were trained for image classification and identification. This page shows the original code of the paper “Automated identification of retinopathy of prematurity by image-based deep learning”.

# Howto
1. Get ROP dataset
2. Clone Tensorflow Models repository

>>>
git clone https://github.com/tensorflow/models
>>>

3. Clone this repository and copy source code to models/research path
4. run object_detection_train_type_7.sh
