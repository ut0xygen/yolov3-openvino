#!/bin/bash
PATH_CD=`dirname ${0}`
cd ${PATH_CD}

##### Clone repos.
mkdir ./repos; cd ./repos
git clone https://github.com/AlexeyAB/darknet.git
git clone https://github.com/david8862/keras-YOLOv3-model-set.git
git clone https://github.com/mystic123/tensorflow-yolo-v3.git
git clone https://github.com/zzh8829/yolov3-tf2.git
cd ${PATH_CD}

##### Download weights.
mkdir ./weights; cd ./weights
wget https://pjreddie.com/media/files/yolov3.weights
wget https://pjreddie.com/media/files/yolov3-tiny.weights
wget https://pjreddie.com/media/files/yolov3-spp.weights
cd ${PATH_CD}

##### Download dataset
mkdir ./dataset; cd ./dataset
wget https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names
wget http://images.cocodataset.org/zips/train2014.zip
wget http://images.cocodataset.org/zips/val2014.zip
