#!/bin/bash
cd `dirname $0`
docker image build --build-arg UNAME=hoge --build-arg UPASS=. --build-arg GNAME=hoge --build-arg UID=1000 --build-arg GID=1000 --tag ubuntu:20.04_openvino_for_user ./


#!/bin/bash
# Convert to TensorFlow format model from YOLOv3 weights.
cd ~/repos/tensorflow-yolo-v3/
python ./convert_weights.py --data_format NHWC --ckpt_file ./model/saved_model --weights_file ./weights/yolov3.weights --class_names ./weights/coco.names
# python ./convert_weights_pb.py --data_format NHWC --output_graph ./model.pb --weights_file ./weights/yolov3.weights --class_names ./weights/coco.names

# Convert to IR format.\model.
cd ~/openvino_tf1
mo \
--input_model ~/repos/tensorflow-yolo-v3/model.pb \
--transformations_config ./lib/python3.7/site-packages/mo/extensions/front/tf/yolo_v3.json \
--input_shape [1,416,416,3] \
--data_type=FP32 \
--output_dir ~/model

# Quantization.
#  > https://docs.openvino.ai/latest/pot_compression_cli_README.html
#  > https://docs.openvino.ai/2020.3/_tools_accuracy_checker_accuracy_checker_data_readers_README.html
pot \
--quantize accuracy_aware  \
--model ~/model/model.xml \
--weights ~/model/model.bin \
--name yolov3 \
--ac-config ./lib/python3.7/site-packages/open_model_zoo/model_tools/models/public/yolo-v3-tf/accuracy-check.yml
