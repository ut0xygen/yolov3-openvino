#!/bin/bash
. ~/openvino_tf1/bin/activate

# Convert to TensorFlow format model from YOLOv3 weights.
cd ~/repos/tensorflow-yolo-v3/

python ./convert_weights_pb.py \
--data_format NHWC \
--output_graph ~/model_tf1.pb \
--weights_file ./weights/yolov3.weights \
--class_names ./weights/coco.names

# python ./convert_weights.py \
# --data_format NHWC \
# --ckpt_file ~/model_tf1/saved_model \
# --weights_file ./weights/yolov3.weights \
# --class_names ./weights/coco.names

# Convert to IR format.\model.
cd ~
mo \
--input_model ~/model_tf1.pb \
--transformations_config ~/openvino_tf1/lib/python3.7/site-packages/mo/extensions/front/tf/yolo_v3.json \
--input_shape [1,416,416,3] \
--data_type=FP32 \
--output_dir ~/model_tf1_ir


# Quantization.
#  > https://docs.openvino.ai/latest/pot_compression_cli_README.html
#  > https://docs.openvino.ai/2020.3/_tools_accuracy_checker_accuracy_checker_data_readers_README.html
cd ~
pot \
--quantize accuracy_aware  \
--model ~/model/model.xml \
--weights ~/model/model.bin \
--name yolov3 \
--ac-config ~/repos/tensorflow-yolo-v3/lib/python3.7/site-packages/open_model_zoo/model_tools/models/public/yolo-v3-tf/accuracy-check.yml
