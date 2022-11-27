Tutorial
  0. Setup.
    wget https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names -P ./weights
    wget https://pjreddie.com/media/files/yolov3.weights -P ./weights
    wget https://pjreddie.com/media/files/yolov3-tiny.weights -P ./weights

  1. Generate model (TensorFlow-format) with darknet-format weight.

    [TensorFlow 1.x]
      git clone https://github.com/mystic123/tensorflow-yolo-v3.git ./repos/tensorflow-yolo-v3

      ##### Normal Model #####
      # python ./repos/tensorflow-yolo-v3/convert_weights_pb.py \
      # --weights_file ./weights/yolov3.weights                 \
      # --class_names  ./weights/coco.name                      \
      # --output_graph ./converted/model_tf.pb

      ##### Tiny Model #####
      python ./repos/tensorflow-yolo-v3/convert_weights_pb.py \
      --weights_file ./weights/yolov3-tiny.weights            \
      --class_names  ./weights/coco.name                      \
      --output_graph ./converted/model_tf.pb                  \
      --tiny

    [TensorFlow 2.x]
      git clone https://github.com/zzh8829/yolov3-tf2.git ./repos/yolov3-tf2

      ##### Normal Model #####
      python ./repos/yolov3-tf2/convert.py    \
      --weights ./weights/yolov3.weights      \
      --num_classes 80                        \
      --output ./converted/model_tf

      ##### Tiny Model #####
      python        ./repos/yolov3-tf2/convert.py \
      --weights     ./weights/yolov3-tiny.weights \
      --num_classes 80                            \
      --output      ./converted/model_tf          \
      --tiny

    [TensroFlow 2.x by Refactored Soruces]
      # Based: https://github.com/zzh8829/yolov3-tf2
      # Specify --checkpoint to save in checkpoint format.

      ##### Normal Model #####
      python ./sources/convert.py                  \
      --weights_file ./weights/yolov3-tiny.weights \
      --out_file     ./converted/model_tf.pb

      # Test.
      python ./sources/test.py                       \
      --ckpt_file    ./converted/model_tf_ckpt/model \
      --classes_file ./weights/coco.names            \
      --img_file     ${IMAGE_PATH}

      ##### Tiny Model #####
      python ./sources/convert.py                  \
      --weights_file ./weights/yolov3-tiny.weights \
      --out_file     ./converted/model_tf.pb       \
      --tiny

      # Test.
      python ./sources/test.py                       \
      --ckpt_file    ./converted/model_tf_ckpt/model \
      --classes_file ./weights/coco.names            \
      --img_file     ${IMAGE_PATH}                   \
      --tiny

  2. Convert model from TensorFlow-format to OpenVINO-format(IR).
    mkdir ./converted/model_ir

    # Only specify --batch option, if input-shape is defined.
    # or
    # Specify --input_shape option without --batch option, if input-shape is not defined.

    [GraphDef]
      mo                                        \
      --framework       tf                      \
      --input_model     ./converted/model_tf.pb \
      --input_shape     [1,416,416,3]           \
      --data_type       FP32                    \
      --model_name      model_ir_fp32           \
      --output_dir      ./converted/model_ir    \
      --progress

    [SavedModel]
      mo                                        \
      --framework       tf                      \
      --saved_model_dir ./converted/model_tf    \
      --input_shape     [1,416,416,3]           \
      --data_type       FP32                    \
      --model_name      model_ir_fp32           \
      --output_dir      ./converted/model_ir    \
      --progress

  3. Inference.
      python ./sources/inference_openvino.py                  ^
      --model_xml_file ./converted/model_ir/model_ir_fp32.xml ^
      --model_bin_file ./converted/model_ir/model_ir_fp32.bin ^
      --classes_file   ./weights/coco.names                   ^
      --img_file       ${IMAGE_PATH}


Reference
  GitHub Repositories
    Darknet Original
      https://github.com/pjreddie/darknet

    Darknet AlexeyAB
      https://github.com/AlexeyAB/darknet

    For TensorFlow 1.x
      https://github.com/david8862/keras-YOLOv3-model-set
      https://github.com/mystic123/tensorflow-yolo-v3

    For TensorFlow 2.x
      https://github.com/zzh8829/yolov3-tf2

  PyPI
    OpenVINO
      https://pypi.org/project/openvino/
      https://pypi.org/project/openvino-dev/

  Others
    Darknet Pretrained Weights
      https://pjreddie.com/media/files/yolov3.weights
      https://pjreddie.com/media/files/yolov3-tiny.weights
      https://pjreddie.com/media/files/yolov3-spp.weights

    Dataset
      https://cocodataset.org/
      http://images.cocodataset.org/zips/train2014.zip
      http://images.cocodataset.org/zips/val2014.zip
      https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names
