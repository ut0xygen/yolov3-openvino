import argparse
import cv2
import numpy as np

import tensorflow as tf
from tensorflow.keras.models    import  load_model
from yolo                       import  YOLOV3, YOLOV3Tiny
from utils                      import  draw_outputs


def main(ckpt_file      :str,
         classes_file   :str,
         img_file       :str,
         width          :int,
         height         :int,
         channels       :int,
         tiny           :bool):
    if not tiny:
        ANCHORS         =   np.array([
            [10, 13 ], [16 , 30 ], [33 , 23 ],
            [30, 61 ], [62 , 45 ], [59 , 119],
            [116, 90], [156, 198], [373, 326],
        ], np.float32)
        MASKS           =   np.array([
            [6, 7, 8],
            [3, 4, 5],
            [0, 1, 2],
        ])

        modelGenerator  =   YOLOV3
    else:
        ANCHORS         =   np.array([
            [10, 14], [23 , 27 ], [37,  58 ],
            [81, 82], [135, 169], [344, 319],
        ], np.float32)
        MASKS           =   np.array([
            [3, 4, 5],
            [0, 1, 2],
        ])

        modelGenerator  =   YOLOV3Tiny

    # Setup GPU.
    LIST_GPU = tf.config.experimental.list_physical_devices('GPU')
    if LIST_GPU:
        tf.config.experimental.set_memory_growth(LIST_GPU[0], True)

    # Load classes.
    CLASSES =   []
    with open(classes_file, 'r', encoding = 'utf-8') as file:
        for elmLine in file.readlines():
            elmLine = elmLine.strip()
            if not elmLine:
                continue

            CLASSES.append(elmLine)

    # Generate model.
    model           =   modelGenerator(
        width       =   width,
        height      =   height,
        channels    =   channels,
        num_classes =   len(CLASSES),
        anchors     =   ANCHORS,
        masks       =   MASKS
    )

    # Load checkpoint.
    model.load_weights(ckpt_file).expect_partial()

    # Load image.
    with open(img_file, 'rb') as file:
        img =   tf.image.decode_image(file.read(), channels = channels)
    img     =   tf.expand_dims(img, 0)
    img     =   tf.image.resize(img, (width, height)) /255

    # Inference
    boxes, scores, classes, nums = model(img)

    # Draw bouding-boxes.
    img     =   cv2.imread(img_file)
    img     =   draw_outputs(img, (boxes, scores, classes, nums), CLASSES)
    cv2.imwrite('test.png', img)


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()

    # Required options.
    argparser.add_argument(
        '--ckpt_file',
        required    =   True,
        type        =   str,
    )
    argparser.add_argument(
        '--classes_file',
        required    =   True,
        type        =   str
    )
    argparser.add_argument(
        '--img_file',
        required    =   True,
        type        =   str
    )

    # Other options.
    argparser.add_argument(
        '--width',
        required    =   False,
        type        =   int,
        default     =   416
    )
    argparser.add_argument(
        '--height',
        required    =   False,
        type        =   int,
        default     =   416
    )
    argparser.add_argument(
        '--channels',
        required    =   False,
        type        =   int,
        default     =   3
    )
    argparser.add_argument(
        '--tiny',
        required    =   False,
        action      =   'store_true'
    )

    args = argparser.parse_args()

    main(
        ckpt_file       =   args.ckpt_file,
        classes_file    =   args.classes_file,
        img_file        =   args.img_file,
        width           =   args.width,
        height          =   args.height,
        channels        =   args.channels,
        tiny            =   args.tiny
    )
