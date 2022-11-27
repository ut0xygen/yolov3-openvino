import os
import argparse
from yolo   import  YOLOV3, YOLOV3Tiny
from utils  import  load_darknet_weights

import numpy as np
import tensorflow as tf
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2


def main(weights_file   :str,
         out_file       :str,
         width          :int,
         height         :int,
         channels       :int,
         num_classes    :int,
         tiny           :bool,
         checkpoint     :bool):
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

    # Generate model.
    model           =   modelGenerator(
        width       =   width,
        height      =   height,
        channels    =   channels,
        num_classes =   num_classes,
        anchors     =   ANCHORS,
        masks       =   MASKS
    )

    # Show summary.
    model.summary()

    # Load and apply weights.
    load_darknet_weights(model, weights_file, tiny)

    # Check sanity.
    model(np.random.random((1, width, height, channels)).astype(np.float32))

    # Save model.
    if not checkpoint:
        if False:
            model.save(out_file)
        else:
            model_  =   tf.function(lambda x: model(x))
            model_  =   model_.get_concrete_function(tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype))
            model_  =   convert_variables_to_constants_v2(model_)
            model_.graph.as_graph_def()

            tf.io.write_graph(
                graph_or_graph_def  =   model_.graph,
                logdir              =   os.path.dirname(out_file),
                name                =   os.path.basename(out_file),
                as_text             =   False
            )
    else:
        model.save_weights(out_file)

    return


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()

    # Required options.
    argparser.add_argument(
        '--weights_file',
        required    =   True,
        type        =   str,
    )
    argparser.add_argument(
        '--out_file',
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
        '--num_classes',
        required    =   False,
        type        =   int,
        default     =   80
    )
    argparser.add_argument(
        '--tiny',
        required    =   False,
        action      =   'store_true'
    )
    argparser.add_argument(
        '--checkpoint',
        required    =   False,
        action      =   'store_true'
    )

    args = argparser.parse_args()

    main(
        weights_file    =   args.weights_file,
        out_file        =   args.out_file,
        width           =   args.width,
        height          =   args.height,
        channels        =   args.channels,
        num_classes     =   args.num_classes,
        tiny            =   args.tiny,
        checkpoint      =   args.checkpoint
    )
