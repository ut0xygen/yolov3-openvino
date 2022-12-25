import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
import tensorflow as tf

print(f'Tensorflow: {tf.__version__}')
print(f'  > Eager Exeution: {tf.executing_eagerly()}')
# tf.config.run_functions_eagerly(True)
from tensorflow.keras           import  Model
from tensorflow.keras.layers    import  (
    Input,
    Concatenate,

    UpSampling2D,

    Lambda,
)
from models.darknet             import (
    _DarknetConv as DarknetConv,
    Darknet,
    DarknetTiny,
)


# NOTE: (Original)
#   > As tensorflow lite doesn't support tf.size used in tf.meshgrid.
def _meshgrid(n_a   :int,
              n_b   :int):
    return tf.stack([
        # Method A
        # tf.reshape(tf.tile(tf.range(n_a), [n_b]), [n_b, n_a]),
        # tf.reshape(tf.repeat(tf.range(n_b), n_a), [n_b, n_a]),

        # Method B
        tf.tile([tf.range(n_a)], [n_b, 1]),
        tf.repeat(tf.reshape(tf.range(n_b), [n_b, 1]), n_a, 1),
    ], -1)

def _YOLOv3Conv(filters :int,
                name    :str    =   None):
    def __internal(x):
        x_  =   x
        if isinstance(x, (list, tuple)):
            inputs  =   [
                Input(
                    shape       =   x_[0].shape[1:],
                    batch_size  =   x_[0].shape[0]
                ),
                Input(
                    shape       =   x_[1].shape[1:],
                    batch_size  =   x_[1].shape[0]
                ),
            ]

            x       =   inputs[0]
            x       =   DarknetConv(
                filters     =   filters,
                kernel_size =   1
            )(x)
            x       =   UpSampling2D(
                size        =   [2, 2]
            )(x)
            x       =   Concatenate()([x, inputs[1]])
        else:
            inputs  =   Input(
                shape       =   x_.shape[1:],
                batch_size  =   x_.shape[0]
            )

            x       =   inputs

        x   =   DarknetConv(
            filters     =   filters,
            kernel_size =   1
        )(x)
        x   =   DarknetConv(
            filters     =   filters * 2,
            kernel_size =   3
        )(x)
        x   =   DarknetConv(
            filters     =   filters,
            kernel_size =   1
        )(x)
        x   =   DarknetConv(
            filters     =   filters * 2,
            kernel_size =   3
        )(x)
        x   =   DarknetConv(
            filters     =   filters,
            kernel_size =   1
        )(x)

        return Model(
            inputs  =   inputs,
            outputs =   x,
            name    =   name
        )(x_)

    return __internal


def _YOLOv3ConvTiny(filters :int,
                    name    :str = None):
    def __internal(x):
        x_  =   x
        if isinstance(x_, (list, tuple)):
            inputs  =   [
                Input(
                    shape       =   x_[0].shape[1:],
                    batch_size  =   x_[0].shape[0]
                ),
                Input(
                    shape       =   x_[1].shape[1:],
                    batch_size  =   x_[1].shape[0]
                ),
            ]

            x       =   inputs[0]
            x       =   DarknetConv(
                filters     =   filters,
                kernel_size =   1
            )(x)
            x       =   UpSampling2D(
                size        =   [2, 2]
            )(x)
            x       =   Concatenate(
            )([x, inputs[1]])
        else:
            inputs  =   Input(
                shape       =   x_.shape[1:],
                batch_size  =   x_.shape[0]
            )

            x       =   inputs
            x       =   DarknetConv(
                filters     =   filters,
                kernel_size =   1
            )(x)

        return Model(
            inputs  =   inputs,
            outputs =   x,
            name    =   name
        )(x_)

    return __internal


def _YOLOv3Output(filters       :int,
                  num_classes   :int,
                  num_anchors   :int,
                  name          :str    =   None):
    NUM_FEATURES    =   5 + num_classes

    def __internal(x):
        x_      =   x
        inputs  =   Input(
            shape       =   x_.shape[1:],
            batch_size  =   x_.shape[0]
        )

        x       =   inputs
        x       =   DarknetConv(
            filters     =   filters,
            kernel_size =   3
        )(x)
        x       =   DarknetConv(
            filters     =   NUM_FEATURES * num_anchors,
            kernel_size =   1,
            batch_norm  =   False
        )(x)
        x       =   Lambda(
            function    =   lambda x: tf.reshape(x, [-1, tf.shape(x)[1], tf.shape(x)[2], num_anchors, NUM_FEATURES])
        )(x)

        return tf.keras.Model(
            inputs  =   inputs,
            outputs =   x,
            name    =   name
        )(x_)

    return __internal


def _YOLOv3Detect(outputs,
                  num_classes   :int,
                  anchors):
    SIZE_GRID   =   tf.roll(tf.shape(outputs)[1:3], 1, 0)

    splited     =   tf.split(outputs, [2, 2, 1, num_classes], -1)
    bboxesXY    =   tf.sigmoid(splited[0])
    bboxesWH    =   tf.exp(splited[1]) * anchors
    bboxesOrig  =   tf.concat((bboxesXY, bboxesWH), -1)
    objectness  =   tf.sigmoid(splited[2])
    probs       =   tf.sigmoid(splited[3])

    # NOTE:
    #   grid[x][y] == (y, x)
    grid        =   _meshgrid(SIZE_GRID[0], SIZE_GRID[1])
    grid        =   tf.expand_dims(grid, 2)
    grid        =   tf.cast(grid, tf.float32)

    bboxesWHH   =   bboxesWH / 2
    bboxesXY   +=   grid
    bboxesXY   /=   tf.cast(SIZE_GRID, tf.float32)
    bboxesX1Y1  =   bboxesXY - bboxesWHH
    bboxesX2Y2  =   bboxesXY + bboxesWHH
    bboxes      =   tf.concat([bboxesX1Y1, bboxesX2Y2], -1)
    bboxes      =   tf.clip_by_value(bboxes, 0, 1)

    return bboxes, objectness, probs, bboxesOrig


def _YOLOv3NMS(outputs,
               batch_size       :int,
               num_classes      :int,
               max_bboxes       :int,
               thr_iou          :float,
               thr_confidence   :float):
    res         =   []
    for idx in range(batch_size):
        bboxes  =   []
        confs   =   []
        probs   =   []
        for elmOut in [elm for elm in outputs]:
            elmOut  =   [elm[idx] for elm in elmOut]

            elmB    =   tf.reshape(elmOut[0], [-1, 4])
            elmC    =   tf.reshape(elmOut[1], [-1, 1])
            elmP    =   tf.reshape(elmOut[2], [-1, num_classes])

            bboxes.append(elmB)
            confs.append(elmC)
            probs.append(elmP)

        bboxes  =   tf.concat(bboxes, 0)
        confs   =   tf.concat(confs, 0)
        probs   =   tf.concat(probs, 0)
        if num_classes > 1:
            confs  *=   probs

        classes =   tf.argmax(confs, 1)
        confs   =   tf.reduce_max(confs, 1)

        nms     =   tf.image.non_max_suppression_with_scores(
            boxes           =   bboxes,
            scores          =   confs,
            max_output_size =   max_bboxes,
            iou_threshold   =   thr_iou,
            score_threshold =   thr_confidence,
            soft_nms_sigma  =   0.5
        )
        numDets         =   tf.gather(tf.shape(nms[0]), [0])
        numDets         =   tf.squeeze(numDets, name = f'RESULT_{idx}_3_NUM_DETECTIONS')
        TENSOR_PAD_I    =   tf.zeros(max_bboxes - numDets, tf.int32)
        TENSOR_PAD_F    =   tf.zeros(max_bboxes - numDets, tf.float32)
        SELECTION       =   tf.concat([nms[0], TENSOR_PAD_I], 0)

        bboxes  =   tf.gather(bboxes, SELECTION, name = f'RESULT_{idx}_0_BBOXES')
        confs   =   tf.concat([nms[1], TENSOR_PAD_F], 0, name = f'RESULT_{idx}_1_CONFIDENCES')
        classes =   tf.gather(classes, SELECTION, name = f'RESULT_{idx}_2_CLASSES')

        res.append([bboxes, confs, classes, numDets])

    return res


def YOLOV3(height       :int,
           width        :int,
           channels     :int,
           batch_size   :int,
           num_classes  :int,
           anchors      :np.ndarray,
           masks        :np.ndarray):
    anchors        /=   np.array([width, height], np.float32)

    inputs          =   Input(
        shape       =   [height, width, channels],
        batch_size  =   batch_size
    )

    x = inputs
    x_36, x_61, x   =   Darknet(
        width       =   width,
        height      =   height,
        channels    =   channels,
        batch_size  =   batch_size,
        name        =   'YOLOv3_DARKNET'
    )(x)

    x               =   _YOLOv3Conv(
        filters     =   512,
        name        =   'YOLOv3_CONVOLUTION_0'
    )(x)
    oA              =   _YOLOv3Output(
        filters     =   1024,
        num_classes =   num_classes,
        num_anchors =   len(masks[0]),
        name        =   'YOLOv3_OUTPUT_0'
    )(x)

    x               =   _YOLOv3Conv(
        filters     =   256,
        name        =   'YOLOv3_CONVOLUTION_1'
    )([x, x_61])
    oB              =   _YOLOv3Output(
        filters     =   512,
        num_classes =   num_classes,
        num_anchors =   len(masks[1]),
        name        =   'YOLOv3_OUTPUT_1'
    )(x)

    x               =   _YOLOv3Conv(
        filters     =   128,
        name        =   'YOLOv3_CONVOLUTION_2'
    )([x, x_36])
    oC              =   _YOLOv3Output(
        filters     =   256,
        num_classes =   num_classes,
        num_anchors =   len(masks[2]),
        name        =   'YOLOv3_OUTPUT_2'
    )(x)

    outputs = Lambda(
        function        =   lambda x: _YOLOv3NMS(x, batch_size, num_classes, 50, 0.5, 0.5),
        name            =   'YOLOv3_NMS'
    )([
        Lambda(
            function    =   lambda x: _YOLOv3Detect(x, num_classes, tf.gather(anchors, masks[0])),
            name        =   'YOLOv3_DETECTION_0'
        )(oA)[:3],
        Lambda(
            function    =   lambda x: _YOLOv3Detect(x, num_classes, tf.gather(anchors, masks[1])),
            name        =   'YOLOv3_DETECTION_1'
        )(oB)[:3],
        Lambda(
            function    =   lambda x: _YOLOv3Detect(x, num_classes, tf.gather(anchors, masks[2])),
            name        =   'YOLOv3_DETECTION_2'
        )(oC)[:3]
    ])

    return Model(
        inputs  =   inputs,
        outputs =   outputs,
        name    =   'YOLOv3'
    )


def YOLOV3Tiny(height       :int,
               width        :int,
               channels     :int,
               batch_size   :int,
               num_classes  :int,
               anchors      :np.ndarray,
               masks        :np.ndarray):
    anchors    /= np.array([width, height], np.float32)

    inputs      =   Input(
        shape       =   [height, width, channels],
        batch_size  =   batch_size
    )

    x           =   inputs
    x_8, x      =   DarknetTiny(
        width       =   width,
        height      =   height,
        channels    =   channels,
        batch_size  =   batch_size,
        name        =   'YOLOv3_DARKNET'
    )(x)

    x           =   _YOLOv3ConvTiny(
        filters     =   256,
        name        =   'YOLOv3_CONVOLUTION_0'
    )(x)
    oA          =   _YOLOv3Output(
        filters     =   512,
        num_classes =   num_classes,
        num_anchors =   len(masks[0]),
        name        =   'YOLOv3_OUTPUT_0'
    )(x)

    x           =   _YOLOv3ConvTiny(
        filters     =   128,
        name        =   'YOLOv3_CONVOLUTION_1'
    )([x, x_8])
    oB          =   _YOLOv3Output(
        filters     =   256,
        num_classes =   num_classes,
        num_anchors =   len(masks[1]),
        name        =   'YOLOv3_OUTPUT_1'
    )(x)

    outputs = Lambda(
        function        =   lambda x: _YOLOv3NMS(x, batch_size, num_classes, 50, 0.5, 0.5),
        name            =   'YOLOv3_NMS'
    )([
        Lambda(
            function    =   lambda x: _YOLOv3Detect(x, num_classes, tf.gather(anchors, masks[0])),
            name        =   'YOLOv3_DETECTION_0'
        )(oA)[:3],
        Lambda(
            function    =   lambda x: _YOLOv3Detect(x, num_classes, tf.gather(anchors, masks[1])),
            name        =   'YOLOv3_DETECTION_1'
        )(oB)[:3],
    ])

    return Model(
        inputs  =   inputs,
        outputs =   outputs,
        name    =   'YOLOv3_TINY'
    )
