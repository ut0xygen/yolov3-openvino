import numpy as np
import tensorflow as tf
from tensorflow.keras           import  Model
from tensorflow.keras.layers    import  Input, Concatenate
from tensorflow.keras.layers    import  UpSampling2D
from tensorflow.keras.layers    import  Lambda
from darknet import (
    _DarknetConvolutional as DarknetConvolutional,
    Darknet,
    DarknetTiny,
)


# As tensorflow lite doesn't support tf.size used in tf.meshgrid,
# we reimplemented a simple meshgrid function that use basic tf function.
def _meshgrid(n_a   :int,
              n_b   :int):
    return [
        tf.reshape(tf.tile(tf.range(n_a), [n_b]), (n_b, n_a)),
        tf.reshape(tf.repeat(tf.range(n_b), n_a), (n_b, n_a)),
    ]

def _YOLOV3Convolutional(filters   :int,
                         name      :str    =   None):
    def __internal(x):
        x_  =   x
        if isinstance(x, (list, tuple)):
            inputs  =   [
                Input(shape = x_[0].shape[1:]),
                Input(shape = x_[1].shape[1:]),
            ]

            x       =   inputs[0]
            x       =   DarknetConvolutional(
                filters     =   filters,
                kernel_size =   1
            )(x)
            x = UpSampling2D(
                size    =   (2, 2)
            )(x)
            x = Concatenate()([x, inputs[1]])
        else:
            inputs  =   Input(shape = x_.shape[1:])

            x       =   inputs

        x   =   DarknetConvolutional(
            filters     =   filters,
            kernel_size =   1
        )(x)
        x   =   DarknetConvolutional(
            filters     =   filters * 2,
            kernel_size =   3
        )(x)
        x   =   DarknetConvolutional(
            filters     =   filters,
            kernel_size =   1
        )(x)
        x   =   DarknetConvolutional(
            filters     =   filters * 2,
            kernel_size =   3
        )(x)
        x   =   DarknetConvolutional(
            filters     =   filters,
            kernel_size =   1
        )(x)

        return Model(
            inputs  =   inputs,
            outputs =   x,
            name    =   name
        )(x_)

    return __internal

def _YOLOV3ConvolutionalTiny(filters   :int,
                             name      :str    =   None):
    def __internal(x):
        x_  =   x
        if isinstance(x_, (list, tuple)):
            inputs  =   [
                Input(shape = x_[0].shape[1:]),
                Input(shape = x_[1].shape[1:]),
            ]

            x       =   inputs[0]
            x       =   DarknetConvolutional(
                filters     =   filters,
                kernel_size =   1
            )(x)
            x       =   UpSampling2D(
                size        =   (2, 2)
            )(x)
            x       =   Concatenate()([x, inputs[1]])
        else:
            inputs = Input(shape = x_.shape[1:])

            x       =   inputs
            x       =   DarknetConvolutional(
                filters     =   filters,
                kernel_size =   1
            )(x)

        return Model(
            inputs  =   inputs,
            outputs =   x,
            name    =   name
        )(x_)

    return __internal

def _YOLOV3Output(filters      :int,
                  num_classes  :int,
                  num_anchors  :int,
                  name         :str    =   None):
    NUM_FEATURES    =   num_classes + 5
    def __internal(x):
        x_      =   x
        inputs  =   Input(shape = x_.shape[1:])

        x       =   inputs
        x       =   DarknetConvolutional(
            filters     =   filters * 2,
            kernel_size =   3
        )(x)
        x       =   DarknetConvolutional(
            filters     =   NUM_FEATURES * num_anchors,
            kernel_size =   1,
            batch_norm  =   False
        )(x)
        x       =   Lambda(
            function    =   lambda x: tf.reshape(x, (-1, tf.shape(x)[1], tf.shape(x)[2], num_anchors, NUM_FEATURES))
        )(x)

        return tf.keras.Model(
            inputs  =   inputs,
            outputs =   x,
            name    =   name
        )(x_)

    return __internal

def _YOLOV3Detect(outputs,
                  anchors,
                  num_classes   :int):
    SIZE_GRID       =   tf.shape(outputs)[1:3]

    tmpSplited      =   tf.split(outputs, (2, 2, 1, num_classes), -1)

    bboxesXY        =   tf.sigmoid(tmpSplited[0])
    bboxesWH        =   tf.exp(tmpSplited[1]) * anchors
    objectness      =   tf.sigmoid(tmpSplited[2])
    clsProbs        =   tf.sigmoid(tmpSplited[3])

    bboxesOrig      =   tf.concat((bboxesXY, bboxesWH), -1)

    # NOTE: grid[x][y] == (y, x)
    grid            =   _meshgrid(SIZE_GRID[1], SIZE_GRID[0])
    grid            =   tf.stack(grid, -1)
    grid            =   tf.expand_dims(grid, 2)
    grid            =   tf.cast(grid, tf.float32)

    bboxesXY       +=   grid
    bboxesXY       /=   tf.cast(SIZE_GRID, tf.float32)

    bboxesWHHalf    =   bboxesWH / 2
    bboxesX1Y1      =   bboxesXY - bboxesWHHalf
    bboxesX2Y2      =   bboxesXY + bboxesWHHalf
    bboxes          =   tf.concat([bboxesX1Y1, bboxesX2Y2], -1)

    return bboxes, objectness, clsProbs, bboxesOrig

def _YOLOV3NMS(outputs,
               num_classes          :int,
               max_bboxes           :int,
               threshold_iou        :float,
               threshold_confidence :float):
    bboxes              =   []
    objectness          =   []
    clsProbs            =   []
    for elmOut in outputs:
        SHAPE_BBOXES        =   tf.shape(elmOut[0])
        SHAPE_OBJECTNESS    =   tf.shape(elmOut[1])
        SHAPE_CLASS_PROBS   =   tf.shape(elmOut[2])

        bboxes.append(tf.reshape(
            elmOut[0],
            (
                SHAPE_BBOXES[0],
                -1,
                SHAPE_BBOXES[-1],
            )
        ))
        objectness.append(tf.reshape(
            elmOut[1],
            (
                SHAPE_OBJECTNESS[0],
                -1,
                SHAPE_OBJECTNESS[-1],
            )
        ))
        clsProbs.append(tf.reshape(
            elmOut[2],
            (
                SHAPE_CLASS_PROBS[0],
                -1,
                SHAPE_CLASS_PROBS[-1],
            )
        ))

    bboxes              =   tf.concat(bboxes, 1)
    bboxes              =   tf.reshape(bboxes, (-1, 4))
    objectness          =   tf.concat(objectness, 1)
    clsProbs            =   tf.concat(clsProbs, 1)

    confidence          =   objectness
    if num_classes > 1:
        confidence     *=   clsProbs
    tmpConfidence       =   tf.squeeze(confidence, 0)
    confidence          =   tf.reduce_max(tmpConfidence, [1])
    classes             =   tf.argmax(tmpConfidence, 1)

    nms                 =   tf.image.non_max_suppression_with_scores(
        boxes           =   bboxes,
        scores          =   confidence,
        max_output_size =   max_bboxes,
        iou_threshold   =   threshold_iou,
        score_threshold =   threshold_confidence,
        soft_nms_sigma  =   0.5
    )
    NUM_VALID_BBOXES    =   tf.shape(nms[0])[0]

    bboxesSelected      =   tf.concat([nms[0], tf.zeros(max_bboxes - NUM_VALID_BBOXES, tf.int32)], 0)
    confidenceSelected  =   tf.concat([nms[1], tf.zeros(max_bboxes - NUM_VALID_BBOXES, tf.float32)], -1)

    bboxes              =   tf.gather(bboxes, bboxesSelected)
    bboxes              =   tf.expand_dims(bboxes, 0, name = 'RESULT_0_BBOXES')
    objectness          =   confidenceSelected
    objectness          =   tf.expand_dims(objectness, 0, name = 'RESULT_1_OBJECTNESS')
    classes             =   tf.gather(classes, bboxesSelected)
    classes             =   tf.expand_dims(classes, 0, name = 'RESULT_2_CLASSES')
    num_detections      =   tf.expand_dims(NUM_VALID_BBOXES, 0, name = 'RESULT_3_DETECTIONS')

    return bboxes, objectness, classes, num_detections


def YOLOV3(width        :int,
           height       :int,
           channels     :int,
           num_classes  :int,
           anchors      :np.ndarray,
           masks        :np.ndarray):
    anchors         =   anchors / np.array([height, width], np.float32)

    inputs          =   Input(
        shape   =   [width, height, channels],
        name    =   'input'
    )

    x               =   inputs
    x_36, x_61, x   =   Darknet(
        width       =   width,
        height      =   height,
        channels    =   channels,
        name        =   'yolo_darknet'
    )(x)
    x       =   _YOLOV3Convolutional(
        filters     =   512,
        name        =   'yolo_conv_0'
    )(x)
    oA      =   _YOLOV3Output(
        filters     =   512,
        num_classes =   num_classes,
        num_anchors =   len(masks[0]),
        name        =   'yolo_output_0'
    )(x)
    x       =   _YOLOV3Convolutional(
        filters     =   256,
        name        =   'yolo_conv_1'
    )([x, x_61])
    oB      =   _YOLOV3Output(
        filters     =   256,
        num_classes =   num_classes,
        num_anchors =   len(masks[1]),
        name        =   'yolo_output_1'
    )(x)
    x       =   _YOLOV3Convolutional(
        filters     =   128,
        name        =   'yolo_conv_2'
    )([x, x_36])
    oC      =   _YOLOV3Output(
        filters     =   128,
        num_classes =   num_classes,
        num_anchors =   len(masks[2]),
        name        =   'yolo_output_2'
    )(x)

    bA      =   Lambda(
        function    =   lambda x: _YOLOV3Detect(x, tf.gather(anchors, masks[0]), num_classes),
        name        =   'yolo_boxes_0'
    )(oA)
    bB      =   Lambda(
        function    =   lambda x: _YOLOV3Detect(x, tf.gather(anchors, masks[1]), num_classes),
        name        =   'yolo_boxes_1'
    )(oB)
    bC      =   Lambda(
        function    =   lambda x: _YOLOV3Detect(x, tf.gather(anchors, masks[2]), num_classes),
        name        =   'yolo_boxes_2'
    )(oC)

    outputs =   Lambda(
        function    =   lambda x: _YOLOV3NMS(x, num_classes, 200, 0.5, 0.5),
        name        =   'yolo_nms'
    )([bA[:3], bB[:3], bC[:3]])

    return Model(
        inputs  =   inputs,
        outputs =   outputs,
        name    =   'yolov3'
    )

def YOLOV3Tiny(width        :int,
               height       :int,
               channels     :int,
               num_classes  :int,
               anchors      :np.ndarray,
               masks        :np.ndarray):
    anchors     =   anchors / np.array([height, width], np.float32)

    inputs      =   Input(
        shape   =   [width, height, channels],
        name    =   'input'
    )

    x       =   inputs
    x_8, x  =   DarknetTiny(
        width       =   width,
        height      =   height,
        channels    =   channels,
        name        =   'yolo_darknet'
    )(x)
    x       =   _YOLOV3ConvolutionalTiny(
        filters     =   256,
        name        =   'yolo_conv_0'
    )(x)
    oA      =   _YOLOV3Output(
        filters     =   256,
        num_classes =   num_classes,
        num_anchors =   len(masks[0]),
        name        =   'yolo_output_0'
    )(x)
    x       =   _YOLOV3ConvolutionalTiny(
        filters     =   128,
        name        =   'yolo_conv_1'
    )([x, x_8])
    oB      =   _YOLOV3Output(
        filters     =   128,
        num_classes =   num_classes,
        num_anchors =   len(masks[1]),
        name        =   'yolo_output_1'
    )(x)

    bA      =   Lambda(
        function    =   lambda x: _YOLOV3Detect(x, tf.gather(anchors, masks[0]), num_classes),
        name        =   'yolo_boxes_0'
    )(oA)
    bB      =   Lambda(
        function    =   lambda x: _YOLOV3Detect(x, tf.gather(anchors, masks[1]), num_classes),
        name        =   'yolo_boxes_1'
    )(oB)
    outputs =   Lambda(
        function    =   lambda x: _YOLOV3NMS(x, num_classes, 200, 0.5, 0.5),
        name        =   'yolo_nms'
    )([bA[:3], bB[:3]])

    return Model(
        inputs  =   inputs,
        outputs =   outputs,
        name    =   'yolov3_tiny'
    )
