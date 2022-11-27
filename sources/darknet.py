import tensorflow as tf
from tensorflow                 import  keras as k
from tensorflow.keras.layers    import  Input, Add
from tensorflow.keras.layers    import  Convolution2D, MaxPooling2D, ZeroPadding2D
from tensorflow.keras.layers    import  BatchNormalization, LeakyReLU


def _DarknetConvolutional(filters     :int,
                          kernel_size :int,
                          strides     :int    =   1,
                          batch_norm  :bool   =   True):
    def __internal(x):
        if strides == 1:
            padding = 'SAME'
        else:
            padding = 'VALID'
            x = ZeroPadding2D(((1, 0), (1, 0)))(x)  # top left half-padding

        x = Convolution2D(
            filters             =   filters,
            kernel_size         =   (kernel_size, kernel_size),
            strides             =   (strides, strides),
            padding             =   padding,
            kernel_regularizer  =   k.regularizers.l2(0.0005),
            use_bias    =   not batch_norm
        )(x)
        if batch_norm:
            x = BatchNormalization()(x)
            x = LeakyReLU(
                alpha           =   0.1
            )(x)

        return x

    return __internal

def _DarknetResidual(filters: int):
    def __internal(x):
        x_  =   x
        x   =   _DarknetConvolutional(
            filters     =   filters // 2,
            kernel_size =   1
        )(x)
        x   =   _DarknetConvolutional(
            filters     =   filters,
            kernel_size =   3
        )(x)
        x   =   Add()([x_, x])

        return x

    return __internal

def _DarknetBlock(filters    :int,
                  blocks     :int):
    def __internal(x):
        x = _DarknetConvolutional(
            filters     =   filters,
            kernel_size =   3,
            strides     =   2
        )(x)
        for _ in range(blocks):
            x = _DarknetResidual(
                filters =   filters
            )(x)

        return x

    return __internal

def Darknet(width       :int,
            height      :int,
            channels    :int,
            name        :str    =   None):
    inputs  =   Input(shape = [width, height, channels])

    x       =   inputs
    x       =   _DarknetConvolutional(
        filters     =   32,
        kernel_size =   3
    )(x)
    x       =   _DarknetBlock(
        filters     =   64,
        blocks      =   1
    )(x)
    x       =   _DarknetBlock(
        filters     =   128,
        blocks      =   2
    )(x)
    x       =   _DarknetBlock(
        filters     =   256,
        blocks      =   8
    )(x)
    x_36    =   x
    x       =   _DarknetBlock(
        filters     =   512,
        blocks      =   8
    )(x)
    x_61    =   x
    x       =   _DarknetBlock(
        filters     =   1024,
        blocks      =   4
    )(x)

    return tf.keras.Model(
        inputs  =   inputs,
        outputs =   [
            x_36,
            x_61,
            x,
        ],
        name    =   name
    )

def DarknetTiny(width       :int,
                height      :int,
                channels    :int,
                name        :str    =   None):
    inputs  =   Input(shape = [width, height, channels])

    x       =   inputs
    x       =   _DarknetConvolutional(
        filters     =   16,
        kernel_size =   3
    )(x)
    x       =   MaxPooling2D(
        pool_size   =   (2, 2),
        strides     =   (2, 2),
        padding     =   'SAME'
    )(x)
    x       =   _DarknetConvolutional(
        filters     =   32,
        kernel_size =   3
    )(x)
    x       =   MaxPooling2D(
        pool_size   =   (2, 2),
        strides     =   (2, 2),
        padding     =   'SAME'
    )(x)
    x       =   _DarknetConvolutional(
        filters     =   64,
        kernel_size =   3
    )(x)
    x       =   MaxPooling2D(
        pool_size   =   (2, 2),
        strides     =   (2, 2),
        padding     =   'SAME'
    )(x)
    x       =   _DarknetConvolutional(
        filters     =   128,
        kernel_size =   3
    )(x)
    x       =   MaxPooling2D(
        pool_size   =   (2, 2),
        strides     =   (2, 2),
        padding     =   'SAME'
    )(x)
    x       =   _DarknetConvolutional(
        filters     =   256,
        kernel_size =   3
    )(x)
    x_8     =   x
    x       =   MaxPooling2D(
        pool_size   =   (2, 2),
        strides     =   (2, 2),
        padding     =   'SAME'
    )(x)
    x       =   _DarknetConvolutional(
        filters     =   512,
        kernel_size =   3
    )(x)
    x       =   MaxPooling2D(
        pool_size   =   (2, 2),
        strides     =   (1, 1),
        padding     =   'SAME'
    )(x)
    x       =   _DarknetConvolutional(
        filters     =   1024,
        kernel_size =   3
    )(x)

    return tf.keras.Model(
        inputs  =   inputs,
        outputs =   [
            x_8,
            x,
        ],
        name    =   name
    )
