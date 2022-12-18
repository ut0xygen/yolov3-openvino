import os
import argparse
from models.yolov3  import  (
    YOLOV3,
    YOLOV3Tiny,
)
from utils  import  load_darknet_weights

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

def visualize(img,
              pr):
    BBOXES          =   pr[0]
    CONFIDENCES     =   pr[1]
    CLASSES         =   pr[2]
    NUM_DETECTIONS  =   pr[3]

    imgWH           =   np.flip(img.shape[0:2])
    for idx in range(NUM_DETECTIONS):
        x1y1    =   (BBOXES[idx][0:2] * imgWH).astype(np.int32)
        x2y2    =   (BBOXES[idx][2:4] * imgWH).astype(np.int32)
        img     =   cv2.rectangle(img, x1y1, x2y2, (255, 0, 0), 2)

    return img


def main(weights_file   :str,
         classes_file   :str,
         width          :int,
         height         :int,
         channels       :int,
         batch_size     :int,
         tiny           :bool,
         out_file       :str,
         test_list_file :str):
    ##### Load classes.
    CLASSES             =   []
    LEN_STR_CLASS       =   0
    with open(classes_file, 'r', encoding = 'utf-8') as file:
        lines           =   file.readlines()
    for elmLine in lines:
        elmLine         =   elmLine.strip()
        if not elmLine:
            continue

        LEN_STR_CLASS   =   max(LEN_STR_CLASS, len(elmLine))
        CLASSES.append(elmLine)

    #####
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

    ##### Construct model.
    model           =   modelGenerator(
        height      =   height,
        width       =   width,
        channels    =   channels,
        batch_size  =   batch_size,
        num_classes =   len(CLASSES),
        anchors     =   ANCHORS,
        masks       =   MASKS
    )
    model.summary()

    # Load and apply weights.
    load_darknet_weights(model, weights_file, tiny)

    # Check sanity.
    model(np.random.random([batch_size, height, width, channels]).astype(np.float32))

    # Save model.
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

    # Test.
    if test_list_file:
        # Get files.
        with open(test_list_file, 'r', encoding = 'utf-8') as file:
            lines   =   file.readlines()
        files       =   []
        for elmLine in lines:
            elmLine =   elmLine.strip()
            if not elmLine:
                continue

            files.append(os.path.abspath(elmLine))
        NUM_DATA    =   len(files)

        # Make output directory.
        PATH_OUT    =   './visualized'
        os.makedirs(PATH_OUT, exist_ok = True)

        #
        for idx in range(0, NUM_DATA, batch_size):
            # Preprocess data.
            batch   =   []
            batch_  =   []
            for idxBat in range(batch_size):
                idxFile =   idx + idxBat
                if idxFile < NUM_DATA:
                    img     =   tf.image.decode_image(open(files[idxFile], 'rb').read(), channels=3)
                    img_    =   tf.image.resize(img, (width, height))
                    img_    =   img_ / 255
                    # img     =   cv2.imread(files[idxFile])
                    # img_    =   cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    # img_    =   cv2.resize(img_, (width, height))
                    # img_    =   img_.astype(np.float32) / 255.

                    batch.append(img)
                else:
                    img_    =   np.random.random([height, width, channels]).astype(np.float32)

                batch_.append(img_)

            batch_  =   np.reshape(batch_, [batch_size, height, width, channels])

            # Inference.
            prs     =   model.predict(batch_)

            # Visualize.
            for idxBat in range(batch_size):
                idxFile =   idx + idxBat
                if idxFile < NUM_DATA:
                    PR_BBOXES       =   prs[idxBat][0]
                    PR_CONFIDENCES  =   prs[idxBat][1]
                    PR_CLASSES      =   prs[idxBat][2]
                    NUM_DETECTIONS  =   prs[idxBat][3]

                    file            =   files[idxFile]
                    fName, fExt     =   os.path.splitext(os.path.basename(file))
                    img             =   batch[idxBat]
                    img             =   cv2.cvtColor(img.numpy(), cv2.COLOR_RGB2BGR)
                    img             =   visualize(img, prs[idxBat])

                    cv2.imwrite(os.path.join(PATH_OUT, f'{fName}_v{fExt}'), img)

                    print(f'FILE: {file}')
                    for idxDet in range(NUM_DETECTIONS):
                        print('    ', end = '')
                        print(CLASSES[PR_CLASSES[idxDet]].ljust(LEN_STR_CLASS, ' '), end = '')
                        print(f' ({(PR_CONFIDENCES[idxDet] * 100):.2f}%)', end = '')
                        print()
                else:
                    break

    del model_
    del model

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
        '--classes_file',
        required    =   True,
        type        =   str,
    )

    # Other options.
    argparser.add_argument(
        '--height',
        required    =   False,
        type        =   int,
        default     =   416
    )
    argparser.add_argument(
        '--width',
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
        '--batch_size',
        required    =   False,
        type        =   int,
        default     =   1
    )
    argparser.add_argument(
        '--tiny',
        required    =   False,
        action      =   'store_true'
    )
    argparser.add_argument(
        '--out_file',
        required    =   False,
        type        =   str,
        default     =   './model.pb'
    )
    argparser.add_argument(
        '--test_list_file',
        required    =   True,
        type        =   str,
    )

    args = argparser.parse_args()

    main(
        weights_file    =   args.weights_file,
        classes_file    =   args.classes_file,
        width           =   args.width,
        height          =   args.height,
        channels        =   args.channels,
        batch_size      =   args.batch_size,
        tiny            =   args.tiny,
        out_file        =   args.out_file,
        test_list_file  =   args.test_list_file
    )
