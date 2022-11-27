import argparse
import cv2
from openvino.inference_engine  import  IECore
from utils                      import  draw_outputs


def main(model_xml_file :str,
         model_bin_file :str,
         classes_file   :str,
         img_file       :str):
    IE                      =   IECore()

    # Load classes.
    CLASSES =   []
    with open(classes_file, 'r', encoding = 'utf-8') as file:
        for elmLine in file.readlines():
            elmLine = elmLine.strip()
            if not elmLine:
                continue

            CLASSES.append(elmLine)

    #
    net                     =   IE.read_network(
        model   =   model_xml_file,
        weights =   model_bin_file
    )
    netExec                 =   IE.load_network(net, device_name = 'CPU', num_requests = 1)

    # Get input/output layer name.
    NAME_INPUT              = list(net.input_info.keys())[0]
    NAME_OUTPUT             = list(net.outputs.keys())[0]

    # Get input shape.
    infoInput               =   net.input_info[NAME_INPUT].tensor_desc.dims
    SHAPE_INPUT_BATCHES     =   infoInput[0]
    SHAPE_INPUT_CHANNELS    =   infoInput[1]
    SHAPE_INPUT_HEIGHT      =   infoInput[2]
    SHAPE_INPUT_WIDTH       =   infoInput[3]

    # Load image.
    img                     =   cv2.imread(img_file)
    img_                    =   cv2.resize(img, (SHAPE_INPUT_WIDTH, SHAPE_INPUT_HEIGHT))
    img_                    =   img_.transpose((2, 0, 1))
    img_                    =   img_.reshape((1, SHAPE_INPUT_CHANNELS, SHAPE_INPUT_HEIGHT, SHAPE_INPUT_WIDTH))

    # Inference.
    prediction              =   netExec.infer(inputs = {NAME_INPUT: img_})
    prediction              =   [elm[1] for elm in sorted(prediction.items())]

    # Draw bouding-boxes.
    img = draw_outputs(img, prediction, CLASSES)
    cv2.imwrite('test.png', img)

    return


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()

    # Required options.
    argparser.add_argument(
        '--model_xml_file',
        required    =   True,
        type        =   str,
    )
    argparser.add_argument(
        '--model_bin_file',
        required    =   True,
        type        =   str
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

    args = argparser.parse_args()

    main(
        model_xml_file  =   args.model_xml_file,
        model_bin_file  =   args.model_bin_file,
        classes_file    =   args.classes_file,
        img_file        =   args.img_file
    )
