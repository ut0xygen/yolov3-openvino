# Refenrence:
#   https://www.intel.com/content/www/us/en/developer/tools/openvino-toolkit/download.html
#   https://docs.openvino.ai/latest/openvino_docs_MO_DG_prepare_model_convert_model_tf_specific_Convert_YOLO_From_Tensorflow.html
#   https://qiita.com/toshitanian/items/5da24c0c0bd473d514c8
#
#   https://pypi.org/project/openvino/
#   https://pypi.org/project/openvino-dev/
#
#   https://github.com/pjreddie/darknet
#   https://github.com/mystic123/tensorflow-yolo-v3

FROM ubuntu:20.04
USER root

# Update system.
RUN apt update -y
# RUN apt upgrade -y
# RUN apt dist-upgrade -y
# RUN apt autoremove -y
# RUN apt autoclean - y

# Setup PPA (Personal Package Archive).
RUN apt install -y software-properties-common
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt update -y

# Install Python.
RUN apt install -y python3-pip

ENV PYTHON_VER=3.7
RUN apt install -y python${PYTHON_VER} python${PYTHON_VER}-venv libpython${PYTHON_VER}
RUN python${PYTHON_VER} -m pip install --upgrade pip

ENV PYTHON_VER=3.8
RUN apt install -y python${PYTHON_VER} python${PYTHON_VER}-venv libpython${PYTHON_VER}
RUN python${PYTHON_VER} -m pip install --upgrade pip
RUN update-alternatives --install /usr/bin/python python /usr/bin/python${PYTHON_VER} 0

# Install others.
RUN apt install -y libgl1-mesa-dev
# RUN apt install -y libglib2.0-0 libsm6 libxrender1 libxext6
RUN apt install -y sudo wget git

# Clone repositories.
WORKDIR /root/repos

# RUN git clone https://github.com/pjreddie/darknet.git
# WORKDIR ./darknet
# RUN make
# WORKDIR /root/repos

RUN git clone https://github.com/mystic123/tensorflow-yolo-v3.git
WORKDIR ./tensorflow-yolo-v3/weights
RUN wget https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names
RUN wget https://pjreddie.com/media/files/yolov3.weights
RUN wget https://pjreddie.com/media/files/yolov3-tiny.weights
RUN wget https://pjreddie.com/media/files/yolov3-spp.weights
WORKDIR /root/repos

RUN git clone https://github.com/zzh8829/yolov3-tf2.git
WORKDIR /root/repos

WORKDIR /root

# Setup OpenVINO for TF 1.x.
RUN python3.7 -m venv openvino_tf1
RUN . ./openvino_tf1/bin/activate && \
    python -m pip install --upgrade pip && \
    pip install openvino[tensorflow]==2021.4.2 && \
    pip install openvino-dev[tensorflow]==2021.4.2 && \
    pip install -r ./openvino_tf1/lib/python3.7/site-packages/mo/requirements_tf.txt && \
    pip install tensorflow==1.15.5 && \
    pip install protobuf==3.20.*

# Setup OpenVINO for TF 2.x.
RUN python3.8 -m venv openvino_tf2
RUN . ./openvino_tf2/bin/activate && \
    python -m pip install --upgrade pip && \
    pip install openvino[tensorflow2]==2021.4.2 && \
    pip install openvino-dev[tensorflow2]==2021.4.2 && \
    pip install -r ./openvino_tf2/lib/python3.8/site-packages/mo/requirements_tf.txt

CMD /bin/bash
