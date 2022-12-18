##### Refenrence:
#   https://www.intel.com/content/www/us/en/developer/tools/openvino-toolkit/download.html
#   https://docs.openvino.ai/latest/openvino_docs_MO_DG_prepare_model_convert_model_tf_specific_Convert_YOLO_From_Tensorflow.html
#   https://penpen-dev.com/blog/upgrade-tigai/
#   https://www.usagi1975.com/2019152355/
#   https://qiita.com/toshitanian/items/5da24c0c0bd473d514c8
#   https://gihyo.jp/admin/serial/01/ubuntu-recipe/0675

FROM ubuntu:20.04
USER root
WORKDIR /root

##### Setup system.
RUN apt update       -y
RUN apt full-upgrade -y

RUN apt install      -y gnupg2 sysstat sudo htop vim
RUN apt install      -y wget git
RUN apt install      -y libgl1-mesa-dev
# RUN apt install      -y libglib2.0-0 libsm6 libxrender1 libxext6

##### Setup PPA (Personal Package Archive).
#   > Search for public-key and more in https://launchpad.net
#     (https://launchpad.net/~deadsnakes/+archive/ubuntu/ppa)
#
#   > https://wiki.archlinux.jp/index.php/GnuPG

# Method A
# RUN apt install -y software-properties-common
# RUN add-apt-repository ppa:deadsnakes/ppa

# Method B
ENV OS_CODENAME=focal
RUN apt-key adv                                          \
    --keyserver keyserver.ubuntu.com                     \
    --recv      F23C5A6CF475977595C89F51BA6932366A755776
RUN : >                                                                                  /etc/apt/sources.list.d/deadsnakes.list && \
    echo "deb     http://ppa.launchpad.net/deadsnakes/ppa/ubuntu ${OS_CODENAME} main" >> /etc/apt/sources.list.d/deadsnakes.list && \
    echo "deb-src http://ppa.launchpad.net/deadsnakes/ppa/ubuntu ${OS_CODENAME} main" >> /etc/apt/sources.list.d/deadsnakes.list

RUN apt update       -y

##### Setup OpenVINO for TF 1.x.
# ENV PYTHON_VER=3.7
# RUN apt install -y python3-pip python${PYTHON_VER} python${PYTHON_VER}-venv libpython${PYTHON_VER}
# RUN python${PYTHON_VER} -m pip install --upgrade pip
# RUN update-alternatives --install /usr/bin/python python /usr/bin/python${PYTHON_VER} 0
#
# RUN python3.7 -m venv openvino_tf1
# RUN . ./openvino_tf1/bin/activate                  && \
#     pip install --upgrade pip                      && \
#     pip install openvino[tensorflow]==2021.4.2     && \
#     pip install openvino-dev[tensorflow]==2021.4.2 && \
#     pip install seaborn matplotlib
#     # pip install -r ./openvino_tf1/lib/python3.7/site-packages/mo/requirements_tf.txt

##### Setup OpenVINO for TF 2.x.
ENV PYTHON_VER=3.8
RUN apt install -y python3-pip python${PYTHON_VER} python${PYTHON_VER}-venv libpython${PYTHON_VER}
RUN python${PYTHON_VER} -m pip install --upgrade pip
RUN update-alternatives --install /usr/bin/python python /usr/bin/python${PYTHON_VER} 0

RUN python3.8 -m venv openvino_tf2
RUN . ./openvino_tf2/bin/activate                   && \
    pip install --upgrade pip                       && \
    pip install openvino[tensorflow2]==2021.4.2     && \
    pip install openvino-dev[tensorflow2]==2021.4.2 && \
    pip install seaborn matplotlib
    # pip install -r ./openvino_tf2/lib/python3.8/site-packages/mo/requirements_tf.txt

#####
RUN chown -R 777 /root
RUN apt autoremove   -y
RUN apt autoclean    -y
CMD /bin/bash
