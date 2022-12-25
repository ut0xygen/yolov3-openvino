##### Refenrence
#   https://www.intel.com/content/www/us/en/developer/tools/openvino-toolkit/download.html
#   https://docs.openvino.ai/latest/openvino_docs_MO_DG_prepare_model_convert_model_tf_specific_Convert_YOLO_From_Tensorflow.html
#   https://penpen-dev.com/blog/upgrade-tigai/
#   https://www.usagi1975.com/2019152355/
#   https://qiita.com/toshitanian/items/5da24c0c0bd473d514c8
#   https://gihyo.jp/admin/serial/01/ubuntu-recipe/0675

##### Setup Host Environment
# ## Set Shell Variables.
# . /etc/os-release
#
# ## Install NVIDIA GPU driver newer than 450.80.02 on host.
# #   > Driver version specified by NVIDIA CUDA 11.8.0.
# # sudo apt -y install ubuntu-drivers-common
# sudo apt -y install nvidia-driver-525
#
# ## Install Docker
# sudo apt -y install ca-certificates curl gnupg lsb-release
#
# sudo mkdir -p /etc/apt/keyrings
# curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
# echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list
# sudo apt -y update
#
# sudo apt -y install docker-ce docker-ce-cli containerd.io docker-compose-plugin
#
# curl -fsSL https://nvidia.github.io/nvidia-container-runtime/gpgkey | sudo apt-key add -
# curl -fsSL https://nvidia.github.io/nvidia-container-runtime/$ID$VERSION_ID/nvidia-container-runtime.list | sudo tee /etc/apt/sources.list.d/nvidia-container-runtime.list
# sudo apt -y update
# sudo apt -y install nvidia-docker2
#
# ## Test
# sudo docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi

#####
# FROM ubuntu:22.04
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04
USER root
WORKDIR /root

##### Setup System
ENV TZ=Asia/Tokyo
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt -y update
RUN apt -y full-upgrade

RUN apt -y install sudo cmake wget curl git sysstat htop vim
RUN apt -y install libopencv-dev libgl1-mesa-dev
# RUN apt -y install libglib2.0-0 libsm6 libxrender1 libxext6
# RUN apt -y install software-properties-common

# ##### Install NVIDIA CUDA 11.8
# # > https://developer.nvidia.com/cuda-11-8-0-download-archive
# #
# # > https://docs.nvidia.com/cuda/
# #
# # > https://zenn.dev/takeshita/articles/a02402e59d72a7
# #   *nvidia-smi shows the maximum CUDA version supported by the driver.
# RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb
# RUN dpkg -i ./cuda-keyring_1.0-1_all.deb
# RUN apt -y update
# RUN apt -y install cuda-toolkit-11-8
# RUN rm ./cuda-keyring_1.0-1_all.deb

# ##### Install NVIDIA cuDNN 8.7
# # > https://docs.nvidia.com/deeplearning/cudnn/
# RUN dpkg -i ./cudnn-local-repo-ubuntu2204-8.7.0.84_1.0-1_amd64.deb
# RUN cp /var/cudnn-local-repo-ubuntu2204-8.7.0.84/cudnn-local-BF23AD8A-keyring.gpg /usr/share/keyrings/
# RUN dpkg -i /var/cudnn-local-repo-ubuntu2204-8.7.0.84/libcudnn8_8.7.0.84-1+cuda11.8_amd64.deb
# RUN dpkg -i /var/cudnn-local-repo-ubuntu2204-8.7.0.84/libcudnn8-dev_8.7.0.84-1+cuda11.8_amd64.deb

##### Setup PPA (Personal Package Archive)
#   > Search for public-key and more in https://launchpad.net
#     (https://launchpad.net/~deadsnakes/+archive/ubuntu/ppa)
#
#   > https://wiki.archlinux.jp/index.php/GnuPG

# Method A
# RUN add-apt-repository ppa:deadsnakes/ppa

# Method B
#   > apt-key is deprecated.
RUN apt-key adv                                          \
    --keyserver keyserver.ubuntu.com                     \
    --recv      F23C5A6CF475977595C89F51BA6932366A755776
RUN . /etc/os-release                                                                                                                 && \
    :                                                                                      >  /etc/apt/sources.list.d/deadsnakes.list && \
    echo "deb     http://ppa.launchpad.net/deadsnakes/ppa/ubuntu ${VERSION_CODENAME} main" >> /etc/apt/sources.list.d/deadsnakes.list && \
    echo "deb-src http://ppa.launchpad.net/deadsnakes/ppa/ubuntu ${VERSION_CODENAME} main" >> /etc/apt/sources.list.d/deadsnakes.list

RUN apt -y update

# ##### Setup OpenVINO for TF 1.x
# ENV PYTHON_VER=3.7
# RUN apt install -y python3-pip python${PYTHON_VER} python${PYTHON_VER}-venv libpython${PYTHON_VER}
# RUN python${PYTHON_VER} -m pip install --upgrade pip
# RUN update-alternatives --install /usr/bin/python python /usr/bin/python${PYTHON_VER} 0
#
# RUN python${PYTHON_VER} -m venv openvino_tf1
# RUN . ./openvino_tf1/bin/activate                  && \
#     pip install --upgrade pip                      && \
#     pip install openvino==2021.4.2                 && \
#     pip install openvino-dev[tensorflow]==2021.4.2 && \
#     pip install seaborn matplotlib
#     # pip install -r ./openvino_tf1/lib/python${PYTHON_VER}/site-packages/openvino/tools/mo/requirements_tf.txt

##### Setup OpenVINO for TF 2.x
ENV PYTHON_VER=3.10
RUN apt install -y python3-pip python${PYTHON_VER} python${PYTHON_VER}-venv libpython${PYTHON_VER}
RUN python${PYTHON_VER} -m pip install --upgrade pip
RUN update-alternatives --install /usr/bin/python python /usr/bin/python${PYTHON_VER} 0

RUN python${PYTHON_VER} -m venv openvino_tf2
RUN . ./openvino_tf2/bin/activate                   && \
    pip install --upgrade pip                       && \
    pip install openvino==2022.3.0                  && \
    pip install openvino-dev[tensorflow2]==2022.3.0 && \
    pip install seaborn matplotlib
    # pip install -r ./openvino_tf2/lib/python${PYTHON_VER}/site-packages/openvino/tools/mo/requirements_tf.txt

##### Setup Environment Variables
# RUN echo 'export PATH="${PATH}:/usr/local/cuda/bin"'                         >> ~/.bashrc
# RUN echo 'export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/usr/local/cuda/lib64"' >> ~/.bashrc

#####
RUN chown -R 777 /root
RUN apt -y autoremove
RUN apt -y autoclean
CMD /bin/bash
