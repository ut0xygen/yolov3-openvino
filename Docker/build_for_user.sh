#!/bin/bash
cd `dirname ${0}`

docker image build .                          \
--file ./Dockerfile_for_user                  \
--tag  openvino:2022.3.0_tf2_ubuntu22.04_hoge \
--build-arg UNAME=hoge                        \
--build-arg UPASS=hoge                        \
--build-arg GNAME=hoge                        \
--build-arg UID=1000                          \
--build-arg GID=1000
