#!/bin/bash
cd `dirname ${0}`

docker image build                \
--file ./Dockerfile_for_user      \
--tag  ubuntu:20.04_openvino_hoge \
--build-arg UNAME=hoge            \
--build-arg UPASS=hoge            \
--build-arg GNAME=hoge            \
--build-arg UID=1000              \
--build-arg GID=1000
