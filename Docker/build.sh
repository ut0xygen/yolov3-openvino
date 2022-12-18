#!/bin/bash
cd `dirname ${0}`

docker image build .         \
--progress=plain             \
--file ./Dockerfile          \
--tag  ubuntu:20.04_openvino
