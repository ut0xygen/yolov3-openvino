#!/bin/bash
cd `dirname $0`
docker image build --build-arg UNAME=hoge --build-arg UPASS=. --build-arg GNAME=hoge --build-arg UID=1000 --build-arg GID=1000 --tag ubuntu:20.04_openvino_for_user ./
