#!/bin/bash
cd `dirname ${0}`
docker image build --tag ubuntu:20.04_openvino ./
