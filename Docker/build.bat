@echo off
cd %~dp0

docker image build .                     ^
--progress=plain                         ^
--file ./Dockerfile                      ^
--tag  openvino:2022.3.0_tf2_ubuntu22.04

@echo on
