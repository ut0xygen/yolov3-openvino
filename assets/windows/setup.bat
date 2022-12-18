@echo off
cd %~dp0

REM py -3.8 -m pip install --upgrade pip
py -3.8 -m venv openvino_tf2
.\openvino_tf2\Scripts\activate                 & ^
pip install --upgrade pip                       & ^
pip install openvino[tensorflow2]==2021.4.2     & ^
pip install openvino-dev[tensorflow2]==2021.4.2 & ^
pip install matplotlib seaborn                  & ^
cd ../../sources

@echo on
