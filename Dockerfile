FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-devel

WORKDIR /workspace

RUN apt-get -y install libgl1-mesa-glx libglib2.0-0

RUN pip install efficientnet_pytorch termcolor timm opencv-python Pillow 
