FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-devel

WORKDIR /workspace

RUN pip install tensorboard \ 
    && pip install efficientnet_pytorch
