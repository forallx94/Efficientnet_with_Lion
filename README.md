# Efficientnet with Lion

Train Efficientnet image classification with Google's Lion Optimizer


* create development docker image
```bash
docker build -t docker.repository/user/efficientnet  -f Dockerfile .
```

* docker environment
```bash
docker run -it --name efficientnet --gpus all --ipc=host -v `pwd`:/workspace docker.repository/user/efficientnet:latest
```

* run command
```bash
python -m torch.distributed.launch --nproc_per_node 3 main.py --data-path {imagenet-path} --batch 512 --label-num 100
```