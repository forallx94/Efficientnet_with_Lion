# Efficientnet with Lion

Train Efficientnet image classification with Google's Lion Optimizer

best test accuarcy at EfficientNet-B0 95.9

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
python main.py --batch 512
```

* Check process TensorBoard
```bash
 tensorboard --logdir=runs
```

## reference 
1. https://keep-steady.tistory.com/35  
2. https://www.kaggle.com/code/nroman/melanoma-pytorch-starter-efficientnet  
3. https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html  
4. https://github.com/google/automl/tree/master/lion  