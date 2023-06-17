import torch
import torchvision
import torch.nn.functional as F
import torch.nn as nn
# import torchtoolbox.transform as transforms
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torch.optim.optimizer import Optimizer

import numpy as np
import os
import time
import warnings
import random
import copy
import torch.optim as optim
from tqdm import tqdm
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from torch.utils.tensorboard import SummaryWriter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # set gpu

warnings.simplefilter('ignore')
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def train_model(args, model, dataloaders, criterion, optimizer,  writer, num_epochs=25):
    """
    model training
    Args:
        model (torch model) : torch model before train
        dataloaders : torch dataloader 
        criterion : loss function, cross-entropy
        optimizer : Lion optimizer
        writer : tensorboard writer
        num_epochs (int) : number of training epochs
    return:
        model : Model at peak performance in valid
        best_idx (int): The epoch of the model at peak performance in valid
        best_acc (float): The accuracy of the model at peak performance in valid
        train_loss (list) : Total train loss list
        train_acc (list) : Total train accuracy list
        valid_loss (list) : Total validation loss list
        valid_acc (list) : Total validation accuracy list
    """
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    train_loss, train_acc, valid_loss, valid_acc = [], [], [], []
    
    for epoch in tqdm(range(num_epochs)):

        # Each epoch has a training and validation phase
        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss, running_corrects, num_cnt = 0.0, 0, 0
            
            with tqdm(dataloaders[phase], unit="batch") as tepoch:
              # Iterate over data.
              for inputs, labels in tepoch:
                  inputs = inputs.to(device)
                  labels = labels.to(device)

                  # zero the parameter gradients
                  optimizer.zero_grad()

                  # forward
                  # track history if only in train
                  with torch.set_grad_enabled(phase == 'train'):
                      outputs = model(inputs)
                      _, preds = torch.max(outputs, 1)
                      loss = criterion(outputs, labels)

                      # backward + optimize only if in training phase
                      if phase == 'train':
                          loss.backward()
                          optimizer.step()

                  # statistics
                  running_loss += loss.item() * inputs.size(0)
                  running_corrects += torch.sum(preds == labels.data)
                  num_cnt += len(labels)

            # if phase == 'train':
            #     scheduler.step()
            
            epoch_loss = float(running_loss / num_cnt)
            epoch_acc  = float((running_corrects.double() / num_cnt).cpu()*100)
            
            if phase == 'train':
                writer.add_scalar('Loss/train', epoch_loss, epoch)
                train_loss.append(epoch_loss)
                writer.add_scalar('Accuracy/train', epoch_acc, epoch)
                train_acc.append(epoch_acc)
            else:
                valid_loss.append(epoch_loss)
                writer.add_scalar('Loss/valid',epoch_loss, epoch)
                valid_acc.append(epoch_acc)
                writer.add_scalar('Accuracy/valid', epoch_acc, epoch)
            print('{} Loss: {:.2f} Acc: {:.1f}'.format(phase, epoch_loss, epoch_acc))
           
            # deep copy the model
            if phase == 'valid' and epoch_acc > best_acc:
                best_idx = epoch
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model)
                torch.save(model,f'{args.save_path}/{args.models}_{best_idx}_{int(best_acc)}.pth')
                # best_model_wts = copy.deepcopy(model.module.state_dict())
                print('==> best model saved - %d / %.1f'%(best_idx, best_acc))

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best valid Acc: %d - %.1f' %(best_idx, best_acc))

    # load best model weights
    model = best_model_wts
    torch.save(model, f'{args.save_path}/{args.models}_president_model.pth')
    print('model saved')
    return model, best_idx, best_acc, train_loss, train_acc, valid_loss, valid_acc


"""
PyTorch implementation of the Lion optimizer.
https://github.com/google/automl/tree/master/lion
"""
class Lion(Optimizer):
  r"""Implements Lion algorithm."""

  def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.0):
    """Initialize the hyperparameters.

    Args:
      params (iterable): iterable of parameters to optimize or dicts defining
        parameter groups
      lr (float, optional): learning rate (default: 1e-4)
      betas (Tuple[float, float], optional): coefficients used for computing
        running averages of gradient and its square (default: (0.9, 0.99))
      weight_decay (float, optional): weight decay coefficient (default: 0)
    """

    if not 0.0 <= lr:
      raise ValueError('Invalid learning rate: {}'.format(lr))
    if not 0.0 <= betas[0] < 1.0:
      raise ValueError('Invalid beta parameter at index 0: {}'.format(betas[0]))
    if not 0.0 <= betas[1] < 1.0:
      raise ValueError('Invalid beta parameter at index 1: {}'.format(betas[1]))
    defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
    super().__init__(params, defaults)

  @torch.no_grad()
  def step(self, closure=None):
    """Performs a single optimization step.

    Args:
      closure (callable, optional): A closure that reevaluates the model
        and returns the loss.

    Returns:
      the loss.
    """
    loss = None
    if closure is not None:
      with torch.enable_grad():
        loss = closure()

    for group in self.param_groups:
      for p in group['params']:
        if p.grad is None:
          continue

        # Perform stepweight decay
        p.data.mul_(1 - group['lr'] * group['weight_decay'])

        grad = p.grad
        state = self.state[p]
        # State initialization
        if len(state) == 0:
          # Exponential moving average of gradient values
          state['exp_avg'] = torch.zeros_like(p)

        exp_avg = state['exp_avg']
        beta1, beta2 = group['betas']

        # Weight update
        update = exp_avg * beta1 + grad * (1 - beta1)
        p.add_(torch.sign(update), alpha=-group['lr'])
        # Decay the momentum running average coefficient
        exp_avg.mul_(beta2).add_(grad, alpha=1 - beta2)

    return loss


def main():    
    # model select
    if args.models in ['efficientnet-b0', 'efficientnet-b1', 'efficientnet-b2', 'efficientnet-b3', 'efficientnet-b4'] :
        from efficientnet_pytorch import EfficientNet
        model = EfficientNet.from_pretrained(args.models, num_classes=args.label_num) # total.csv 라벨 수
        # input_size = 256

    # data augmentation
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(size=args.input_size, scale=(0.8, 1.0)),   
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                p=0.8
            ),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize((args.input_size,args.input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    ])

    # dataset_load
    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)

    batch_size = args.batch

    # train_sampler = torch.utils.data.DistributedSampler(train_dataset, shuffle=True)
    # test_sampler = torch.utils.data.DistributedSampler(test_dataset, shuffle=False)

    dataloaders = {}
    dataloaders['train'] = DataLoader(dataset=train_dataset, 
                                      batch_size=batch_size, 
                                      shuffle=True,
                                      num_workers=12
                                      )
    dataloaders['valid'] = DataLoader(dataset=test_dataset, 
                                      batch_size=batch_size, 
                                      shuffle=False,
                                      num_workers=12
                                      )

    # hyper parameter
    criterion = nn.CrossEntropyLoss()
    optim = Lion(model.parameters(), lr=args.learning_rate)

    # tensorboard
    writer = SummaryWriter(f'runs/{args.models}') # default `log_dir` is "runs" - we'll be more specific here

    # get some random training images
    dataiter = iter(dataloaders['train'])
    images, labels = next(dataiter)
    img_grid = torchvision.utils.make_grid(images) # create grid of images
    writer.add_image(f'{args.models}', img_grid) # write to tensorboard

    writer.add_graph(model, images) # model structure

    # model save folder
    os.makedirs(args.save_path, exist_ok=True) 

    # multi gpu
    if torch.cuda.device_count() > 1:
      print("Let's use", torch.cuda.device_count(), "GPUs!")
      # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
      model = nn.DataParallel(model)

    model.to(device)
    model, best_idx, best_acc, train_loss, train_acc, valid_loss, valid_acc = \
      train_model(args, model, dataloaders, criterion, optim ,  writer , num_epochs=args.epochs)

    writer.close()

if __name__ == "__main__":
    parser = ArgumentParser(description="Training script for ImageClassification",formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--models', default='efficientnet-b0', type=str, 
                        choices= ['efficientnet-b0', 'efficientnet-b1', 'efficientnet-b2', 'efficientnet-b3', 'efficientnet-b4'], 
                        help='Base model name')
    parser.add_argument('--learning-rate', default= 1e-4, type=float, help='base learning rate')
    parser.add_argument('--input-size', default=256, type=int, help='images input size')
    parser.add_argument('--epochs', default=120, type=int, help='epochs')
    parser.add_argument('--batch', default=64, type=int, help='Batch by GPU')
    parser.add_argument('--save-path', default='models', type=str, help='model save path')
    parser.add_argument('--label-num', default=10, type=str, help='label number')
    args = parser.parse_args()
    seed_everything(47)
    main()