import torch

import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter


class Discriminator(nn.Module)
    def __init__(self,img):
        super(Discriminator,self).__init__()
        self.disc=nn.Sequential(
            nn.Linear(img,128),
            nn.LeakyReLU(0.1),
            nn.Linear(128,1),
            nn.Sigmoid()
        )

    def forward(self,x):
        return self.disc(x)


class Generator(nn.Module)
    def __init__(self,z_dim,img):
        super(Generator,self).__init__()
        self.gen=nn.Sequential(
            nn.Linear(z_dim,256),
            nn.LeakyReLU(0.1),
            nn.Linear(256,img),
            nn.Tanh()
        )

    def forward(self,x):
        return self.gen(x)

