# use best practice
# set up neural networ
# set up loss function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim.lr_scheduler import StepLR
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim.lr_scheduler import CyclicLR
# import mnist
import torch.datasets.mnist as mnist

# load data
from utils.dataset import BasicDataset
# load mnist
dataset = mnist.MNIST('./data', download=True)
# split data into training and validation
train, val = random_split(data, [int(0.8*len(data)), int(0.2*len(data))])
# set up data loaders
train_loader = DataLoader(train, batch_size=128, shuffle=True, num_workers=8, pin_memory=True)
val_loader = DataLoader(val, batch_size=128, shuffle=False, num_workers=8, pin_memory=True, drop_last=True)

def train():
    # set up neural network
    net = UNet(n_channels=1, n_classes=3, bilinear=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net.to(device=device)
    # set up loss function
    criterion = nn.CrossEntropyLoss()
    # set up optimizer
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    # set up scheduler
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=10)
    # set up tensorboard
    writer = SummaryWriter(comment=f'LR_0.001_Batch_size_128')
    global_step = 0
    # set up training loop
    for epoch in range(500):
        net.train()
        epoch_loss = 0
        for i, (image, mask) in enumerate(train_loader):
            image = image.to(device=device, dtype=torch.float32)
            mask = mask.to(device=device, dtype=torch.long)
            # forward pass