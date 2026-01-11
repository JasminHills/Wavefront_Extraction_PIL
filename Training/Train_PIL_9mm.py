# utilities and paths
import sys
import os

# torch imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from collections import OrderedDict

#from train import *
from Training.train import train

# Maths -ey imports
import scipy 
import math
from scipy.ndimage import *
import numpy as np
import json

# Even more utilities
 # utility functions
#Import model and learning rate scheduler 
import torch.optim.lr_scheduler as lr_scheduler
from Utilities.lr_analyzer import *
from Utilities.criterion import * #
from pytorch.models.U1D_dropout import U1_drop
# Import physics informed loss - i.e. laser propagation included in loss function
from pytorch.PIL import PhysicsInformedLoss
#Import training function
from Training.Train_PIL_utility import *
from Utilities.dataset import psf_dataset, splitDataLoader, ToTensor, Normalize, resize  # dataset loader 

data_dir='example_data256/' # Inputs directory
dataset_size=len(os.listdir(data_dir)) # Number of samples in the dataset
dataset = psf_dataset(root_dir = data_dir, 
                      size = dataset_size,
                      transform = transforms.Compose([ Normalize(), ToTensor(), resize()]))# 

# dataset loader with normalisation, tensor conversion and resizing

import matplotlib.pyplot as plt
from Utilities.dataset import psf_dataset, splitDataLoader, ToTensor, Normalize
dataloaders = {}                  
dataloaders['train'], dataloaders['val'] = splitDataLoader(dataset, split=[0.9, 0.1], batch_size=10, random_seed=2)
# for _, sample in enumerate(dataloaders['train']):
#     print(np.shape(sample['image']))
    

# setting all the parameters for training the model 
optimizer_name = "SGD" # Optimizers
lr = 0.0006    # Learning rates
stpsz = 70
lr=0.003
momentum=0.821
stpsz=40
gmma = 0.7
momentum=0.14
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
n_channels_in = 1
n_channels_out = 1
dropout=0.012 # from 0.02
dropout_bottleneck=0.02 # from 0.07

#Defining the model
model = U1_drop(n_channels_in, n_channels_out, dropout_enc=dropout_bottleneck,dropout_bottleneck=dropout_bottleneck,dropout_dec=dropout_bottleneck).to(device)

#Setting up scheudler and optimiser 
optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=stpsz, gamma=gmma)

#Define relative min and max for normalisation, and relative extents
int_max, int_min, phase_min, phase_max = min_max_ext(dataset, dataset_size)

f0=9
criterion=PhysicsInformedLoss(f0, plot=True, device="cuda", int_max=int_max, int_min=int_min, phase_max=phase_max, phase_min=phase_min)

device='cuda'
train(model, 
      dataset, 
      optimizer, 
      criterion, # RMSELoss
      split = [0.90, 0.1], #dataset split
      batch_size=8, #64 - vary depending on GPU memory
      n_epochs = 10000, # maximum epoch to terminate at
      device=device, 
      scheduler=scheduler,
      int_min=int_min, 
      int_max=int_max, # normalisation values
      
      model_dir = 'mod_comp', # Model save directory    
      visdom = False)
# model = nn.DataParallel(model)
