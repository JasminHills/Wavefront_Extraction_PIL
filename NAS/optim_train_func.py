import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.cuda.amp import autocast, GradScaler
from torchvision import transforms
from collections import OrderedDict

from Utilities.dataset import psf_dataset, splitDataLoader, ToTensor, Normalize, resize
from Training.Train_PIL_utility import normalize_to 
import numpy as np
import os




def train(model, model_dir, dataset, optimizer, criterion, split=[0.9, 0.1], batch_size=32, stpsz=150, gmma=0.1, epoch=0, int_min=0, int_max=1e23, lmda_ph=0.1): #, scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=155, gamma=0.85)):
    #(model, model_dir, dataset, optimizer, criterion, split=[0.9, 0.1], batch_size=32, stpsz=150, gmma=0.1, epoch=0,random_seed=None, visdom=False)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # print(device)
    random_seed = 1
    # Create directory if doesn't exist
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    # Logging
    log_path = os.path.join(model_dir, 'logs.log')
    
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=stpsz, gamma=gmma)
    # Dataset
    dataloaders = {}
    dataloaders['train'], dataloaders['val'] = splitDataLoader(dataset, split=split, 
                                                             batch_size=batch_size, random_seed=random_seed)

    
    best_loss = 0.0
    scaler = GradScaler()
    val_loss=10
    # Each epoch has a training and validation phase
    for phase in ['train', 'val']:
        if phase == 'train':
            model.train()  # Set model to training mode
        else:
            model.eval()   # Set model to evaluate mode

        running_loss = 0.0
        zernike_loss =0.0
        current_loss=0.0
        train_loss=100.0
        for _, sample in enumerate(dataloaders[phase]):
            

#             print(np.shape(sample['image']))
            # print(sample['image'].shape)
            # print(sample['phase'].shape)
            inputs = sample['image'][:, 1:,:,].to(device) #.unsqueeze(0)
            phase_0 = sample['phase'].to(device)#.unsqueeze(0)
            #             print(np.shape(sample['image']))
#             print(inputs.shape)
#             print(phase_0.shape)


            if (not inputs[0].isnan().any()):
              

               # zero the parameter gradients
               optimizer.zero_grad()
               
               #logging.info(' individual loss: %f %f' % (phase_0, phase_estimation))
            
               # forward: track history if only in train
               with torch.set_grad_enabled(phase == 'train'):

                   # Network return phase and zernike coeffs
                   nwIn=inputs#changed #[:, 1:,:,]
                   
                   with autocast():

                       phase_estimation = model(normalize_to(nwIn, int_min, int_max))
                  
                       rgg=phase_estimation.squeeze(1)
                       # print('phase_est', rgg.shape)
                       # print('nwIn', nwIn.shape)
                       # print('phase_0', phase_0.shape)
                       loss =  criterion(rgg, nwIn,phase_true=phase_0, lambda_phase=lmda_ph)
                          
                       phse_loss=criterion.phse_loss
                       int_loss=criterion.int_loss
#                        loss = criterion(torch.squeeze(phase_estimation), phase_0) #+l1_regularizer(model,lambda_l1=lambda_l1)+orth_regularizer(model, device, lambda_orth=lambda_orth)  
                       # print(loss)
                   if loss.item()!=np.nan:
                         #running_loss += 1 * loss.item() * inputs[0].size(0)
                         current_loss += 1 * abs(loss.item())*1 # * inputs[0].size(0)
#                          print('either', current_loss/batch_size)
                       
                       
                   if phase == 'train':
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update() 
                        train_loss=current_loss / batch_size

        
        # Adaptive learning rate
        if phase == 'val':
            val_loss=current_loss / batch_size
            int_val_loss=int_loss/ batch_size

    
    del dataloaders
    del inputs
    del phase_0
    del phase_estimation
    return model, int_val_loss #+train_loss
