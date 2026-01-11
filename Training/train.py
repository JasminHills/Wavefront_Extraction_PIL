from Utilities.datasetNw import psf_dataset, splitDataLoader, ToTensor, Normalize
from Utilities.dataset import psf_dataset, splitDataLoader, ToTensor, Normalize
import logging
import json
# torch imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from collections import OrderedDict
from torch.cuda.amp import autocast, GradScaler

from Training.Train_PIL_utility import normalize_to

import os
import Utilities.logging_utils as utils
import time
import numpy as np

def train(model, dataset, optimizer, criterion, device, scheduler, split=[0.9, 0.1], batch_size=32, 
          n_epochs=1, model_dir='./', int_min=0, int_max=1e23, random_seed=None, visdom=False): #, scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=155, gamma=0.85)):

    # Create directory if doesn't exist
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    # Logging
    log_path = os.path.join(model_dir, 'logs.log')
    utils.set_logger(log_path)
    
    # Dataset
    dataloaders = {}                  
    dataloaders['train'], dataloaders['val'] = splitDataLoader(dataset, split=split, batch_size=batch_size, random_seed=random_seed)
    
    # Metrics
    # print('Metrics Here')
    metrics_path = os.path.join(model_dir, 'metrics.json')
    #Items to save in this particular training run
    metrics = {
        'model': model_dir,
        'optimizer': optimizer.__class__.__name__,
        'criterion': criterion.__class__.__name__,
        'scheduler': scheduler.__class__.__name__,
        'dataset_size': int(len(dataset)),
        'train_size': int(split[0]*len(dataset)),
        'test_size': int(split[1]*len(dataset)),
        'n_epoch': n_epochs,
        'batch_size': batch_size,
        'learning_rate': [],
        'train_loss': [],
        'val_loss': [],
        'train_phs_loss': [],
        'val_phs_loss': [],
        'train_int_loss': [],
        'train_nm_loss': [],
        'val_nm_loss': [],
        'val_int_loss': [],
        'zernike_train_loss': [],
        'zernike_val_loss': []
    }
    
    # Device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    torch.autograd.set_detect_anomaly(True)

    # Training
    since = time.time()
    dataset_size = {
        'train':int(split[0]*len(dataset)),
        'val':int(split[1]*len(dataset))
    }
    
    best_loss = 0.0

    for epoch in range(n_epochs):
        epoch_time = time.time()

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
#             print('here')
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode
            #Reset Loss values
            running_loss = 0.0
            zernike_loss =0.0
            current_loss=0.0
            train_loss=100.0

            for _, sample in enumerate(dataloaders[phase]):
                #print(np.shape(sample['image']))
                # print(np.shape(sample['image']))
                inputs = sample['image'][:, 1:,:,].to(device) #, nan=1e-3) #[:, 1:,:,]
                # print(np.shape(inputs))
                phase_0 = sample['phase'].to(device) #, nan=1e-3)
                optimizer.zero_grad(set_to_none=True)
                with torch.set_grad_enabled(phase == 'train'):
                       nwIn=inputs #changed #[:, 1:,:,]
                       
                       with autocast():
                           norm = (nwIn - torch.min(nwIn)) / (torch.max(nwIn) - torch.min(nwIn))
                           # print(np.shape(nwIn))
                           phase_estimation = model(normalize_to(nwIn, int_min, int_max))
                           rgg=torch.squeeze(phase_estimation)

                           #Compare values to ground truth with criterion PIL loss function - includes all the propagation physics
                           loss =  criterion(rgg, nwIn,phase_true=phase_0, lambda_phase=0)  #How much of phase and intensity to include                        
                           phse_loss=criterion.phse_loss
                           int_loss=criterion.int_loss
                           int_nm_loss=criterion.int_normalized_intensity
                           logging.info(' where loss and that: %f' % (loss.item()))   

                       if loss.item()!=np.nan:
                             current_loss += 1 * abs(loss.item()) #*1/inputs[0].size(0)
                             logging.info(' where train: %f' % (current_loss))                     
                           
                           
                       if phase == 'train':
                            loss.backward()
                            optimizer.step()
                            train_loss=current_loss/ batch_size
            

            if phase == 'val':
                val_loss=current_loss/batch_size

                with torch.no_grad():
                    scheduler.step()
                    model_path = os.path.join(model_dir, 'modelfinal.pth')
                    torch.save(model.to('cpu').state_dict(), model_path)
                    model.to(device)

                    if epoch == 0 or current_loss < best_loss:
                        best_loss = current_loss
                        model_path = os.path.join(model_dir, 'model.pth')
                        
                        torch.save(model.to('cpu').state_dict(), model_path)
                        model.to(device)
                        
                    # Save metrics
                    with open(metrics_path, 'w') as f:
                        json.dump(metrics, f, indent=4) 
            
            metrics[phase+'_loss'].append(current_loss)
            metrics[phase+'_int_loss'].append(float(int_loss))
            metrics[phase+'_phs_loss'].append(float(phse_loss))
            metrics[phase+'_nm_loss'].append(float(int_nm_loss))
            
            
        logging.info(' where val: %f %f' % (current_loss, dataset_size[phase]))                    
                   
        logging.info('[%i/%i] Time: %f s' % (epoch + 1, n_epochs, time.time()-epoch_time))
    
        
        if phase=='train':
                metrics['learning_rate'].append(get_lr(optimizer))
        logging.info(' where val: %f %f' % (current_loss, dataset_size[phase]))                    
#       
    model_path = os.path.join(model_dir, 'modelfinal.pth')
    torch.save(model.to('cpu').state_dict(), model_path)
    model.to(device)
    torch.cuda.empty_cache()
        
        
    time_elapsed = time.time() - since    
    logging.info('[-----] All epochs completed in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    
  