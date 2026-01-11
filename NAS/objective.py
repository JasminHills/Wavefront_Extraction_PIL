from Utilities.dataset import psf_dataset, splitDataLoader, ToTensor, Normalize, resize

import os
from torchvision import transforms
import time
import torch
import pytorch.PIL as torchLASY_V4

from pytorch.models.U1D_dropout import U1_drop

import numpy as np
import optuna
from optuna.trial import TrialState
import torch.optim as optim
from Training.Train_PIL_utility import normalize_to
from NAS.optim_train_func import train

def objective(trial):
    n_channels_in = 1
    n_channels_out = 1
    random_seed = 1
    data_dir='example_data256/'
    len_training=len(os.listdir(data_dir))
#     print(len1)
    n_epochs=100
    device="cuda"

  
    dataset = psf_dataset(root_dir = data_dir, 
                          size = len_training,
                          transform = transforms.Compose([Normalize(), ToTensor(), resize()]))

#     model.to(device)
    # Training
    split=[0.9, 0.1]
    since = time.time()
    dataset_size = { 'train':int(split[0]*len(dataset)),'val':int(split[1]*len(dataset))}
    f0=9
    p_min=[]
    p_max=[]
    i_min=[]
    i_max=[]
    for di in range(len_training):
        d=dataset.__getitem__(di)
        p_max.append(np.max(d['phase']))
        p_min.append(np.min(d['phase']))
        i_max.append(float((d['image'][0]).max().numpy()))
        i_min.append(float((d['image'][0]).min().numpy()))

    int_max=np.max(i_max)
    int_min=np.min(i_min)
    phase_min=np.min(p_min)
    phase_max=np.max(p_max)
    # print(int_min, int_max, phase_min, phase_max)

    criterion=torchLASY_V4.PhysicsInformedLoss(f0, plot=False, device="cuda", int_max=int_max, int_min=int_min, phase_max=phase_max, phase_min=phase_min)

    momentum=trial.suggest_float("momentum", 0.1, 0.9, log=True) 
    # --- Optimizer & learning rate ---
    optimizer_name = "SGD"  # Fixed for now; tune later once architecture stabilizes
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)  # narrower, more realistic for SGD

    # --- Train/val split ---
    split = [0.9, 0.1]  # keep fixed; avoid wasting trials here

    # --- Batch size ---
    batch = 32  # fixed (don’t tune early unless you have plenty of GPU headroom)

    # --- Scheduler params ---
    stpsz = trial.suggest_int("stpsz", 50, 180)  # smaller range, 3x fewer possibilities
    gmma = trial.suggest_float("gmma", 0.4, 0.95)  # smaller lower bound, more practical decay

    # --- Network architecture ---
#     n_layers = trial.suggest_int("n_layers", 3, 4)

#     kern_down = trial.suggest_categorical("kern_down", [3])  # fix one kernel size first
#     kern_up = trial.suggest_categorical("kern_up", [3])
#     kerns = [kern_down, kern_up]

#     bilinear = trial.suggest_categorical("bilinear", [True, False])
#     dropout = trial.suggest_float("dropout", 0.0, 0.25)  # slightly tighter range

    # --- Channel widths ---
    # Use a single rule for channel progression instead of full combinatorial explosion
#     base_ch = trial.suggest_categorical("base_ch", [32, 64])
#     downs = [base_ch * (2 ** i) for i in range(n_layers)]  # e.g. [64,128,256,...]

    # Mirror ups
#     ups = downs[::-1]
    dropout_bottleneck = trial.suggest_float("dropout_bottleneck", 0.0, 0.6)  # smaller lower bound, more practical decay
#     dropout_dec = trial.suggest_float("dropout_dec", 0.0, 0.6)  # smaller lower bound, more practical decay
#     dropout_enc = trial.suggest_float("dropout_enc", 0.0, 0.6)
#     cos_dec = trial.suggest_categorical("cos_dec", [True, False])
#     if cos_dec:
        
    
#     lmda_ph = trial.suggest_float("lmda_ph", 0.0, 0.8) 

        # smaller lower bound, more practical decay



    # Create model

    model = U1_drop(1, 1, dropout_enc=dropout_bottleneck,dropout_bottleneck=dropout_bottleneck,dropout_dec=dropout_bottleneck).to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    
#     if optimizer_name=="Adam":
#          optimizer = optim.Adam(model.parameters(), lr=lr)
#     if optimizer_name=="SGD":
# #          mom = trial.suggest_float("mom", 0.1, 0.99)     # Learning rates
#          optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    model_dir='opt'

    # Train + evaluate
    vl_mets=[]
    for n in range(n_epochs):
      model, met=train(model, model_dir, dataset, optimizer, criterion, split=[0.9, 0.1], batch_size=16, stpsz=stpsz, gmma=gmma, epoch=0, int_min=int_min, int_max=int_max,lmda_ph=0.0)
      vl_mets.append(met.to('cpu'))
#     print(torch.cuda.memory_summary())
    # Delete all variables you no longer need
    del model
#     del inputs
#     del outputs
#     torch.cuda.empty_cache()
#     torch.cuda.ipc_collect()
    return np.mean(vl_mets[-5:])  # e.g., Dice or IoU

