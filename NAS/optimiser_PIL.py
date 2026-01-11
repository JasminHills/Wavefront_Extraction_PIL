
import sys
import os

from pytorch.models.U1D import U1
from pytorch.models.U1D_dropout import U1_drop
from pytorch.models.Flex import U1

#Torch imports
import torch

# maths and science imports
import math

# Optuna + pLotting and visualising results
import optuna
from optuna.trial import TrialState

# Load dataset
from NAS.optimiser_PIL import *
from NAS.objective import *
from NAS.optimiser_utils import clbk

# import torchLASY
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=300, callbacks=[clbk])

# === Print Best Trial ===
# print("\nBest trial:")
# print(f"  Score: {study.best_trial.value:.4f}")
for key, value in study.best_trial.params.items():
    # print(f"  {key}: {value}")